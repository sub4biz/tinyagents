//! Steering a running detached subagent, and cancelling subagents by the
//! thread that spawned them.
//!
//! [`steer_detached`] delivers an injected message through the process-local
//! steering handle when the child run registered one, and otherwise hands the
//! message to a host-supplied fallback (a host run queue, say). The host keeps
//! the queue type; this module only owns the lane rules, the ownership /
//! terminal checks and the message framing.

use std::future::Future;

use tinyagents_harness::ids::TaskId;
use tinyagents_harness::run_queue::QueueLane;
use tinyagents_harness::steering::{RecentRequestIds, SteeringCommand, SteeringHandle};
use tinyagents_tasks::{CancelledDetachedTask, DetachedTaskRegistry, DetachedTaskRegistryError};
use tinyinference_llm::message::Message;

use super::types::{DetachedSubagentStatus, SubagentIdentity, WaitError};

/// Why a steer could not be delivered.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteerError {
    /// No such subagent: never existed, or already finished and pruned.
    Unknown,
    /// The caller does not own this subagent.
    NotOwned,
    /// The subagent already reached a terminal status.
    AlreadyDone,
    /// Detached subagents only accept an injected instruction or collected
    /// context; follow-up work cannot be dispatched through a steer.
    UnsupportedLane,
    /// The `request_id` exceeds [`RecentRequestIds::MAX_REQUEST_ID_BYTES`].
    RequestIdTooLong,
}

impl From<DetachedTaskRegistryError> for SteerError {
    fn from(error: DetachedTaskRegistryError) -> Self {
        match error {
            DetachedTaskRegistryError::NotOwned => Self::NotOwned,
            DetachedTaskRegistryError::AlreadyDone => Self::AlreadyDone,
            _ => Self::Unknown,
        }
    }
}

/// Which path delivered a steer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteerRoute {
    /// The child run's live steering handle.
    Registry,
    /// The host fallback (the child had no live handle).
    Fallback,
}

/// Result of an idempotent steer ([`steer_detached_with_request_id`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SteerReceipt {
    /// The path that delivered the steer; `None` when `duplicate` is set,
    /// because nothing was delivered this time.
    pub route: Option<SteerRoute>,
    /// The `request_id` was already applied to this task, so the steer was
    /// acknowledged without being enqueued again.
    pub duplicate: bool,
}

/// Whose authority a steer is made under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteerAccess<'a> {
    /// Only the owning parent session may steer.
    Owner(&'a str),
    /// A trusted control surface: no ownership check.
    Trusted,
}

/// Stable lane name for logs (`steer` / `followup` / `collect`).
pub fn queue_lane_name(lane: QueueLane) -> &'static str {
    match lane {
        QueueLane::Steer => "steer",
        QueueLane::Followup => "followup",
        QueueLane::Collect => "collect",
    }
}

/// The steering command a lane maps to; `None` for a lane a detached subagent
/// does not accept.
pub fn steering_command_for_lane(lane: QueueLane, text: &str) -> Option<SteeringCommand> {
    match lane {
        QueueLane::Steer => Some(SteeringCommand::InjectMessage(Message::user(format!(
            "[User steering message]: {text}"
        )))),
        QueueLane::Collect => Some(SteeringCommand::InjectMessage(Message::user(format!(
            "[Additional context from user]: {text}"
        )))),
        QueueLane::Followup => None,
    }
}

fn send_registered(handle: &SteeringHandle, text: &str, lane: QueueLane) -> bool {
    match steering_command_for_lane(lane, text) {
        Some(command) => {
            handle.send(command);
            true
        }
        None => false,
    }
}

/// Inject `text` into the running subagent `task_id`.
///
/// Rejects an unsupported lane, an unknown / unowned / already-terminal task,
/// then prefers the live steering handle and falls back to
/// `fallback(metadata, lane, text)` when there is none. The text is never
/// logged here.
pub async fn steer_detached<M, F, Fut>(
    registry: &DetachedTaskRegistry<M, DetachedSubagentStatus>,
    task_id: &str,
    access: SteerAccess<'_>,
    text: String,
    lane: QueueLane,
    fallback: F,
) -> Result<SteerRoute, SteerError>
where
    M: Clone + Send + Sync + 'static,
    F: FnOnce(M, QueueLane, String) -> Fut,
    Fut: Future<Output = ()>,
{
    let receipt =
        steer_detached_with_request_id(registry, task_id, access, text, lane, None, fallback)
            .await?;
    Ok(receipt
        .route
        .expect("a steer without a request id is never a duplicate"))
}

/// Idempotent [`steer_detached`]: when `request_id` was already applied to
/// `task_id`, returns a [`SteerReceipt`] with `duplicate: true` and delivers
/// nothing. The last [`tinyagents_harness::steering::RecentRequestIds::DEFAULT_CAPACITY`] ids per task are
/// remembered. Validation (lane, unknown / unowned / terminal) runs first, so
/// a rejected steer never consumes its id and can be retried. A `request_id`
/// longer than [`RecentRequestIds::MAX_REQUEST_ID_BYTES`] is rejected with
/// [`SteerError::RequestIdTooLong`].
///
/// # Delivery semantics
///
/// The id is claimed *before* delivery, so a steer is delivered **at most
/// once** per `request_id`: if delivery is interrupted after the claim (the
/// future is dropped, or the host fallback fails to enqueue), a retry with the
/// same id is acknowledged as a duplicate and is not re-delivered. A caller
/// that needs retry-until-delivered must use a fresh `request_id`.
pub async fn steer_detached_with_request_id<M, F, Fut>(
    registry: &DetachedTaskRegistry<M, DetachedSubagentStatus>,
    task_id: &str,
    access: SteerAccess<'_>,
    text: String,
    lane: QueueLane,
    request_id: Option<&str>,
    fallback: F,
) -> Result<SteerReceipt, SteerError>
where
    M: Clone + Send + Sync + 'static,
    F: FnOnce(M, QueueLane, String) -> Fut,
    Fut: Future<Output = ()>,
{
    if !matches!(lane, QueueLane::Steer | QueueLane::Collect) {
        return Err(SteerError::UnsupportedLane);
    }
    let key = TaskId::new(task_id);
    let snapshot = match access {
        SteerAccess::Owner(owner) => registry.snapshot(&key, owner)?,
        SteerAccess::Trusted => registry.snapshot_trusted(&key)?,
    };
    if snapshot.status.is_terminal() {
        return Err(SteerError::AlreadyDone);
    }
    if request_id.is_some_and(|id| id.len() > RecentRequestIds::MAX_REQUEST_ID_BYTES) {
        return Err(SteerError::RequestIdTooLong);
    }
    if let Some(request_id) = request_id
        && !registry.claim_steer_request(&key, request_id)?
    {
        tracing::debug!("[subagent-steer] duplicate request_id task_id={task_id}");
        return Ok(SteerReceipt {
            route: None,
            duplicate: true,
        });
    }
    let handle = match access {
        SteerAccess::Owner(owner) => registry.steering_handle(&key, owner),
        SteerAccess::Trusted => registry.steering_handle_trusted(&key),
    };
    if handle
        .map(|handle| send_registered(&handle, &text, lane))
        .unwrap_or(false)
    {
        return Ok(SteerReceipt {
            route: Some(SteerRoute::Registry),
            duplicate: false,
        });
    }
    fallback(snapshot.metadata, lane, text).await;
    Ok(SteerReceipt {
        route: Some(SteerRoute::Fallback),
        duplicate: false,
    })
}

/// Abort every registered subagent whose [`SubagentIdentity::parent_thread_id`]
/// is `thread_id`, returning the cancelled entries so the host can settle its
/// own state (ledger rows, durable sessions, notices).
pub fn cancel_for_thread<M>(
    registry: &DetachedTaskRegistry<M, DetachedSubagentStatus>,
    thread_id: &str,
) -> Result<Vec<CancelledDetachedTask<M, DetachedSubagentStatus>>, WaitError>
where
    M: SubagentIdentity + Clone + Send + Sync + 'static,
{
    Ok(registry.cancel_where(|metadata| metadata.parent_thread_id() == Some(thread_id))?)
}

/// The distinct parent thread ids of `cancelled`, in first-seen order. Entries
/// with no parent thread (headless spawns) contribute none.
pub fn distinct_parent_threads<M>(
    cancelled: &[CancelledDetachedTask<M, DetachedSubagentStatus>],
) -> Vec<String>
where
    M: SubagentIdentity,
{
    let mut out: Vec<String> = Vec::new();
    for entry in cancelled {
        if let Some(thread) = entry.metadata.parent_thread_id()
            && !out.iter().any(|seen| seen == thread)
        {
            out.push(thread.to_string());
        }
    }
    out
}
