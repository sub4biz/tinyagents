use std::sync::Arc;

use tinyagents_harness::ids::TaskId;
use tinyagents_tasks::{
    OrchestrationTaskFilter, OrchestrationTaskKind, OrchestrationTaskRecord,
    OrchestrationTaskResult, OrchestrationTaskSpec, OrchestrationTaskStatus, TaskStore,
};
use tokio::sync::watch;

use super::types::{DetachedSubagentStatus, SpawnedSubagent, WaitError, WaitOutcome};

/// Metadata-only timeout mirrored into the ledger spec. Matches a waiter's
/// default window; execution is governed by the detached task itself.
pub const DETACHED_LEDGER_TIMEOUT_MS: u64 = 120_000;

/// How many times the status watcher tries to persist a terminal status.
pub const STATUS_WRITE_ATTEMPTS: usize = 3;

fn record_status_with_retries(
    store: &dyn TaskStore,
    task_id: &str,
    status: &DetachedSubagentStatus,
) {
    for _ in 0..STATUS_WRITE_ATTEMPTS {
        if record_status(store, task_id, status).is_ok() {
            return;
        }
    }
}

/// Record a freshly-spawned subagent in `store` (`Pending` then `Running`).
///
/// An insert failure (e.g. a task id still present in the durable store) is
/// returned and the record is left untouched, so the caller can stop the
/// spawn instead of advancing a stale record.
pub fn record_spawned(
    store: &dyn TaskStore,
    spawned: &SpawnedSubagent<'_>,
) -> tinyagents_harness::Result<()> {
    let root_run_id = spawned
        .session_parent_prefix
        .and_then(|prefix| prefix.split("__").next())
        .filter(|root| !root.is_empty())
        .unwrap_or(spawned.parent_session);
    let mut spec = OrchestrationTaskSpec::new(
        spawned.task_id.to_string(),
        OrchestrationTaskKind::SubAgent {
            agent: spawned.agent_id.to_string(),
        },
    )
    .with_lineage(spawned.parent_session.to_string(), root_run_id.to_string())
    .with_timeout_ms(DETACHED_LEDGER_TIMEOUT_MS)
    .with_metadata("parentSession", spawned.parent_session.to_string())
    .with_metadata("rootSession", root_run_id.to_string())
    .with_metadata(
        "defaultWaitTimeoutMs",
        DETACHED_LEDGER_TIMEOUT_MS.to_string(),
    )
    .with_metadata("workspaceDir", spawned.workspace_dir.to_string());
    if let Some(prefix) = spawned.session_parent_prefix {
        spec = spec.with_metadata("sessionParentPrefix", prefix.to_string());
    }
    if let Some(thread) = spawned.parent_thread_id {
        spec = spec
            .with_thread(thread.to_string())
            .with_metadata("parentThreadId", thread.to_string());
    }
    if let Some(session) = spawned.subagent_session_id {
        spec = spec.with_metadata("subagentSessionId", session.to_string());
    }
    store.insert(spec)?;
    store.mark_running(&TaskId::new(spawned.task_id))?;
    Ok(())
}

/// Mirror a published status into the store.
///
/// First writer wins: when the record is already terminal (or gone) the call
/// is a no-op. Any other store failure, such as a durable backend that cannot
/// persist the transition, is returned so the caller can retry or report it
/// instead of losing the terminal result.
pub fn record_status(
    store: &dyn TaskStore,
    task_id: &str,
    status: &DetachedSubagentStatus,
) -> tinyagents_harness::Result<()> {
    let id = TaskId::new(task_id);
    if matches!(status, DetachedSubagentStatus::Running) {
        return Ok(());
    }
    match store.get(&id) {
        None => return Ok(()),
        Some(record) if record.is_terminal() => return Ok(()),
        Some(_) => {}
    }
    match status {
        DetachedSubagentStatus::Completed { output, .. } => {
            store.complete(&id, OrchestrationTaskResult::text(output.clone()))?;
        }
        DetachedSubagentStatus::Failed { error } => {
            store.fail(&id, error.clone())?;
        }
        DetachedSubagentStatus::AwaitingUser { question } => {
            store.mark_awaiting_with_question(&id, question.clone())?;
        }
        DetachedSubagentStatus::Running => {}
    }
    Ok(())
}

/// Record a cancellation (`CancelRequested` then `Cancelled`). Store failures
/// are returned; a record that is already terminal (or gone) is a no-op.
pub fn record_cancelled(store: &dyn TaskStore, task_id: &str) -> tinyagents_harness::Result<()> {
    let id = TaskId::new(task_id);
    match store.get(&id) {
        None => return Ok(()),
        Some(record) if record.is_terminal() => return Ok(()),
        Some(_) => {}
    }
    store.request_cancel(&id)?;
    store.mark_cancelled(&id)?;
    Ok(())
}

/// Watch a child's status channel and mirror its first terminal status into
/// `store`. A dropped sender without a terminal status is recorded as
/// [`DetachedSubagentStatus::ended_without_result`]. A failed write is retried
/// up to [`STATUS_WRITE_ATTEMPTS`] times before the watcher gives up; hosts
/// that must observe persistence failures call [`record_status`] themselves.
pub fn spawn_status_watcher(
    store: Arc<dyn TaskStore>,
    task_id: String,
    mut status: watch::Receiver<DetachedSubagentStatus>,
) {
    tokio::spawn(async move {
        loop {
            let snapshot = status.borrow_and_update().clone();
            if snapshot.is_terminal() {
                record_status_with_retries(store.as_ref(), &task_id, &snapshot);
                break;
            }
            if status.changed().await.is_err() {
                record_status_with_retries(
                    store.as_ref(),
                    &task_id,
                    &DetachedSubagentStatus::ended_without_result(),
                );
                break;
            }
        }
    });
}

/// Every subagent record in `store`.
pub fn list_subagent_records(store: &dyn TaskStore) -> Vec<OrchestrationTaskRecord> {
    store.list(OrchestrationTaskFilter::default().with_kind("sub_agent"))
}

/// The `parentSession` a record was spawned under.
pub fn record_parent_session(record: &OrchestrationTaskRecord) -> Option<&str> {
    record
        .spec
        .metadata
        .get("parentSession")
        .map(String::as_str)
}

/// The durable `subagentSessionId` a record carries.
pub fn record_subagent_session_id(record: &OrchestrationTaskRecord) -> Option<&str> {
    record
        .spec
        .metadata
        .get("subagentSessionId")
        .map(String::as_str)
}

/// The worker type of a record (`subagent` for a non-subagent kind).
pub fn record_agent_id(record: &OrchestrationTaskRecord) -> String {
    match &record.spec.kind {
        OrchestrationTaskKind::SubAgent { agent } => agent.clone(),
        _ => "subagent".to_string(),
    }
}

/// Fetch the subagent record for `task_id`, enforcing parent-session ownership.
pub fn subagent_record_for_task(
    store: &dyn TaskStore,
    task_id: &str,
    parent_session: &str,
) -> Result<OrchestrationTaskRecord, WaitError> {
    let Some(record) = store.get(&TaskId::new(task_id)) else {
        return Err(WaitError::Unknown);
    };
    if !matches!(record.spec.kind, OrchestrationTaskKind::SubAgent { .. }) {
        return Err(WaitError::Unknown);
    }
    if record_parent_session(&record) != Some(parent_session) {
        return Err(WaitError::NotOwned);
    }
    Ok(record)
}

/// Read a durable record back as the wait outcome a live run would have given.
pub fn record_to_wait_outcome(record: OrchestrationTaskRecord) -> WaitOutcome {
    use DetachedSubagentStatus as S;
    match record.status {
        OrchestrationTaskStatus::Completed => {
            let output = record
                .result
                .and_then(|result| {
                    result
                        .text
                        .or_else(|| result.output.map(|output| output.to_string()))
                })
                .unwrap_or_default();
            WaitOutcome::Terminal(S::Completed {
                output,
                iterations: 0,
            })
        }
        OrchestrationTaskStatus::Awaiting => WaitOutcome::Terminal(S::AwaitingUser {
            question: record.error.unwrap_or_else(|| {
                "sub-agent is awaiting user input; no clarification text was available from the durable task store".to_string()
            }),
        }),
        OrchestrationTaskStatus::Failed
        | OrchestrationTaskStatus::TimedOut
        | OrchestrationTaskStatus::Abandoned => WaitOutcome::Terminal(S::Failed {
            error: record.error.unwrap_or_else(|| {
                format!(
                    "sub-agent reached durable task status `{}`",
                    task_status_label(record.status)
                )
            }),
        }),
        OrchestrationTaskStatus::Cancelled => WaitOutcome::Terminal(S::Failed {
            error: "sub-agent was cancelled".to_string(),
        }),
        OrchestrationTaskStatus::Pending
        | OrchestrationTaskStatus::Running
        | OrchestrationTaskStatus::CancelRequested => WaitOutcome::TimedOut(S::Running),
    }
}

/// Stable snake_case label for a durable task status.
pub fn task_status_label(status: OrchestrationTaskStatus) -> &'static str {
    match status {
        OrchestrationTaskStatus::Pending => "pending",
        OrchestrationTaskStatus::Running => "running",
        OrchestrationTaskStatus::Awaiting => "awaiting",
        OrchestrationTaskStatus::Completed => "completed",
        OrchestrationTaskStatus::Failed => "failed",
        OrchestrationTaskStatus::CancelRequested => "cancel_requested",
        OrchestrationTaskStatus::Cancelled => "cancelled",
        OrchestrationTaskStatus::TimedOut => "timed_out",
        OrchestrationTaskStatus::Abandoned => "abandoned",
    }
}

/// The reason an orphaned subagent record is settled with. Built in one place
/// because it is written both into the store and into the lifecycle event.
pub fn orphaned_subagent_reason(prior_status: OrchestrationTaskStatus) -> String {
    format!(
        "sub-agent orphaned by core restart (was `{}`)",
        task_status_label(prior_status)
    )
}
