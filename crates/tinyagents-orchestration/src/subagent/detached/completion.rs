//! Recording a detached subagent's terminal status with the completion router.
//!
//! The driver records children it runs itself. A host that tracks detached
//! children through a status channel (see [`spawn_status_watcher`]) uses
//! [`spawn_status_watcher_with_completions`] to get the same durable push:
//! the first terminal status is mirrored into the [`TaskStore`] and then
//! recorded with the router.
//!
//! [`spawn_status_watcher`]: super::spawn_status_watcher

use std::sync::Arc;

use tinyagents_harness::ids::TaskId;
use tinyagents_tasks::{
    CompletionRecord, CompletionResult, CompletionRouter, CompletionStatus, NotifyMode,
    OrchestrationTaskStatus, TaskStore,
};
use tokio::sync::watch;

use super::ledger::record_status_with_retries;
use super::types::DetachedSubagentStatus;

const LOG_PREFIX: &str = "[detached-completion]";

/// Where a detached child's completion goes and how it announces itself.
#[derive(Clone)]
pub struct DetachedCompletionTarget {
    /// The router to record with.
    pub router: Arc<CompletionRouter>,
    /// The parent key (a thread id or session key that survives a restart).
    pub parent_key: String,
    /// The child's agent id.
    pub agent_id: String,
    /// Optional human label.
    pub label: Option<String>,
    /// How the parent is told.
    pub notify_mode: NotifyMode,
}

impl DetachedCompletionTarget {
    /// A target with the default [`NotifyMode`] and no label.
    pub fn new(
        router: Arc<CompletionRouter>,
        parent_key: impl Into<String>,
        agent_id: impl Into<String>,
    ) -> Self {
        Self {
            router,
            parent_key: parent_key.into(),
            agent_id: agent_id.into(),
            label: None,
            notify_mode: NotifyMode::default(),
        }
    }

    /// Sets the human label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the notify mode.
    pub fn with_notify_mode(mut self, mode: NotifyMode) -> Self {
        self.notify_mode = mode;
        self
    }

    /// The completion for `status`, or `None` for a status that is not a
    /// completion: still running, or paused on a question (the same task
    /// completes later, so a pause must not occupy its dedupe slot).
    pub fn record_for(
        &self,
        task_id: &str,
        status: &DetachedSubagentStatus,
    ) -> Option<CompletionRecord> {
        let (state, text) = match status {
            DetachedSubagentStatus::Completed { output, .. } => {
                (CompletionStatus::Success, output.clone())
            }
            DetachedSubagentStatus::Failed { error } => (CompletionStatus::Failed, error.clone()),
            DetachedSubagentStatus::Running | DetachedSubagentStatus::AwaitingUser { .. } => {
                return None;
            }
        };
        let mut record = CompletionRecord::new(
            task_id,
            self.parent_key.clone(),
            self.agent_id.clone(),
            state,
            CompletionResult::text(text),
        )
        .with_notify_mode(self.notify_mode);
        record.label = self.label.clone();
        Some(record)
    }
}

/// Records the completion `status` stands for, if it is one. A router failure
/// is logged and returned as `false`.
pub async fn record_detached_completion(
    target: &DetachedCompletionTarget,
    task_id: &str,
    status: &DetachedSubagentStatus,
) -> bool {
    let Some(record) = target.record_for(task_id, status) else {
        tracing::debug!(
            "{LOG_PREFIX} task_id={task_id} status={} is not a completion",
            status.label()
        );
        return true;
    };
    match target.router.record(record).await {
        Ok(outcome) => {
            tracing::debug!("{LOG_PREFIX} task_id={task_id} outcome={outcome:?}");
            true
        }
        Err(error) => {
            tracing::warn!("{LOG_PREFIX} task_id={task_id} could not record completion: {error}");
            false
        }
    }
}

/// Like [`spawn_status_watcher`](super::spawn_status_watcher), and additionally
/// records the first final status with `target`'s router after it is mirrored
/// into `store`.
///
/// A pause (`AwaitingUser`) is mirrored into the store and the watcher keeps
/// waiting, so a resumed child that reports on the same channel is still
/// recorded when it finishes. A dropped sender is a failed completion unless
/// the ledger already shows a requested or recorded cancellation: the parent
/// asked for that, so it is not pushed back to it. (A host that aborts a child
/// without marking the ledger should [`CompletionRouter::tombstone`] it first.)
pub fn spawn_status_watcher_with_completions(
    store: Arc<dyn TaskStore>,
    task_id: String,
    mut status: watch::Receiver<DetachedSubagentStatus>,
    target: DetachedCompletionTarget,
) {
    tokio::spawn(async move {
        let terminal = loop {
            let snapshot = status.borrow_and_update().clone();
            match snapshot {
                DetachedSubagentStatus::Running => {}
                DetachedSubagentStatus::AwaitingUser { .. } => {
                    record_status_with_retries(store.as_ref(), &task_id, &snapshot);
                }
                _ => break snapshot,
            }
            if status.changed().await.is_err() {
                break DetachedSubagentStatus::ended_without_result();
            }
        };
        let cancelled_now = || {
            matches!(
                store.get(&TaskId::new(&task_id)).map(|r| r.status),
                Some(OrchestrationTaskStatus::CancelRequested | OrchestrationTaskStatus::Cancelled)
            )
        };
        // Before the write, a requested cancel would be overwritten by it; after
        // it, a cancel that won the race leaves the record cancelled.
        let cancelled_before = cancelled_now();
        // A failed ledger write is logged, not fatal: the completion store is
        // independent, and the parent should hear about a finished child even
        // if the task ledger lags (restart reconciliation settles that record).
        if !record_status_with_retries(store.as_ref(), &task_id, &terminal) {
            tracing::warn!("{LOG_PREFIX} task_id={task_id} terminal status not written to ledger");
        }
        if cancelled_before || cancelled_now() {
            tracing::debug!("{LOG_PREFIX} task_id={task_id} cancelled; not recorded");
            return;
        }
        record_detached_completion(&target, &task_id, &terminal).await;
    });
}
