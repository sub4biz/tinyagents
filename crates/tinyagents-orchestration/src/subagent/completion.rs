//! Recording a finished subagent with the durable completion router: normal
//! finishes, cancellations and executor errors; never a pause.
//!
//! Opt-in: the driver only touches any of this when a
//! [`CompletionRouter`](tinyagents_tasks::CompletionRouter) was configured with
//! [`SubagentDriver::with_completion_router`](super::SubagentDriver::with_completion_router).

use std::sync::Arc;

use tinyagents_tasks::{
    CompletionArtifact, CompletionRecord, CompletionResult, CompletionRouter, CompletionStatus,
    NotifyMode,
};

use super::{
    AppliedResult, ArtifactReference, PreparedSubagent, SubagentError, SubagentOutcome,
    SubagentOutcomeKind, SubagentTaskKey,
};

const LOG_PREFIX: &str = "[subagent-completion]";

impl From<&ArtifactReference> for CompletionArtifact {
    fn from(artifact: &ArtifactReference) -> Self {
        Self {
            id: artifact.id.clone(),
            media_type: artifact.media_type.clone(),
            metadata: artifact.metadata.clone(),
        }
    }
}

impl From<&AppliedResult> for CompletionResult {
    /// The result policy's output as the parent will be shown it.
    fn from(applied: &AppliedResult) -> Self {
        Self {
            text: applied.text.clone(),
            omitted_chars: applied.omitted_chars,
            artifact: applied.artifact.as_ref().map(CompletionArtifact::from),
        }
    }
}

/// Who a finished child reports to, captured when the child is launched.
pub(crate) struct CompletionOrigin {
    task_id: String,
    parent_key: String,
    agent_id: String,
    notify_mode: NotifyMode,
}

impl CompletionOrigin {
    /// `None` when the spawn did not ask to be recorded.
    pub(crate) fn new<C>(
        task_key: &SubagentTaskKey,
        prepared: &PreparedSubagent<C>,
    ) -> Option<Self> {
        let notify_mode = prepared.notify_mode?;
        // The thread outlives a restart; the parent run id does not, so it is
        // only the fallback for a thread-less parent.
        let parent_key = prepared
            .completion_parent
            .clone()
            .or_else(|| task_key.thread_id.clone())
            .unwrap_or_else(|| task_key.parent_run_id.clone());
        Some(Self {
            task_id: task_key.task_id.clone(),
            parent_key,
            agent_id: prepared.agent_key.clone(),
            notify_mode,
        })
    }

    fn record(&self, status: CompletionStatus, result: CompletionResult) -> CompletionRecord {
        CompletionRecord::new(
            self.task_id.clone(),
            self.parent_key.clone(),
            self.agent_id.clone(),
            status,
            result,
        )
        .with_notify_mode(self.notify_mode)
    }

    /// The record for a finished lifecycle, or `None` when the outcome is not a
    /// completion: a pause is not final (the same task id completes later, and
    /// that finish is what gets recorded). A cancellation is recorded as
    /// [`CompletionStatus::Cancelled`] with an empty result.
    pub(crate) fn record_for_outcome(
        &self,
        outcome: &SubagentOutcome,
        omitted_chars: usize,
    ) -> Option<CompletionRecord> {
        let status = CompletionStatus::try_from(&outcome.status).ok()?;
        if matches!(&outcome.status, SubagentOutcomeKind::Cancelled) {
            // A cancel that lands after the executor returned keeps the child's
            // late output on the outcome (`cancelled_preserving`); it must not
            // reach the parent as a usable answer.
            return Some(self.record(status, CompletionResult::default()));
        }
        let text = match &outcome.status {
            SubagentOutcomeKind::Incomplete(incomplete) if outcome.output.is_empty() => {
                incomplete.reason.clone()
            }
            _ => outcome.output.clone(),
        };
        let result = CompletionResult {
            text,
            omitted_chars,
            // An overflow artifact is appended by the result policy, so the last
            // reference is the one holding the full output.
            artifact: outcome.artifacts.last().map(CompletionArtifact::from),
        };
        tracing::debug!(
            "{LOG_PREFIX} task_id={} status={} building record from outcome",
            self.task_id,
            status.as_str()
        );
        Some(self.record(status, result))
    }

    /// The record for an executor error, or `None` when the error is not the
    /// child failing: a host seam fault (a task id mismatch, a persistence
    /// failure, a missing capability) says nothing about the child's own run.
    pub(crate) fn record_for_error(&self, error: &SubagentError) -> Option<CompletionRecord> {
        if !matches!(
            error,
            SubagentError::Execution(_) | SubagentError::Transient { .. }
        ) {
            return None;
        }
        tracing::debug!(
            "{LOG_PREFIX} task_id={} status=failed building record from executor error",
            self.task_id
        );
        Some(self.record(
            CompletionStatus::Failed,
            CompletionResult {
                text: error.to_string(),
                omitted_chars: 0,
                artifact: None,
            },
        ))
    }
}

/// Hands `record` to the router. A router failure is logged, never raised: the
/// child's own result is already persisted and must not be lost to a
/// notification problem.
pub(crate) async fn deliver(router: &Arc<CompletionRouter>, record: CompletionRecord) {
    let task_id = record.task_id.clone();
    match router.record_with_retries(record, 3).await {
        Ok(outcome) => {
            tracing::debug!("{LOG_PREFIX} task_id={task_id} outcome={outcome:?}");
        }
        Err(error) => {
            tracing::warn!("{LOG_PREFIX} task_id={task_id} could not record completion: {error}");
        }
    }
}
