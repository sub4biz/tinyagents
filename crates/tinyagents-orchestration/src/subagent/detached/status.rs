use std::time::Duration;

use tinyagents_tasks::{
    DetachedTaskRegistry, DetachedTaskRegistryError, DetachedTaskWaitOutcome,
};
use tinyagents_harness::ids::TaskId;

use super::types::{DetachedSubagentStatus, FinishedOutcome, WaitError, WaitOutcome};

impl DetachedSubagentStatus {
    /// Everything except [`Self::Running`] is terminal (an awaiting run is
    /// paused, but it will not progress on its own).
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::Running)
    }

    /// Stable wire label: `running` / `completed` / `awaiting_user` / `failed`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed { .. } => "completed",
            Self::AwaitingUser { .. } => "awaiting_user",
            Self::Failed { .. } => "failed",
        }
    }

    /// The status reported when a run's sender was dropped without a result.
    pub fn ended_without_result() -> Self {
        Self::Failed {
            error: "sub-agent task ended without reporting a result".to_string(),
        }
    }

    /// How a run had already ended, if it had. `AwaitingUser` is paused, not
    /// finished, so it yields `None`.
    pub fn finished_outcome(&self) -> Option<FinishedOutcome> {
        match self {
            Self::Completed { .. } => Some(FinishedOutcome::Completed),
            Self::Failed { .. } => Some(FinishedOutcome::Failed),
            Self::Running | Self::AwaitingUser { .. } => None,
        }
    }
}

impl FinishedOutcome {
    /// Wire name (`completed` / `failed`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

/// Map a registry error onto [`WaitError`] (see its `From` impl).
pub fn wait_error_from_registry(error: DetachedTaskRegistryError) -> WaitError {
    WaitError::from(error)
}

/// Block until `task_id` reaches a terminal status or `timeout` elapses.
///
/// A closed status channel (aborted/panicked task) surfaces as a
/// [`DetachedSubagentStatus::ended_without_result`] failure instead of hanging.
pub async fn wait_detached<M>(
    registry: &DetachedTaskRegistry<M, DetachedSubagentStatus>,
    task_id: &str,
    owner: &str,
    timeout: Duration,
) -> Result<WaitOutcome, WaitError>
where
    M: Clone + Send + Sync + 'static,
{
    match registry.wait(&TaskId::new(task_id), owner, timeout).await {
        Ok(DetachedTaskWaitOutcome::Terminal(status)) => Ok(WaitOutcome::Terminal(status)),
        Ok(DetachedTaskWaitOutcome::TimedOut(status)) => Ok(WaitOutcome::TimedOut(status)),
        Err(DetachedTaskRegistryError::StatusChannelClosed) => Ok(WaitOutcome::Terminal(
            DetachedSubagentStatus::ended_without_result(),
        )),
        Err(error) => Err(wait_error_from_registry(error)),
    }
}
