//! Conversions from [`DetachedSubagentStatus`] to the other status
//! vocabularies. The payloads (output, question, error) are dropped; only the
//! lifecycle state maps. See the `tinyagents-tasks` README for the full map.

use tinyagents_session::run_ledger::AgentRunStatus;
use tinyagents_tasks::{CompletionStatus, NoEquivalentStatus, OrchestrationTaskStatus};

use super::types::DetachedSubagentStatus;
use crate::subagent::SubAgentJobStatus;

impl DetachedSubagentStatus {
    /// The managed-task status for this state; same as
    /// `OrchestrationTaskStatus::from(&status)`. `AwaitingUser` maps to
    /// `Awaiting`, which (unlike [`Self::is_terminal`]) is a live task state.
    pub fn to_task_status(&self) -> OrchestrationTaskStatus {
        self.into()
    }

    /// The run-ledger status for this state; same as
    /// `AgentRunStatus::from(&status)`. `AwaitingUser` maps to `AwaitingUser`,
    /// a live ledger state.
    pub fn to_run_status(&self) -> AgentRunStatus {
        self.into()
    }
}

/// Drops the payload; `AwaitingUser` -> `Awaiting`.
impl From<&DetachedSubagentStatus> for OrchestrationTaskStatus {
    fn from(status: &DetachedSubagentStatus) -> Self {
        match status {
            DetachedSubagentStatus::Running => Self::Running,
            DetachedSubagentStatus::Completed { .. } => Self::Completed,
            DetachedSubagentStatus::AwaitingUser { .. } => Self::Awaiting,
            DetachedSubagentStatus::Failed { .. } => Self::Failed,
        }
    }
}

/// Drops the payload; `AwaitingUser` -> `AwaitingUser`.
impl From<&DetachedSubagentStatus> for AgentRunStatus {
    fn from(status: &DetachedSubagentStatus) -> Self {
        match status {
            DetachedSubagentStatus::Running => Self::Running,
            DetachedSubagentStatus::Completed { .. } => Self::Completed,
            DetachedSubagentStatus::AwaitingUser { .. } => Self::AwaitingUser,
            DetachedSubagentStatus::Failed { .. } => Self::Failed,
        }
    }
}

/// Only a finished run is a completion. `Running` and `AwaitingUser` (a pause:
/// the same task completes later) fail with [`NoEquivalentStatus`].
impl TryFrom<&DetachedSubagentStatus> for CompletionStatus {
    type Error = NoEquivalentStatus;

    fn try_from(status: &DetachedSubagentStatus) -> Result<Self, Self::Error> {
        match status {
            DetachedSubagentStatus::Completed { .. } => Ok(Self::Success),
            DetachedSubagentStatus::Failed { .. } => Ok(Self::Failed),
            DetachedSubagentStatus::Running | DetachedSubagentStatus::AwaitingUser { .. } => {
                Err(NoEquivalentStatus::new(status.label(), "CompletionStatus"))
            }
        }
    }
}

/// Converts a detached state to a job status. `AwaitingUser` has no job
/// equivalent (a job never pauses for input) and fails with
/// [`NoEquivalentStatus`]; payloads are dropped.
impl TryFrom<&DetachedSubagentStatus> for SubAgentJobStatus {
    type Error = NoEquivalentStatus;

    fn try_from(status: &DetachedSubagentStatus) -> Result<Self, Self::Error> {
        match status {
            DetachedSubagentStatus::Running => Ok(Self::Running),
            DetachedSubagentStatus::Completed { .. } => Ok(Self::Completed),
            DetachedSubagentStatus::Failed { .. } => Ok(Self::Failed),
            DetachedSubagentStatus::AwaitingUser { .. } => Err(NoEquivalentStatus::new(
                "awaiting_user",
                "SubAgentJobStatus",
            )),
        }
    }
}

#[cfg(test)]
#[path = "status_map_tests.rs"]
mod tests;
