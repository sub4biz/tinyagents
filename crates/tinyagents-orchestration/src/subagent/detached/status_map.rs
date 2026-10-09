//! Conversions from [`DetachedSubagentStatus`] to the other status
//! vocabularies. The payloads (output, question, error) are dropped; only the
//! lifecycle state maps. See the crate README for the full map.

use tinyagents_session::run_ledger::AgentRunStatus;
use tinyagents_tasks::OrchestrationTaskStatus;

use super::types::DetachedSubagentStatus;
use crate::status::NoEquivalentStatus;
use crate::subagent::SubAgentJobStatus;

impl DetachedSubagentStatus {
    /// The managed-task status for this state. `AwaitingUser` maps to
    /// `Awaiting`, which (unlike [`Self::is_terminal`]) is a live task state.
    pub fn to_task_status(&self) -> OrchestrationTaskStatus {
        match self {
            Self::Running => OrchestrationTaskStatus::Running,
            Self::Completed { .. } => OrchestrationTaskStatus::Completed,
            Self::AwaitingUser { .. } => OrchestrationTaskStatus::Awaiting,
            Self::Failed { .. } => OrchestrationTaskStatus::Failed,
        }
    }

    /// The run-ledger status for this state. `AwaitingUser` maps to
    /// `AwaitingUser`, a live ledger state.
    pub fn to_run_status(&self) -> AgentRunStatus {
        match self {
            Self::Running => AgentRunStatus::Running,
            Self::Completed { .. } => AgentRunStatus::Completed,
            Self::AwaitingUser { .. } => AgentRunStatus::AwaitingUser,
            Self::Failed { .. } => AgentRunStatus::Failed,
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
