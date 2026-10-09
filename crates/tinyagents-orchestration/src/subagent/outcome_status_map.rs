//! Conversions from [`SubagentOutcomeKind`] (what a driver-level run
//! returned, with its payload) to the other status vocabularies. The payloads
//! (pause state, incomplete reason) are dropped; only the lifecycle state
//! maps, so there is no reverse direction. The serde form of
//! `SubagentOutcomeKind` is unchanged.

use tinyagents_session::run_ledger::AgentRunStatus;
use tinyagents_tasks::{CompletionStatus, NoEquivalentStatus, OrchestrationTaskStatus};

use super::SubAgentJobStatus;
use super::types::{IncompleteKind, SubagentOutcomeKind};

/// `AwaitingInput` -> `Awaiting` (live); `Incomplete` with
/// [`IncompleteKind::Timeout`] -> `TimedOut`, any other incomplete -> `Failed`
/// (the same refinement as `SubAgentJob::task_status`).
impl From<&SubagentOutcomeKind> for OrchestrationTaskStatus {
    fn from(outcome: &SubagentOutcomeKind) -> Self {
        match outcome {
            SubagentOutcomeKind::Completed => Self::Completed,
            SubagentOutcomeKind::AwaitingInput(_) => Self::Awaiting,
            SubagentOutcomeKind::Incomplete(incomplete) => match incomplete.kind {
                IncompleteKind::Timeout => Self::TimedOut,
                _ => Self::Failed,
            },
            SubagentOutcomeKind::Cancelled => Self::Cancelled,
        }
    }
}

/// Goes through the task status: see `From<OrchestrationTaskStatus> for
/// AgentRunStatus` for the collapsed cases.
impl From<&SubagentOutcomeKind> for AgentRunStatus {
    fn from(outcome: &SubagentOutcomeKind) -> Self {
        OrchestrationTaskStatus::from(outcome).into()
    }
}

/// `AwaitingInput` has no job equivalent (a job never pauses) and fails with
/// [`NoEquivalentStatus`]; the incomplete cause is dropped.
impl TryFrom<&SubagentOutcomeKind> for SubAgentJobStatus {
    type Error = NoEquivalentStatus;

    fn try_from(outcome: &SubagentOutcomeKind) -> Result<Self, Self::Error> {
        match outcome {
            SubagentOutcomeKind::Completed => Ok(Self::Completed),
            SubagentOutcomeKind::Incomplete(_) => Ok(Self::Incomplete),
            SubagentOutcomeKind::Cancelled => Ok(Self::Cancelled),
            SubagentOutcomeKind::AwaitingInput(_) => Err(NoEquivalentStatus::new(
                "awaiting_input",
                "SubAgentJobStatus",
            )),
        }
    }
}

/// `AwaitingInput` is a pause, not a completion, and fails with
/// [`NoEquivalentStatus`]. (The completion router separately declines to record
/// a `Cancelled` outcome; that is a routing policy, not a mapping gap.)
impl TryFrom<&SubagentOutcomeKind> for CompletionStatus {
    type Error = NoEquivalentStatus;

    fn try_from(outcome: &SubagentOutcomeKind) -> Result<Self, Self::Error> {
        match outcome {
            SubagentOutcomeKind::Completed => Ok(Self::Success),
            SubagentOutcomeKind::Incomplete(_) => Ok(Self::Incomplete),
            SubagentOutcomeKind::Cancelled => Ok(Self::Cancelled),
            SubagentOutcomeKind::AwaitingInput(_) => Err(NoEquivalentStatus::new(
                "awaiting_input",
                "CompletionStatus",
            )),
        }
    }
}

#[cfg(test)]
#[path = "outcome_status_map_tests.rs"]
mod tests;
