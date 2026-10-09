//! Conversions from [`SubAgentJobStatus`] to the other status vocabularies.
//! See the crate README for the full map.

use tinyagents_session::run_ledger::AgentRunStatus;
use tinyagents_tasks::OrchestrationTaskStatus;

use super::types::{SubAgentJob, SubAgentJobStatus};
use crate::status::NoEquivalentStatus;
use crate::subagent::IncompleteKind;

impl SubAgentJobStatus {
    /// The managed-task status for this job status.
    ///
    /// Lossy: `Incomplete` maps to `Failed` because the status alone does not
    /// say whether the child timed out or exhausted a budget. Use
    /// [`SubAgentJob::task_status`] to refine a timeout to `TimedOut`.
    pub fn to_task_status(self) -> OrchestrationTaskStatus {
        match self {
            Self::Queued => OrchestrationTaskStatus::Pending,
            Self::Running => OrchestrationTaskStatus::Running,
            Self::Completed => OrchestrationTaskStatus::Completed,
            Self::Failed | Self::Incomplete => OrchestrationTaskStatus::Failed,
            Self::Cancelled => OrchestrationTaskStatus::Cancelled,
        }
    }

    /// The run-ledger status for this job status.
    ///
    /// Lossy: `Incomplete` maps to `Failed` (the ledger has no incomplete
    /// state).
    pub fn to_run_status(self) -> AgentRunStatus {
        match self {
            Self::Queued => AgentRunStatus::Pending,
            Self::Running => AgentRunStatus::Running,
            Self::Completed => AgentRunStatus::Completed,
            Self::Failed | Self::Incomplete => AgentRunStatus::Failed,
            Self::Cancelled => AgentRunStatus::Cancelled,
        }
    }
}

impl SubAgentJob {
    /// The managed-task status for this job, using
    /// [`incomplete_kind`](Self::incomplete_kind) to tell a timeout
    /// (`TimedOut`) from any other incomplete run (`Failed`).
    pub fn task_status(&self) -> OrchestrationTaskStatus {
        match (self.status, self.incomplete_kind) {
            (SubAgentJobStatus::Incomplete, Some(IncompleteKind::Timeout)) => {
                OrchestrationTaskStatus::TimedOut
            }
            (status, _) => status.to_task_status(),
        }
    }
}

/// Converts a managed-task status to a job status.
///
/// Lossy: `CancelRequested` becomes `Running` (the job is live until the
/// cancel is observed) and `TimedOut` becomes `Incomplete`. `Awaiting` and
/// `Abandoned` have no job equivalent and fail with [`NoEquivalentStatus`].
///
/// `TimedOut` -> `Incomplete` drops the cause: the job status carries no
/// `IncompleteKind`, so callers must set
/// `IncompleteKind::Timeout` themselves when they convert a timed-out task.
impl TryFrom<OrchestrationTaskStatus> for SubAgentJobStatus {
    type Error = NoEquivalentStatus;

    fn try_from(status: OrchestrationTaskStatus) -> Result<Self, Self::Error> {
        use OrchestrationTaskStatus as T;
        match status {
            T::Pending => Ok(Self::Queued),
            T::Running | T::CancelRequested => Ok(Self::Running),
            T::Completed => Ok(Self::Completed),
            T::Failed => Ok(Self::Failed),
            T::Cancelled => Ok(Self::Cancelled),
            T::TimedOut => Ok(Self::Incomplete),
            T::Awaiting => Err(NoEquivalentStatus::new("awaiting", "SubAgentJobStatus")),
            T::Abandoned => Err(NoEquivalentStatus::new("abandoned", "SubAgentJobStatus")),
        }
    }
}

#[cfg(test)]
#[path = "status_map_tests.rs"]
mod tests;
