//! Conversions from [`SubAgentJobStatus`] to the other status vocabularies.
//! See the `tinyagents-tasks` README for the full map.

use tinyagents_session::run_ledger::AgentRunStatus;
use tinyagents_tasks::{CompletionStatus, NoEquivalentStatus, OrchestrationTaskStatus};

use super::types::{SubAgentJob, SubAgentJobStatus};
use crate::subagent::IncompleteKind;

impl SubAgentJobStatus {
    /// The managed-task status for this job status; same as
    /// `OrchestrationTaskStatus::from(self)`.
    ///
    /// Lossy: `Incomplete` maps to `Failed` because the status alone does not
    /// say whether the child timed out or exhausted a budget. Use
    /// [`SubAgentJob::task_status`] to refine a timeout to `TimedOut`.
    pub fn to_task_status(self) -> OrchestrationTaskStatus {
        self.into()
    }

    /// The run-ledger status for this job status; same as
    /// `AgentRunStatus::from(self)`.
    ///
    /// Lossy: `Incomplete` maps to `Failed` (the ledger has no incomplete
    /// state).
    pub fn to_run_status(self) -> AgentRunStatus {
        self.into()
    }
}

/// Lossy: `Incomplete` becomes `Failed` (see [`SubAgentJob::task_status`] for
/// the timeout refinement).
impl From<SubAgentJobStatus> for OrchestrationTaskStatus {
    fn from(status: SubAgentJobStatus) -> Self {
        match status {
            SubAgentJobStatus::Queued => Self::Pending,
            SubAgentJobStatus::Running => Self::Running,
            SubAgentJobStatus::Completed => Self::Completed,
            SubAgentJobStatus::Failed | SubAgentJobStatus::Incomplete => Self::Failed,
            SubAgentJobStatus::Cancelled => Self::Cancelled,
        }
    }
}

/// Lossy: `Incomplete` becomes `Failed` (the ledger has no incomplete state).
impl From<SubAgentJobStatus> for AgentRunStatus {
    fn from(status: SubAgentJobStatus) -> Self {
        match status {
            SubAgentJobStatus::Queued => Self::Pending,
            SubAgentJobStatus::Running => Self::Running,
            SubAgentJobStatus::Completed => Self::Completed,
            SubAgentJobStatus::Failed | SubAgentJobStatus::Incomplete => Self::Failed,
            SubAgentJobStatus::Cancelled => Self::Cancelled,
        }
    }
}

/// Converts a run-ledger status to a job status. `AwaitingUser`, `Paused` and
/// `Interrupted` have no job equivalent (a job never pauses, and an interrupted
/// run was never observed to finish) and fail with [`NoEquivalentStatus`].
impl TryFrom<AgentRunStatus> for SubAgentJobStatus {
    type Error = NoEquivalentStatus;

    fn try_from(status: AgentRunStatus) -> Result<Self, Self::Error> {
        match status {
            AgentRunStatus::Pending => Ok(Self::Queued),
            AgentRunStatus::Running => Ok(Self::Running),
            AgentRunStatus::Completed => Ok(Self::Completed),
            AgentRunStatus::Failed => Ok(Self::Failed),
            AgentRunStatus::Cancelled => Ok(Self::Cancelled),
            AgentRunStatus::AwaitingUser | AgentRunStatus::Paused | AgentRunStatus::Interrupted => {
                Err(NoEquivalentStatus::new(
                    status.as_str(),
                    "SubAgentJobStatus",
                ))
            }
        }
    }
}

/// Total and lossless: every completion status has a job status of the same
/// meaning.
impl From<CompletionStatus> for SubAgentJobStatus {
    fn from(status: CompletionStatus) -> Self {
        match status {
            CompletionStatus::Success => Self::Completed,
            CompletionStatus::Failed => Self::Failed,
            CompletionStatus::Cancelled => Self::Cancelled,
            CompletionStatus::Incomplete => Self::Incomplete,
        }
    }
}

/// Only terminal jobs are completions; `Queued` and `Running` fail with
/// [`NoEquivalentStatus`].
impl TryFrom<SubAgentJobStatus> for CompletionStatus {
    type Error = NoEquivalentStatus;

    fn try_from(status: SubAgentJobStatus) -> Result<Self, Self::Error> {
        match status {
            SubAgentJobStatus::Completed => Ok(Self::Success),
            SubAgentJobStatus::Failed => Ok(Self::Failed),
            SubAgentJobStatus::Incomplete => Ok(Self::Incomplete),
            SubAgentJobStatus::Cancelled => Ok(Self::Cancelled),
            SubAgentJobStatus::Queued => Err(NoEquivalentStatus::new("queued", "CompletionStatus")),
            SubAgentJobStatus::Running => {
                Err(NoEquivalentStatus::new("running", "CompletionStatus"))
            }
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
