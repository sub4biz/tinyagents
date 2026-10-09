//! Conversions between [`AgentRunStatus`] (the durable run-ledger row status)
//! and [`OrchestrationTaskStatus`] (the canonical lifecycle status).
//!
//! Both are total. The wire formats of both enums are unchanged; see the
//! mapping table in the `tinyagents-tasks` README.

use tinyagents_tasks::OrchestrationTaskStatus;

use super::types::AgentRunStatus;

/// Maps a run-ledger status onto the canonical task status.
///
/// Lossy (many-to-one):
/// - `AwaitingUser` and `Paused` both become `Awaiting`; reading it back gives
///   `AwaitingUser`.
/// - `Interrupted` -> `Abandoned`: both are terminal "stopped without a
///   result"; the task vocabulary has no restart-specific state.
///
/// Terminality is preserved.
impl From<AgentRunStatus> for OrchestrationTaskStatus {
    fn from(status: AgentRunStatus) -> Self {
        match status {
            AgentRunStatus::Pending => Self::Pending,
            AgentRunStatus::Running => Self::Running,
            AgentRunStatus::AwaitingUser | AgentRunStatus::Paused => Self::Awaiting,
            AgentRunStatus::Completed => Self::Completed,
            AgentRunStatus::Failed => Self::Failed,
            AgentRunStatus::Cancelled => Self::Cancelled,
            AgentRunStatus::Interrupted => Self::Abandoned,
        }
    }
}

/// Maps a canonical task status onto the run-ledger status.
///
/// Lossy (many-to-one):
/// - `CancelRequested` -> `Running`: the ledger has no "cancelling" state, and
///   the work is live until the cancel is observed.
/// - `TimedOut` -> `Failed`: the ledger records a deadline as a failure.
/// - `Abandoned` -> `Interrupted`.
/// - `Awaiting` -> `AwaitingUser`: the task may be waiting on a child rather
///   than a person; the ledger cannot tell them apart.
///
/// Terminality is preserved.
impl From<OrchestrationTaskStatus> for AgentRunStatus {
    fn from(status: OrchestrationTaskStatus) -> Self {
        use OrchestrationTaskStatus as T;
        match status {
            T::Pending => Self::Pending,
            T::Running | T::CancelRequested => Self::Running,
            T::Awaiting => Self::AwaitingUser,
            T::Completed => Self::Completed,
            T::Failed | T::TimedOut => Self::Failed,
            T::Cancelled => Self::Cancelled,
            T::Abandoned => Self::Interrupted,
        }
    }
}

#[cfg(test)]
#[path = "status_map_tests.rs"]
mod tests;
