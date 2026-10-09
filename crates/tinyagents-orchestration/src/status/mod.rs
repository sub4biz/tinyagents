//! Conversions between the status vocabularies that describe one unit of
//! background agent work.
//!
//! Several layers each grew their own lifecycle enum. They are *not* merged:
//! each answers a different question and keeps its own wire format. This module
//! (and the `to_*_status` methods / `TryFrom` impls next to the job and
//! detached types) is the one explicit, tested place where they map onto each
//! other. The full table is in this crate's `README.md`.
//!
//! Dependency direction decides where a conversion lives. `tinyagents-graph`
//! (task status) and `tinyagents-session` (run-ledger status) do not depend on
//! each other, so the pair of them is mapped here, in the first crate that sees
//! both. The job and detached statuses are owned by this crate, so their
//! conversions sit beside their types.
//!
//! Every mapping is total or fails with [`NoEquivalentStatus`]; lossy ones say
//! so in their documentation.

use tinyagents_graph::orchestration::OrchestrationTaskStatus;
use tinyagents_session::run_ledger::AgentRunStatus;

mod types;

pub use types::NoEquivalentStatus;

/// Maps a managed-task status onto the run-ledger status.
///
/// Lossy (many-to-one):
/// - `CancelRequested` -> `Running`: the ledger has no "cancelling" state, and
///   the work is still live until the cancel is observed.
/// - `TimedOut` -> `Failed`: the ledger records a deadline as a failure.
/// - `Abandoned` -> `Interrupted`: the supervisor stopped waiting; the ledger's
///   nearest terminal state for "stopped without an answer" is `Interrupted`.
/// - `Awaiting` -> `AwaitingUser`: the task may be waiting on a child rather
///   than a person; the ledger cannot tell them apart.
///
/// Terminality is always preserved.
pub fn task_status_to_run_status(status: OrchestrationTaskStatus) -> AgentRunStatus {
    use OrchestrationTaskStatus as T;
    match status {
        T::Pending => AgentRunStatus::Pending,
        T::Running | T::CancelRequested => AgentRunStatus::Running,
        T::Awaiting => AgentRunStatus::AwaitingUser,
        T::Completed => AgentRunStatus::Completed,
        T::Failed | T::TimedOut => AgentRunStatus::Failed,
        T::Cancelled => AgentRunStatus::Cancelled,
        T::Abandoned => AgentRunStatus::Interrupted,
    }
}

/// Maps a run-ledger status onto the managed-task status.
///
/// Lossy (many-to-one):
/// - `AwaitingUser` and `Paused` both become `Awaiting`; a round trip through
///   [`task_status_to_run_status`] reads `Paused` back as `AwaitingUser`.
/// - `Interrupted` -> `Abandoned`: both are terminal "stopped without a
///   result"; the task vocabulary has no restart-specific state.
///
/// Terminality is always preserved.
pub fn run_status_to_task_status(status: AgentRunStatus) -> OrchestrationTaskStatus {
    use OrchestrationTaskStatus as T;
    match status {
        AgentRunStatus::Pending => T::Pending,
        AgentRunStatus::Running => T::Running,
        AgentRunStatus::AwaitingUser | AgentRunStatus::Paused => T::Awaiting,
        AgentRunStatus::Completed => T::Completed,
        AgentRunStatus::Failed => T::Failed,
        AgentRunStatus::Cancelled => T::Cancelled,
        AgentRunStatus::Interrupted => T::Abandoned,
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
