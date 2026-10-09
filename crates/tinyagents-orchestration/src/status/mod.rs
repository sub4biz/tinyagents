//! Conversions between the status vocabularies that describe one unit of
//! background agent work.
//!
//! [`OrchestrationTaskStatus`] is the one durable lifecycle status. The other
//! enums stay because their serialized forms are persisted or consumed
//! elsewhere (ledger rows, completion records, job snapshots, projected
//! transcripts), but each is expressed in terms of the canonical one through
//! `From` (total) or `TryFrom` (fallible, [`NoEquivalentStatus`]) impls. The
//! single mapping table is in the `tinyagents-tasks` README.
//!
//! Dependency direction decides where an impl lives: the ledger and
//! transcript-view conversions sit in `tinyagents-session`, the completion
//! conversions in `tinyagents-tasks`, and the job, detached and outcome ones
//! beside their types in this crate. The free functions below are the former
//! entry points and now delegate to those impls.

use tinyagents_session::run_ledger::AgentRunStatus;
use tinyagents_tasks::OrchestrationTaskStatus;

pub use tinyagents_tasks::NoEquivalentStatus;

/// Maps a managed-task status onto the run-ledger status.
///
/// Lossy (many-to-one): `CancelRequested` -> `Running`, `TimedOut` ->
/// `Failed`, `Abandoned` -> `Interrupted`, `Awaiting` -> `AwaitingUser`.
#[deprecated(since = "2.1.4", note = "use `AgentRunStatus::from(task_status)`")]
pub fn task_status_to_run_status(status: OrchestrationTaskStatus) -> AgentRunStatus {
    AgentRunStatus::from(status)
}

/// Maps a run-ledger status onto the managed-task status.
///
/// Lossy (many-to-one): `AwaitingUser` and `Paused` -> `Awaiting`,
/// `Interrupted` -> `Abandoned`.
#[deprecated(
    since = "2.1.4",
    note = "use `OrchestrationTaskStatus::from(run_status)`"
)]
pub fn run_status_to_task_status(status: AgentRunStatus) -> OrchestrationTaskStatus {
    OrchestrationTaskStatus::from(status)
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
