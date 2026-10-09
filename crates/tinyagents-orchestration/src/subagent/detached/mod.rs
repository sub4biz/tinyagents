//! Detached (background) subagent runtime vocabulary.
//!
//! The process-local mechanics (watch channel, cancel token, abort handle,
//! ownership) live in `tinyagents_graph::orchestration::DetachedTaskRegistry`;
//! the durable lifecycle lives in an `orchestration::TaskStore`. This module
//! is the host-neutral layer between them for subagents:
//!
//! - [`DetachedSubagentStatus`], the status a detached run publishes, with its
//!   stable wire labels, and [`FinishedOutcome`] for a run that ended before a
//!   cancel arrived;
//! - [`WaitError`] / [`WaitOutcome`] and [`wait_detached`], a registry wait
//!   that treats a dropped sender as a failure rather than a hang;
//! - the ledger helpers that mirror a status into a `TaskStore` and read a
//!   durable record back as a wait outcome (so a run this process never
//!   registered, e.g. after a restart, still resolves);
//! - [`steer_detached`] / [`SteerError`] to inject a message into a running
//!   subagent (live handle first, host fallback second), and
//!   [`cancel_for_thread`] to abort the subagents of one parent thread;
//! - [`SubagentIdentity`] with roster snapshots, session-id resolution and
//!   resume references over a registry or the durable records.
//!
//! Hosts own the registry instance, its metadata type (via
//! [`SubagentIdentity`]), where the store lives, progress projection, and
//! policy.

mod ledger;
mod roster;
mod status;
mod status_map;
mod steer;
mod types;

pub use ledger::{
    DETACHED_LEDGER_TIMEOUT_MS, list_subagent_records, orphaned_subagent_reason, record_agent_id,
    record_cancelled, record_parent_session, record_spawned, record_status,
    record_subagent_session_id, record_to_wait_outcome, spawn_status_watcher,
    subagent_record_for_task, task_status_label,
};
pub use roster::{
    resume_ref_for_task, resume_ref_from_record, snapshot_for_owner, task_id_for_session,
    task_id_for_session_in_records,
};
pub use status::{wait_detached, wait_error_from_registry};
pub use steer::{
    SteerAccess, SteerError, SteerReceipt, SteerRoute, cancel_for_thread, distinct_parent_threads,
    queue_lane_name, steer_detached, steer_detached_with_request_id, steering_command_for_lane,
};
pub use types::{
    DetachedSubagentStatus, FinishedOutcome, SpawnedSubagent, SubagentIdentity, SubagentResumeRef,
    SubagentSnapshot, WaitError, WaitOutcome,
};

#[cfg(test)]
#[path = "steer_tests.rs"]
mod steer_test;
#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
