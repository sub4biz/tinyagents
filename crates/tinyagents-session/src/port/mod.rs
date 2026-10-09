//! The session store port: where a host plugs in its own storage.
//!
//! Everything an agent persists while it runs — its transcripts, its
//! turn-state snapshots, its key-value records and its event journal — is
//! reached through one [`SessionStoreProvider`]. The host installs a provider
//! once; the runtime asks it for each agent's [`AgentStores`] and never names
//! a file, a database or a directory itself.
//!
//! That is what lets one process serve many users from a shared database: a
//! cloud host implements the provider over its own store (MongoDB, Postgres,
//! …) and scopes every query by agent id, while a desktop host keeps today's
//! on-disk layout behind the same traits.
//!
//! # Isolation
//!
//! [`SessionStoreProvider::for_agent`] is the only way to obtain stores, and
//! the handles it returns are bound to that one agent. A provider that serves
//! several users from one backend must make it impossible for one agent's
//! handles to read or write another's data; [`InMemorySessionStores`] does,
//! and the conformance suite
//! ([`crate::testkit::conformance::session_store_isolation_conformance`])
//! checks it. A single-user host (the desktop app) may legitimately share one
//! store between its agents, and simply does not claim isolation.
//!
//! # Sync and async
//!
//! The transcript and turn-state seams are synchronous, as they already were:
//! the turn path that persists a transcript is a chain of sync `&mut self`
//! methods. An implementation over an async client bridges at its own
//! boundary. The key-value and journal seams are the harness's async
//! [`Store`] and [`AppendStore`].
//!
//! # Bundled implementations
//!
//! - [`InMemorySessionStores`]: per-agent, process-lifetime storage, for
//!   tests and for hosts that deliberately keep nothing.
//! - `DriverSessionStores` (feature `storage-drivers`): every agent's stores
//!   in one `tinystoragedrivers` backend (SQLite, MongoDB, ...), each agent
//!   under its own storage scope.
//! - The on-disk layout (`session_raw/`, `tinyagents_store/`, turn-state
//!   files) is wrapped by the host that owns it, not here: this crate keeps
//!   the file and SQLite building blocks, and a host crate decides to use them.

#[cfg(feature = "storage-drivers")]
mod drivers;
mod memory;
mod types;

use std::sync::Arc;

#[cfg(feature = "storage-drivers")]
pub use drivers::{
    DriverSessionStores, DriverTranscriptHistory, DriverTranscriptLocator, DriverTurnStates,
};
pub use memory::{InMemorySessionStores, InMemoryTranscriptLocator, InMemoryTurnStates};
pub use types::AgentStores;

pub use tinyagents_harness::store::{AppendStore, Store};

use crate::turn_state::{TurnLifecycle, TurnState, TurnStateStore};

/// Hands out each agent's stores. Installed once per runtime by the host.
pub trait SessionStoreProvider: Send + Sync {
    /// The stores of the agent `agent_id`.
    ///
    /// Called whenever the runtime needs an agent's stores, so it should be
    /// cheap: hand out shared handles rather than opening connections.
    fn for_agent(&self, agent_id: &str) -> AgentStores;

    /// Repairs state an unclean shutdown left behind, once, before any agent
    /// runs: turns that were in flight are marked interrupted, and so on.
    ///
    /// Defaults to doing nothing, for stores that keep no in-flight state or
    /// whose host repairs it elsewhere.
    ///
    /// # Errors
    ///
    /// Whatever the backend reports; the runtime logs it and carries on.
    fn recover(&self) -> anyhow::Result<()> {
        Ok(())
    }

    /// A stable name for where this provider keeps its data, for logs and
    /// for telling two providers apart. `None` when it cannot say.
    fn destination_key(&self) -> Option<String> {
        None
    }

    /// The workspace directory a file-backed provider keeps the classic
    /// on-disk layout in (`session_raw/`, `tinyagents_store/`, …), or `None`
    /// for any other store.
    ///
    /// A host uses it to keep file-era companions of that layout running —
    /// mirrors that read transcript files back, say — only where there are
    /// files to read.
    fn workspace_dir(&self) -> Option<std::path::PathBuf> {
        None
    }
}

/// Restart-survivable snapshots of an agent's in-flight turns.
///
/// The contract of [`TurnStateStore`], the on-disk implementation, as a
/// trait. Every method is scoped to the handle's own agent.
pub trait TurnStates: Send + Sync {
    /// Writes `state` as the snapshot of its turn.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn put(&self, state: &TurnState) -> Result<(), String>;

    /// Writes `state` unless its turn is already recorded as completed;
    /// returns whether it wrote.
    ///
    /// # Errors
    ///
    /// When the stored snapshot cannot be read or the backend cannot write.
    fn put_unless_completed(&self, state: &TurnState) -> Result<bool, String>;

    /// The latest turn's snapshot on `thread_id`.
    ///
    /// # Errors
    ///
    /// When the backend cannot read.
    fn get(&self, thread_id: &str) -> Result<Option<TurnState>, String>;

    /// The snapshot of one turn.
    ///
    /// # Errors
    ///
    /// When the backend cannot read.
    fn get_turn(&self, thread_id: &str, request_id: &str) -> Result<Option<TurnState>, String>;

    /// Removes every snapshot on `thread_id`; returns whether any existed.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn delete(&self, thread_id: &str) -> Result<bool, String>;

    /// Removes one turn's snapshot; returns whether it existed.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn delete_turn(&self, thread_id: &str, request_id: &str) -> Result<bool, String>;

    /// The latest snapshot of every thread.
    ///
    /// # Errors
    ///
    /// When the backend cannot read.
    fn list(&self) -> Result<Vec<TurnState>, String>;

    /// Every turn snapshot on `thread_id`.
    ///
    /// # Errors
    ///
    /// When the backend cannot read.
    fn list_thread(&self, thread_id: &str) -> Result<Vec<TurnState>, String>;

    /// Removes every snapshot; returns how many there were.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn clear_all(&self) -> Result<usize, String>;

    /// Marks every non-terminal snapshot interrupted, as of `now_rfc3339`;
    /// returns how many changed.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn mark_all_interrupted(&self, now_rfc3339: &str) -> Result<usize, String>;

    /// Records a terminal `lifecycle` on a turn that is still in flight;
    /// returns whether it changed anything.
    ///
    /// # Errors
    ///
    /// When the backend cannot read or write.
    fn settle_turn(
        &self,
        thread_id: &str,
        request_id: &str,
        lifecycle: TurnLifecycle,
        now_rfc3339: &str,
    ) -> Result<bool, String>;
}

impl TurnStates for TurnStateStore {
    fn put(&self, state: &TurnState) -> Result<(), String> {
        TurnStateStore::put(self, state)
    }

    fn put_unless_completed(&self, state: &TurnState) -> Result<bool, String> {
        TurnStateStore::put_unless_completed(self, state)
    }

    fn get(&self, thread_id: &str) -> Result<Option<TurnState>, String> {
        TurnStateStore::get(self, thread_id)
    }

    fn get_turn(&self, thread_id: &str, request_id: &str) -> Result<Option<TurnState>, String> {
        TurnStateStore::get_turn(self, thread_id, request_id)
    }

    fn delete(&self, thread_id: &str) -> Result<bool, String> {
        TurnStateStore::delete(self, thread_id)
    }

    fn delete_turn(&self, thread_id: &str, request_id: &str) -> Result<bool, String> {
        TurnStateStore::delete_turn(self, thread_id, request_id)
    }

    fn list(&self) -> Result<Vec<TurnState>, String> {
        TurnStateStore::list(self)
    }

    fn list_thread(&self, thread_id: &str) -> Result<Vec<TurnState>, String> {
        TurnStateStore::list_thread(self, thread_id)
    }

    fn clear_all(&self) -> Result<usize, String> {
        TurnStateStore::clear_all(self)
    }

    fn mark_all_interrupted(&self, now_rfc3339: &str) -> Result<usize, String> {
        TurnStateStore::mark_all_interrupted(self, now_rfc3339)
    }

    fn settle_turn(
        &self,
        thread_id: &str,
        request_id: &str,
        lifecycle: TurnLifecycle,
        now_rfc3339: &str,
    ) -> Result<bool, String> {
        TurnStateStore::settle_turn(self, thread_id, request_id, lifecycle, now_rfc3339)
    }
}

/// A provider shared behind an [`Arc`] is still a provider.
impl<P: SessionStoreProvider + ?Sized> SessionStoreProvider for Arc<P> {
    fn for_agent(&self, agent_id: &str) -> AgentStores {
        (**self).for_agent(agent_id)
    }

    fn recover(&self) -> anyhow::Result<()> {
        (**self).recover()
    }

    fn destination_key(&self) -> Option<String> {
        (**self).destination_key()
    }

    fn workspace_dir(&self) -> Option<std::path::PathBuf> {
        (**self).workspace_dir()
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
