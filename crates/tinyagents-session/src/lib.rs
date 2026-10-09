//! Durable session database and run ledger.
//!
//! SQLite-backed store (WAL + FTS5) for sessions, messages, tool calls, cost
//! metadata, and parent/child lineage, plus a [`run_ledger`] for background
//! agent/workflow execution state. This is the runtime's *history* layer: what
//! ran, what it cost, what it called, and how runs nest.
//!
//! # Why this is a top-level module
//!
//! Session history is a persistence domain in its own right, not a part of the
//! agent loop. Nothing in `tinyagents_harness` reads from it, and a host can use
//! it without running a harness at all — indexing sessions produced elsewhere,
//! or recovering orchestration state at boot before any agent exists. Filing it
//! under `harness::` would imply a dependency that does not exist in either
//! direction.
//!
//! # Relationship to the other persistence layers
//!
//! - `tinyagents_harness::store` is namespaced key-value storage for live
//!   runtime data. It is a substrate runs read and write during execution.
//! - `tinyagents_graph::checkpoint` is durability for *resuming* an interrupted
//!   graph run.
//! - This module is queryable history. Nothing resumes from it; it answers
//!   "what happened", supports cross-session search, and lets a host recover
//!   orchestration state after a restart.
//!
//! A host that keeps its own transcript files (the source of truth for
//! KV-cache resume) still wants this module for indexing and search over them.
//!
//! # The entry tree
//!
//! [`entry_tree`] adds a second, opt-in shape over the same session
//! database: an append-only, branchable tree of entries (`id`/`parent_id`)
//! rather than a flat list. It exists alongside the linear
//! `record_message`/[`transcript`] paths above, not in place of them — a
//! host that never forks a conversation can ignore it entirely, and the
//! linear JSONL/SQLite writers are unchanged. See
//! `docs/modules/session/README.md` for the full design (entry kinds, the
//! context-projection rule, fork semantics) and [`entry_tree::legacy`] for
//! how pre-tree data is deterministically read into the same model.
//!
//! # Chat threads
//!
//! [`threads`] is the product-facing chat log: JSONL threads and messages
//! under `{workspace_dir}/memory/conversations/` with a cross-thread inverted
//! index. It shares no files with the session database or transcripts; see
//! `src/threads/README.md`.
//!
//! # Layout
//!
//! Every entry point takes the workspace root and derives the database path,
//! so a host chooses only where its workspace lives:
//!
//! ```text
//! {workspace_dir}/session_db/sessions.db
//! ```
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use tinyagents_session::{self as session, SessionStatus};
//!
//! # fn main() -> tinyagents_session::Result<()> {
//! let workspace = Path::new("/tmp/workspace");
//!
//! session::record_session_start(
//!     workspace, "sess-1", "researcher", "Researcher", "sess-1",
//!     None, None, None, Some("gpt-5"), None,
//! )?;
//! session::record_message(
//!     workspace, "sess-1", "user", "summarize the repo", None, None, None, None,
//! )?;
//! session::record_session_end(
//!     workspace, "sess-1", SessionStatus::Completed, 1, 120, 340, 0, 0.004,
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! See [`README.md`](./README.md) for the schema, the FTS behaviour, and the
//! coordination guarantees.

mod context;
pub mod entry_tree;
mod migrations;
pub mod ops;
mod paging;
pub mod port;
pub mod retention;
pub mod run_ledger;
mod store;
pub mod testkit;
pub mod threads;
pub mod transcript;
pub mod turn_state;
pub mod types;

pub use tinyagents_harness::error::{Result, TinyAgentsError};

pub use entry_tree::{
    Branch, BranchSummaryEntry, CompactionEntry, CustomEntry, Entry, EntryId, EntryKind, EntryTree,
    Fork, ForkPosition, ForkScope, LabelEntry, SessionCompactionSink,
};
pub use ops::{
    DEFAULT_FTS_SNIPPET_BYTES, fts_snippet_bytes, get_session, list_children, list_messages,
    list_sessions, list_tool_calls, mark_interrupted, record_message,
    record_message_with_reasoning, record_session_end, record_session_start, record_tool_call,
    search_sessions, set_fts_snippet_bytes,
};
pub use port::{
    AgentStores, InMemorySessionStores, InMemoryTranscriptLocator, InMemoryTurnStates,
    SessionStoreProvider, TurnStates,
};
#[cfg(feature = "storage-drivers")]
pub use port::{
    DriverSessionStores, DriverTranscriptHistory, DriverTranscriptLocator, DriverTurnStates,
};
pub use retention::{
    RetentionReport, apply_retention, prune_run_events_before, prune_run_telemetry_before,
    prune_sessions_before, prune_tool_calls_before, reindex_fts, trim_session_messages,
};
pub use run_ledger::command_center::{
    CommandCenterView, ControlError, ControlVerb, apply_control, build_view, list_agent_work,
};
pub use store::{db_path, with_connection, with_transaction};
pub use threads::{
    ConversationPurgeStats, ConversationStore, ConversationThread, CreateConversationThread,
    CrossThreadHit, ThreadMessage, ThreadMessagePatch,
};
pub use transcript::spend::{ThreadSpend, TranscriptSpend, thread_spend, transcript_spend};
pub use turn_state::TurnStateMirror;
pub use types::{
    SessionMessage, SessionRecord, SessionSearchParams, SessionSearchResult, SessionStatus,
    SessionToolCall,
};

#[cfg(test)]
#[path = "lib_tests.rs"]
mod test;
