//! Durable push-completion for detached children.
//!
//! A detached child runs on its own and finishes whenever it finishes. The
//! parent still has to hear about it, exactly once, even if the parent is idle,
//! mid-turn, or the process restarts in between. This module owns that
//! hand-off:
//!
//! - [`CompletionRecord`]: one finished child, keyed by task id, with a
//!   delivery [`CompletionState`] and an attempt counter.
//! - [`CompletionStore`]: [`InMemoryCompletionStore`] and the crash-safe
//!   [`JsonlCompletionStore`].
//! - [`CompletionRouter`]: dedupe, tombstones, batch claims with attempt
//!   counting, give-up, and routing by [`NotifyMode`] onto the parent's queue
//!   lanes.
//! - [`CompletionFormatter`]: the host's wording. The harness ships only a
//!   neutral default.
//!
//! What stays with the host: when a parent is idle enough to receive a
//! delivery turn, the turn itself, and what to do with a record that gave up.
//! Nothing here relaunches a child.

mod format;
mod router;
mod store;
mod types;

pub use format::{CompletionFormatter, NeutralCompletionFormatter};
pub use router::{CompletionRouter, DEFAULT_MAX_ATTEMPTS, RecordOutcome, TombstoneOutcome};
pub use store::{CompletionStore, InMemoryCompletionStore, JsonlCompletionStore};
pub use types::{
    CompletionArtifact, CompletionRecord, CompletionResult, CompletionState, CompletionStatus,
    NotifyMode,
};

#[cfg(test)]
#[path = "router_tests.rs"]
mod router_tests;
#[cfg(test)]
#[path = "store_tests.rs"]
mod store_tests;
