//! Transcript-derived view: project the append-only `session_raw/*.jsonl`
//! source of truth into typed display items for the chat renderer, with
//! newest-first pagination over a bounded in-memory cache.
//!
//! Entry point: [`get_page`] (used by the `threads.transcript_get` RPC).

mod cache;
mod project;
mod prompt_tools;
mod resolve;
mod status_map;
mod subagents;
pub mod types;

use std::path::Path;

use serde::Serialize;

pub use project::{project_records, project_thread, project_thread_scoped, resolve_files_scoped};
#[allow(deprecated)]
pub use types::SubagentStatus;
pub use types::{DisplayItem, ProjectedTranscript, ToolCallStatus, TranscriptSubagentStatus};

/// Key under which the writer stamps per-result tool failures into a
/// transcript message's extra metadata (`{call_id: {detail}}`); the
/// projection reads it back to mark a tool row as failed. The writer side
/// lives in the host's transcript codec and imports this constant, so the two
/// halves cannot drift.
pub const TOOL_RESULT_FAILURES_METADATA_KEY: &str = "openhuman_tool_failures";

const LOG_PREFIX: &str = "[threads][transcript]";

/// Default page size — roughly one screen of chat items.
pub const DEFAULT_LIMIT: usize = 50;
/// Hard upper bound on a single page so a client can't request an unbounded
/// projection slice.
pub const MAX_LIMIT: usize = 500;

/// One newest-first page of a thread's projected transcript.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptPage {
    pub thread_id: String,
    /// Display items for this page, **newest-first**.
    pub items: Vec<DisplayItem>,
    /// Total top-level items available for the thread.
    pub total: usize,
    /// Opaque cursor to pass back for the next (older) page; `null` at the end.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// `true` when more (older) items remain beyond this page.
    pub has_more: bool,
    /// `false` when the thread has no persisted transcript yet (empty page).
    pub has_transcript: bool,
}

/// Project `thread_id`'s transcript and return one newest-first page.
///
/// `cursor` is the opaque token returned as `next_cursor` by a previous call
/// (an offset from the newest item); `None`/empty starts at the newest item.
/// `limit` defaults to [`DEFAULT_LIMIT`] and is clamped to [`MAX_LIMIT`].
pub fn get_page(
    workspace_dir: &Path,
    thread_id: &str,
    cursor: Option<&str>,
    limit: Option<usize>,
) -> TranscriptPage {
    get_page_scoped(workspace_dir, thread_id, None, cursor, limit)
}

/// Fetch a page scoped to an owning agent. Use this when thread IDs can be
/// supplied by callers and shared by multiple agents.
pub fn get_page_scoped(
    workspace_dir: &Path,
    thread_id: &str,
    agent_id: Option<&str>,
    cursor: Option<&str>,
    limit: Option<usize>,
) -> TranscriptPage {
    let limit = limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let upper_cursor = parse_cursor(cursor);

    let projected = if let Some(agent_id) = agent_id {
        project::project_thread_scoped(workspace_dir, thread_id, Some(agent_id))
            .map(std::sync::Arc::new)
    } else {
        cache::global().get_or_project(workspace_dir, thread_id)
    };
    let Some(projected) = projected else {
        tracing::debug!("{LOG_PREFIX} get_page thread={thread_id}: no transcript");
        return TranscriptPage {
            thread_id: thread_id.to_string(),
            items: Vec::new(),
            total: 0,
            next_cursor: None,
            has_more: false,
            has_transcript: false,
        };
    };

    let total = projected.items.len();
    // Cursor is an exclusive chronological upper bound, so appending at the
    // end does not move the boundary for the next older page.
    let upper = upper_cursor.map_or(total, |cursor| cursor.min(total));
    let lower = upper.saturating_sub(limit);
    let items: Vec<DisplayItem> = projected.items[lower..upper]
        .iter()
        .rev()
        .cloned()
        .collect();
    let has_more = lower > 0;
    let next_cursor = has_more.then(|| lower.to_string());

    tracing::debug!(
        "{LOG_PREFIX} get_page thread={thread_id} total={total} upper={upper} returned={} has_more={has_more}",
        items.len()
    );

    TranscriptPage {
        thread_id: thread_id.to_string(),
        items,
        total,
        next_cursor,
        has_more,
        has_transcript: true,
    }
}

/// Parse the opaque cursor into an exclusive chronological upper bound
/// (`None` on absent/invalid, meaning "start from the newest item").
fn parse_cursor(cursor: Option<&str>) -> Option<usize> {
    cursor
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .and_then(|c| c.parse::<usize>().ok())
}

#[cfg(test)]
#[path = "transcript_subagent_anchor_tests.rs"]
mod subagent_anchor_tests;
#[cfg(test)]
#[path = "transcript_view_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "prompt_tools_tests.rs"]
mod prompt_tools_tests;

#[cfg(test)]
#[path = "transcript_ordering_tests.rs"]
mod ordering_tests;

#[cfg(test)]
#[path = "mod_status_alias_tests.rs"]
mod status_alias_tests;
