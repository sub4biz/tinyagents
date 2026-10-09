# Transcript view

This module projects persisted transcript JSONL files into display records and
serves newest-first pages. The transcript files remain the source of truth;
the bounded projection cache in `cache.rs` is an optimization for unscoped
reads. Agent-scoped reads resolve roots separately so one agent's request does
not reuse another scope's cached projection.

## Public surface

`get_page` and `get_page_scoped` return `TranscriptPage` values. The `project`
module exposes record and thread projection helpers, while `types.rs` defines
the serialized display item and status shapes. Cursors are opaque exclusive
chronological upper bounds; callers should pass back `next_cursor` unchanged.

## Resolution and ordering

`resolve.rs` selects root generations and sub-agent transcript files.
`subagents.rs` correlates child sessions with spawning calls and places their
records in the root conversation. A scoped child must have a durable ownership
relationship to an accepted root; a matching thread identifier alone is not
sufficient. Projection retains chronology before pagination reverses each page
for display.

## Operational constraints

Transcript reads are best-effort over append-only JSONL sources. Malformed or
unreadable candidates are skipped with diagnostics. Keep pagination cursors
stable and do not treat a cache hit as an authorization decision.

`status_map.rs` expresses `TranscriptSubagentStatus` in terms of the canonical `OrchestrationTaskStatus` (`From` in, `TryFrom` out; `Cancelled` has no projection). Its serialized strings are unchanged. Mapping table: `tinyagents-tasks` README.
