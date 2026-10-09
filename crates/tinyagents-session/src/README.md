# `session` — durable session history, run ledger, and transcripts

SQLite-backed history for agent sessions (requires the `sqlite` feature), a
restart-survivable ledger for background agent/workflow execution, a
JSONL-backed transcript store for KV-cache-stable resume, and a JSONL chat
thread/message store for the product-facing chat log.

## Why this is a top-level module

Session history is a persistence domain in its own right, not part of the agent
loop. Nothing in `harness` reads from it, and a host can use it without running
a harness at all — indexing sessions produced elsewhere, or recovering
orchestration state at boot before any agent exists. Filing it under `harness::`
would imply a dependency that exists in neither direction.

## How it differs from the other persistence layers

| Layer | Question it answers | Lifetime |
| --- | --- | --- |
| `harness::store` | "what is this run working with right now?" | during a run |
| `graph::checkpoint` | "how do I resume this interrupted run?" | until resumed |
| **`session` (SQLite)** | "what happened, what did it cost, how did runs nest?" | indefinitely |
| **`session::transcript` (JSONL)** | "what exact messages did the model see?" | indefinitely, resumed verbatim |
| **`session::threads` (JSONL)** | "which chats does the user have, and what is in them?" | indefinitely |

The SQLite-backed history (`ops`, `run_ledger`) is queryable history that
nothing resumes *from*: cross-session search, cost attribution, and
orchestration recovery. `session::transcript` is the exception — it exists
specifically so a restarted run can resume with the byte-identical message
stream a provider (and its prompt cache) already saw; see its own
[README](./transcript/README.md) for the format.

## Layout

Every entry point takes the workspace root and derives the path itself, so a
host chooses only where its workspace lives:

```text
{workspace_dir}/session_db/sessions.db          ← SQLite: sessions, run ledger
{workspace_dir}/session_raw/{stem}.jsonl        ← JSONL: transcripts (source of truth)
{workspace_dir}/sessions/YYYY_MM_DD/{stem}.md   ← human-readable transcript view
{workspace_dir}/memory/conversations/           ← JSONL: chat threads + messages
```

## Public surface

Re-exported from the crate root (see `src/lib.rs`); the full surface stays
reachable under `session::`, `session::run_ledger::`, and
`session::transcript::`.

- **Recording** — `record_session_start`, `record_message`,
  `record_message_with_reasoning`, `record_tool_call`, `record_session_end`.
  Use `record_message_with_reasoning` when an assistant response has hidden
  reasoning that must remain separate from its visible content;
  `record_message` remains the compatibility wrapper for content-only callers.
- **Querying** — `get_session`, `list_sessions`, `search_sessions`,
  `list_messages`, `list_tool_calls`, `list_children`
- **Recovery** — `mark_interrupted`
- **Retention** — `apply_retention`, `prune_sessions_before`,
  `prune_tool_calls_before`, `prune_run_events_before`,
  `prune_run_telemetry_before`, `trim_session_messages`, `reindex_fts`
- **Run ledger** — agent runs, workflow runs, teams, members, tasks, run events,
  and telemetry, with the claim/completion coordination primitives — see its
  own [README](./run_ledger/README.md)
- **Transcripts** — full-rewrite and append-only writers, model-context and
  display readers, thread lookups and usage summaries — see its own
  [README](./transcript/README.md)
- **Chat threads** — `ConversationStore` and its wire types (re-exported at
  the root), plus the free functions and channel subscriber under
  `session::threads` — see its own [README](./threads/README.md)
- **Store port** — `SessionStoreProvider`, `AgentStores`, `TurnStates`,
  `InMemorySessionStores`, and with feature `storage-drivers`
  `DriverSessionStores` over a `tinystoragedrivers` backend — see
  [`docs/modules/session/store-port.md`](../../../docs/modules/session/store-port.md)
- **Connections** — `with_connection` (autocommit) and `with_transaction`
  (`BEGIN IMMEDIATE`)
- **Testkit** — `testkit::conformance::run_ledger_conformance` and
  `transcript_history_conformance`: contract suites certifying that a run
  ledger workspace or a `TranscriptHistory` implementation behaves like the
  bundled ones. Run against the SQLite run ledger (two independent
  workspaces) and both `FileTranscriptHistory` and the in-memory
  `testkit::InMemoryTranscriptHistory` double in
  `crates/tinyagents-integration-tests/tests/session_conformance.rs`.

## Schema

Six tables plus one FTS5 virtual table, created on demand and idempotently:

| Table | Holds |
| --- | --- |
| `sessions` | one row per session; lineage via `parent_session_id` |
| `session_messages` | per-message visible content, optional assistant reasoning, model, tokens, cost |
| `session_tool_calls` | tool name, input, bounded output, status, duration |
| `sessions_fts` | FTS5 index over session name, message content, tool name |
| `agent_runs` / `workflow_runs` | background execution state |
| `run_events` / `run_telemetry` | per-run event stream and rollups |
| `agent_teams` / `agent_team_members` / `agent_team_tasks` | team coordination |

The file is opened through the tinystoragedrivers SQLite driver's native
mode (`SqliteNative`): one shared connection per file per process, which any
other handle the host opens on `sessions.db` also uses. WAL journaling,
`foreign_keys = ON` and a 5 s busy timeout.

**Durability:** the driver runs WAL with `synchronous = NORMAL`. A commit
survives a process crash, but after an OS crash or power loss the most
recent committed transactions can be rolled back. The database itself stays
consistent and opens normally; what is lost is the tail of recent writes
(the last turns' history rows, run-ledger updates). Anything that must
survive power loss byte-for-byte belongs in the transcript files, which
remain the source of truth for resume. FTS5 comes from `rusqlite`'s `bundled`
build — there is no separate `fts5` cargo feature at 0.40, so do not add one.

## Operational constraints

These are the non-obvious rules; each is pinned by a test in `lib_tests.rs` and `store_tests.rs`.

**Search input is plain text, not FTS5 syntax.** `SessionSearchParams::query` is
translated to a quoted FTS5 expression before it reaches `MATCH`. Binding raw
user input made ordinary strings (`C++`, `foo-bar`, `file.rs`) fail with a
syntax or `no such column` error instead of searching.

**Indexed content is truncated on a character boundary.** Slicing at a raw byte
offset panics on multi-byte input, and because the message insert has already
committed, the row would survive with no FTS entry — silently unsearchable.

**Tool output is bounded** to `MAX_TOOL_OUTPUT_BYTES`, truncated on a character
boundary with a marker appended.

**Telemetry counters are `Option` for partial updates.** The columns are
`NOT NULL DEFAULT`, and SQLite does not apply a column default to an explicitly
supplied `NULL`, so the insert path coalesces to the default while the update
path coalesces to the stored value. `excluded.*` cannot serve the update side —
it observes the already-coalesced row, so `None` would read as `0` and clobber a
stored counter.

**Run-event sequences are allocated by the INSERT itself.** Reading
`MAX(sequence) + 1` and then inserting is a read-modify-write race; the loser
fails the primary key and the event is lost.

**Coordination operations need `with_transaction`, not `with_connection`.**
`with_connection` is autocommit, which gives ordering but no isolation. Claim
and completion read state and then act on it, so they take the write lock up
front with `BEGIN IMMEDIATE` — racing claims serialize at `BEGIN` rather than
failing at `COMMIT` after one has already decided it won.

**That serialization depends on a busy timeout, which we now set ourselves.**
SQLite's own default is zero — with no busy handler a `BEGIN IMMEDIATE` that
meets a competing writer fails immediately with `SQLITE_BUSY` instead of
waiting. It was never actually zero here: `rusqlite`'s `Connection::open`
installs a 5s timeout unconditionally. `store::BUSY_TIMEOUT` sets the same value
explicitly, so a correctness property the claim/gate/sequence logic relies on is
not silently supplied by a transitive dependency's undocumented default.

**An upsert reads its own write back inside the same transaction.** Inserting on
an autocommit connection, closing it, then re-opening to `get_*` returns
whatever a concurrent writer left behind rather than what this call wrote.

**Schema changes go through the versioned migration list** in `migrations.rs`,
not through more `CREATE TABLE IF NOT EXISTS` at the top of an operation. The
list index *is* the version, so it may only be appended to; retire a migration
by replacing its body with `"SELECT 1;"` rather than deleting it. Without a
version marker no column could ever be added to a workspace database that
already existed.

**Nothing is deleted unless a host asks.** `retention.rs` owns the only delete
paths (sessions, messages, tool calls, run events, telemetry) plus
`reindex_fts`, which rebuilds the search index for rows whose FTS entry was lost
before the row/index pairs became transactional. Retention is a policy decision,
so none of it runs on a schedule of its own.

**A claim is meaningful only while a task is `in_progress`.** An upsert that
moves a task off that status clears `claimed_by_member_id` and `claim_token`;
leaving them set strands the task, since a new claim sees `AlreadyClaimed`,
completion sees `NotClaimed`, and release/shutdown skip it.

**Evidence accumulates across completion attempts,** including attempts whose
gate fails — otherwise a retry after fixing an unrelated gate would fail
`require_evidence` on evidence already submitted.

## Layout of this module

| File | Role |
| --- | --- |
| `lib.rs` | crate docs and public surface |
| `types.rs` | serde record types for sessions/messages/tool calls |
| `store.rs` | connection/transaction helpers and pragma setup |
| `migrations.rs` | versioned, append-only schema migration list |
| `ops.rs` | session recording and querying |
| `retention.rs` | pruning, trimming, and FTS index repair |
| `context.rs` | `StorageContext`, the error-context shim |
| `run_ledger/` | background run + team coordination — see its own [README](./run_ledger/README.md) |
| `threads/` | JSONL chat thread/message store with a cross-thread search index — see its own [README](./threads/README.md) |
| `transcript.rs` + `transcript/` | durable, provider-neutral transcripts for KV-cache-stable resume — see its own [README](./transcript/README.md) |
| `lib_tests.rs`, `store_tests.rs`, `migrations_test_tests.rs` | module-local unit tests |
