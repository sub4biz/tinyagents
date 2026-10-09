# The session store port

`tinyagents_session::port` is where a host plugs in its own storage for
everything an agent persists while it runs. A host implements one
`SessionStoreProvider`; the runtime asks it for each agent's `AgentStores`
and never names a file, a database or a directory itself.

```rust
pub trait SessionStoreProvider: Send + Sync {
    fn for_agent(&self, agent_id: &str) -> AgentStores;
    fn recover(&self) -> anyhow::Result<()> { Ok(()) }          // once, at boot
    fn destination_key(&self) -> Option<String> { None }        // for logs
    fn workspace_dir(&self) -> Option<PathBuf> { None }         // file-backed only
}

pub struct AgentStores {
    pub transcripts: Arc<dyn TranscriptLocator>,  // what the model sees
    pub turn_states: Arc<dyn TurnStates>,         // snapshots of turns in flight
    pub kv: Arc<dyn Store>,                       // run status, goals, todos
    pub journal: Arc<dyn AppendStore>,            // each run's events
}
```

## Why a port

The file and SQLite layout (`session_raw/`, `session_db/sessions.db`,
`tinyagents_store/`, turn-state files) suits one operator on one machine.
A cloud host serving many users from one process needs every agent's state
in a shared database, scoped by agent, so a conversation outlives the
process that served it. The port lets both use the same runtime.

## Isolation

`for_agent` is the only way to obtain stores, and the handles it returns are
bound to that agent. A provider serving several users from one backend must
make it impossible for one agent's handles to reach another's data; the
single-user desktop layout shares one workspace between its agents and does
not claim isolation.

## Sync and async

`TranscriptLocator`/`TranscriptHistory` and `TurnStates` are synchronous:
the turn path that commits a transcript is a chain of sync methods. An
implementation over an async client bridges at its own boundary (`DriverSessionStores` uses a dedicated runtime thread;
`block_in_place` also works, on a multi-threaded tokio runtime only). `Store` and `AppendStore` are
the harness's async traits; `Arc<dyn Store>` and `Arc<dyn AppendStore>`
implement them, so code generic over a store accepts injected handles, and
`FileStatusStore::over` keeps run status in any `Store`.

## Bundled pieces

- `InMemorySessionStores`: per-agent, process-lifetime stores
  (`InMemoryTranscriptLocator`, `InMemoryTurnStates`, the harness's
  in-memory stores). For tests and hosts that keep nothing.
- `TurnStateStore` implements `TurnStates`, `FileTranscriptLocator`
  implements `TranscriptLocator`; together with `open_session_stores` they
  are the building blocks of a file-backed provider. This crate does not
  assemble one: the host that owns a layout does (OpenHuman's
  `openhuman_rpc::session_store`).
- `TranscriptLocator::append_interrupted_partial` carries the display-only
  partial of an interrupted turn without a file path.
- `DriverSessionStores` (feature `storage-drivers`): every agent's stores in
  one `tinystoragedrivers` backend, described below.

## `DriverSessionStores`

With the `storage-drivers` feature, a host that has opened a
`tinystoragedrivers` backend (SQLite on a desktop, MongoDB in the cloud,
memory in tests) gets a complete provider from it:

```rust
let backend: Arc<dyn StorageBackend> = Arc::new(SqliteStorage::open(path)?);
let provider = DriverSessionStores::new(backend)?.recover_on_open(true);
```

Each agent id becomes a storage `Scope`. An id that is not a valid scope,
or that starts with the reserved `sha256:` prefix, maps to `sha256:<hex>` of
itself, so no raw id can name a hashed scope, and the driver enforces scopes, so the
provider claims isolation and passes `session_store_isolation_conformance`.

| Store | Backend shape |
| --- | --- |
| transcripts | `session_transcripts`: one index document per stem (thread, agent, sub-agent flag, creation time); `session_transcript_entries`: an append-only log per stem |
| turn states | `session_turn_states`: one document per `(thread, request)` |
| key-value | `session_kv` through the harness `DriverStore` |
| journal | streams `16:session_journal/<name>` (the harness `DriverAppendStore` always writes `<prefix len>:` first) |

- **Transcript log.** Each write is one entry, numbered from 0 and claimed
  with an insert-only write. An ordinary turn stores only its new rows
  (`extend`); a first write or a compaction stores the whole set (`set`). Two
  writers on one transcript (two processes on one database) serialize: the
  loser re-reads and decides again, which is also how a write racing a seal is
  refused. As in the JSONL writer, a turn whose `prev` is not what is stored
  is refused as stale rather than allowed to replace newer rows, and the
  written rows carry the turn's usage, request id and step stamps, built by
  the writer's own code (`transcript::stamped_rows`).
- **Index.** Each index document records the log entry it was built from
  (`indexed_seq`). A handle with an older replay leaves it alone, and the
  update is a compare-and-swap. A failed index refresh never fails the write
  that preceded it; the next write retries it.
- **Generations.** `begin_generation` reserves the successor in the index
  before sealing the predecessor, so two compactions cannot both open it. A
  reservation left unwritten for 30 seconds (its process stopped mid-way) may
  be taken over, so a sealed head never strands the conversation.
  `begin_generation_from_baseline` checks the baseline inside the seal
  itself, so a turn committed after the caller read it fails the compaction
  instead of vanishing from the successor. A failed seal releases the
  reservation only at the version this call wrote.
- **Turn states.** Conditional writes, settling, the interruption sweep and
  retention pruning are all conditional on the document version.
- **Sync seams.** Transcript and turn-state calls run on a
  `tinystoragedrivers` `Blocking` bridge (one dedicated runtime thread), so
  they work from any caller, inside a runtime or not.
- **Recovery.** A backend cannot list its scopes, so `recover` covers the
  agents this provider has opened. `recover_on_open(true)` also interrupts an
  agent's in-flight turns the first time it is opened. The agent counts as
  recovered only once the sweep succeeds, and a concurrent first open waits
  for that sweep. Use it only when a
  single process owns the database (the desktop app).
- **Failing closed.** If the backend cannot bind an agent's scope, `for_agent`
  returns stores that refuse every call with that error and does not cache
  them. `try_for_agent` returns the error itself.

## Conformance

`testkit::conformance::session_store_conformance(&provider)` exercises a
whole provider: transcripts (sessions, thread and agent lookups, compaction
generations, partials kept out of the replay), turn states (conditional
writes, settling, interrupted-marking) and the key-value and journal stores.
`session_store_conformance` runs against `InMemorySessionStores`, the file
building blocks and `DriverSessionStores` over the memory and SQLite drivers. It checks the common session-store behavior, not provider
isolation. `session_store_isolation_conformance` is a separate check that two
agents cannot see each other's data; it runs only against
`InMemorySessionStores`, `DriverSessionStores` and hosts whose providers claim
isolation.
