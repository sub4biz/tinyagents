# harness::store

Long-term key-value and append-only stream storage backends for the harness
runtime.

## Why this exists

In the recursive architecture the store is the durable, shared substrate that
outlives any single run: parent and child runs, sub-agents, and
nested executions read and write the same namespaced values, so a
deeply nested call can persist a result a sibling or a later turn picks up.
It is the harness-side persistence layer for runtime data — events, model and
tool call records, message history, artifacts, memory — and is intentionally
separate from graph checkpointing (which belongs to `tinyagents-graph`'s
`checkpoint` module) and from prompt/model context assembly (`prompt`,
`context`).

## Public surface

### Flat key-value: [`Store`]

The original get/put/delete/list trait over a flat `&str` namespace.

- [`InMemoryStore`] — thread-safe, non-durable, for tests and prototyping.
- [`FileStore`] — one file per key under `<root>/<namespace>/<key>.json`, with
  name sanitization (path-traversal guard) and atomic writes (write-to-temp +
  rename).
- [`StoreRegistry`] — a named bag of `Arc<dyn Store>` backends, always
  carrying a built-in default in-memory store, injected into `RunContext`.

### Append-only streams: [`AppendStore`]

Where `Store` answers "what is the current value?", an `AppendStore` answers
"what happened, in order?" — the durable backbone for event journals.

- [`InMemoryAppendStore`] — thread-safe, non-durable, with an optional
  per-stream retention cap (`with_max_entries_per_stream`); offsets stay
  monotonic across eviction.
- [`JsonlAppendStore`] — one `<stream>.jsonl` file per stream, one JSON line
  per entry. Derives the next offset from the tail of the file (not a
  remembered counter), so it stays correct across multiple instances/process
  restarts addressing the same directory, and tolerates a write torn
  mid-character or mid-line.
- [`StoreRecord`] — a decoded entry (`offset`, `value`, `created_at_ms`)
  returned by `JsonlAppendStore` internals and available as a convenience type.

### Hierarchical store: [`namespaced`]

The richer, TTL-aware, batch-oriented sibling trait — see its own README.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Implementations of `Store` and `AppendStore` for every backend, plus `StoreRegistry`. |
| `types.rs` | The `Store`/`AppendStore` traits and every backend struct/field. |
| `namespaced/` | `NamespacedStore` — hierarchical namespaces, TTL, filtering, pagination, batching. See `namespaced/README.md`. |
| `test.rs` | Coverage for every backend: round-tripping, namespace isolation, sanitization, atomicity, offset monotonicity/tailing, retention eviction, and JSONL crash-recovery edge cases. |

## Operational constraints

- `FileStore` and `JsonlAppendStore` validate namespace/key/stream names
  (ASCII alphanumerics, `-`, `_`, `.` only; no all-dot names) to block path
  traversal — untrusted names must go through these backends, never be
  interpolated into a path directly.
- `FileStore::put` writes to a uniquely named temp file in the same directory
  and renames over the destination, so a reader never observes a partial write
  and a crash mid-write leaves the previous value intact.
- `JsonlAppendStore::append` runs its blocking I/O via `spawn_blocking` when a
  tokio runtime is present (inline otherwise), and holds a per-instance guard
  across the tail-read-then-write so concurrent appends through *one*
  instance get distinct offsets. Across separate processes writing the same
  directory, duplicate offsets are unlikely but not impossible — a real
  server backend is the right answer for genuinely concurrent multi-process
  writers.
- None of the in-memory backends are durable: data is lost when the value is
  dropped.

## Storage-driver backends (`storage-drivers` feature)

`DriverStore` and `DriverAppendStore` adapt the harness `Store` and
`AppendStore` traits onto [tinystoragedrivers](https://github.com/tinyhumansai/tinystoragedrivers)
ports. A host opens one storage backend (SQLite on desktop, MongoDB in the cloud,
memory in tests), takes a scoped `DocumentStore` / `StreamStore` for the tenant,
and wraps it:

```rust,ignore
let scoped = backend.for_scope(&Scope::new(agent_id)?)?;
let kv = DriverStore::new(scoped.documents().clone());
let journal = DriverAppendStore::with_prefix(scoped.streams().clone(), "journal/");
```

- `DriverStore` keeps every namespace in one collection (`harness_store` by
  default) as `{ns, key, value}` documents, so namespaces and keys may hold any
  characters, and `list` is an indexed query.
- `DriverAppendStore` maps each stream to a driver stream (optionally
  prefixed). Driver offsets are already dense and zero-based, matching the
  `AppendStore` contract.
- Driver `InvalidInput` errors surface as `TinyAgentsError::Validation`. Every
  other driver error surfaces as `TinyAgentsError::Storage`.
