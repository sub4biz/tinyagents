# graph::todos

A per-thread **todo list**: an ordered checklist of steps per thread, the
shape Claude Code and Codex keep. The model rewrites the whole list as it
works; each item is a line of text in one of three states, and nothing else
hangs off an item — no ids, no approvals, no assignment, no run log.
Provider-neutral, offline-testable, with no app-specific coupling (no progress
events, RPC envelopes, or scratch fallback).

Distinct from [`graph::goals`](../goals) (a *single* durable objective per
thread): a goal is the completion contract the graph pursues; the list holds
the concrete steps.

## Data model (`types.rs`)

- `TodoStatus { Pending, InProgress, Completed }` (`as_str`, serde
  `snake_case`).
- `TodoItem { content, status }` (serde `camelCase`; `status` defaults to
  `pending` when absent).
- `TodoList { thread_id, items, updated_at }` — the stored value.
- `TodosSnapshot { thread_id, items, markdown }` — every store op returns one.
- `parse_status` (accepts aliases like `todo`→`Pending`, `done`→`Completed`),
  `render_markdown` (`[ ]`/`[~]`/`[x]` markers, one line per item),
  `normalise_list` (trimming, empty-content drop, `updated_at` stamp).

`updated_at` is unix-epoch millis as a string (dependency-free, no `chrono`).

## Persistence (`store.rs`)

One serialized `TodoList` per thread under the `graph.todos` namespace of a
`crate::harness::store::Store`, keyed by `hex(thread_id)`. Mutations run
`load → mutate → normalise → put` under a per-thread async mutex (a weak-value
`graph::thread_locks::ThreadLockMap`, so idle threads' mutexes are reclaimed
instead of leaking) — atomic within one process (same single-process caveat as
`graph::goals::store`). Ops: `replace` / `clear` / `list`, plus the raw
`get` (absent versus present-empty, un-normalised) and `delete` for hosts.

Invariant: **single in-progress** — at most one item may be `InProgress`; a
violation is a `Validation` error on `replace`, never silently fixed, so the
model is told to narrow its focus.

## Tool (`tool.rs`)

`TodoTool` is a harness `Tool` named `todo`. One call writes the whole list
(`{"todos": [{"content", "status"}]}`); omitting `todos` reads it back. Build
it with `todo_tools(store)` or `register_todo_tools`. The target thread comes
from `ToolExecutionContext::thread_id` (never a tool argument); the bare
`Tool::call` entry point errors without a thread. Domain errors (blank
content, unknown status, two in-progress) are surfaced to the model as tool
errors rather than failing the run.

## Example

```rust,ignore
use std::sync::Arc;
use tinyagents_graph::{TodoItem, TodoStatus, TodoTool, todo_store};
use tinyagents_harness::store::{InMemoryStore, Store};

let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());

// Programmatic:
let snap = todo_store::replace(&store, "thread-1", vec![
    TodoItem::with_status("Write the RFC", TodoStatus::InProgress),
    TodoItem::new("Review it"),
]).await?;
println!("{}", snap.markdown); // - [~] Write the RFC\n- [ ] Review it

// Or register the `todo` tool for a model to drive:
let tool = TodoTool::new(store.clone());
```

## Files

| File | Role |
| --- | --- |
| `types.rs` | Item/list model, `parse_status`, `render_markdown`, `normalise_list`, `TodosSnapshot`. |
| `store.rs` | `Store`-backed `replace`/`list`/`clear`, per-thread RMW lock, single-in-progress invariant. |
| `tool.rs` | The `todo` tool. |
| `mod_tests.rs` | Unit tests (types, store, tool). |
