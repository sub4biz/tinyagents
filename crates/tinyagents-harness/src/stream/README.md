# harness::stream

Higher-level streaming projections for the harness — the typed lens an
observer uses to watch a run (and, transitively, the sub-agents and
sub-graphs it spawns) unfold in real time.

## Why this exists

State snapshots, diffs, model deltas, debug traces, and interrupts from a run
are projected as filtered [`StreamChunk`]s so a parent driving the run can
consume only the categories it cares about,
instead of every caller re-implementing delta reassembly and mode filtering
over raw [`crate::events::AgentEvent`]s.

The chunk *types* (`types.rs`) are independent of `crate::events`; only the
projection (`project.rs`) depends on it, so callers that never touch events
pay nothing for the coupling.

## Public surface

- [`StreamMode`] — selects which chunk categories a consumer wants: `Values`,
  `Updates`, `Messages`, `Debug`, `Interrupts`, `Custom`.
- [`StreamChunk`] — the typed union of chunk categories, adjacently tagged
  for serialization (`{"type": …, "content": …}`). `mode()` returns the
  `StreamMode` that gates a given chunk.
- [`StreamSink`] — a synchronous, mode-filtered buffer. `new`/`all` construct
  it; `push`/`push_event` submit chunks/events; `drain`/`peek` read the
  buffer; `enable`/`disable` adjust the active mode set at runtime.
- [`stream`] — a synchronous helper filtering an already-collected slice of
  chunks by a set of modes (tests, post-processing).
- [`project_event`] / [`project_event_for_modes`] / [`projected_mode`] — the
  `AgentEvent` → `StreamChunk` projection. `project_event_for_modes` is the
  one a streaming run loop should call per event: it skips
  clone/serialization work for any mode nobody subscribed to.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `StreamSink` method implementations and the standalone `stream()` filter helper. |
| `project.rs` | The `AgentEvent` → `StreamChunk` projection: `project_event`, `project_event_for_modes`, `projected_mode`, and the routing table in its module docs. |
| `types.rs` | `StreamMode`, `StreamChunk`, `StreamSink` type definitions. |
| `mod_tests.rs` | `StreamSink` filtering/push/drain/peek/enable/disable, `StreamChunk::mode` mapping, and the `stream()` helper. |

## Operational constraints

- Each `AgentEvent` projects to **at most one** `StreamChunk`, so a consumer
  subscribed to several modes never receives the same event twice in two
  shapes (see the routing table in `project.rs`).
- `StreamMode::Values` is **never** produced by the projection — a full state
  snapshot is graph state, which the event stream does not carry; the graph
  runtime pushes `StreamChunk::Values` itself via `StreamSink::push`.
- `StreamMode::Custom` is likewise never produced by projection — by
  definition it is the caller's own extension channel, pushed directly.
- `StreamSink` is synchronous and single-threaded (`RefCell`-backed buffer);
  wrap it in `Arc<Mutex<_>>` to share across threads.
