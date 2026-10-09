# graph::stream

The live, in-process event surface: pluggable sinks and the wire vocabulary
(`GraphEvent`, `StreamMode`) the durable executor uses to narrate its own
execution as it happens.

The executor emits a `GraphEvent` at every meaningful boundary — run
start/end, superstep start/end, node lifecycle, retries, routing decisions,
checkpoints, interrupts, subgraph enter/exit, forked branches, recursion depth
changes, and arbitrary custom writes. Routing those events into a
`GraphEventSink` is what lets a UI or an enclosing graph watch a
subgraph or sub-agent execute in real time; because every event carries its
node/step, nested runs' streams can be merged and attributed back up the run
tree. `observability` is what makes this same event vocabulary durable
(journaled, replayable) rather than merely live — see
`observability/README.md`.

## Public surface

- `GraphEventSink` (trait) — `emit(&self, event: GraphEvent)` receives one
  event and must not block the executor; `flush(&self)` (default no-op) lets
  an async-persisting sink let callers wait for the durable log to catch up
  (the executor calls it after a terminal run event).
- `NoopSink` — drops every event.
- `CollectingSink` — records every event (in emission order) for inspection
  in tests and UIs: `new()`, `events()`, `len()`, `is_empty()`.
- `GraphEvent` — the event enum itself (see `types.rs` for every variant);
  `kind()` returns a stable dot-separated name (e.g. `"node.completed"`) for
  logging/filtering; `step()` returns the associated superstep number when the
  variant carries one.
- `StreamMode` — LangGraph-style selection of which projection of the stream a
  caller wants: `Values`, `Updates`, `Messages`, `Debug`, `Interrupts`,
  `Custom`. The milestone executor exposes this as a plain selection enum;
  richer typed `StreamPart` projection is noted as future work.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `GraphEvent` (every emitted variant), `StreamMode`. |
| `mod.rs` | `GraphEventSink` trait, `NoopSink`, `CollectingSink`. |
| `mod_tests.rs` | Unit tests for the two built-in sinks. |

## How it fits together

- `CompiledGraph::with_event_sink` attaches a `GraphEventSink`; the executor
  (`compiled::executor`, `compiled::routing`) calls `self.emit(...)`
  throughout the superstep loop.
- `GraphEvent` is `serde`-(de)serializable so a single event can be wrapped
  into a durable `observability::GraphObservation` envelope, journaled, and
  replayed — the live stream and the durable journal share this same event
  shape rather than maintaining parallel vocabularies.
- `GraphEvent::step()` is what stamps the `step` field of a
  `GraphObservation` when journaling.

## Operational constraints

- A `GraphEventSink::emit` implementation must not block the executor thread
  — a sink that persists asynchronously should hand off work (e.g. to a
  channel or background task) and implement `flush` to let callers wait for
  it to settle, rather than doing the write inline.
- `CollectingSink` is unbounded and in-memory; it is meant for tests and
  short-lived UIs, not as a durability mechanism for long-running production
  runs.
