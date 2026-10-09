# events

Typed, in-process observability layer for the harness. This is the *live*
event vocabulary — every model call, tool call, middleware hook, routing
decision, and (crucially) sub-agent boundary the harness passes through
during a run — fanned out synchronously to registered listeners. It is the
foundation `observability/` builds durability on top of: that module wraps
each [`AgentEvent`] in a durable [`AgentObservation`](../observability/README.md)
and persists it; this module only distributes it while a run is live.

Because TinyAgents is recursive (agents call agents, graphs run graphs), a
single top-level request fans out into a tree of runs. `AgentEvent` variants
like `SubAgentStarted`/`SubAgentReused`/`SubAgentCompleted`/`SubAgentFailed`
carry recursion `depth` so a listener can reconstruct that tree from one
flat event stream, and `HarnessRunStatus` threads `root_run_id` /
`parent_run_id` lineage for the same purpose in a compact, poll-friendly
snapshot rather than a stream.

## Public surface

- [`AgentEvent`] — the tagged (`"kind"`) enum of every lifecycle transition:
  run boundaries, model calls (including streaming deltas), tool calls,
  middleware start/complete/fail, budget/cost/usage accounting, routing and
  fallback decisions, retries, caching, compression, memory, workspace, and
  sub-agent recursion. Most `*Started`/`*Completed` pairs also have a
  `*Failed` terminal partner so an exporter pairing calls by id never sees an
  open span for a call that actually errored.
  Run boundaries carry a typed [`TerminalOutcome`](crate::terminal::TerminalOutcome) on
  `RunCompleted`/`RunFailed`, and `TurnStarted`/`TurnCompleted`/
  `MessageAppended` let a consumer mirror the transcript turn by turn; see
  `docs/modules/harness/terminal-outcome.md`.
- [`EventRecord`] — an [`AgentEvent`] paired with a stable [`EventId`] and a
  monotonic stream `offset`.
- [`EventListener`] — the `Send + Sync` trait a pluggable observer
  implements (`fn on_event(&self, record: &EventRecord)`); must stay
  low-latency since it runs synchronously on the emitting thread.
- [`EventSink`] — the cloneable fan-out bus: `emit` assigns an id/offset and
  delivers to every subscribed listener, `subscribe`/`unsubscribe` manage
  listeners. Delivery is globally ordered by offset across concurrent
  emitters, and a panicking listener never permanently wedges the sink.
- [`RecordingListener`] — an in-memory listener that buffers every record it
  receives, for tests and ad hoc inspection.
- [`EventJournal`] — an append-only in-memory record of everything emitted
  through its internal sink, replayable from any offset. Distinct from
  `observability::HarnessEventJournal`: this one is process-local and not
  pluggable/durable.
- [`HarnessRunStatus`] — a compact "what is running now?" snapshot (phase,
  counters, active calls, cumulative usage/cost, lineage) with lifecycle
  helpers (`mark_running`, `mark_completed`, `mark_failed`,
  `mark_interrupted`).

## File map

| File | Role |
| --- | --- |
| `types.rs` | Every public type listed above, including the full `AgentEvent` variant vocabulary; this is the module's entire public surface. |
| `mod.rs` | Behavior: `EventSink`'s ordered-dispatch/panic-safety implementation, `RecordingListener`/`EventJournal`/`HarnessRunStatus` impls, and poisoned-lock recovery helpers shared across them. |
| `test.rs` | Fan-out/replay ordering (including under concurrency and a panicking listener), poisoned-lock recovery, `HarnessRunStatus` transitions, and `AgentEvent` serde round-trips. |

## Operational constraints

- **Listeners must be cheap.** `EventSink::emit` calls every listener
  synchronously on the emitting thread before returning (or, for a
  re-entrant emit while another emitter is draining, before that emitter's
  drain loop reaches it); heavy work (I/O, serialization, network) belongs in
  a listener that hands off to a background task — see
  `observability::worker::AppendWorker`, which every durable sink in
  `observability/` uses for exactly this reason.
- **Delivery is offset-ordered, not per-thread-ordered.** Concurrent `emit`
  calls interleave under one critical section for id/offset assignment, then
  a single draining emitter delivers queued records in offset order, so a
  listener never observes offset `n + 1` before offset `n`.
- **A poisoned lock is recovered, not fatal.** Every `Mutex` in this module
  guards a plain buffer/counter with no cross-field invariant a half-finished
  update could break, so a panicking listener poisoning a lock does not take
  down the event bus for the rest of the process.
- **`EventSink` vs. `EventJournal` vs. `observability::HarnessEventJournal`:**
  a sink only fans out live; a journal (this module) additionally buffers
  everything for in-process replay but is neither durable nor pluggable; only
  `observability::HarnessEventJournal` persists across process restarts. Most
  callers construct one `EventSink`, subscribe whatever combination of a
  `RecordingListener` (tests), an `EventJournal` (in-process replay), and an
  `observability` sink (durability) they need, and emit through the sink.

## Tool progress

`AgentEvent::ToolProgressDetail { call_id, message, fraction, partial }` (wire kind
`tool.progress`) is emitted while a tool is still running, when it reports
through `tinytools::ToolRunContext::report_progress`. Every progress event for a
call lies between its `ToolStarted` and its terminal `ToolCompleted` /
`ToolFailed`; updates reported after the call settles are dropped. Calls in one
concurrent batch interleave. A flooding tool is coalesced (default 32 events per
second per call), so the stream is a thinned view; the held final update is flushed when the call settles, even past the window limit. The
mechanics live in [`../tool/progress/mod.rs`](../tool/progress/mod.rs).
