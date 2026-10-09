# harness::testkit

Deterministic test doubles and trajectory assertions that make nested,
model-driven behaviour testable without a live provider.

## Why this exists

In the recursive architecture this is how nested, model-driven behaviour is
made *deterministically testable*: scripted/streaming model doubles, fake
tools, controllable clocks/ids, and a [`Trajectory`] over recorded
[`crate::events::AgentEvent`]s let tests assert exactly what an agent — and
the sub-agents and sub-graphs it spawns — did, all without a live provider.
The same [`EventRecorder`] that observes a top-level run also captures
child-run events fanned onto a shared sink, so recursion is observable in
tests.

## Public surface

| Type | Purpose |
| --- | --- |
| [`StreamingMock`] | A `ChatModel` replaying scripted `ModelStreamItem`s verbatim on `stream`, and the merged response those items fold into on `invoke`. |
| [`SlowModel`] | A `ChatModel` that sleeps a fixed delay before replying — for deterministically triggering a wall-clock timeout. |
| [`ScriptedModel`] | A `ChatModel` returning pre-loaded responses in order; records every received `ModelRequest`; errors (not panics) when the queue is exhausted. |
| [`FakeTool`] | A configurable `Tool` (`new`/`returning`/`failing`) that records every invocation's arguments. |
| [`DeterministicClock`] | A controllable millisecond clock that only advances when told to. |
| [`DeterministicIds`] | A monotonic `"{prefix}-N"` id generator. |
| [`EventRecorder`] | Subscribes an internal listener to an `EventSink` and captures every emitted `AgentEvent`. |
| [`Trajectory`] | Structural (tool-called, model-call-count, ordering, completed/failed) assertions over a sequence of events, with both predicate and panicking `assert_*` forms. |

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Implementations for every double and for `Trajectory`'s assertion methods. |
| `types.rs` | Struct/enum definitions for every type in the table above. |
| `mod_tests.rs` | Exercises every double and the trajectory assertions with synthetic inputs. |

## Operational constraints

- Every double is intended for tests only; none of them are exported outside
  `#[cfg(test)]`-adjacent usage conventions, but the module itself is a
  regular `pub mod` so integration tests in
  `crates/tinyagents-integration-tests/` can use it too.
- `ScriptedModel` and `StreamingMock`/`SlowModel` implement
  `ChatModel<State>` generically over any `State: Send + Sync`, so they work
  against both the harness agent loop and graph-embedded model calls.
- `Trajectory::assert_order` checks a **subsequence** match (other events may
  appear between matched labels), not exact adjacency — see its doc comment
  for the exact label-matching rules (`kind()` string, or `tool_name` for
  `ToolStarted`/`ToolCompleted`, or `route` for `RouteSelected`).
