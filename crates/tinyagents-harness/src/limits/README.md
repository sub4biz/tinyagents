# harness::limits

Run-scoped limit enforcement: the policy type, the live tracker that enforces
it, and the outcome/kind vocabulary that lets callers see a cap decision as
data.

## Why this exists

Limits are what keep recursion bounded: because agents can call agents and
graphs can run graphs, an unbounded run tree could fan out forever or burn a
provider budget. `RunLimits::max_depth` caps how deep the sub-agent/sub-graph
recursion may go, while the call and wall-clock caps bound the work within
each run. Every model call and tool call must go through `LimitTracker` so
limits are fail-closed.

## Public surface

- [`RunLimits`] (`types.rs` + `mod.rs`) — the policy: `max_model_calls`,
  `max_tool_calls`, `max_wall_clock_ms` (whole-run deadline),
  `max_model_call_ms` (per-call ceiling, bounds an individual hung call
  independent of the run deadline), `stream_idle_timeout_ms` (default 120 s,
  streaming silence bound that applies after the first output event) /
  `stream_first_event_timeout_ms` (opt-in, default `None`: no separate bound on
  the first output event) with the per-model
  `max_consecutive_stream_idle_timeouts` breaker (default 5),
  `max_retries_per_call`, `max_depth`
  (default `RunLimits::DEFAULT_MAX_DEPTH` = 8), and `behavior`
  ([`LimitBehavior`]). Built with `with_*` chained setters.
- [`LimitBehavior`] (`types.rs`) — what happens when a call cap is reached:
  `Error` (historical default; fails the run) or `StopWithPartial` (stops the
  loop cleanly and keeps the transcript accumulated so far).
- [`LimitOutcome`] (`types.rs`) — the cap decision as data (`Proceed` or
  `Stop(LimitKind)`), returned by the `try_record_*` tracker methods.
- [`LimitKind`] (`types.rs`) — which cap tripped (`ModelCalls` / `ToolCalls`).
  Deliberately mirrors `crate::events::LimitKind` rather than reusing it, so
  this leaf module has no dependency on the observability layer.
- [`LimitTracker`] (`mod.rs`) — the live counters plus a wall-clock start
  instant. `record_model_call`/`record_tool_call` are the hard-error form;
  `try_record_model_call`/`try_record_tool_call` return a `LimitOutcome` and
  honor `RunLimits::behavior`. `rollback_tool_calls` un-counts calls that were
  requested but never executed (the `StopWithPartial` tool path).
  `check_wall_clock`/`elapsed`/`remaining_wall_clock` are the deadline
  queries. `sync_call_limits` (fail-open, plain assignment in both
  directions) and `tighten_call_limits` (fail-closed, keeps the stricter cap)
  are the two ways a second limit source (a harness-wide `RunPolicy`) can
  reconcile with the tracker's current caps.

## Files

| File       | Role                                                              |
| ---------- | ---------------------------------------------------------------------- |
| `types.rs` | `RunLimits`, `LimitBehavior`, `LimitOutcome`, `LimitKind`, and `RunLimits::default()`. |
| `mod.rs`   | `RunLimits` builders, `LimitTracker` and its methods.                 |
| `mod_tests.rs`  | Unit tests: counter/cap smoke path, fail-open vs. fail-closed reconciliation, error-vs-stop-with-partial exhaustion. |

## Operational constraints

- All limits are checked fail-closed: the first call that exceeds a cap
  returns an error (or, under `StopWithPartial`, a clean stop) and the run
  should not proceed further on that axis.
- `sync_call_limits` and `tighten_call_limits` are **not** interchangeable.
  `sync_call_limits` is a plain assignment in both directions (can raise or
  lower a cap) and is fail-open with respect to widening; `tighten_call_limits`
  only ever keeps the stricter of the tracker's current cap and the supplied
  one. The harness agent loop resolves the effective cap itself per axis
  (config explicitly set → stricter of config/policy; config merely defaulted
  → policy wins outright) and then calls `sync_call_limits` once with the
  already-reconciled values — see `RunLimits`'s note on
  `LimitTracker::sync_call_limits` for why `tighten_call_limits` is not used
  there instead.
- `max_model_call_ms` deliberately does **not** apply to tool calls: a
  sub-agent delegation tool call wraps an entire child run and must not
  inherit a model-call-sized timeout. Tools carry their own
  `ToolTimeoutSettings` deadlines and remain bounded by the run's remaining
  wall-clock budget.
- `record_model_call`/`record_tool_call` increment the counter **before**
  checking the cap, so a cap of `N` allows exactly `N` calls (inclusive), not
  `N - 1`. Preserve that ordering in any future change.
