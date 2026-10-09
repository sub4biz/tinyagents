# harness::context

Run configuration and the live runtime context threaded through every step of
a harness run.

## Why this exists

`RunContext` is the unit of recursion in the runtime: every nested layer — a
sub-agent or a sub-graph — runs inside its own context.
`RunConfig::depth`/`RunConfig::max_depth` plus `RunConfig::child` track and
bound how deep that recursion may go, while a shared `CancellationToken` and
event sink let cancellation signals and observability flow across the whole
tree. This module owns the authoritative run-context contract that downstream
middleware, the agent loop, and graph nodes code against.

## Public surface

- [`RunConfig`] (`types.rs` + `mod.rs`) — the declarative, serializable
  description of a run: identity, thread, tags, metadata, limits, and
  [`RunLineage`]. Built with `RunConfig::new` and refined with `with_*`
  builders. `effective_max_model_calls`/`effective_max_tool_calls` resolve an
  unset cap to the crate-default `RunLimits`; `child`/`checked_child_depth`
  are the single source of truth for the sub-agent recursion-depth guard.
- [`RunLineage`] (`types.rs`) — immutable, data-only ancestry (root run,
  parent run, depth, max depth) that can be persisted or replayed without a
  live `RunContext`.
- [`RunContext<Ctx>`] (`types.rs` + `mod.rs`) — the live, non-serializable
  handle bundling `RunConfig` with a `StoreRegistry`, `EventSink`,
  `LimitTracker`, cancellation token, steering handle, workspace descriptor,
  and arbitrary user `data: Ctx`. `new`/`child` construct top-level and nested
  contexts; `with_*` builders attach shared dependencies; `record_model_call`/
  `record_tool_call`/`check_deadline`/`remaining_wall_clock` enforce and query
  the run's limits.
- [`MiddlewareControl`] (`types.rs`) — a structured control outcome
  (`StopWithFinal` / `Interrupt`) middleware requests via
  `RunContext::request_control`; the agent loop drains it via
  `RunContext::take_control` at its safe checkpoints. Competing requests
  resolve by `precedence()`, never last-writer-wins.
- [`ContextStatistics`], [`context_statistics`], [`estimate_context_tokens`]
  (`stats.rs`) — host-free, tokenizer-free transcript statistics and an
  optional caller-supplied-tokenizer token estimate.

## Files

| File       | Role                                                                |
| ---------- | ---------------------------------------------------------------------- |
| `types.rs` | All type definitions: `RunConfig`, `RunLineage`, `RunContext`, `MiddlewareControl`. |
| `mod.rs`   | `RunConfig`/`RunContext` method implementations, custom `RunConfig` `Deserialize` (with legacy wire-field migration), instance-id minting, metadata merge helper. |
| `stats.rs` | `ContextStatistics` and the two statistics-computing free functions.  |
| `mod_tests.rs`  | Unit tests for defaults, builders, child derivation, control precedence, deadline/limit tracking. |

## Operational constraints

- `RunConfig::max_model_calls`/`max_tool_calls` are `Option<usize>`, not a
  bare `usize`, specifically so "caller asked for N" is distinguishable from
  "nobody asked, so it defaulted." An explicitly-set cap is a ceiling the
  agent loop reconciles with the harness `RunPolicy` by taking the *stricter*
  of the two; an unset cap lets the policy raise or lower it freely. Do not
  collapse this back to a plain default — that reintroduced a real bug once
  (see the field doc on `max_model_calls`).
- Every recursion surface (`SubAgent` and its reuse-session tool) must route
  its `depth + 1` check through
  `RunConfig::checked_child_depth` rather than reimplementing the comparison,
  so the fail-closed depth guard cannot drift out of sync between call sites.
- `RunContext::child` shallow-merges metadata automatically (object keys
  overlay the parent's; a non-null non-object value replaces it; `null`
  inherits the parent's) — see `shallow_merge_metadata` in `mod.rs`.
- `RunContext::instance_id` is process-unique and distinct from
  `RunConfig::run_id`, which is a caller-supplied label two concurrent runs
  may share. Key per-run, cross-clone bookkeeping on `instance_id`, not
  `run_id`.
- `RunConfig`'s custom `Deserialize` impl accepts legacy wire fields
  (`depth`, `max_depth` at the top level, predating `RunLineage`) so old
  persisted configs still load; new serialization always emits `lineage`.
- `RunContext` is intentionally not `Serialize`/`Deserialize` — it owns live
  counters, listener lists, and user handles. Only `RunConfig` (and
  `RunLineage`) are meant to be persisted or replayed.
