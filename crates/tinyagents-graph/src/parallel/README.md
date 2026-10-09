# graph::parallel

Ordered, bounded-concurrency parallel map/reduce with a configurable failure
policy, plus (in `claims/`) the shared-workspace claim arbitration that
decides when fanned-out workers are safe to run concurrently at all.

Graph `Send` fanout is the low-level primitive for parallel supersteps, but
callers frequently want a *reusable* "run these N items concurrently and
reduce the results" helper independent of the graph executor: deterministic
input-order results, a concurrency cap, per-item success/failure isolation,
and a policy for what to do when some items fail (fail-fast, collect-all,
quorum, best-effort). That is what [`map_reduce`] provides.

## Public surface

- `map_reduce(items, options, f) -> Result<ParallelOutcome<T>>` — runs `f`
  over `items` with bounded concurrency (`ParallelOptions::max_concurrency`,
  `0` = unbounded), collecting per-item outcomes in **input order** regardless
  of completion order, and applying `options.failure_policy`.
- `ParallelOptions` — builder-style options: `with_max_concurrency`,
  `with_failure_policy`, `with_item_timeout`, `with_total_timeout`,
  `with_cancellation`.
- `FailurePolicy` — `FailFast` (return the first input-order error, cancel the
  rest), `CollectAll` (default; always `Ok`, per-item outcomes), `Quorum(n)`
  (error unless at least `n` items succeed), `BestEffort` (always `Ok`, keep
  only successes).
- `ItemOutcome<T>` / `ParallelOutcome<T>` — per-item and aggregate results,
  with `success_count`/`failure_count`/`successes`/`into_successes` helpers.
- `claims` (re-exported): `WorkspaceClaim`, `DispatchPlan`, `DispatchMode`,
  `ClaimConflict`, `ClaimPathError`, `parse_relative_claim_paths`,
  `paths_overlap`, `writes_shared_workspace`,
  `plan_shared_workspace_dispatch` — see `claims/README.md`.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `FailurePolicy`, `ParallelOptions`, `ItemOutcome<T>`, `ParallelOutcome<T>`. |
| `mod.rs` | `map_reduce`: the bounded-concurrency, input-order-preserving driver. |
| `mod_tests.rs` | Unit tests (ordering under out-of-order completion, each failure policy, timeouts, cancellation). |
| `claims/` | Shared-workspace claim arbitration (own `README.md`). |

## Operational constraints

- `map_reduce` re-orders by input index after collection; it does **not**
  stream partial results — callers needing incremental output should not use
  this helper.
- `FailFast` cancels remaining in-flight work by dropping the underlying
  stream once every item with a smaller input index has resolved, not merely
  on the first error observed — see the `fail_fast_error` bookkeeping in
  `mod.rs` for why the first-completed error is not necessarily the one
  returned.
- `item_timeout` and `total_timeout` are independent: an item exceeding
  `item_timeout` becomes a per-item failure handled by `FailurePolicy`; the
  whole call exceeding `total_timeout` aborts unconditionally with
  `TinyAgentsError::Timeout`, even under `CollectAll`/`BestEffort`.

## Relation to neighbours

`map_reduce` answers "run these N items concurrently"; `claims::plan_shared_workspace_dispatch`
(see `claims/README.md`) answers the orthogonal question of whether running
fanned-out workers concurrently is *safe* against a shared filesystem root.
Neither module depends on the other at the type level — a caller composes
them: plan dispatch first, then drive the resulting parallel/serial groups
through `map_reduce` (or an equivalent scheduler).
