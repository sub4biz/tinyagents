# graph::parallel::claims

Shared-workspace claim arbitration for parallel agent fan-out — a decision
module with **no I/O and no scheduling**: given a set of `WorkspaceClaim`s it
returns a `DispatchPlan` saying which workers may run concurrently, which must
be serialized, and which cannot be scheduled at all.

`parallel::map_reduce` answers "run these N items with bounded concurrency."
It cannot answer whether running them concurrently is *safe*. When fanned-out
workers share one filesystem root, two of them mutating the same file at the
same time corrupts it silently — there is no error, just a plausible-looking
result built on a torn tree. This module owns that decision, and only that
decision.

## The rule

A worker is safe to run in parallel when it either has its own root
(`isolated`) or never writes (`!writes`). A worker that writes a *shared* root
must declare the paths it owns; it is then serialized, and its claim is
checked against every claim already granted. Claims are granted in **input
order, first-writer-wins**, which makes the plan a pure function of the input
regardless of which worker happens to finish first — the same request always
yields the same rejection.

## What stays with the caller

The crate reports a `ClaimConflict` as data; it never decides what a host does
about one. Whether an unbounded write is a hard rejection or a warning, and
what sentence the user reads, are product decisions. Likewise the syntax a
claim arrives in: `parse_relative_claim_paths` takes the *body* of a claim
list, not whatever prefix or key a host wraps it in.

## Public surface

- `WorkspaceClaim` — one worker's declared relationship to the shared
  workspace (`isolated`, `read_only`, `writing` constructors).
- `plan_shared_workspace_dispatch(claims) -> DispatchPlan` — the planner.
- `DispatchPlan` — `modes` (index-aligned `Option<DispatchMode>`, `None` =
  rejected) plus `conflicts` (index-keyed rejections); `has_serial_work`,
  `parallel_indices`, `serial_indices`.
- `DispatchMode` — `Parallel` / `Serial`.
- `ClaimConflict` — `UnboundedWrite` (writer with no declared paths) /
  `Overlap` (claim collides with an earlier one); `worker_id()` accessor.
- `ClaimPathError` — `Absolute` / `Escaping`, returned by
  `parse_relative_claim_paths`.
- `parse_relative_claim_paths(spec) -> Result<Vec<PathBuf>, ClaimPathError>` —
  parses a comma/newline-separated, optionally bulleted claim list into safe,
  sorted, deduplicated relative paths.
- `paths_overlap(left, right) -> bool` — component-wise (not textual) overlap
  check: `src/a` and `src/ab` are distinct, `src/a` and `src/a/inner.rs`
  collide.
- `writes_shared_workspace(effects: &ToolSideEffects) -> bool` — derives a
  "does this tool write?" claim input from tool side-effect metadata.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `WorkspaceClaim`, `ClaimPathError`, `ClaimConflict`, `DispatchMode`, `DispatchPlan`. |
| `mod.rs` | `parse_relative_claim_paths`, `paths_overlap`, `writes_shared_workspace`, `plan_shared_workspace_dispatch`. |
| `mod_tests.rs` | Unit tests (path parsing/safety, overlap semantics, planner ordering and conflict reporting). |

## Operational constraints

- What a host *does* with a `ClaimConflict` (hard rejection vs. warning, and
  what a user reads) is a product decision this crate deliberately does not
  make — it only reports the conflict as data.
- `parse_relative_claim_paths` rejects absolute paths and anything containing
  `..` or a platform path prefix, because either would let a claim reach
  outside the shared root it is meant to partition. This is a security
  boundary, not a convenience check — do not bypass it by constructing
  `WorkspaceClaim::writing` paths from unvalidated input.
- Claim order matters: `plan_shared_workspace_dispatch` grants first-writer-
  wins in the order `claims` is given, so callers that need deterministic,
  reproducible plans must pass claims in a stable order.
- This module performs no I/O and spawns nothing; it is pure decision logic
  the caller (e.g. `parallel::map_reduce`) uses to decide dispatch, not an
  executor.

## Relation to neighbours

Sits alongside `parallel::map_reduce` (parent module) as the "is this safe?"
half of parallel fan-out; `map_reduce` handles "run it," this module handles
"may it run concurrently." Consumed by hosts that fan out agent/tool workers
over a shared filesystem root before dispatching them through `map_reduce` or
an equivalent scheduler.
