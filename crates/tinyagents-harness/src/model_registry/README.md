# harness::model_registry

Runtime-owned executable model registry and selection policy.

## Why this exists

Provider-neutral inference types and calls live in `tinyinference`; this
module owns only the harness-side concerns: registering executable models
under runtime names, and deterministically picking one for a given call.

## Public surface

- [`ModelRegistry<State>`] (`types.rs` + `mod.rs`) — name-keyed registry of
  `Arc<dyn ChatModel<State>>` handles plus one designated default (the first
  model registered, unless overridden with `set_default`). `register`,
  `get`, `default_model`, `default_name`, `names` are the basic accessors;
  `resolve`/`resolve_request` are the selection entry points.
- [`ModelSelection`] (`types.rs`) — the input policy for one resolution:
  explicit override, reusable previous selection, priority-ordered runtime
  hints, agent-level default, and a capability requirement, each field
  corresponding to one precedence tier `resolve` walks.
- [`ResolvedModelBinding<State>`] (`types.rs`) — the resolution result: the
  executable `model` handle plus `resolved`
  (`tinyinference_llm::model::ResolvedModel`) metadata recording which name
  and source won.
- [`model_eligible`] (`mod.rs`, crate-private) — the shared eligibility check
  every candidate in `resolve` passes through: capability satisfaction plus,
  unless `allow_retired` is set, usability.

## Files

| File       | Role                                                              |
| ---------- | ---------------------------------------------------------------------- |
| `types.rs` | `ModelSelection`, `ModelRegistry`, `ResolvedModelBinding` definitions (plus their `Debug` impls). |
| `mod.rs`   | `ModelRegistry` methods, the `resolve` selection algorithm, `model_eligible`/`model_satisfies`/`binding` helpers. |
| `mod_tests.rs` | Unit tests for registration, default handling, and each resolution precedence tier. |

## Operational constraints

- `resolve`'s precedence order is a contract, not an implementation detail:
  explicit request override → reusable previous selection → priority-sorted
  hints → agent default → registry-wide default. A source that names an
  unregistered or ineligible model is skipped, not treated as a hard failure
  — resolution only returns `None` once every tier is exhausted. Any change
  to this order is a behavior change, not a docs change.
- Hint ordering is a **stable** sort: equal-priority hints keep the caller's
  original order rather than an arbitrary one. Preserve that stability if the
  sort is ever touched.
- `allow_retired` only ever *widens* eligibility (letting a retired model be
  selected); it must never be used to route fresh, non-pinned selections to a
  retired model — that path exists specifically for finishing an in-flight
  conversation already pinned to one.
- `ModelRegistry::register` sets the default implicitly on the *first*
  registration only; subsequent registrations never change it unless
  `set_default` is called explicitly. A test or caller relying on
  registration order to pick the default should register the intended
  default first.
