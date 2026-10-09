# `tinyagents-registry` — named capability catalog

The **registry** is the named-addressable catalog that makes TinyAgents
recursive: a host session can reference a model, tool, graph, or agent *by
name* rather than hardcoding it, and the registry
resolves that name to a real handle — with the allowlist guarantee that
only capabilities a human explicitly registered can be invoked.

## Public surface

- **`CapabilityRegistry<State>`** — the core registry (generic over application
  state `State`). Stores executable model and tool capabilities, agent
  definitions, and metadata for descriptor kinds. Enforces duplicate detection
  and provides aliasing and snapshot export.
- **`ComponentKind`, `ComponentId`, `ComponentMetadata`** — the vocabulary for
  registering and discovering capabilities by name and kind. Metadata is
  durable and serializable; [`RegistrySnapshot`] projects it for audit logs,
  UIs, and discovery.
- **`ModelCatalog`** — a checked-in offline snapshot of provider model prices,
  context windows, and capability flags. Lets recursive runs estimate costs and
  gate features (tool calling, vision, etc.) before dispatch, without a network
  round-trip.
- **`ModelRouter`** — the declarative workload-tier layer that maps host tier
  aliases (`chat-v1`, `vision-v1`, …) to concrete registered models, with
  per-tier capability gates and same-family fallback chains.
- **`RegistryDiagnostic`, `RegistrySnapshot`** — machine-readable registry
  state for introspection, diffing, and health checks.

## Design and invariants

- **Execution vs. metadata:** The registry stores live model/tool handles
  (execution) but only names and metadata for routers/reducers/stores (discovery).
  This decoupling lets metadata survive process exits and be rendered in UIs.
- **Name-safe recursion:** By owning the allowlist of legal names, the registry
  is the boundary that makes agent-authored plans safe to compile — a
  self-authored workflow can only reference what a human pre-registered.
- **State genericity:** The registry is generic over an application `State`
  because models and tools are generic over it. Stateless capabilities use
  `State = ()`.
- **Deterministic metadata:** Every successful registration records
  [`ComponentMetadata`] keyed by `(kind, name)`. Duplicate checks,
  alias resolution, and presence queries all work uniformly across kinds.
- **Offline model facts:** The [`ModelCatalog`] is a compile-time embedded
  snapshot, so recursive runs have model costs and capabilities available
  offline, deterministically, without depending on a live provider API.

## File and module map

- **`capability/`** — the [`CapabilityRegistry`] implementation: registration,
  lookup, aliasing, duplicate detection, and bridges to the harness
  ([`ModelRegistry`], [`ToolRegistry`]).
  - `types.rs` — the [`CapabilityRegistry`] struct and storage maps.
  - `mod.rs` — registration and accessor methods.
  - `mod_tests.rs` — tests for registration, alias, and lookup.
- **`component/`** — identity and discovery types: [`ComponentKind`],
  [`ComponentId`], [`ComponentMetadata`]. Used by every other part of the
  registry and by host applications.
  - `types.rs` — the data types.
  - `mod.rs` — constructors and string conversions.
  - `mod_tests.rs` — tests for kind, id, and metadata.
- **`router/`** — [`ModelRouter`], the declarative workload-tier layer.
  Maps host tier aliases to concrete models with capability gates and
  fallback chains. Holds policy, not models.
  - `types.rs` — [`WorkloadRoute`] struct.
  - `mod.rs` — [`ModelRouter`] registration, resolution, and fallback
    construction.
  - `mod_tests.rs` — tests for routing, fallback, and capability gating.
  - `README.md` — detailed design and example (see existing).
- **`catalog.rs`** — [`ModelCatalog`]: offline model facts (prices, context
  windows, capabilities). Embedded at build time from a JSON snapshot.
- **`diagnostics.rs`** — [`RegistrySnapshot`] (serializable point-in-time view)
  and [`RegistryDiagnostic`] (health checks: dangling aliases, name collisions).
- **`lib.rs`** — the crate's public surface and re-exports.

## Relationship to other modules

- **Depends on:** `tinyagents-harness` (model/tool registries, error types),
  `tinyagents-definition`
  (agent definitions), `tinyinference-llm` (chat models, capabilities).
- **Used by:** host code for registration, discovery, and model/tool dispatch.
- **Test coverage:** Integration tests in `crates/tinyagents-integration-tests/`
  cover registry binding and serialization.
