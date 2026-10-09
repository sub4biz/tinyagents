# graph::builder

The authoring/compile contract: `GraphBuilder` accumulates nodes, edges,
conditional routing, and a reducer; `.compile()` validates the topology and
freezes it into an immutable `CompiledGraph`.

This is the entry point every workflow uses before it can run. Because a node handler can itself
drive another compiled graph or a sub-agent, the builder is also what a
recursively-generated sub-workflow compiles through, one level down.

## Public surface

### `GraphBuilder<State, Update>`

- `new()` / `overwrite()` (the latter is `State == Update` with the built-in
  overwrite reducer) — construction.
- `set_reducer`, `set_entry`, `set_finish` — required wiring: a reducer and an
  entry node are validated at `compile()`.
- `add_node`, `add_edge`, `add_sequence`, `add_waiting_edge`,
  `add_conditional_edges`, `add_barrier_relief` — topology construction.
- `mark_command_routing`, `with_command_destinations` — declares a node
  routes exclusively via `Command::goto` rather than static/conditional
  edges; `compile()` rejects nodes that mix the two.
- `with_node_kind`, `with_node_metadata`, `mark_subgraph` — behavior-free
  introspection markers surfaced by `graph::export`. `mark_deferred` and
  `mark_interrupt` also set the marker but are *not* behavior-free:
  `mark_deferred` is `NodePolicy::defer`, and `mark_interrupt` is an alias
  for `interrupt_before`.
- `interrupt_before`, `interrupt_after` — executor-level pauses at named
  nodes (before the handler runs / after it runs but before its result is
  applied); see `docs/modules/graph/interrupts.md`.
- `with_parallel`, `with_max_concurrency`, `with_node_timeout`,
  `with_recursion_limit`, `with_graph_id`, `with_name`, `set_defaults` —
  per-graph configuration, either called directly or bundled via
  `GraphDefaults`.
- `compile() -> Result<CompiledGraph<State, Update>>` — validates topology
  (missing nodes, dangling edges, START/END misuse, command-routing conflicts,
  missing reducer) and freezes the builder into an immutable, cheaply-clonable
  `CompiledGraph`.

### Supporting types (`types.rs`)

- `START` / `END` — reserved virtual entry/terminal node names.
- `NodeHandler<State, Update>` / `NodeFuture<Update>` — the boxed async
  handler shape every node closure is stored as.
- `NodeContext` — per-task runtime context passed to a handler: run identity,
  step, resume value, fork identity (`ForkId`) in a parallel step, `send_arg`
  for `Send`/`GraphInput`-scheduled activations, recursion frames, the
  `ChildRunSink`, and an optional `AgentInvocationBinding`.
- `ForkId` — identifies one branch of a concurrent superstep (`branch_index`
  + `node`).
- `Route` — an optional newtype for a conditional-route label, for routers
  that prefer a typed value over a bare string.
- `GraphDefaults` — an optional bundle of per-graph defaults applied in one
  `set_defaults` call; every field is additive (`Some` overrides, `None`
  leaves the current value).
- `RouterFn<State>` — the conditional-router closure shape.
- `BarrierRelief` (in `mod.rs`) — a relief registration for a mixed fan-in
  barrier, so a barrier fed by one unconditional and one conditional-only
  predecessor doesn't deadlock when the conditional branch isn't taken.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `GraphBuilder` struct, `NodeContext`, `ForkId`, `Route`, `GraphDefaults`, `NodeHandler`/`NodeFuture`/`RouterFn` aliases, `START`/`END`, and the crate-private `NodeMeta`/`BuilderNode`/`Branch`. |
| `mod.rs` | All `GraphBuilder` methods (construction, topology, configuration, `compile`) and `BarrierRelief`. |
| `mod_tests.rs` | Unit tests for the compile contract: reducer requirement, START/END validation, missing-node/route detection, command-routing conflicts. |

## How it fits together

- `compile()` is the sole producer of `compiled::CompiledGraph` — see
  `compiled/README.md` for what runs after a builder is compiled.
- `command::Command`/`RouteTarget` are what a node handler returns at
  runtime; `mark_command_routing`/`with_command_destinations` only declare
  the *intent* to route that way so `compile()` can reject a mixed topology
  and `export` can draw the advisory destinations — the runtime always
  resolves the real successor from the emitted `Command`.
- `reducer::StateReducer` is the trait `set_reducer` accepts; `overwrite()`
  is shorthand for the built-in `OverwriteStateReducer`.
- `recursion::RecursionFrame` values seeded into a `NodeContext` come from
  `subgraph`/`subagent_node`, not from this module — the builder only carries
  the `recursion_limit` cap through to the compiled graph.
## Operational constraints

- `compile()` fails closed: no reducer, a missing/self-referential entry,
  dangling edges, START/END misuse, or a node that mixes command routing with
  static/conditional edges all fail validation rather than compiling a graph
  that would misbehave at runtime.
- A `BarrierRelief`'s `relief_node` must actually be one of `barrier_node`'s
  registered waiting predecessors (via `add_waiting_edge`) — a relief
  registered against a barrier with no matching waiting registration is a
  silent no-op at execution time, not a compile error.
- `add_sequence` only wires edges between nodes that must already exist via
  `add_node`; it does not itself add nodes.
