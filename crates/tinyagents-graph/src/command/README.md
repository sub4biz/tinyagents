# graph::command

The vocabulary a node handler returns to steer the durable executor: partial
state updates, explicit routing, dynamic fanout, and human-in-the-loop
interrupts.

A node in the durable runtime no longer returns whole state directly. It
returns a [`NodeResult`], and this module is where every shape that result can
take is defined and constructed. `compiled::routing` consumes `Command::goto`
targets when deciding the next superstep's active set; `compiled::executor`
matches on `NodeResult::Interrupt` to pause a run and persist a resumable
checkpoint.

## Public surface

- `NodeResult<Update>` — what a node handler returns: `Update` (merge through
  the reducer), `Command` (update + explicit routing + optional resume value),
  or `Interrupt` (pause for human input).
- `Command<Update>` — combines an optional partial update with explicit `goto`
  routing (one or more targets, overriding static/conditional edges) and an
  optional resume value. Constructors: `Command::new`, `::goto`, `::send`,
  `::update`, `::resume`; builder methods `with_update`, `with_goto`,
  `with_sends`, `with_resume`.
- `RouteTarget` — one routing target: `Node` (plain activation against shared
  state) or `Send` (a fanout packet carrying a per-invocation argument).
- `Send` — schedules a node for the next superstep with a custom `arg`
  delivered via `NodeContext::send_arg`, independent of the shared committed
  state; the map-reduce / search-fanout / per-item-scoring primitive. A single
  `Command::send` call may target the same node many times, each with its own
  `arg`.
- `Interrupt` — a human-in-the-loop pause point (`id`, `node`, `payload`).
  Requires a checkpointer: the executor persists a checkpoint at the
  boundary and returns control to the caller; `CompiledGraph::resume` re-runs
  the interrupted node with the resume value attached.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `NodeResult`, `Command`, `RouteTarget`, `Send`, `Interrupt` definitions. |
| `mod.rs` | Constructors and builder methods on those types; interrupt id generation. |
| `mod_tests.rs` | Unit tests for command/interrupt construction. |

## How it fits together

- `builder` nodes are `NodeHandler<State, Update>` closures whose `Result`
  future resolves to a `NodeResult<Update>` — this module is the return type
  of every node in the graph.
- `compiled::routing::route` resolves `Command::goto` targets (when a node
  returned a command) ahead of static/conditional edges; `RouteTarget::Send`
  entries become distinct activations of the same node, each carrying its own
  argument, rather than being deduplicated like plain `Node` targets.
- `compiled::executor` checkpoints and pauses a run on `NodeResult::Interrupt`,
  and resumes it by feeding a `Command::resume` value back into the
  interrupted node's `NodeContext::resume`.

## Operational constraints

- An interrupt with no configured checkpointer (or no thread id) is rejected
  before it can pause a run — see `compiled::routing::require_interrupt_durability`
  — because an interrupt that cannot be persisted cannot be resumed.
- `Send` targets are never deduplicated by node id the way plain `goto`
  targets are; a node that wants to fan out N times to the same target must
  emit N distinct `Send` packets, each with its own `arg`.
