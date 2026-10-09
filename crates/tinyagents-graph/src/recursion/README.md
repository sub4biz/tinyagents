# graph::recursion

Recursion policy and depth tracking: the bound-and-observe contract for
nested graph/subgraph/sub-agent recursion.

The graph runtime allows recursion — a graph can embed itself as a subgraph, a
subgraph can call back into its parent, an agent node can call another agent
that re-enters the same graph, a router can loop nodes until state converges,
and a `Send` fanout can schedule many recursive child tasks — but only under
explicit, enforced limits. This module is that contract; `compiled::executor`
is the sole enforcer, building one `RecursionStack` per run, pushing/popping
frames around every nested call, and stamping the live stack into checkpoint
metadata and `GraphEvent::RecursionDepthChanged`.

## Public surface

- `RecursionFrame` — one level of the run tree: `graph_id`, `node_id` (the
  hosting node, `None` for a root frame), `run_id`, `task_id`, `namespace`,
  `depth`, and `parent` (the enclosing frame's run id). Serialized into
  checkpoint metadata so a UI can render nested runs without replaying event
  logs.
- `RecursionPolicy` — the configured caps, tracked independently:
  `max_depth` (run-tree depth, default 25), `max_visits_per_node` (optional
  cap on activations of one node in a run, default unbounded),
  `max_total_steps` (supersteps per run, default 1000). `Default` gives the
  conservative defaults above.
- `RecursionStack` — the live frame stack for one executing run, paired with
  the policy that bounds it:
  - `new(policy)` / `with_frames(frames, policy)` (the latter seeds inherited
    parent frames for a subgraph/sub-agent child run).
  - `push`/`pop` — symmetric frame push/pop; `push` enforces `max_depth`
    (`TinyAgentsError::SubAgentDepth`) and leaves the frame unpushed on
    failure.
  - `depth()` / `frames()` — current depth and the frame list, root-first.
  - `check_total_steps(steps)` — enforces `max_total_steps`
    (`TinyAgentsError::RecursionLimit`).
  - `record_node_visit(counts, node)` — increments and enforces
    `max_visits_per_node` (`TinyAgentsError::NodeVisitLimit`) when configured.
- `ChildRun` — a reference to a child run spawned from a subgraph/sub-agent
  node: `node`, `graph_id`, `run_id`, `root_run_id`, and rolled-up `usage`
  (only populated for sub-agent children; subgraph children track usage
  through their own model calls).
- `ChildRunSink` — a thread-safe, per-run collector the executor hands to node
  contexts so a subgraph/sub-agent node can report the `ChildRun` it spawned;
  `record`/`drain` (drained once per superstep boundary).
- `RunTree` — a flat, after-the-fact parent/child lineage view derived from a
  completed run: `run_id`, `root_run_id`, `parent_run_id`, `children`;
  `is_root()` reports whether `parent_run_id` is `None`. The run-id
  counterpart to the live `RecursionStack`.

## Files

| File | Role |
| --- | --- |
| `types.rs` | All type definitions and their `impl` blocks. |
| `mod.rs` | Module doc only (re-exports; no additional behavior). |
| `mod_tests.rs` | Unit tests for the stack contract and its enforcement inside the executor. |

## How it fits together

- `builder::types::GraphBuilder`/`compiled::types::CompiledGraph` carry
  `recursion_policy`, `recursion_frames` (inherited from an enclosing run),
  and `recursion_node` (the hosting node when embedded); `subgraph` and
  `subagent_node` are what seed `recursion_frames` so a nested run extends the
  parent's tree instead of starting fresh.
- `compiled::executor` enforces the three caps with distinct errors
  (`SubAgentDepth`, `NodeVisitLimit`, `RecursionLimit`), records the current
  stack into checkpoint metadata under a `recursion` array, and emits
  `GraphEvent::RecursionDepthChanged` on depth changes.
- `compiled::types::GraphExecution::run_tree()` builds a `RunTree` from the
  run's own ids plus its accumulated `child_runs` — the completed-run
  counterpart to the live `RecursionStack`.
- `status::GraphRunStatus` carries `root_run_id`/`parent_run_id` in the same
  shape, so the live-status and completed-run views of the recursion tree
  stay consistent.

## Operational constraints

- `RecursionStack::push`/`pop` must stay balanced around every nested call —
  an executor path that pushes without a matching pop (e.g. on an early
  return from an error branch) leaves the stack permanently deeper than the
  actual call depth for the rest of that run.
- `max_visits_per_node` counts are per-run, not persisted across a
  resume/retry from a checkpoint — a long-running resumed thread does not
  inherit visit counts from before the checkpoint boundary.
