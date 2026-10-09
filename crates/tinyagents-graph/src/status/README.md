# graph::status

`GraphRunStatus` — a compact, cheaply-pollable summary of a graph run,
distinct from a checkpoint.

Where a `checkpoint::Checkpoint` preserves the *resumable state* of a run, a
`GraphRunStatus` is a lightweight record an observer can read to answer "is
this run active?", "which node is executing?", "which interrupt is waiting?"
— without deserializing full graph state. Because each status carries
`root_run_id` / `parent_run_id`, these records form the same run-tree shape
as `recursion::RunTree`, connecting a parent run to every subgraph/sub-agent
run it spawns, so an observer can roll up progress, interrupts, and errors
across levels of recursion.

## Public surface

- `GraphRunStatus` — the record itself: run/root/parent run ids, thread id,
  graph id, latest checkpoint id and namespace, coarse `ExecutionStatus`,
  current superstep, active nodes, pending interrupts, last event id,
  timestamps (`started_at`/`updated_at`/`ended_at`), and a rendered error
  summary for failed runs.
  - `GraphRunStatus::new(run_id, graph_id, status)` — a fresh status for a
    top-level run with no recorded progress (`root_run_id` mirrors `run_id`,
    zeroed step/nodes/interrupts).
  - `GraphRunStatus::is_terminal()` — true for `Completed`, `Failed`, or
    `Cancelled`.

## Files

| File | Role |
| --- | --- |
| `types.rs` | The `GraphRunStatus` struct definition. |
| `mod.rs` | `GraphRunStatus::new` and `::is_terminal`. |
| `mod_tests.rs` | Unit tests for the constructor's defaults and terminal-state detection. |

## How it fits together

- `compiled::types::GraphExecution` carries a `status: GraphRunStatus`
  snapshot at the final boundary of a run.
- `observability::GraphStatusStore` (opt-in via
  `CompiledGraph::with_status_store`) persists `GraphRunStatus` updates as a
  run progresses, so an external poller doesn't need to replay the event
  journal to answer "is this run still running?".
- This module has no dependency on `checkpoint`, `stream`, or `observability`
  — it defines only the record shape; the executor and observability layer
  are what populate and persist it.
