# `orchestration::workflow` — durable phase DAG execution

A **workflow** is a directed acyclic graph of phases, each with associated
agents, dependencies, and concurrency limits. The [`WorkflowEngine`] schedules
phases topologically, spawns bounded-concurrent child tasks per phase,
collects results, and manages state transitions durably.

## Public surface

- **`WorkflowEngine<E: WorkflowExecutor>`** — the execution engine: schedules
  runnable phases, spawns child tasks, handles concurrency limits, collects
  results, and persists state. Generic over a host-supplied executor.
- **`WorkflowExecutor`** — the trait for host-supplied work: create and
  monitor child tasks, cancel them, and query their status.
- **`WorkflowStore`, `SessionWorkflowStore`** — durable run state: loads,
  saves, claims (lease), and compare-and-swap updates. [`SessionWorkflowStore`]
  wraps the session run ledger.
- **`WorkflowDefinition`, `WorkflowPhase`** — the declarative phase DAG: phase
  names, descriptions, agent ids, dependencies, concurrency settings.
- **`PhaseStatus`** — phase state machine: Pending → Running → (Completed |
  Failed). Interrupted phases can reset to Pending for retry.
- **State projection functions** — query and mutate the JSON phase-state
  document: `phase_status()`, `next_runnable_phase()`, `all_phases_completed()`,
  `reset_running_phases()`, `phase_prompt()`, `synthesize_summary()`.

## Design and invariants

- **Durable at every boundary:** Workflow runs and phase states are durably
  persisted via [`WorkflowStore`]. The engine is safe to interrupt and resume.
- **Topological scheduling:** Phases are scheduled in dependency order. The
  engine finds the next runnable phase (all dependencies met), preventing
  partial execution and cycles.
- **Bounded parallelism:** The engine respects `default_concurrency` (agents
  per phase in parallel) and `max_children` (total spawned children at once).
  Phase work is fan-out (parallel agents) + fan-in (collect results).
- **Deterministic retry:** If the engine is interrupted while running phases,
  all running phases are reset to Pending (outputs cleared). Completed phases
  remain immutable and are never retried.
- **Result aggregation:** Upstream outputs (from dependency phases) are
  collected and passed to downstream phases' prompts, allowing workflows to
  reason over prior results.
- **JSON phase state:** The phase-state document is JSON (a BTreeMap of phase
  names to `{ status, outputs, reason, ... }`). This projection is durable,
  queryable, and renderable in UIs.

## File map

- **`engine.rs`** — [`WorkflowEngine`] implementation: phase scheduling, child
  task spawning and monitoring, state persistence, concurrency enforcement.
  [`WorkflowStore`], [`SessionWorkflowStore`], [`WorkflowExecutor`] trait.
- **`types.rs`** — data types: [`WorkflowDefinition`], [`WorkflowPhase`],
  [`DefinitionError`], [`WorkflowDefinitionListResponse`].
- **`state.rs`** — phase-state projection: JSON document queries and mutations.
  [`PhaseStatus`], `next_runnable_phase()`, `phase_prompt()`,
  `synthesize_summary()`, etc.
- **`graph.rs`** — scheduler DAG: models workflow phases as a directed graph
  for topological sorting and dependency resolution.
- **`child_step.rs`** — runs each agent child through `SubagentDriver`
  (`subagent::run_agent_step`). `WorkflowEngine::with_step_config` applies an
  `AgentStepConfig` (spawn admission scoped per workflow run id and targeted
  at the agent id, timeout/retry/budget, result policy, role). The default is
  inert; a refused, timed-out or over-budget child fails its phase with the
  reason, and a result-policy-trimmed output replaces the raw output.
- **`validate.rs`** — structural validation: no duplicate phases, valid
  dependencies, no cycles, valid concurrency settings, etc.
- **`tests.rs`** — tests for scheduling, phase transitions, concurrency,
  result aggregation, and interruption/retry behavior.

## Relationship to other modules

- **Depends on:** `tinyagents-graph` (DAG validation), `tinyagents-session`
  (run ledger), `tinyagents-harness` (cancellation, error types).
- **Used by:** host orchestration logic (workflow management and execution).
- **Integration:** Can model multi-agent phases as teams (via
  `orchestration::teams`).

## Typical usage

1. Define a [`WorkflowDefinition`] with phases and dependencies.
2. Create a [`WorkflowEngine`] with a host-supplied [`WorkflowExecutor`] and
   [`WorkflowStore`].
3. Call `engine.run()`: the engine schedules phases, spawns bounded child
   tasks, collects results, and persists state.
4. On interruption, retry `engine.run()`: running phases reset to Pending;
   completed phases remain done.
