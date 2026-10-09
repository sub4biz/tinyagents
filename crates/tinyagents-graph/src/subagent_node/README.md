# graph::subagent_node

> **Deprecated.** `SubAgentNode` and `subagent_node` are marked
> `#[deprecated(note = "use tinyagents_orchestration SubagentDriver")]`. Graph
> cannot depend on orchestration, so the node cannot adapt the driver; its
> behaviour is unchanged. Use `SubagentDriver` for new work.

Sub-agent nodes — the graph node that delegates to a harness *agent* (a
model-driven agent loop) invoked through an explicit host capability.

Where `graph::subgraph` embeds an entire `CompiledGraph` as a node (graph
recursion), this module embeds a *harness agent* as a node (graph-to-agent
recursion): a graph step hands its work to a host-selected,
independently-observable agent and folds the agent's answer back into the
parent graph state. A graph never synthesizes a harness `RunContext` itself;
it only ever calls out through a host-bound `AgentInvoker`, so a graph without
one fails closed rather than silently fabricating agent behavior.

## Public surface

- `SubAgentNode<State, Update>` — binds an agent name to an `InputMapper`
  (parent `State` → `SubAgentInput`), an `OutputMapper` (`SubAgentOutput` →
  parent `Update`), and a `SubAgentPolicy`. `::new` / `::from_fns` construct
  it; `.with_policy` sets the policy.
- `subagent_node(node) -> Handler<State, Update>` — lowers a `SubAgentNode`
  into an ordinary graph node handler: resolves the agent's `AgentInvoker`
  from the node context, projects state into `SubAgentInput`, mints a child
  `run_id` parented to the enclosing run, runs under the policy
  (timeout/retry/budget), records the child run (with its usage) onto the
  parent's execution rollup, forwards the child's events, and folds the
  output into a parent update.
- `AgentInvoker` (trait) — the object-safe host boundary for graph-to-agent
  recursion; implementations must dispatch through the same host entry point
  used for a top-level harness run and create the child via
  `RunContext::child`.
- `AgentInvocation` — an explicit request for a host-owned recursive agent
  invocation (agent id, mapped input, graph/node identity, parent/root run
  ids, forwarded event sink and cancellation).
- `AgentInvocationBinding` — the atomic, execution-scoped capability supplied
  to one graph run (`invoker` + forwarded `events` + `cancellation`); never
  stored on a reusable `CompiledGraph`.
- `SubAgentInput` / `SubAgentOutput` — the structured carriers crossing the
  graph↔agent boundary; `SubAgentOutput` also carries `UsageTotals` and call
  counts so usage rolls up and a budget can be enforced.
- `SubAgentPolicy` — timeout/retry/budget policy for one invocation; defers to
  the harness `RetryPolicy` for backoff. Default is a conservative *single
  attempt, no timeout, no budget* so a node never silently re-runs a
  non-idempotent agent.
- `SubAgentBudget` — optional caps on model/tool calls and input/output tokens,
  checked *after* the child run returns; violating one fails the node with
  `TinyAgentsError::LimitExceeded`. `apply_call_caps` tightens a `RunConfig`
  for in-run enforcement; `max_cost` is carried but not enforced here
  (`to_budget_limits` hands it to the harness `BudgetMiddleware`).
  `SubAgentPolicy::retry_after_tool_calls` (default `false`) is honoured by the
  orchestration paths; the graph node cannot see whether a failed attempt ran
  tools. The policy types are re-exported from
  `tinyagents_orchestration::subagent`.
- `InputMapper<State>` / `OutputMapper<Update>` — type aliases for the parent↔
  child mapping closures.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `AgentInvocation`, `AgentInvocationBinding`, `AgentInvoker`, `SubAgentInput`, `SubAgentOutput`, `SubAgentBudget`, `SubAgentPolicy`, `InputMapper`, `OutputMapper`, `SubAgentNode`. |
| `mod.rs` | `SubAgentNode` builders, `subagent_node` (the handler lowering), retry/timeout/budget execution (`run_with_policy`), and child-run recording. |
| `mod_tests.rs` | Unit tests (delegation, mapper wiring, retry/timeout behavior, budget enforcement, child-run recording). |

## Operational constraints

- `AgentInvocationBinding` is execution-scoped: it is supplied per graph run
  (via `NodeContext::agent_binding`), never cached on a `CompiledGraph`.
  Cloning it is only meant for descendants of the same execution tree — do
  not reuse one binding across unrelated top-level executions, or their
  cancellation/event signals bleed into each other.
- A node without an attached `AgentInvocationBinding` fails immediately with
  `TinyAgentsError::Capability` rather than falling back to any default
  behavior — this is deliberate fail-closed design, not a bug to work around
  by fabricating a binding.
- `SubAgentPolicy`'s default retry policy allows exactly one attempt; a caller
  that wants retries must opt in explicitly with `with_retry`, since retrying
  a non-idempotent agent call by default would be unsafe.
- The minted child `run_id` always preserves the run tree's `root_run_id` and
  is parented to the enclosing graph run, so observability/tracing (see
  `graph::observability`) can nest a sub-agent's model/tool calls under the
  graph run that spawned it.

## Relation to neighbours

Structural counterpart to `graph::subgraph` (graph-to-graph recursion) and
`graph::parallel` (fan-out without an agent boundary). Child-run bookkeeping
here mirrors what `graph::subgraph` records via `crate::recursion::ChildRun`,
so both recursion styles show up uniformly in `GraphExecution::child_runs` /
`run_tree`. `graph::testkit::subagent_fake_node` fakes this module's
child-run recording for tests without a live agent or registry.
