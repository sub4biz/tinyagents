# Subagent orchestration

`tinyagents_orchestration::subagent` is the public home for child-agent work.
It contains two complementary surfaces.

The invocation surface provides `SubAgent`, `SubAgentTool`,
`SubAgentJobRegistry`, `SubAgentSession`, and `ChildDataPolicy`. It wraps an
`AgentHarness` as a child run, enforces recursion depth, and inherits live
parent cancellation, events, and host authority. Model-facing delegation is
background by default: `SubAgentTool` returns a job id immediately while the
child continues in the background. Callers can request `mode: "inline"` to
wait for the child and receive its final result in the same tool call. A host
registers `SubAgentJobsTool` and
`SubAgentMessageTool` over the same registry to query status/results and send
messages to queued or running jobs. Message delivery is cooperative at the
child loop's next safe steering checkpoint. The registry is process-local and
host-owned; durable lifecycle persistence remains the separate surface below.

The durable lifecycle surface coordinates a host-resolved subagent lifecycle
without making product-policy decisions. Its dependency direction is:

```text
orchestration::subagent -> tinyagents-runtime -> {tinyagents-harness, tinyagents-session}
```

Hosts provide three object-safe lifecycle seams. `SubagentPlanner<C, H>`
transforms a live `SubagentRequest<C, H>` into a complete
`PreparedSubagent<C>`: resolved
agent identity, model messages, immutable `ToolSnapshot`, and an explicit,
owned `RunContext<C>`. `H` is opaque per-call host options and is forwarded
unchanged only to the planner. `SubagentExecutor<C>` executes precisely that
plan. The
planner and executor own prompts, model choice, tool authorization, workspace
policy, artifact resolution, and host context data; this module owns none of
them. `SubagentPersistence` owns a host's durable resume and lifecycle records.

`SubagentTaskKey` is the durable lifecycle identity. It combines the original
root run, immediate parent run, optional thread, and host-local task id.
Fresh-task callers use `SubagentRequest::fresh_from_parent`, which derives the
key from an actual parent and validates the owned child context. Continuations
use `continue_with_key`: hosts must authorize and recover the original key,
then retain it even with a fresh owned execution context. Request fields are
private, so callers cannot bypass these identity checks. The key is the sole
durable thread identity; a request carries no duplicate thread field.

Hosts must assign unique durable `RunConfig` ids to parent and child runs. The
constructors reject an equal parent/child id but cannot prove global uniqueness.
Persistence, in-flight coalescing, and terminal caching all use this scoped key: two parents
may reuse a task id without sharing state, while repeat calls for the same
durable scope deduplicate and resume correctly.

`SubagentDriver<C, H>::run` uses the validated request key before it loads or
prepares anything. It then loads a resume only when the caller did not provide one,
then prepares and executes. An awaiting-input result calls `save_pause`; all
other results call `record_terminal`. These operations are mutually exclusive.
The driver caches only successfully persisted terminal outcomes by scoped
`SubagentTaskKey`, so repeated calls through the same driver return them without executing or
recording again. A pause is never cached: the next call loads its resume state
and executes the continuation. Concurrent calls for the same task coalesce;
different scoped task keys, including nested child tasks on the same driver,
execute independently. Persistence implementations must also enforce
idempotency by `SubagentTaskKey` across processes and driver instances.

Cancellation is cooperative. It is observed before planning, after resume
load and planning, and immediately after execution. A cancellation that wins
after the executor produced an outcome preserves that outcome's output,
history, usage, and neutral artifact references while reporting `Cancelled`.
The supplied cancellation token is installed on the prepared `RunContext`, so
the executor and the actual harness context observe one cancellation tree. If
host context `C` itself embeds a separate cancellation token, the executor is
responsible for synchronizing that host-owned token with this supplied token;
the neutral lifecycle cannot inspect opaque host data.
Persistence `Ok(())` is the commit boundary: if cancellation wins before it,
the driver abandons that uncommitted operation and records one `Cancelled`
terminal outcome; if persistence commits first, the committed result remains
truthful. Planner and executor task ids must exactly match the durable key task id
or the driver returns a typed error before persistence or caching.
Each coalesced caller retains its own cancellation token: cancelling a follower
returns a local cancelled outcome promptly, without cancelling the leader or
creating an additional persistence action.
An absent host seam is rejected at driver construction with a typed
`MissingCapability` error; no partial lifecycle runs.

## Spawn admission control

`SpawnPolicy` bounds child fan-out; `SpawnAdmission` is the injected ledger
that enforces it (no global state: clones share one ledger, and a host shares
one instance across every tool/driver whose spawns should count together).

**What the caps bound.** Hosts mint run ids fresh per turn, and background
children outlive the turn that spawned them, so counts are keyed on a stable
*scope key* resolved from the **parent's** `RunConfig`: its `thread_id` (the
conversation) when set, else its `run_id` (keys are `thread:<id>` / `run:<id>`, so the two id spaces never alias). Two turns with different run ids on
one thread share their caps; a child from turn 1 still alive in turn 2 still
holds its slot. Override the rule with `SpawnAdmission::with_scope_key(|cfg|
...)` (for example one key per tenant, or one key for a whole run tree). The
driver resolves the scope from its `SubagentTaskKey` (`thread_id`, else
`parent_run_id`), identically for fresh spawns and continuations.

| Field | Bounds (per scope) | Released |
| --- | --- | --- |
| `max_children_per_parent` | children live at once | at the child's terminal state; the driver releases when its lifecycle call returns, so a child paused awaiting input holds no slot until it is resumed |
| `max_total_per_root` | children ever spawned (a conversation-wide budget) | only if the spawn never happened |
| `allowed_targets` | sub-agent names that may be spawned (`Some(vec![])` allows none) | n/a |

Under the default rule, nested agents have their own threads and so their own
scopes; a tree-wide budget needs a resolver that maps descendants to the root's
key.

Enforcement uses *reservation* semantics: `try_reserve` checks every limit and
claims the slot under one lock, returning a `SpawnReservation` guard, so
concurrent spawns cannot race past a cap. Dropping the guard releases the live
slot; dropping it without `commit()` (the spawn failed before the child
launched) also refunds the total budget. `SubAgentTool` (background and inline)
reserves before creating the child and moves the guard into the child's task
for its whole lifetime (a panic, abort, or dropped inline call still releases
it); `SubagentDriver::run` reserves after the cancellation and resume checks and
before the planner runs, and commits just before the executor launches.
Coalesced followers and cached terminal results never reserve; a resumed
lifecycle takes a live slot but no new total budget.

An over-limit `SubAgentTool` call returns a tool error worded as a limit signal
("...treat this as a delegated-agent limit signal, not a completed answer"),
like the depth-limit error; the driver returns `SubagentError::SpawnRejected`.
Because the neutral `SubagentRequest` carries no agent identity, a driver host
that configures `allowed_targets` must name each request with
`SubagentRequest::with_target`; an unnamed request is refused (fail closed).

**Bypass.** Only `SubAgentTool` and `SubagentDriver` enforce admission. Calling
`SubAgent::invoke_in_parent` / `invoke_hosted_in_parent` (or `SubAgentSession`)
directly bypasses it; a host exposing those paths must reserve itself.

Every limit defaults to `None` (unlimited), so nothing changes until a host opts
in with `SubAgentTool::with_spawn_admission` /
`SubagentDriver::with_spawn_admission`. Recommended guard-rail starting point:
`max_children_per_parent: Some(5)` (OpenClaw's default) and a
`max_total_per_root` of a few dozen.

**API note.** `SubagentError` gained a `SpawnRejected` variant and is now
`#[non_exhaustive]`: downstream code that matches it exhaustively must add a
wildcard arm. There is no CHANGELOG in this repository.

## Completion push (D10)

`SubagentDriver::with_completion_router(Arc<CompletionRouter>)` makes the driver
record each child it finishes with a `tinyagents_tasks::CompletionRouter`, so the
parent is told according to the child's `NotifyMode`. Recording is per spawn:
`PreparedSubagent::with_notify_mode(mode)` opts a child in (a detached spawn
passes `NotifyMode::default()`, follow-up); a child with no notify mode, such as
a foreground one whose result already returns to the parent, is never recorded.
The parent key is `PreparedSubagent::with_completion_parent`, else the request's
thread id, else the parent run id. Only the invocation that wins the durable
terminal write records, so coalesced followers and replayed terminals add
nothing. Recorded: `Completed` (success) and `Incomplete`. Not recorded: a
cancellation (the parent's own doing), a pause (the same task id completes
later), and an executor error (nothing terminal was persisted and the task may be
re-run; `run` returns the error, and a host that wants a failed push records it
itself). A router failure is logged, never raised. With no router configured the
driver is unchanged. Detached children tracked by a status channel use
`spawn_status_watcher_with_completions` (see `detached/README.md`), which keeps
watching across a pause and skips a child whose ledger shows a cancellation.

## Policy, result and role (D5/D6/D9)

One `SubAgentPolicy { timeout, retry, budget, retry_after_tool_calls }` governs
every path (it is defined in `tinyagents-graph` and re-exported here, because
orchestration depends on graph).

- **Timeout.** `SubagentDriver` runs the executor under its own child token; on
  elapse the child is cancelled (the caller's token is not) and the outcome is
  `Incomplete(kind: Timeout)`. `SubAgentTool` cancels the job token and marks
  the job `Incomplete` (`incomplete_kind: "timeout"`). A timeout is never retried.
- **Retry.** Only a failure the `RetryPolicy` deems retryable, and never after
  the attempt ran tools unless `retry_after_tool_calls`. The driver retries
  `SubagentError::Transient { tools_ran, .. }` (adapters classify; anything else
  is never retried) and needs `PreparedSubagent::retry_context`, a factory for a
  fresh run context per attempt; without it one attempt runs. `SubAgentTool`
  pre-mints one child per attempt and watches `ToolStarted` events to learn
  whether tools ran. A host-visible side effect of retry: the job's recorded
  `subagent_run_id` is the first attempt's.
- **Budget.** Call caps tighten the child `RunConfig` (stricter wins) so the
  harness enforces them during the run. Input/output token caps are checked on
  the reported usage after the run: the driver ends `Incomplete(BudgetExceeded)`
  (output kept), the tool path fails the job `Incomplete`. `max_cost` is **not
  enforced** here; `SubAgentBudget::to_budget_limits()` gives the harness
  `BudgetMiddleware` the same caps for in-run cost enforcement.
- **Incomplete status.** `SubagentOutcomeKind::Incomplete(SubagentIncomplete { reason,
  kind: IncompleteKind })` and `SubAgentJobStatus::Incomplete` replace the old
  `[SUBAGENT_INCOMPLETE]` text marker. The transcript view reads
  `"status": "incomplete"` in a spawn result and still reads the legacy marker;
  both project as `SubagentOutcomeKind::Incomplete` (previously `Failed`).
- **Result policy.** `ResultPolicy { max_chars, overflow: Truncate | Artifact,
  schema }` (builders; `artifact_store` is a host `ArtifactStore` callback, the
  orchestration layer owns no store). `Truncate` keeps head and tail around
  `[… N chars omitted …]`; `Artifact` stores the full output through the host
  store and returns the preview plus a path-free `ArtifactReference` (no store:
  falls back to truncate and reports `artifact_error`). `schema` is checked with
  the tool-call boundary's structural validator (`type`, `properties`,
  `required`, `additionalProperties`, `items`, `enum`; no `$ref`/combinators)
  and a mismatch is surfaced as `schema_error`, never a failure. Defaults: no
  cap, no schema.
- **Role and framing.** `SubagentRole::{Orchestrator (default), Leaf}`.
  `restrict_tools` drops `subagent_jobs`, `subagent_message` and the host's
  `delegation_tools` for a leaf and intersects with `tool_ceiling` so a child
  never widens what it inherited; the driver applies it to
  `PreparedSubagent::tools`. `SubAgentTool` cannot filter a shared harness, so a
  leaf tool refuses to spawn when the child harness exposes a delegation tool.
  `subagent_framing(role, depth, task)` is an optional neutral preamble hosts
  may prepend; nothing applies it by default.

**Retry notes.** Subagent retry compounds with the harness's per-call model
retry (each attempt may retry model calls first), so keep one of the two low.
`SubAgentTool` attempts after the first get run ids `{first}-a{n}` (no extra
child ordinals are consumed, so sibling ids do not depend on retry) and the job's
`subagent_run_id` follows the attempt that is running. The retry predicate
(`RetryPolicy::retry_on`) sees the real error on the tool path; the driver only
has the `Transient` message, so it sees a synthetic tool error. Usage of a failed
or timed-out attempt is not reported (an error carries no run).

## Agent steps: one engine for teams, workflows and the driver (D11)

`run_agent_step(&AgentStepConfig, AgentStepIdentity, CancellationToken, work)`
adapts an opaque host worker (a team member closure, a workflow child
executor) to the driver's planner / executor / in-memory persistence seams and
runs it on a real `SubagentDriver`. `teams::run_member_graph_with` and
`WorkflowEngine::with_step_config` call it, so those steps honour `SpawnPolicy`
admission (scope = team id / workflow run id, target = member / agent id),
`SubAgentPolicy::timeout` and `ResultPolicy` and the typed
`SubagentOutcomeKind`. Retry (`StepWorkError::Transient`), token budget (`StepSuccess::usage`) and
`SubagentRole` are supported by `run_agent_step` itself, but the team and
workflow adapters have no source for them (their workers report no transient
failures or usage and expose no tools), so they only take effect for callers
that use `run_agent_step` directly.

`AgentStepConfig::default()` is inert (unlimited admission, no timeout, one
attempt, no trimming), so `run_member_graph` and a `WorkflowEngine` without
`with_step_config` behave as before. A worker `Err` is returned unchanged as
`AgentStepError::Worker`.

`tinyagents_graph::SubAgentNode` / `subagent_node` are `#[deprecated]` rather
than adapters: graph cannot depend on orchestration. Their behaviour is
unchanged; `SubAgentPolicy` is shared and not deprecated.

`StepContext` hands the worker the role and the model/tool call caps (`max_model_calls`, `max_tool_calls`) to enforce, since the crate cannot count an opaque worker's calls.

## Breaking changes

None of the new structs hosts build is `#[non_exhaustive]`; use the constructors
and `..Default::default()`.

- `PreparedSubagent` gained `role`, `tool_ceiling`, `delegation_tools`,
  `policy`, `result_policy`, `retry_context`: build it with
  `PreparedSubagent::new(..)` and the `with_*` builders.
- `SubagentOutcome` gained `schema_error` and `artifact_error` (use
  `SubagentOutcome::completed` / `cancelled` / `incomplete`).
- `SubagentIncomplete` gained `kind: IncompleteKind` (use
  `SubagentIncomplete::new(..).with_kind(..)`).
- `SubagentError` gained `Transient` (already `#[non_exhaustive]`).
- `SubAgentBudget` gained `max_input_tokens`, `max_output_tokens`, `max_cost`
  (struct literals need `..SubAgentBudget::unlimited()`) and **lost `Eq`**
  (the cost cap is an `f64`). `SubAgentPolicy` gained `retry_after_tool_calls`.
- `SubAgentJob` gained `incomplete_kind`, `artifacts`, `schema_error`,
  `artifact_error`; `SubAgentJobStatus` gained `Incomplete`. **Behaviour change:**
  a job whose child fails with `LimitExceeded` or `Timeout` now ends
  `Incomplete` (with `incomplete_kind`) instead of `Failed`; it is still
  terminal and still returned as a tool error.
- The transcript view's `TranscriptSubagentStatus` gained `Incomplete`, and a legacy
  `[SUBAGENT_INCOMPLETE]` result now projects as `Incomplete` instead of
  `Failed`.
- `tinyagents-runtime`: new `ToolSnapshot::retaining`.
