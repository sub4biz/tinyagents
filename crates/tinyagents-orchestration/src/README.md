# `tinyagents-orchestration` — subagent orchestration

This crate is the single home for TinyAgents subagent behavior. It builds on
the lower-level harness and runtime crates without making those crates depend
on orchestration policy.

## Public surface

- `subagent::SubAgent` runs an `AgentHarness` as a child with inherited
  lineage, cancellation, events, host authority, and recursion limits.
- `subagent::SubAgentTool` starts a child asynchronously through typed
  parent-context tool dispatch and immediately returns a stable job id.
- `subagent::SubAgentJobRegistry` records queued, running, completed, failed,
  and cancelled jobs. `SubAgentJobsTool` queries them and
  `SubAgentMessageTool` sends messages to live children.
- `subagent::SubAgentSession` reuses one child and its transcript across turns.
- `subagent::SpawnPolicy` / `SpawnAdmission` bound child fan-out (per-parent
  live cap, per-root total budget, target allowlist) with atomic reservation;
  see `subagent/README.md`.
- `subagent::SubagentDriver` coordinates durable resume, preparation,
  execution, pause, and terminal persistence through host-supplied traits.

The direct invocation implementation and its tool-focused tests live in
`subagent/invocation/`. Durable lifecycle files live beside it in `subagent/`.
End-to-end and live subagent tests live in the crate-level `tests/` directory,
and `tests/live_orchestrator_subagents.rs` exercises network-backed job-based
delegation.

## Status vocabularies

Background agent work is described by several lifecycle enums. They are kept
separate on purpose (different questions, different wire formats); conversions
between them are explicit, documented and exhaustively tested.

| Type | Crate | Answers | Variants |
| --- | --- | --- | --- |
| `SubAgentJobStatus` | orchestration (`subagent::invocation`) | Where is an asynchronous spawned job? Serialized `snake_case` on `SubAgentJob`. | `Queued`, `Running`, `Completed`, `Failed`, `Incomplete`, `Cancelled` |
| `DetachedSubagentStatus` | orchestration (`subagent::detached`) | What did a detached run publish to its waiters? Carries payloads (output, question, error). Not serialized. | `Running`, `Completed`, `AwaitingUser`, `Failed` |
| `OrchestrationTaskStatus` | graph (`orchestration`) | Supervisor view of any managed task (the superset: adds cancellation, deadline and abandonment states). | `Pending`, `Running`, `Awaiting`, `Completed`, `Failed`, `CancelRequested`, `Cancelled`, `TimedOut`, `Abandoned` |
| `AgentRunStatus` | session (`run_ledger`) | Durable ledger row for a background run, shown in a command center. | `Pending`, `Running`, `AwaitingUser`, `Paused`, `Completed`, `Failed`, `Cancelled`, `Interrupted` |
| `SubagentOutcomeKind` | orchestration (`subagent`) | What a driver-level run *returned*: the outcome plus its payload (pause state, incomplete reason). Formerly `SubagentStatus`. | `Completed`, `AwaitingInput`, `Incomplete`, `Cancelled` |
| `TranscriptSubagentStatus` | session (`transcript::view`) | Display projection of a sub-agent run read back from a persisted transcript. Formerly `SubagentStatus`. | `Completed`, `Failed`, `Incomplete`, `Interrupted`, `Running` |

The old `SubagentStatus` names remain as `#[deprecated]` type aliases at their
original paths; the wire formats did not change.

`SubagentOutcomeKind` and `TranscriptSubagentStatus` are not lifecycle states
of a job and have no conversions: the first is a result value, the second a
read-only projection of recorded text.

### Conversions

`graph` and `session` do not depend on each other, so the task <-> run-ledger
mapping lives in this crate (`status`), the first one that sees both. The job
and detached mappings sit beside their types. Fallible conversions return
`status::NoEquivalentStatus`.

| From | To | API | Lossy cases |
| --- | --- | --- | --- |
| `SubAgentJobStatus` | `OrchestrationTaskStatus` | `to_task_status()` | `Incomplete` -> `Failed` |
| `SubAgentJob` | `OrchestrationTaskStatus` | `task_status()` | `Incomplete` + `Timeout` -> `TimedOut`; other incomplete -> `Failed` |
| `SubAgentJobStatus` | `AgentRunStatus` | `to_run_status()` | `Incomplete` -> `Failed` |
| `OrchestrationTaskStatus` | `SubAgentJobStatus` | `TryFrom` | `CancelRequested` -> `Running`; `TimedOut` -> `Incomplete`; `Awaiting`, `Abandoned` -> error |
| `DetachedSubagentStatus` | `OrchestrationTaskStatus` | `to_task_status()` | payload dropped; `AwaitingUser` -> `Awaiting` |
| `DetachedSubagentStatus` | `AgentRunStatus` | `to_run_status()` | payload dropped; `AwaitingUser` -> `AwaitingUser` |
| `&DetachedSubagentStatus` | `SubAgentJobStatus` | `TryFrom` | payload dropped; `AwaitingUser` -> error |
| `OrchestrationTaskStatus` | `AgentRunStatus` | `status::task_status_to_run_status` | `CancelRequested` -> `Running`; `TimedOut` -> `Failed`; `Abandoned` -> `Interrupted`; `Awaiting` -> `AwaitingUser` (caveat: `Awaiting` may mean waiting on a child task rather than on the user) |
| `AgentRunStatus` | `OrchestrationTaskStatus` | `status::run_status_to_task_status` | `Paused`, `AwaitingUser` -> `Awaiting`; `Interrupted` -> `Abandoned` |

Terminality is preserved by every mapping, with one deliberate exception:
`DetachedSubagentStatus::is_terminal` treats `AwaitingUser` as terminal (the
run will not progress by itself), while the task and ledger statuses keep it
live so a follow-up can resume it.

## Boundaries

The crate does not own teams or workflow DAGs. Graph-specific node lowering
remains in `tinyagents-graph`, while provider calls, tool dispatch mechanics,
events, and run contexts remain in `tinyagents-harness`. This crate composes
those primitives into child-agent behavior.

Hosts remain responsible for agent definitions, model selection, credentials,
workspace policy, durable storage implementations, and authorization. Hosted
child invocations always re-enter through the exact capability bundle carried
by their parent context.
