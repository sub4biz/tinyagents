# `session::run_ledger` — background run and team coordination

A durable, restart-survivable ledger for background agent/workflow execution
state, sharing the session database with `crate::store`. See the top-level
`session` crate README for how this fits next to session history and the
graph checkpointer.

## Why this is separate from session history

`crate::ops` (session history) records what a conversation *said*. This
module records what the runtime was *doing*: which background agent runs,
worker threads, and workflows were in flight, what state they held, and how
they coordinate. A run and the session that produced it are queryable
together (both go through `crate::store`), but they answer different
questions and have different write patterns — session history is append-only
per turn; the run ledger is upserted and compare-and-swapped as a run's
state changes.

## Public surface

Re-exported from `run_ledger::` (see `mod.rs`); also reachable through the
crate root as `session::run_ledger::`.

### Agent runs

- `upsert_agent_run` / `get_agent_run` / `list_agent_runs` — CRUD for
  `AgentRun` (subagent, worker-thread, background-agent, team-member, or
  workflow-child runs).
- `transition_agent_run_status` — the only path that can *clear* `error` /
  `completed_at` (the upsert can only set them); used by control verbs like
  "retry".
- `interrupt_orphaned_agent_runs` — startup sweep that settles any run still
  `running`/`pending` from a dead process to `interrupted`.

### Workflow runs

- `upsert_workflow_run` / `get_workflow_run` / `list_workflow_runs` — CRUD for
  `WorkflowRun`.
- `try_claim_workflow_run` — acquires (or observes) the compare-and-swap
  driver lease that ensures at most one process drives a given workflow.
- `compare_and_swap_workflow_run` / `compare_and_swap_workflow_run_lifecycle`
  — revision-fenced writes for a live driver (the former renews the lease and
  checks the caller still owns it; the latter is for host lifecycle commands
  that may fence an in-flight driver and always clears the lease).
- `renew_workflow_run_lease` — heartbeat that extends a lease without
  advancing `revision`.

### Events and telemetry

- `append_run_event` / `list_recent_run_events` — an append-only,
  atomically-sequenced per-run event log.
- `upsert_run_telemetry` — partial-update token/cost/status rollup per run.

### Agent teams

- `upsert_agent_team` / `get_agent_team` / `list_agent_teams` — CRUD for
  `AgentTeam`.
- `upsert_agent_team_member` / `get_agent_team_member` /
  `list_agent_team_members` — CRUD for `AgentTeamMember`.
- `mark_agent_team_member_running` / `mark_agent_team_member_idle` /
  `shutdown_agent_team_member` — member lifecycle transitions; shutdown also
  releases any task the member had claimed.
- `upsert_agent_team_task` / `get_agent_team_task` / `list_agent_team_tasks`
  — CRUD for `AgentTeamTask`.
- `claim_agent_team_task` / `complete_agent_team_task` /
  `release_agent_team_task` — the claim/completion protocol: an atomic
  compare-and-swap claim, a quality-gated completion, and an explicit
  release for a member that abandons a task without shutting down.

### Restart recovery classification (`recovery.rs`)

- `classify_recovery(tail, tool_effects)` — pure and advisory. Given the
  transcript tail's dangling tool calls (`DanglingToolCall`, no recorded
  result) and the per-call `tool_effects` rows, returns one `CallRecovery` per
  call, matched on `(run_id, call_id)`:

  | Effect record | Class | Meaning |
  | --- | --- | --- |
  | none | `Resume` | the call never began; continue normally |
  | `completed`, `failed` | `ResumeReportOnly` | status is recorded (the result/error itself may not be persisted); report it, do not re-execute |
  | `deferred` | `AwaitingAnswer` | the call paused for approval/a result; resume by answering the deferral, never re-execute |
  | `started` only, `interrupted` | `NeedsVerification` | may have committed; verify real-world state, never blindly re-run |

- When several effect rows match one call, the most cautious class wins.
- `overall_recovery(&classified)` — the most cautious class (`Resume <
  ResumeReportOnly < AwaitingAnswer < NeedsVerification`), `Resume` when nothing dangles.

The effect ledger writes `started` before a call executes, which is why "no
row" means "never began"; hosts that did not attach the ledger must not rely on
`Resume`. Nothing here re-runs a tool or writes to the ledger. For the
parent-facing note about interrupted *children* see
`tinyagents_tasks::build_restart_recovery_note`.

## Layout of this module

| File | Role |
| --- | --- |
| `mod.rs` | Module docs and public surface (re-exports). |
| `types.rs` | Serde record types, one `*Upsert` companion per persisted type, and list request/response shapes. |
| `ops.rs` | CRUD, listing, and coordination primitives (see below). |
| `store.rs` | Schema entry point — a no-op now that all DDL lives in `crate::migrations`; kept as the conventional call site. |
| `recovery.rs` | `classify_recovery` / `overall_recovery`: advisory restart-recovery verdicts for dangling tool calls (tests in `recovery_tests.rs`). |
| `mod_tests.rs` | Module-local unit tests. |

## Operational constraints

These are the non-obvious rules; most are pinned by a test in `mod_tests.rs`.

**An upsert reads its own write back inside the same transaction.** Every
`upsert_*` function opens one `crate::store::with_transaction` call that
both writes the row and re-fetches it via a connection-scoped `*_inner`
helper — never a fresh `with_connection` after the write, which could
observe a concurrent writer's state instead of the caller's own.

**Workflow-run upserts favor `COALESCE` on plain `upsert_workflow_run`, but
compare-and-swap for a live driver.** `upsert_workflow_run` has no revision
fencing and can clobber a concurrent driver's write; a driver holding a
lease must use `compare_and_swap_workflow_run` (revision + lease-owner
checked) or `compare_and_swap_workflow_run_lifecycle` (revision only, always
clears the lease) instead.

**A workflow lease is cleared whenever `status` becomes terminal.** Both the
plain upsert and the compare-and-swap paths null out `lease_owner` /
`lease_expires_at` on a transition to `completed` / `failed` / `cancelled` /
`interrupted`, since a finished run has nothing left to drive.

**Run-event sequences are allocated by the INSERT itself**, via a
sub-`SELECT MAX(sequence) + 1` inside the same statement — never a separate
read followed by an insert, which would race two concurrent appenders onto
the same sequence number and lose one event to a primary-key conflict.

**Telemetry counters are `Option` for partial updates.** The columns are
`NOT NULL DEFAULT`, and SQLite does not apply that default to an explicit
`NULL`, so `upsert_run_telemetry`'s insert side coalesces to the column
default while its update side coalesces to the *stored* value — `excluded.*`
cannot serve the update side, since it observes the already-coalesced insert
row and would read a caller's `None` as `0`.

**A team-task claim is meaningful only while `status = "in_progress"`.**
`claimed_by_member_id` and `claim_token` are cleared whenever
`upsert_agent_team_task` moves a task off that status; leaving them set
strands the task — a new claim sees `AlreadyClaimed`, completion sees
`NotClaimed`, and release/shutdown skip it.

**Completion evidence accumulates across attempts, including failed ones.**
`complete_agent_team_task` merges submitted evidence into what is already
stored rather than replacing it, so a retry after fixing one gate failure
does not have to resubmit evidence already accepted.

**`shutdown_agent_team_member` releases the member's in-progress tasks back
to `todo`** (clearing claimant + token) in the same transaction that marks
the member `stopped` — the bulk analogue of `release_agent_team_task`.

`status_map.rs` holds the total `From` conversions between `AgentRunStatus` and the canonical `tinyagents_tasks::OrchestrationTaskStatus` (`Paused` reads as `Awaiting`, `Interrupted` as `Abandoned`); the `AgentRunStatus` wire strings are unchanged. Mapping table: `tinyagents-tasks` README.
