# `orchestration::teams` — durable agent-team coordination

A **team** is a durable group of worker agents (members) who claim and
complete collaborative tasks. The module enforces team structure invariants
(no duplicate member names, valid task dependencies) and manages the event
log that records every state change: members added, tasks created, claims,
completions, and inter-member messages.

## Public surface

- **`TeamService<L: TeamLedger>`** — the main API: creates teams, adds members,
  creates tasks, handles claims and completions, and routes messages. Generic
  over a durable ledger.
- **`TeamLedger`** — the trait for durable team state: CRUD operations for
  teams, members, tasks, and events. Implementations handle all persistence.
- **`SessionTeamLedger`** — built-in ledger backed by `tinyagents-session`'s
  run ledger at a caller-supplied workspace root.
- **`NewMember`, `TeamView`, `MemberShutdown`** — data types for snapshots and
  operations.
- **`TeamError`** — validation errors: duplicate members, unknown members,
  cyclic task dependencies, etc.
- **`run_member_graph`** — executes a member's work as a generic DAG (execute →
  complete/fail → done), bridging host worker callbacks and the graph layer.
  The `execute` step runs the worker through `SubagentDriver`
  (`subagent::run_agent_step`) under an inert default policy, so existing
  callers behave exactly as before.
- **`run_member_graph_with` / `MemberStep`** — the same graph with explicit
  driver policy (`AgentStepConfig`: `SpawnPolicy` admission scoped to the team
  id with the member id as target, `SubAgentPolicy::timeout`, `ResultPolicy`).
  Retry, token budget and role need worker-reported transient failures / usage /
  a tool surface that `MemberOutcome` does not carry, so they do not take
  effect through this adapter (use `subagent::run_agent_step` directly).
  `MemberStep::with_cancellation` ties a step to the run's token (a pre-cancelled member never runs and goes to `on_failed`). `MemberStep::default()` shares one scope and has no target: give a limited
  config a real `MemberStep::new(config, team, member)`.
  A completed run goes to `on_complete` with the policy-trimmed output; a
  spawn rejection, timeout, exceeded budget or worker-reported failure goes to
  `on_failed` with the reason; a worker `Err` still fails the graph run.

## Design and invariants

- **Durable state at every step:** Team members, tasks, events, and watermarks
  are all durably persisted via the ledger. Hosts can use any storage backend
  that implements [`TeamLedger`].
- **Event-sourced messaging:** All team events (messages, member adds, task
  completions) are logged in the durable run event stream. Members read their
  undelivered messages and advance a delivery watermark — making repeated
  reads idempotent and safe for retries.
- **Dependency-aware tasks:** Tasks form a directed acyclic graph with
  dependencies. The service validates this at creation time (no cycles, no
  self-dependencies, no unknown member/task references).
- **Atomic task claims:** Task claiming uses atomic compare-and-swap to ensure
  exactly one member claims each task. Completion is similarly atomic,
  advancing task status and optionally requiring evidence.
- **Watermark-based delivery:** Members have a message delivery watermark
  (sequence number) that advances durably as messages are read. Hosts can call
  message-delivery functions repeatedly without duplicating messages.

## File map

- **`service.rs`** — [`TeamService`] implementation: team/member/task CRUD,
  validation, and coordination. [`TeamLedger`] trait definition.
  [`SessionTeamLedger`] — session run-ledger adapter.
- **`types.rs`** — data types: [`NewMember`], [`TeamView`], [`MemberShutdown`],
  [`TeamError`].
- **`graph.rs`** — member worker graph: `execute` → `complete`/`fail` → `done`
  DAG. Host supplies `run_worker`, `on_complete`, `on_failed` callbacks.
  [`MemberOutcome`] routes on success/failure.
- **`runtime.rs`** — message delivery and prompt composition:
  `deliver_pending_messages()` reads undelivered messages and advances
  watermarks. `build_member_prompt()` composes the stable worker prompt from
  task and delivered messages.
- **`tests.rs`** — tests for team creation, member lifecycle, task coordination,
  concurrency, and message delivery.

## Relationship to other modules

- **Depends on:** `tinyagents-graph` (DAG execution), `tinyagents-session` (run
  ledger), `tinyagents-harness` (error types).
- **Used by:** `orchestration::workflow` (to model multi-agent phases),
  host code (for team management).
- **Integration:** The workflow engine can model a phase as a team of agents.
