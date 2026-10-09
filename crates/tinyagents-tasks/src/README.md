# tinyagents-tasks

Graph-independent detached-task machinery: task model, `TaskStore`
implementations, the detached task registry, restart reconciliation and the
orchestration control tools. `tinyagents-graph` and `tinyagents-orchestration`
both depend on this crate; it does not depend on the graph engine. The old
`tinyagents_graph::orchestration::*` paths remain as plain re-exports.

This crate is the managed child-work surface. It gives
language-model orchestrators stable task ids and typed controls — `spawn`,
`await`, `cancel`, `kill`, `status`, `list`, `timeout`, `race`, `yield`, and
`steer` — without exposing raw executor handles such as `tokio::JoinHandle`.
A model asks for work by task id and observes lifecycle status; it never
touches an in-process future/handle directly, so the same tool surface works
whether the "child" is an in-process subgraph, a sub-agent, or (in principle)
out-of-process work fronted by a `TaskStore` implementation.

The controls are ordinary harness tools. Use `OrchestrationTool` directly,
call `orchestration_tools` to build the full set, or call
`register_orchestration_tools` to insert them into a
`harness::tool::ToolRegistry` alongside any other tools.

## Public surface

### Tools (`tool.rs`)

- `OrchestrationTool` — a single control, constructed with an
  `OrchestrationToolKind` and a `TaskStore`; `.with_steering(..)` wires it to a
  `SteeringRegistry` for controls that need to reach a running task (e.g.
  `steer`, `cancel`).
- `orchestration_tools(store)` — builds the full default set of controls
  against one store.
- `orchestration_tools_with_steering(store, steering)` — same, with steering
  wired.
- `register_orchestration_tools(registry, store, ..)` — inserts the full set
  into a `ToolRegistry`.
- `orchestration_tool_schema(kind)` / `orchestration_tool_schemas()` — the
  JSON tool schemas, independent of a concrete store (useful for prompting or
  schema inspection without constructing tools).
- `SteeringRegistry` — a concurrent map from `TaskId` to `SteeringHandle`
  (`register` / `deregister` / `get`), letting a `steer`/`cancel` control reach
  a task that is currently running.

### Task model (`types.rs`)

- `OrchestrationTaskKind` — what a task *is* (subgraph run, sub-agent run,
  ...); `as_str()`.
- `OrchestrationTaskStatus` — lifecycle state; `is_terminal()` / `is_live()`
  predicates.
- `OrchestrationTaskSpec` — the request to spawn a task: kind, lineage
  (thread/node), timeout, input payload, metadata. Built with `new` +
  `with_lineage` / `with_thread` / `with_node` / `with_timeout_ms` /
  `with_input` / `with_metadata`.
- `OrchestrationTaskResult` — a completed task's output (`text` / `output`
  constructors for the common shapes).
- `OrchestrationTaskRecord` — the durable record a `TaskStore` holds: spec +
  current status + result once terminal. `pending(spec)` constructs the
  initial record; `task_id()` / `is_terminal()` accessors.
- `OrchestrationTaskFilter` — filters for `list` (`with_kind`,
  `created_between`, `matches(&record)`).
- `OrchestrationToolKind` — the control identifiers (`spawn`, `await`,
  `cancel`, `kill`, `status`, `list`, `timeout`, `race`, `yield`, `steer`);
  `name()` / `description()` give the tool-facing strings.
- `OrchestrationControlOutcome` — the typed result of invoking a control.

### Storage (`store.rs`)

- `TaskStore` (trait) — durable task bookkeeping: create/update/list/get
  records by id, apply an `OrchestrationTaskFilter`.
  - `InMemoryTaskStore` — in-process implementation; `from_records(..)` seeds
    it for tests.
  - `JsonlTaskStore` — append-only JSONL-backed implementation;
    `JsonlTaskStore::open(path)`.
- `TaskStoreRegistry<K>` (`store_registry.rs`) — process-wide cache mapping a
  host-defined scope key to a lazily-opened `Arc<dyn TaskStore>`, so a
  multi-tenant host keeps exactly one store per scope. `get_or_open` /
  `get` / `values` / `len` / `is_empty` / `clear`.
  `open_jsonl_task_store_or_memory(path)` is the standard opener: it degrades
  to an `InMemoryTaskStore` if the durable log can't be created or read.

### Process-local runtime (`runtime.rs`)

- `DetachedTaskRegistry<Metadata, Status>` — tracks the executor-only pieces
  of a detached task that cannot survive a process restart: status watch
  channel, cancellation token, abort handle, owner id, and live steering
  lookup. `TaskStore` remains the durable source of lifecycle truth; this
  registry is what an executor consults to `wait`, `cancel`, `cancel_where`,
  `cancel_all`, `snapshot(s)`, or fetch a `steering_handle` for a task it
  still owns in-process. `sweep_terminal` / the `soft_cap` passed to `new`
  bound unbounded growth from tasks nobody ever waited on.

### Orphan reconciliation (`reconcile.rs`)

- `reconcile_orphaned_tasks(store, filter, reason)` — settles every live task
  matching `filter` into a terminal state (`Cancelled` if a cancellation was
  already requested, otherwise `Failed` with the caller-supplied `reason`).
  Meant to run once at host startup against a `TaskStore` whose executor
  process may have died since the last run, before any `DetachedTaskRegistry`
  is repopulated.
- `ReconcileReport` / `ReconciledTask` / `ReconcileOutcome` — the sweep's
  per-task and aggregate results; `task_status_label(status)` gives the
  stable lowercase label used in logs.

### Restart recovery note (`recovery.rs`)

Reconciliation fixes the store; the *parent agent* still has to be told its
children died. These pure functions turn a `ReconcileReport` into a note for the
parent's next turn (wiring it into a host is the host's job):

- `recovery_children(report, reason)` — selects the interrupted children
  (everything the sweep failed or could not settle; tasks settled as cancelled
  are user intent and are skipped), ordered by creation time then task id. Each
  `RecoveryChild` carries `task_id`, `kind`, `label` (the `label` metadata, else
  the agent/graph/tool name), `last_status`, and `interrupted_reason` (the sweep's persisted reason).
  Labels and reasons are truncated to `MAX_RECOVERY_LABEL_CHARS` = 256 chars
  here, in `recovery_children`; the builder does not re-truncate rows a caller
  constructs by hand.
- `build_restart_recovery_note(children)` — a capped roster (at most
  `MAX_RECOVERY_CHILDREN` = 32 rows, then a `+N more` line) as JSON inside
  a `<child_task_facts>` block (`<` is escaped so labels cannot close it),
  followed by `RESTART_RECOVERY_INSTRUCTION`: reconcile against saved results,
  verify uncertain side effects, never blindly re-run. Empty input gives `""`.

Nothing here ever relaunches a task. For the per-tool-call side of recovery see
`tinyagents_session::run_ledger::classify_recovery`.

## Files

| File | Role |
| --- | --- |
| `lib.rs` | Module declarations and re-exports. |
| `types.rs` | Task kind/status/spec/result/record/filter types, `OrchestrationToolKind`, `OrchestrationControlOutcome`, and the `DetachedTaskRegistry` snapshot/error types. |
| `tool.rs` | `OrchestrationTool`, `SteeringRegistry`, tool constructors and schemas. |
| `store.rs` | `TaskStore` trait, `InMemoryTaskStore`, `JsonlTaskStore`. |
| `store_registry.rs` | `TaskStoreRegistry<K>`, `open_jsonl_task_store_or_memory`. |
| `runtime.rs` | `DetachedTaskRegistry<Metadata, Status>` — process-local executor handles keyed by task id. |
| `reconcile.rs` | `reconcile_orphaned_tasks` and its report types, for settling orphans left by a dead executor. |
| `recovery.rs` | `recovery_children` / `build_restart_recovery_note`: the parent-facing roster of interrupted children after a reconcile sweep (tests in `recovery_tests.rs`). |
| `lib_tests.rs` | Unit tests (spawn/await/cancel/timeout/race semantics, store round-trips, filters, reconciliation, detached-task registry). |

## Operational constraints

- Controls that reach a *running* task (`steer`, `cancel`) require a
  `SteeringRegistry` populated with that task's `SteeringHandle`; without one,
  those controls only affect terminal-state bookkeeping in the `TaskStore`,
  not the in-flight task itself.
- `JsonlTaskStore` is append-only: task-record updates are appended, not
  rewritten in place, so a long-lived store grows monotonically. Compact or
  rotate externally if that matters for your deployment. Each transition
  appends one line; inside a multi-thread tokio runtime the write runs under
  `block_in_place` so the blocking file I/O does not stall other tasks on
  that worker (`TaskStore` is a synchronous trait, so it cannot await a
  `spawn_blocking` handle).
- `OrchestrationTaskFilter::matches` is evaluated in-process against loaded
  records; a `TaskStore` backend is not required to push the filter down, so
  `list` cost scales with total record count unless a given implementation
  optimizes it.
