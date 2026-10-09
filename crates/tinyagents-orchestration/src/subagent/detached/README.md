# Detached subagent runtime

Host-neutral vocabulary and helpers for background ("detached") subagents. The
process-local mechanics (status watch channel, cancel token, abort handle,
ownership) live in `tinyagents_tasks::DetachedTaskRegistry`; the
durable lifecycle lives in an `orchestration::TaskStore`. This module sits
between them. Hosts own the registry instance, its metadata type, where the
store lives, progress projection and policy.

## Layout

| File | Contents |
| --- | --- |
| `types.rs` | Every data definition: `DetachedSubagentStatus`, `FinishedOutcome`, `WaitError`, `WaitOutcome`, `SpawnedSubagent`, `SubagentIdentity`, `SubagentSnapshot`, `SubagentResumeRef` |
| `status.rs` | Status behavior (labels, terminal check) and `wait_detached` |
| `completion.rs` | `DetachedCompletionTarget`, `record_detached_completion`, `spawn_status_watcher_with_completions`: record a detached child's final status with the completion router |
| `ledger.rs` | Mirroring a status into a `TaskStore` and reading a durable record back |
| `roster.rs` | Roster snapshots, session-id resolution and resume references |
| `test.rs` | Unit tests |

## Public surface

- `DetachedSubagentStatus` with stable wire labels (`running`, `completed`,
  `awaiting_user`, `failed`); `awaiting_user` is paused, not finished.
- `wait_detached`: a registry wait that treats a dropped status sender as a
  failed result instead of hanging; a timeout leaves the entry intact.
- Ledger helpers: `record_spawned`, `record_status`, `record_cancelled`,
  `spawn_status_watcher`, `record_to_wait_outcome`, `subagent_record_for_task`.
- Completion helpers: `spawn_status_watcher_with_completions` is
  `spawn_status_watcher` plus a durable push to the parent through a
  `tinyagents_tasks::CompletionRouter` (a paused `awaiting_user` child records
  nothing; a dropped sender records a failure). `SubagentDriver` does the same
  for children it runs itself, via `with_completion_router`.
- Roster helpers: `snapshot_for_owner`, `task_id_for_session`,
  `task_id_for_session_in_records`, `resume_ref_for_task`,
  `resume_ref_from_record`.

## Operational constraints

- Ownership is enforced on every lookup (`WaitError::NotOwned`); a poisoned
  registry lock is reported as `WaitError::RegistryPoisoned`, never a panic.
- `record_spawned` returns the store's insert error (for example a task id that
  is still present in the durable store) and does not advance the existing
  record; callers must stop the spawn on error.
- `record_status` is first-writer-wins: transitions out of a terminal state are
  ignored. An awaiting run stores its question through
  `TaskStore::mark_awaiting_with_question`, so a restarted process can still
  surface the clarification text.
- `spawn_status_watcher` retries a failed terminal write a bounded number of times; call `record_status` directly to observe persistence failures.
- A run the process never registered (for example after a restart) resolves
  from its durable record, with `iterations` reported as 0.
