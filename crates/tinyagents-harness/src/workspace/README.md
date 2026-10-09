# harness::workspace

Workspace isolation and sandbox hooks for tools that run over real files or
command executors.

## Why this exists

Application-specific worktree/sandbox providers need a common seam: a
`tinytools::WorkspaceDescriptor` tells a tool which filesystem root it may
touch, and a [`WorkspaceIsolation`] provider trait prepares/tears down a
per-agent environment. TinyAgents does not own any concrete isolation policy
— it owns the interface, plus two providers: a trivial single-shared-root
default and a real git-worktree-backed implementation.

## Public surface

- [`WorkspaceIsolation`] — the provider trait: `prepare(run_id, agent) ->
  WorkspaceDescriptor` and `cleanup(&descriptor)`.
- [`prepare_workspace`] / [`cleanup_workspace`] — free functions that drive a
  `WorkspaceIsolation` provider *and* emit the corresponding
  `AgentEvent::WorkspacePrepared` / `AgentEvent::WorkspaceCleanup` on the
  run's event sink, so isolation setup/teardown is observable.
- [`SharedRootWorkspace`] — the trivial provider: scopes every agent to one
  shared root without per-agent copying (`prepare` is a descriptor
  construction, `cleanup` is a no-op). A sensible default and a test double.
- [`enforce_workspace_path`] — the fail-closed path gate a tool calls
  *before* touching a path: emits `AgentEvent::WorkspaceViolation` and returns
  a validation error when the path is outside every allowed root.
- From `git` (git-worktree-backed isolation):
  - [`GitWorktreeIsolation`] — a real `WorkspaceIsolation` implementation:
    each run gets its own `git worktree` checkout under
    `<repo>/.claude/worktrees/<run_id>`. Builder methods:
    `with_base_ref`, `with_sandbox`, `with_trusted_root`.
  - [`GitWorktreeBaseRef`] — which ref a new worktree branches from (`Head` or
    `Fresh`, i.e. the repo's default branch).
  - [`GitWorktreeStatus`] — a snapshot (`path`, `branch`, `is_dirty`,
    `changed_files`) of one worktree's state.
  - [`GitWorktreeError`] — errors from the worktree manager (`NotAGitRepo`,
    `DirtyRefused`, `GitFailed`, `Io`).
  - Standalone functions usable without the `WorkspaceIsolation` trait:
    `create_git_worktree`, `list_git_worktrees`, `git_worktree_status`,
    `git_worktree_diff_summary`, `remove_git_worktree`,
    `detect_worktree_overlaps` (finds files touched by more than one sibling
    worker — useful for merge-conflict-avoidance heuristics).
  - `GIT_WORKTREE_SUBDIR` — the fixed subdirectory (`.claude/worktrees`)
    worktrees are created under.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `WorkspaceIsolation` driving helpers (`prepare_workspace`/`cleanup_workspace`) and `SharedRootWorkspace`. |
| `types.rs` | The `WorkspaceIsolation` trait. |
| `policy.rs` | `enforce_workspace_path`, the fail-closed path gate (paired with the descriptor's lexical `allows` check, which lives in `tinytools`). |
| `git.rs` | `GitWorktreeIsolation` and the standalone git-worktree management functions. |
| `git/git_tests.rs` | Tests for the git-worktree isolation provider (spawns real `git` subprocesses against a temp repo). |
| `mod_tests.rs` | Tests for the descriptor-allows and `SharedRootWorkspace`/event-emission hooks. |

## Operational constraints

- `WorkspaceDescriptor`'s allowed-root policy and its lexical `allows` check
  live in `tinytools`, which owns the tool vocabulary; this module only
  supplies the event-emitting wrapper (`enforce_workspace_path`) and the
  isolation providers.
- `GitWorktreeIsolation::cleanup` refuses to remove a dirty worktree unless
  forced — `remove_git_worktree`'s `force` parameter is `false` in the
  `WorkspaceIsolation::cleanup` path, so uncommitted work is never silently
  discarded by the isolation lifecycle.
- Run ids are sanitized (`sanitize_run_id`) into a filesystem- and
  git-ref-safe slug before being used in a worktree path or branch name.
- All git operations shell out to the `git` binary via `std::process::Command`
  — there is no `libgit2`/`gix` dependency. `git.rs`'s private `git`/`git_raw`
  helpers are the only two call sites that invoke it.
