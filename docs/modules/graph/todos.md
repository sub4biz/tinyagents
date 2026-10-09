# Per-thread todo list

`graph::todos` gives a graph a per-thread **todo list**: an ordered checklist
of steps, each `pending`, `in_progress` or `completed`. It is the
concrete-steps counterpart to the single-objective [`graph::goals`](goals.md),
and takes the shape Claude Code and Codex use: the model rewrites the whole
list as it works, and nothing else hangs off an item.

See the source module README at `crates/tinyagents-graph/src/todos/README.md` for the full public
surface; this spec captures the design contract.

## Model

- `TodoItem { content, status }`; `TodoStatus`: `Pending`, `InProgress`,
  `Completed`.
- `TodoList { thread_id, items, updated_at }`; `TodosSnapshot` (items +
  markdown) is returned by every store op.
- `render_markdown` renders GitHub-flavored markers (`[ ]`/`[~]`/`[x]`), one
  line per item; `parse_status` accepts aliases (`todo`, `done`, ...);
  `normalise_list` trims and drops empty items.

## Persistence

One serialized `TodoList` per thread in the `graph.todos` namespace of a
`harness::store::Store`, keyed by `hex(thread_id)`. Mutations run
`load → mutate → normalise → put` under a per-thread async mutex (atomic within
one process, same caveat as `graph::goals`). The list is rewritten wholesale
(`replace`) or emptied (`clear`); `list` snapshots it. Hosts also get the raw
`get` (absent versus present-empty, un-normalised) and `delete`.

### Invariant

- **Single in-progress:** at most one item may be `InProgress`; a violation is
  a `Validation` error on `replace` — never silently fixed.

## Tool

`TodoTool` is the harness `Tool` named `todo`, built with `todo_tools` /
`register_todo_tools`. One call writes the whole list
(`{"todos": [{"content", "status"}]}`); omitting `todos` reads it back. The
list is bound to `ToolExecutionContext::thread_id` (never a tool argument).
Domain errors (blank content, unknown status, two in-progress) are surfaced to
the model as tool errors rather than failing the run.

## Testing

Unit tests in `crates/tinyagents-graph/src/todos/mod_tests.rs` (types, store
invariant, tool) and an end-to-end model-driven tool run in
`tests/e2e_graph_todos.rs`.
