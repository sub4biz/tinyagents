# SDK Gaps: Streaming And Event Journals

> Part of [SDK Gaps](README.md). Covers reasoning/tool-argument streaming
> and production event/status journals.

## Backlog

### 3. Reasoning And Tool-Argument Streaming

Status: mostly implemented (runtime-comparison Phase 3, C1).

`MessageDelta { text, reasoning, tool_call }` carries reasoning alongside
visible text. Block start/end channels now exist too:
`ModelStreamItem::{BlockStart, BlockDelta, BlockEnd}` give a
`BlockKind::{Text, Thinking, ToolCall { id, name }}` per index, so a consumer
knows exactly when a tool-call block opens/closes instead of inferring it
from delta content; `ToolDelta::content_index` carries the same index on the
flat compatibility channel. Anthropic maps `content_block_start`/`_delta`/
`_stop` 1:1 onto these; the OpenAI chat-completions adapter now derives them
too, tracking the currently open block (text, reasoning, or each tool call by
wire index) and emitting `BlockStart`/`BlockDelta`/`BlockEnd` as it switches
or on `finish_reason`, sharing one dense index space across all three kinds
so `content_index` matches the terminal message's `content` ordering. The
OpenAI Responses API still has no true incremental SSE path in this crate
(`stream()` does one unary call and replays it as `Started`/one
`MessageDelta`/`Completed`), so there is nothing to derive blocks from there
yet. `ProviderFailed` now also carries `partial_message`/`stop_reason` for a
mid-stream failure on both adapters. Mid-execution *tool* progress streaming
is now wired (C2): `tinytools::ToolRunContext::report_progress` is the surface,
and the loop emits `AgentEvent::ToolProgressDetail` and calls `run_on_tool_delta`.

Remaining work:

- Give the OpenAI Responses API a true incremental SSE path (currently
  simulated as one unary call), then derive `BlockStart`/`BlockEnd` from its
  `response.output_text.delta` / `response.function_call_arguments.delta` /
  reasoning-summary delta events.
- Attribute every delta to run id, model call id, optional thread id, parent
  run id, and root run id (partially covered by `ModelStreamMetadata`).

Acceptance criteria:

- OpenHuman can delete `ThinkingForwarder`.
- UI consumers can render visible text, reasoning, and tool argument assembly
  from TinyAgents events alone.
- Non-streaming providers can still emit post-hoc reasoning as one event.

### 6. Production Event And Status Journals

Status: partially present (runtime-comparison Phase 3, C2/C3, narrowed the
late-attach-replay gap).

TinyAgents has `HarnessEventJournal`, `StoreEventJournal`, `HarnessStatusStore`,
and `HarnessRunStatus`. OpenHuman still bridges TinyAgents events into its own
progress system, cost tracker, run ledger, and UI status stream.

Late attach is now partially solved. `tinyagents_harness::stream::{AssistantFrame,
FrameEncoder, reduce_frames}` give a durable per-block frame codec:
`FrameEncoder` turns a `ModelStreamItem` sequence into frames (periodic
`ToolArgsCheckpoint` snapshots bound replay depth), and `reduce_frames` folds
a — possibly truncated — sequence into a `PartialAssistantMessage`. Every
`GraphEvent` is now wrapped in a `GraphEventEnvelope { run_id, task_id, ns,
seq, event }` (`seq` monotonic per emitting graph instance, fresh for an
embedded subgraph), and `tinyagents_graph::stream::StreamProjection` folds
graph envelopes plus harness `AgentEvent`s into cursor-ordered
`messages`/`tool_calls`/`subagents` views; `StreamProjection::since(cursor)`
is the late-attach replay primitive. `seq` does not chain across a subgraph
boundary into one run-tree-wide sequence yet (D4's `TaskId` is the natural
place for that). `JournalGraphSink::dropped()` exposes its best-effort drop
counter so lossy-under-load is observable; the harness-side
`HarnessEventJournal` has no equivalent yet. Filters/compaction/redaction
hooks are still missing on both sides.

Remaining work:

- Replay windows, filters, compaction, and redaction hooks on the durable
  journals (cursors/`since` now exist for `StreamProjection`; the journals
  themselves still lack cursor-addressable replay).
- Status stores with parent/root lineage, thread-scoped listing, phase details,
  active tool/model call ids, usage totals, cost totals, and terminal summaries.
- Redaction policies for prompts, tool args, tool results, PII, secrets, and
  provider payloads.

Acceptance criteria:

- A UI can attach late and reconstruct a run without subscribing at start time.
- A supervisor can query every active descendant of a root run.
- OpenHuman event bridges become mostly format adapters, not state owners.

