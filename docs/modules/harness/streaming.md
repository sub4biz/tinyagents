# Harness Streaming Feature

The harness must support streaming independently from the graph. Graph streaming
can forward harness events, but direct harness users should also be able to
consume streams.

## Responsibilities

- Stream model token/message deltas.
- Stream tool progress.
- Stream usage and cost updates.
- Stream cache hit/miss events.
- Stream summary events.
- Stream final outputs.
- Forward events into the registry event bus.
- Merge provider chunks into a final assistant message.
- Preserve streamed tool-call chunks where providers support them.
- Support cancellation and backpressure.
- Support stream replay from event stores.

## Source Inspiration

LangChain exposes model streams, runnable event streams, callback streaming, and
tracer streams:

- chat model stream:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/chat_model_stream.py>
- runnable event streams:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/runnables>
- streaming tracers:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/tracers>
- provider streaming adapters such as OpenAI:
  <https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py>

## Stream Modes

- `messages`: model deltas and final messages
- `tools`: tool lifecycle and progress
- `usage`: token updates
- `cost`: price updates
- `events`: all harness events
- `final`: final result only

Every stream item should carry run ids and component ids so web UIs can merge
harness streams with graph streams.

## Stream Items

```rust
pub enum HarnessStreamItem {
    Event(HarnessEvent),
    MessageDelta(MessageDelta),
    ToolCallDelta(ToolCallDelta),
    ToolProgress(ToolProgress),
    Usage(UsageRecord),
    Cost(CostRecord),
    Final(AgentRun),
}
```

A stream consumer should be able to subscribe to a subset of modes without
changing execution. Dropping a consumer must not cancel the run unless the
consumer owns the run cancellation token.

## Chunk Merging

Streaming adapters must merge chunks deterministically:

- text chunks preserve order
- reasoning/thinking chunks preserve order on a side channel
- content block indexes are respected
- tool-call chunks are correlated by id or index
- cumulative usage is converted into deltas or clearly marked cumulative
- final message equals the merged stream
- invalid partial tool calls are surfaced as repairable parse errors

## Implementation status (runtime-comparison Phase 3, C1/C2)

The design above names the target shape; here is what exists in code today,
one layer down at `tinyinference_llm::model::ModelStreamItem` and
`tinyagents_harness::stream`.

**Block-indexed streaming (C1, vendor `tinyinference-llm`).** Content block
indexes are respected via new `ModelStreamItem` variants: `BlockStart {
index, kind: BlockKind::{Text, Thinking, ToolCall { id, name }} }`,
`BlockDelta { index, delta: BlockDelta::{Text, Thinking, ToolArgs} }`, and
`BlockEnd { index, block: ContentBlock }`. The pre-existing flat
`MessageDelta`/`ToolCallDelta` channel is still emitted alongside them
(`model::block_delta_to_message_delta` is the shared derivation), so nothing
that only understood the old shape breaks. `ToolDelta::content_index` carries
the wire block index on the flat channel too. The Anthropic adapter maps
`content_block_start`/`_delta`/`_stop` 1:1 onto the new items. The OpenAI
chat-completions adapter derives the same items from its delta shape: it
tracks which block (text, reasoning, or a given tool call's wire index) is
currently open, emits `BlockStart` the first time a new one is seen (a tool
call opens as soon as its id or name arrives, even before its first argument
fragment), `BlockDelta` per fragment, and `BlockEnd` when the open block
switches or `finish_reason` arrives — all sharing one dense index space, so
`ToolDelta::content_index` lines up with the terminal message's `content`
ordering the same way it does for Anthropic. The OpenAI Responses API has no
true incremental SSE path in this crate yet (`stream()` simulates one with a
single unary call replayed as `Started`/one `MessageDelta`/`Completed`), so
there are no block boundaries to derive there (tracked in
`docs/sdk-gaps/streaming.md` §3). `ModelStreamItem::{Failed, ProviderFailed}` —
specifically `ProviderError` — now carries `partial_message:
Option<AssistantMessage>` and `stop_reason: Option<String>` on both
adapters, so a mid-stream failure does not discard whatever content had
already arrived.

**Frame codec (C2, `crates/tinyagents-harness/src/stream/frame.rs`).**
`AssistantFrame` is the durable, journal-friendly encoding of a block-aware
stream: `FrameEncoder` turns a sequence of `ModelStreamItem`s into frames
(only the block-indexed and terminal items are framed — `Started` and the
flat compatibility deltas carry no information a reducer needs beyond what
the block items already have), emitting a periodic
`AssistantFrame::ToolArgsCheckpoint { index, json_so_far }` full snapshot for
long tool-argument streams. `reduce_frames(&[AssistantFrame]) ->
PartialAssistantMessage` folds a — possibly truncated — frame sequence back
into closed `content` blocks plus whatever `open_blocks` were still
in-progress, without needing the original provider stream. A checkpoint is a
full snapshot, not a delta, so a reader that only has frames from a
checkpoint onward (earlier per-fragment frames pruned from the journal) still
reduces to a consistent result.

**Mid-tool progress (C2).** A running tool reports through
`tinytools::ToolRunContext::report_progress(ToolProgress { message, fraction,
partial })` (a default no-op, so existing tools are unchanged). The loop gives
every executing call a progress gate (`crates/tinyagents-harness/src/tool/progress/mod.rs`)
that emits `AgentEvent::ToolProgressDetail { call_id, message, fraction, partial }`
live and queues a `ToolDelta` for `Middleware::on_tool_delta`. Guarantees:

- Every `ToolProgressDetail` for a call falls between its `ToolStarted` and its
  terminal `ToolCompleted`/`ToolFailed`, in the order the tool reported it.
- The gate closes when the tool's future settles (return, error, timeout),
  before the terminal event. An update reported afterwards (for example from a
  task the tool spawned) is dropped, never emitted late; the open check and the
  emit share one lock, so there is no race window.
- A concurrent batch interleaves progress across calls; each call's progress
  still precedes its own terminal event (terminals are emitted by the fold,
  in call order, after the batch).
- `on_tool_delta` needs `&mut RunContext`, which the executing tool holds, so
  middleware sees a call's deltas replayed in order immediately after the call
  settles and before `after_tool` and the terminal event. It observes; the live
  event is already out. A failing hook is logged and does not fail the call.
- The gate also closes if the dispatch future is dropped (run cancelled or
  timed out mid-call), so a task the tool spawned cannot emit afterwards.
- Flooding is coalesced: at most 32 events per second per call (fixed,
  crate-private `ToolProgressLimits`); beyond that the newest value of each
  field replaces the held update. There is no timer: the held update is emitted
  on the first update of the next window or when the call settles (that final
  flush is exempt from the window limit), so the final reported state is never
  lost.
- The middleware replay queue is bounded (newest 64 deltas, each `content`
  capped at 4 KiB, partials serialized only up to the cap) and is not filled at
  all when the run has no middleware.
- An event listener must not call `report_progress` re-entrantly (events are
  emitted under the call's gate lock).

The typed
`HarnessStreamItem` enum, `StreamMode::{tools, usage, cost, events, final}`,
and stream replay from event stores are still design-only.
