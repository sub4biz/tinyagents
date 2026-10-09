# Tool Execution Context and Rich Returns

Plan items B1 and B2 (`docs/runtime-comparison/plan.md`, Phase 2): what a
canonical tool can see about the run that invoked it, and what the agent
loop does with the host-facing halves of a `tinytools::ToolResult`. Source:
`crates/tinyagents-harness/src/tool/types.rs` and
`crates/tinyagents-harness/src/agent_loop/tools.rs` (fold phase).

## Tool execution context (B1)

Every canonical tool call receives a `ToolExecutionContext` — the harness's
`tinytools::ToolRunContext` implementation — built by the loop for that
exact call:

```rust
pub struct ToolExecutionContext {
    pub run_id: RunId,
    pub call_id: CallId,                 // the admitted call's id
    pub thread_id: Option<ThreadId>,
    pub depth: usize,
    pub max_turn_output_tokens: Option<u32>,
    pub events: EventSink,
    pub cancellation: CancellationToken,
    pub streaming: bool,
    pub workspace: Option<tinytools::WorkspaceDescriptor>,
    pub store: Option<Arc<dyn NamespacedStore>>,        // RunContext::with_namespaced_store
    pub state_view: Option<Arc<dyn Any + Send + Sync>>, // RunContext::with_state_view
    pub progress: Option<tinytools::ProgressSink>,      // sink behind report_progress (captured from the loop's gate)
    pub nested: Option<Arc<dyn NestedToolRunner>>,      // behind call_tool; None outside the agent loop
}
impl ToolExecutionContext {
    pub fn state<S: 'static>(&self) -> Option<&S>;   // None on absent or mismatched type
    pub fn custom(&self, payload: serde_json::Value); // emits AgentEvent::Custom { call_id, payload }
    pub async fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<ToolResult>; // nested call (C9)
}
```

`call_tool` runs another tool through the same admission, wrap onion and
`max_tool_calls` budget as a model-issued call; see
[nested-tool-calls.md](nested-tool-calls.md).

A tool streams progress with the portable
`ToolRunContext::report_progress(ToolProgress::message("...").with_fraction(0.4))`
(no downcast needed); the loop emits `AgentEvent::ToolProgressDetail` between the
call's `ToolStarted` and terminal event and drops anything reported after the
call has settled (see `streaming.md`). `custom` remains the channel for
application-defined payloads the harness attaches no meaning to.

`call_id` is the same id the transcript row and the `ToolStarted` /
`ToolCompleted` events carry, so anything a tool records or emits correlates
with the call (each call of a concurrent batch gets its own). `store` and
`state_view` are `None` unless the host attached them on the `RunContext`;
both are inherited by child contexts, like `stores`. `state_view` is an owned
`Arc<S>` snapshot the host supplies because the loop only ever holds a
borrowed `&State` it cannot lend to a concurrent tool future.

The portable `ToolRunContext` methods cover only workspace, thread id, and
output cap. The rest is reachable from a `tinytools::Tool` by downcasting the
erased host extension:

```rust
async fn execute_with_context(&self, args: Value, options: ToolCallOptions,
                              context: Option<&dyn ToolRunContext>) -> anyhow::Result<ToolResult> {
    let harness = context
        .and_then(ToolRunContext::host_extension)
        .and_then(|any| any.downcast_ref::<ToolExecutionContext>());
    if let Some(h) = harness {
        h.custom(json!({"stage": "fetching"}));
        if let Some(store) = &h.store { store.put(&ns, h.call_id.as_str(), value).await?; }
        let tenant = h.state::<AppState>().map(|s| s.tenant);
    }
    ...
}
```

A `ToolDispatch` implementor receives the same `call_id` as an explicit
`execute` parameter alongside the typed parent `RunContext`.

## Rich returns: `follow_up` and `metadata` (B2)

`tinytools::ToolResult` carries two host-facing fields beyond `content`:

- **`follow_up: Vec<ToolContent>`** — content the model should see as a
  *separate user message* after the tool result (a screenshot after a click,
  a document a later turn should read). The loop appends one user message per
  result that has any, **after the batch's last tool row**, in call order —
  never between two tool rows, because a provider requires every tool row to
  sit directly after the assistant row that requested it. Blocks map as:
  `Text`/`Json` verbatim; `Image` → `ContentBlock::Image(ImageRef)` with a
  URL as-is or inline bytes as a `data:<media_type>;base64,<bytes>` URI;
  `File` → the text placeholder `[file <name> (<media_type>)]` (the message
  model has no file block yet). The tool row itself never includes it.
- **`metadata: Option<Value>`** — host-only. It is copied onto the call's
  `AgentEvent::ToolCompleted { metadata }` and onto
  `AgentRun::tool_metadata` (`ToolResultMetadata { call_id, tool_name,
  metadata }`, one entry per answered call that carried any), and is never
  rendered into the transcript — not into the tool row's content, nor its
  host-side `artifact`, nor any model request.

The tool row's `artifact` still carries the full ordered `tinytools_content`
block list, the markdown rendering, and the reported-error bit for host
consumers and persistence; model-facing content stays bounded to `content`
(or the markdown rendering when `prefer_markdown` selects it).
