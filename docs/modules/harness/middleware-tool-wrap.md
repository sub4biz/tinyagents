# Tool-wrap middleware and concurrent batches

`ToolMiddleware::wrap_tool` wraps one tool call in an onion of layers around
the real tool. Until this change it took `&mut RunContext`, so a harness with
any registered wrap ran every multi-call batch **serially**. It now takes a
shared `&RunContext`, and the wrap onion runs inside each concurrent call.

## Breaking changes (migration entry)

No CHANGELOG exists in this repo; breaking changes are recorded here.

| Item | Before | After |
| --- | --- | --- |
| `ToolMiddleware::wrap_tool` | `ctx: &mut RunContext<Ctx>` | `ctx: &RunContext<Ctx>` |
| `ToolHandler::run` | `ctx: &mut RunContext<Ctx>` | `ctx: &RunContext<Ctx>` |
| `ToolBaseCall::call` | `ctx: &'a mut RunContext<Ctx>` | `ctx: &'a RunContext<Ctx>` |
| `MiddlewareStack::run_wrapped_tool` | `ctx: &mut RunContext<Ctx>` | `ctx: &RunContext<Ctx>` |
| `AgentEvent::MiddlewareStarted` / `MiddlewareCompleted` | `{ name }` | `{ name, call_id: Option<CallId> }` (serde default, omitted when `None`) |
| `ToolMiddleware::concurrent_safe` | n/a | new, defaults to `true` |

Struct-literal constructions and exhaustive patterns of the two events need
`call_id` (or `..`). Details and migration steps follow.

## Breaking signature change

```rust
// before
async fn wrap_tool(&self, ctx: &mut RunContext<Ctx>, state: &State,
                   call: ToolCall, next: ToolHandler<'_, State, Ctx>)
    -> Result<MiddlewareToolOutcome>;
// after
async fn wrap_tool(&self, ctx: &RunContext<Ctx>, state: &State,
                   call: ToolCall, next: ToolHandler<'_, State, Ctx>)
    -> Result<MiddlewareToolOutcome>;
```

The same change applies to:

- `ToolHandler::run(&self, ctx: &RunContext<Ctx>, ..)`;
- `ToolBaseCall::call(&self, ctx: &RunContext<Ctx>, ..)` (the innermost real
  call; the dispatch underneath already took `&RunContext`);
- `MiddlewareStack::run_wrapped_tool(&self, ctx: &RunContext<Ctx>, ..)`.

### Migration

For almost every wrap the migration is mechanical: change `&mut RunContext` to
`&RunContext` in the `wrap_tool` signature (and in any helper you forward `ctx`
to). Everything a wrap could legitimately do already works on `&RunContext`:

- read run data (`ctx.run_id()`, `ctx.data`, limits, cancellation);
- `ctx.emit(..)` events and `ctx.request_control(..)`;
- use interior-mutable handles (`Arc<Mutex<..>>`, atomics, channels).

A wrap that **mutated** the context (assigning a field on `ctx`) must move that
state into an interior-mutable handle it owns, or into a lifecycle hook
(`before_tool` / `after_tool`, which still receive `&mut RunContext` and run
serially during admission and fold). `ModelMiddleware` and `AgentMiddleware`
are unchanged.

## When a batch runs concurrently

A turn with two or more calls runs them concurrently when every call's tool is
`is_concurrency_safe`, arguments need no host preparation, and **every
registered wrap reports `concurrent_safe() == true`** (the default). The
registered wraps no longer matter on their own.

- Admission (`before_tool`, limits, validation) stays serial and in call order.
- Each call's future runs the full wrap onion around the tool; calls overlap.
- Results fold serially in call order; `ToolCompleted` events are emitted in
  call order.
- `MiddlewareStarted` / `MiddlewareCompleted` are emitted per call and stay
  balanced, including when a wrap errors or short-circuits. Events from
  different calls interleave, so correlate by call, not by adjacency: the
  tool-wrap onion tags both events with `call_id: Some(<tool call id>)` (the
  other middleware hooks leave it `None`).
- In concurrent mode every `ToolStarted` is emitted at admission, **before**
  any wrap runs, so never infer "the current call" from event order; use the
  `call` argument of `wrap_tool` and the `call_id` on the events.
- A short-circuit or `Err` in one call's wrap does not touch its siblings;
  they run to completion. A fatal `Err` still fails the turn at the first such
  call in call order (siblings get their terminal events). An
  `ApprovalRequired` / `CallDeferred` raised inside a wrap defers just that
  call, like a deferral raised by the tool.
- `RunLimits::max_tool_concurrency` bounds how many wrapped calls overlap.

## Escape hatch: `concurrent_safe`

```rust
fn concurrent_safe(&self) -> bool { false }
```

Return `false` when the wrap needs to see one call at a time (a non-reentrant
lock held across `next.run`, a strictly ordered log, a single-slot resource).
If **any** registered wrap returns `false`, every multi-call batch runs
serially in call order, exactly as before. The stack exposes the aggregate as
`MiddlewareStack::tool_middleware_concurrent_safe()`.

## Approval prompts stay one at a time

The library `ApprovalGateMiddleware` and `ToolPolicyGateMiddleware` hold a
per-middleware `tokio::sync::Mutex<()>` around `ApprovalResolver::resolve`. A
host that routes one prompt per chat thread therefore never sees two prompts
collide, even though the batch's calls overlap. Only the interactive part is
serialised: calls that need no approval (`requires_approval == false`, or a
policy `Allow`/`Deny`) never take the lock, and an approved call runs its tool
outside it. A host that calls `ToolPolicyGate::evaluate` directly (instead of
the middleware) owns that serialisation itself. These middleware do **not**
set `concurrent_safe() == false`.

## Nested calls

A call a tool makes with `ToolExecutionContext::call_tool` runs the wrap onion
too, so a policy wrap can deny it. `before_tool`/`after_tool` proper take
`&mut RunContext` and do not run for nested calls; `Middleware::check_nested_tool`
and `observe_nested_result` are their shared-reference counterparts. See
[nested-tool-calls.md](nested-tool-calls.md).
