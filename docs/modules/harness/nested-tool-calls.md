# Nested tool calls (C9)

> **Opt-in; off by default.** `RunLimits::max_nested_depth` defaults to `0`, so
> `call_tool` fails with `nested tool calls are disabled (max_nested_depth = 0)`.
> Enable it with `with_max_nested_depth(n)` **only after** every `before_tool`
> enforcement your host registers (allowlists, policy gates, approval gates,
> plan mode, host hook middlewares) also implements
> `Middleware::check_nested_tool`. `before_tool` never runs for nested calls, so
> an enforcement without that method is bypassed by `call_tool`.

A tool that needs another tool's result calls it through the harness instead of
reaching into the registry:

```rust
async fn execute_with_context(&self, args, _opts, context) -> anyhow::Result<ToolResult> {
    let harness = context
        .and_then(tinytools::ToolRunContext::host_extension)
        .and_then(|any| any.downcast_ref::<ToolExecutionContext>())
        .expect("installed by the agent loop");
    let found = harness.call_tool("search", json!({"q": "tinyagents"})).await?;
    // ...
}
```

`ToolExecutionContext::call_tool(name, arguments) -> Result<ToolResult>`
(`crates/tinyagents-harness/src/tool/types.rs`). The call is admitted and
executed much like a model-issued call, **with exceptions** (below), so read
"What applies" before relying on a middleware to bind nested calls. Source: `src/tool/nested.rs` (the seam)
and `src/agent_loop/nested.rs` (the runner).

## What applies to a nested call

| Stage | Behaviour |
| --- | --- |
| Lookup | `ToolRegistry::model_dispatch` plus the hosted run's tool allow-list. An unknown or disallowed name is `Err(ToolNotFound)`. |
| Arguments | Injected-argument preparation, the `InvalidArgsPolicy` normalisation, schema validation. Invalid arguments are an `Err` naming the tool. |
| Approval | A tool that is external or declares `approval_required` **fails**: `nested call '<name>' requires approval; nested calls cannot be deferred`. The parent is never deferred. A `CallDeferred`/`ApprovalRequired` raised during execution is the same refusal. |
| Middleware admission | `Middleware::check_nested_tool(&RunContext, &State, &ToolCall)` on every registered middleware, in order, after validation and before host authorization; the first `Err` refuses the call. This is how a middleware's `before_tool` **enforcement** reaches nested calls: `ToolAllowlistMiddleware`, `ToolPolicyMiddleware` (deny mask, sandbox, classification, approval grants), `HumanApprovalMiddleware` (a flagged tool fails with the approval error unless its callback allows; it never interrupts or defers) and `PlanModeMiddleware` implement it over the same decision as `before_tool`. The default admits. |
| Host authorization | `SecurityGate::authorize_tool` for a hosted run; a denial is an `Err`. The `ToolCallRequest` has `parent_call_id: Some(..)` (`is_nested()`): a nested call cannot be prompted for, so a host gate that would normally ask a human must **fail closed** for it. |
| Tool-wrap onion | `ToolMiddleware::wrap_tool` runs around the tool, so a policy middleware can deny or rewrite a nested call. |
| Timeouts | The per-tool timeout policy and the run's remaining wall-clock budget. |
| Budget | One `max_tool_calls` pool shared with model-issued calls (below). |
| Result observation | `Middleware::observe_nested_result(&RunContext, &State, &ToolCall, &ToolResult)` on every middleware after a nested call produced a result. `after_tool` never runs for nested calls, so a budget or repeated-failure middleware (research budget, failure counters, result auditing) implements this to account for them. It cannot rewrite the result; keep state behind interior mutability. A refused or raised call has no result and is not observed. |
| Effect ledger | One `started`/`settled` row per nested call, keyed by the nested id, so recovery sees the effects a parent had through `call_tool`. A ledger `started` failure follows `LedgerFailure` (`Abort` refuses the call). A call dropped mid-flight leaves its row `started`. |
| Refusal cap | A refused call releases its budget slot, so a parent gets at most 8 refused nested calls; every later call is refused without being evaluated. |
| Cancellation | The run's token: a cancelled run refuses the next nested call, and dropping the parent drops its in-flight nested calls. |

A tool that reports its own failure (`ToolResult::is_error`) is an `Ok` result,
exactly as for a model-issued call. A refusal is an `Err(TinyAgentsError)`; a
tool that forwards it with `?` is answered to the model as
`TinyAgentsError::ToolFailed` (the message above reaches the model verbatim).

Outside the agent loop (no runner installed) `call_tool` returns
`ToolFailed("cannot call tool '<name>' ...: nested tool calls are only available
while the agent loop executes the calling tool")`.

Open follow-up for hosts: a host's own `before_tool` enforcement (for example
a hooks middleware that vetoes tools) must implement `check_nested_tool`
too, or `call_tool` is a way around it.

## Dropped calls

If a parent settles, times out or is cancelled while a nested call is in
flight, or its tool drops the `call_tool` future, the nested call is dropped and
reports `ToolFailed("parent settled: ...")`, so every `ToolStarted` has exactly
one terminal event. Its summary status is `abandoned` when the tool stopped
waiting. A `ToolDispatch` that builds the `ToolExecutionContext` must do so
inside the future the loop polls, or the context carries no runner.

## Budget

`LimitTracker` holds an atomic `nested_tool_calls` next to `tool_calls`
(`tool_calls()` is the model-issued count only; both count against the cap). A
nested call reserves a slot with `try_reserve_nested_tool_call` (`&self`, so it
works from a tool future that holds only `&RunContext`); model-issued admission
counts `tool_calls + nested_tool_calls` against `max_tool_calls`. Concurrent
parents therefore cannot overspend the cap, and a cap spent by nested calls
trips the next model-issued call. A nested call that is refused at admission
releases its slot; one that ran keeps it. A spent cap is always an error for
the nested caller (`LimitExceeded`, plus `LimitReached { ToolCalls }`), whatever
`LimitBehavior` says: there is no loop boundary at which to stop with a partial
result.

## Depth

`RunLimits::max_nested_depth` (default `0` = disabled; `with_max_nested_depth(n)`
enables up to `n` levels, `3` is a reasonable choice). A call the
model issued is level 0; what its tool calls is level 1; and so on. A call whose
level exceeds the cap fails with `nested call '<name>' exceeds
max_nested_depth (<n>)`. This is distinct from `max_depth`, the sub-agent
recursion cap.

## Ids and events

A nested call's id is `<parent call id>/<n>` (`n` counts from 1 per parent;
nested-of-nested ids extend the path: `p1/1/1`). `ToolStarted`, `ToolCompleted`
and `ToolFailed` carry `parent_call_id: Option<CallId>` (serde default, omitted
when `None`) naming the **immediate** parent (`p1/1/1` has parent `p1/1`, not
`p1`), so exporters can nest the span. Every nested `ToolStarted` has
exactly one terminal partner.

## Transcript and metadata

Nested calls get **no transcript rows**: a nested call is never answered to the
provider, so adding one would break tool-call/tool-result pairing. The parent's
result metadata (host-only; `ToolCompleted.metadata` and
`AgentRun::tool_metadata`) gains a capped summary:

```json
{
  "nested_calls": [
    {"id": "p1/1", "name": "leaf", "status": "ok", "duration_ms": 3,
     "args": "{\"n\":7}"},
    {"id": "p1/2", "name": "ghost", "status": "failed", "duration_ms": 0,
     "args": "{}", "error": "tool not found: ghost"}
  ]
}
```

(`nested_calls_truncated` is added, as a count, only when more than 32 calls ran.)

`status` is `ok`, `error` (the tool returned `is_error`), `failed` (refused or
raised) or `abandoned` (the tool stopped waiting). `args` is the serialized arguments cut at 1 KiB, present only when tool payload capture (`PayloadCapture::tool_io`) is on; `error` (also only under `tool_io` capture, except for `abandoned`) is cut at 256
bytes and omitted when empty. At most 32 entries are kept; `nested_calls_truncated`
counts the rest. The summary is attached only when the metadata is absent or an
object.

## What does not apply

- **Per-call approvals by id.** A nested id (`<parent>/<n>`) never inherits an
  approval granted for a model call with the same id: `HumanApprovalMiddleware`
  ignores `is_call_approved` for nested calls. Approval callbacks
  (`ApprovalFn`, `ApprovalOutcomeFn`) receive only the `ToolCall`; a nested call
  is recognizable by its id format `<parent>/<n>` (the `ToolCallRequest` a host
  security gate sees carries an explicit `parent_call_id`).
- **`RepeatProgressMiddleware` and observe/redaction middleware.** They key on
  `before_tool`/`after_tool`, so the repeat guard never counts nested calls and
  middleware that redacts or scrubs tool arguments and results (credential
  scrubbing, observe redaction) does not run on them. With payload capture on
  (`PayloadCapture::tool_io`), a nested call's arguments reach the nested
  `ToolStarted`/`ToolCompleted` events and the effect ledger's idempotency key
  unredacted. Keep capture off, or redact in an event listener, when nested
  arguments may be sensitive.
- **`before_tool` / `after_tool` proper.** They take `&mut RunContext`, which a
  tool future (holding `&RunContext`) cannot lend, so they never run for nested
  calls. Enforcement belongs in `check_nested_tool` or a
  `ToolMiddleware::wrap_tool`; accounting in `observe_nested_result`. The
  repeat-progress and no-progress guards key on `before_tool`/`after_tool`, so
  they never see nested calls and cannot count them as model repeats of the
  parent.
- **Progress gate.** A nested tool's `report_progress` is not forwarded: the
  parent's gate is keyed to the parent's call id, and a nested call does not open
  one of its own on the parent's stream. Use the nested call's result and the
  parent's own progress.
- **Host output screening and tool control.** The screening of the parent's
  final output covers the call as a whole. A nested result's `ToolControl` (`terminate`, `return_direct`, `goto`) and a
  wrap middleware's control request are ignored.

## Shape (why it is a channel)

The harness installs a `NestedToolRunner` (a type-erased `Arc<dyn ...>`) in a
task-local while it drives the call, and `ToolExecutionContext::from_run_context`
captures it for exactly that call id. The runner is a channel: the loop that
drives the tool's future also services the nested requests it sends, borrowing
`&AgentHarness`, `&State` and `&RunContext` as the model-issued path does.
Servicing a nested call is a plain call-base execution, so it can nest again.

## Concurrency

Nested calls honor `Tool::is_concurrency_safe` and `ToolMiddleware::concurrent_safe`
among themselves: one parent's fan-out and every concurrent parent of the run share a
shared/exclusive gate (safe calls shared, unsafe ones exclusive). A concurrency-safe
nested call that itself calls an unsafe tool is refused (a shared hold cannot be
upgraded). The gate covers nested calls only: it does not stop a nested unsafe tool
from overlapping a *model-issued* sibling in the same batch, because that batch's
safety was declared per call by the model-issued path; a tool that nests an unsafe
tool takes that composition on itself.

## Replay of a parent with nested effects

Nested ledger rows are keyed by the deterministic nested id, and the ledger only
lists unresolved rows. A nested call to a non-replay-safe tool is refused over an
unresolved row of the same id, but a row that already settled cannot be seen
through the ledger trait, so a parent that is replayed after its nested effect
completed would issue the same id again. A parent that nests non-replayable
effects must therefore declare `ToolReplay::Never` itself.
