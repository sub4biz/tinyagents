# Harness Middleware Feature

Middleware is the main extension point for behavior that cuts across models,
tools, memory, stores, streaming, and graph nodes.

## Source Inspiration

LangChain v1 middleware provides typed model requests, tool-call wrappers,
before/after hooks, wrap hooks, dynamic prompts, human-in-the-loop control, PII
redaction, retry/fallback, summarization, context editing, and tool selection:

- middleware types:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py>
- built-in middleware:
  <https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/agents/middleware>
- agent factory composition:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py>

TinyAgents should provide equivalent extension power without requiring users to
understand graph internals for normal harness usage.

## Responsibilities

- Provide stable middleware ordering.
- Support before/after hooks for observation and simple mutation.
- Support wrap hooks for replacement, retry, fallback, short-circuit, and
  human-interrupt behavior.
- Support streaming hooks for model deltas and tool progress so middleware can
  act during long-running calls.
- Support steering hooks so parent orchestrators or humans can safely guide
  agent loops and sub-agent runs without direct state mutation.
- Support prompt/cache-layout hooks so middleware can compress context without
  accidentally invalidating provider prompt/KV-cache prefixes.
- Allow middleware to modify model requests, tool calls, and responses.
- Allow middleware to emit events.
- Allow middleware to add local state updates without mutating unrelated state.
- Allow middleware to jump to model, tools, or end when used inside an agent loop.
- Translate middleware control outcomes into state-graph commands when a run is
  graph-backed.
- Expose errors to middleware for logging, redaction, fallback, or recovery.
- Keep middleware testable with fake models, tools, and event sinks.

## Hook Types

```rust
#[async_trait]
pub trait Middleware<State, Ctx = ()>: Send + Sync {
    async fn before_agent(&self, state: &State, ctx: &mut RunContext<Ctx>) -> Result<()>;
    async fn after_agent(&self, state: &State, ctx: &mut RunContext<Ctx>, run: &mut AgentRun) -> Result<()>;
    async fn before_steering(&self, state: &State, ctx: &mut RunContext<Ctx>, command: &mut SteeringCommand) -> Result<()>;
    async fn after_steering(&self, state: &State, ctx: &mut RunContext<Ctx>, outcome: &mut SteeringOutcome) -> Result<()>;

    async fn before_model(&self, state: &State, ctx: &mut RunContext<Ctx>, request: &mut ModelRequest) -> Result<()>;
    async fn before_model_stream(&self, state: &State, ctx: &mut RunContext<Ctx>, request: &mut ModelRequest) -> Result<()>;
    async fn on_model_delta(&self, state: &State, ctx: &mut RunContext<Ctx>, delta: &mut ModelDelta) -> Result<()>;
    async fn after_model(&self, state: &State, ctx: &mut RunContext<Ctx>, response: &mut ModelResponse) -> Result<()>;

    async fn before_tool(&self, state: &State, ctx: &mut RunContext<Ctx>, call: &mut ToolCall) -> Result<()>;
    async fn on_tool_delta(&self, state: &State, ctx: &mut RunContext<Ctx>, delta: &mut ToolDelta) -> Result<()>;
    async fn after_tool(&self, state: &State, ctx: &mut RunContext<Ctx>, invocation: &ToolInvocationIdentity, result: &mut ToolResult) -> Result<()>;

    async fn on_error(&self, state: &State, ctx: &mut RunContext<Ctx>, error: &TinyAgentsError) -> Result<()>;
}
```

Wrap hooks need separate traits because they receive a handler. These are
implemented in `crate::harness::middleware`:

```rust
#[async_trait]
pub trait ModelMiddleware<State, Ctx = ()>: Send + Sync {
    fn name(&self) -> &str;
    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome>;
}

#[async_trait]
pub trait ToolMiddleware<State, Ctx = ()>: Send + Sync {
    fn name(&self) -> &str;
    /// Defaults to `true`; `false` keeps multi-call batches serial.
    fn concurrent_safe(&self) -> bool { true }
    async fn wrap_tool(
        &self,
        ctx: &RunContext<Ctx>, // shared: calls of one batch overlap
        state: &State,
        call: ToolCall,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareToolOutcome>;
}
```

`next` is a borrowed handle to the rest of the onion (`ModelHandler` /
`ToolHandler`); calling `next.run(ctx, state, request_or_call)` proceeds to the
inner layer and ultimately the real model/tool call. Because `run` borrows
`&self`, a wrap middleware can call it **zero** times (short-circuit /
replace), **once** (proceed), or **many** times (retry / fallback). The
innermost layer is supplied by the agent loop via the `ModelBaseCall` /
`ToolBaseCall` traits, and the stack composes the onion through
`MiddlewareStack::run_wrapped_model` / `run_wrapped_tool` (registration order =
outermost first). `MiddlewareModelOutcome::Response(ModelResponse)` and
`MiddlewareToolOutcome::Result(ToolResult)` carry the resolved value; both are
`#[non_exhaustive]`. The agent loop runs each lifecycle `before_*` hook, then
the wrap onion, then each lifecycle `after_*` hook. `wrap_tool` and
`ToolHandler::run` take a shared `&RunContext`; see
[tool-wrap concurrency](middleware-tool-wrap.md) for what that means for
multi-call batches and how to migrate a `&mut RunContext` wrap.

## Ordering

Before hooks run in registration order. After hooks run in reverse registration
order. Wrap hooks compose so the first registered middleware is the outermost
layer. This mirrors common web middleware stacks and keeps cleanup symmetrical.

Streaming hooks run in registration order for each delta before the delta is
forwarded to subscribers or accumulated into the final response. Middleware that
needs symmetrical setup and teardown for a stream should use `wrap_model`; delta
hooks are for per-chunk inspection or transformation.

Prompt/cache-layout middleware should run after static prompt rendering and
before model dispatch. It must declare whether it changed stable prefix segments
or only volatile tail segments so provider prompt-cache behavior is observable.

## Control Outcomes

Middleware should be able to return:

- continue with modified request
- replace model/tool response
- replace or suppress a streaming delta
- emit state update
- retry current call
- fallback to another model or tool
- accept, reject, transform, or defer a steering command
- jump to `model`
- jump to `tools`
- jump to `end`
- interrupt for human input
- persist checkpoint
- resume from checkpoint
- fail with classified error

Graph-specific commands should be translated at the graph boundary. The harness
should expose harness-native control outcomes so it remains usable without a
graph.

## Graph Boundary

Middleware must not need to know whether the caller is using the simple loop or
the state-graph runtime. The runtime adapter maps harness-native outcomes onto
graph commands:

- continue -> `Command::Continue`
- jump to model/tools/end -> `Command::Goto(...)` or `Command::End`
- human interrupt -> `Command::Interrupt`
- accepted steering -> `Command::Update`, `Command::Goto`,
  `Command::Interrupt`, or queued child-run delivery depending on target
- branch/fan-out middleware -> `Command::Fork`
- retry/fallback -> handled inside the node or wrap hook before command return

When middleware mutates graph-visible state, it must emit an explicit state
update event so checkpoint replay can explain the change.

## Built-In Middleware

Initial built-ins should include:

- tracing/event middleware
- timeout middleware
- retry middleware
- model fallback middleware
- model-call limit middleware
- tool-call limit middleware
- rate limiter middleware
- dynamic prompt middleware
- prompt cache layout guard middleware
- context editing middleware
- context compression middleware
- transcript compression middleware
- retrieval compression middleware
- streaming delta compression middleware
- output compression middleware
- message trimming middleware
- summarization middleware
- structured output validation middleware
- PII detection/redaction middleware
- tool allowlist middleware
- dynamic tool selection middleware
- human-in-the-loop middleware
- privileged shell/filesystem guard middleware

Each built-in must document:

- hook points used
- mutation behavior
- emitted events
- failure mode
- interaction with streaming
- interaction with provider prompt/KV-cache layout
- interaction with retries and fallbacks

The repeat-progress guard (`RepeatProgressMiddleware`) is documented in [repeat-progress.md](repeat-progress.md).

## Tool policy enforcement

`ToolPolicyMiddleware` (`crates/tinyagents-harness/src/middleware/library/`) enforces the
per-tool [`ToolPolicy`](tool.md#tool-policy-enforcement) metadata at two hooks:
`before_model` (exposure — a blocked tool is hidden from the model) and
`before_tool` (execution — a blocked call is rejected with
`TinyAgentsError::Validation`). Both hooks share one decision so a hidden tool
can never be executed by a divergent path.

Build it from a registry snapshot (`ToolRegistry::policies()`), then compose
enforcement builders:

- `ToolPolicyMiddleware::strict(policies)` — fail-closed baseline: unclassified
  or unknown tools are rejected, and `destructive`/`payment` side effects denied.
- `.require_classification(bool)` / `.require_background_safe(bool)`
- `.deny_side_effects(mask)` — deny any tool declaring a side effect in `mask`.
- `.require_sandbox(true)` — block a tool whose `runtime.sandbox ==
  SandboxMode::Required` **unless** the run carries a workspace whose `sandbox`
  is `Required`; hosts attach that descriptor through around-agent middleware,
  and enforcement fails closed otherwise.
- `.require_approval([names])` — block any tool declaring
  `access.approval_required` unless its name is in the approved set.
- `.enforce_result_bytes(true)` — in `after_tool`, truncate a result exceeding
  the tool's `runtime.max_result_bytes` and flag it (`result.error` mentions
  `max_result_bytes`).

```rust
use std::sync::Arc;
use tinyagents_harness::middleware::{MiddlewareStack, ToolPolicyMiddleware};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinytools::{SandboxMode, ToolPolicy, ToolRuntime, WorkspaceDescriptor};

let mut policies = std::collections::HashMap::new();
policies.insert(
    "shell".to_string(),
    ToolPolicy::classified().with_runtime(ToolRuntime {
        sandbox: SandboxMode::Required,
        ..ToolRuntime::default()
    }),
);
use tinyinference_llm::tool::ToolCall;
let call = || ToolCall::new("c1", "shell", serde_json::json!({}));

let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
stack.push(Arc::new(ToolPolicyMiddleware::new(policies).require_sandbox(true)));

// No workspace -> the sandboxed tool is blocked (fail closed).
let mut bare: RunContext = RunContext::new(RunConfig::new("no-sandbox"), ());
assert!(stack.run_before_tool(&mut bare, &(), &mut call()).await.is_err());

// A sandboxed workspace satisfies the requirement.
let mut ok: RunContext = RunContext::new(RunConfig::new("sandboxed"), ())
    .with_workspace(WorkspaceDescriptor::new("/work").with_sandbox(SandboxMode::Required));
stack.run_before_tool(&mut ok, &(), &mut call()).await?; // admitted
```

Emitted events: rejections surface as `TinyAgentsError::Validation` (not events).
`enforce_result_bytes` mutates the `ToolResult` in place.

## Tool exposure

`ContextualToolSelectionMiddleware` filters the model-visible tool set on each
`before_model`, using a predicate that sees both the `ToolSchema` and a live
`ToolSelectionContext { run_id, depth, tags, requested_model }` — so exposure can
vary by recursion depth, run tags (security tier / background marker), or the
target model. When it withholds any tools it emits
`AgentEvent::ToolsFiltered { by, excluded, remaining }`, making the exposure
decision auditable.

Two constructors:

- `from_lists(allow, deny)` — deny always hides; when `allow` is `Some`, a tool
  must be listed to be exposed (fail-closed for unknown tools).
- `inheriting(parent_allow, parent_deny, child_allow, child_deny)` — composes a
  child policy against an inherited parent policy so a sub-agent can only
  **narrow**, never widen: **deny is additive** (`parent ∪ child`) and **allow
  is intersective** (the effective allowlist is the intersection when both
  restrict; the single restriction when only one does; unrestricted otherwise).

```rust
use tinyagents_harness::middleware::ContextualToolSelectionMiddleware;

// Parent allows {a,b,c} and denies {c}; child tries to allow {b,c,d}.
// Effective allow = {a,b,c} ∩ {b,c,d} = {b,c}; deny adds parent's c -> {c}.
// So only `b` survives (`d` was never parent-allowed, `c` is parent-denied).
let mw = ContextualToolSelectionMiddleware::inheriting(
    Some(["a", "b", "c"]), ["c"],
    Some(["b", "c", "d"]), Vec::<String>::new(),
);
// After run_before_model, request.tools == [schema("b")].
```

Exposure only changes what the model *sees*; pair it with
[tool policy enforcement](#tool-policy-enforcement) or `ToolAllowlistMiddleware`
so a model that calls a hidden tool is still stopped at execution.

## Plan mode

`PlanModeMiddleware` (built with `plan_mode_middleware`) is a ready-made
combination of the two mechanisms above, gated by a live, host-settable
`RunMode`: `Build` (the default) leaves every tool exposed and executable;
`Plan` hides every side-effecting tool from the model at `before_model` (the
same `AgentEvent::ToolsFiltered` auditing `ContextualToolSelectionMiddleware`
emits) and denies it at `before_tool` (the same `ToolPolicy`/`ToolSideEffects`
classification `ToolPolicyMiddleware::deny_side_effects` enforces) — except for
tools on the middleware's own allowlist, which a host uses to keep read-only
tools and plan-mode-specific tools (e.g. `plan_exit`, `request_plan_review`,
`todo`) available while planning.

A tool counts as side-effecting when its policy declares any of
`writes_files`, `network`, `installs_dependencies`, `destructive`,
`external_service`, or `payment` — or when `policies` has no entry for it at
all: an unclassified tool is assumed capable of side effects until it is
either classified read-only or added to the allowlist.

`RunModeHandle` is `Clone` and cheap to share: a host keeps one clone to flip
modes (a UI toggle, a `plan_exit` tool call, a slash command) while the
middleware holds another to read it. `set` takes effect on the very next tool
exposure or execution check, so a host can switch modes mid-run, between
turns, without restarting it.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tinyagents_harness::middleware::{plan_mode_middleware, MiddlewareStack, RunMode, RunModeHandle};
use tinytools::{ToolPolicy, ToolSideEffects};

let mut policies = HashMap::new();
policies.insert("read_file".to_string(), ToolPolicy::read_only());
policies.insert(
    "write_file".to_string(),
    ToolPolicy::classified().with_side_effects(ToolSideEffects {
        writes_files: true,
        ..ToolSideEffects::default()
    }),
);

let mode = RunModeHandle::new(RunMode::Plan);
let mw = plan_mode_middleware(mode.clone(), policies).allow(["plan_exit"]);

let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
stack.push(Arc::new(mw));
// While `mode.get() == RunMode::Plan`: `write_file` is hidden and denied,
// `read_file` and `plan_exit` stay available. `mode.set(RunMode::Build)`
// lifts the restriction on the very next check.
```

## Middleware control (A1)

Any middleware (or step) can steer the loop out-of-band via
`RunContext::request_control(MiddlewareControl)`
([context feature](context.md#middleware-control-outcomes)). `MiddlewareControl`
has five variants:

- `MiddlewareControl::Continue` — no instruction; never installed as a pending
  request (`request_control` treats it as a no-op).
- `MiddlewareControl::JumpTo(LoopTarget)` — reroute the loop. `LoopTarget::Model`
  abandons the rest of the current turn (closing out any unanswered tool calls
  on the last assistant row) and restarts from a fresh model call;
  `LoopTarget::Tools` is a no-op — the turn's tool calls already run whenever
  they exist, so there is nothing else to route to; `LoopTarget::End` finishes
  the run, using `AgentRun::final_response` if already set, otherwise the most
  recent assistant text.
- `MiddlewareControl::UpdateState(StateUpdate)` — queue a typed state mutation.
  The loop only ever holds `state: &State` (a shared reference), so it cannot
  apply this itself: it pushes the `StateUpdate` onto
  `RunContext::push_state_update`, and a host that owns `&mut State` between
  runs drains the queue with `RunContext::take_state_updates` and applies each
  one. `StateUpdate::new::<S>(f: impl Fn(&mut S))` captures the closure behind
  an `Arc` (so `MiddlewareControl` stays `Clone`) and is a no-op if applied
  against a mismatched `State` type.
- `MiddlewareControl::StopWithFinal(text)` — stop now, using `text` as the final
  assistant response.
- `MiddlewareControl::Interrupt { node, message }` — pause at the next safe
  checkpoint, surfacing `TinyAgentsError::Interrupted` so a caller can checkpoint
  and resume.

Requests are resolved by **precedence, not last-writer**: `request_control`
keeps the highest-`precedence()` pending request within a turn — `Interrupt`
(4) outranks `StopWithFinal` (3), which outranks `JumpTo`/`UpdateState` (2/1),
which outranks `Continue` (0) — so a stronger pause is never silently
downgraded to a stop by a later weaker request. The agent loop drains the
request at its safe checkpoints (before dispatching a model call, after the
model response, and after tool execution) via `RunContext::take_control` and,
when it honors one, emits `AgentEvent::ControlApplied { control, detail }`
where `control` is the outcome's `kind()` label (`"continue"`,
`"jump_to:model"`, `"jump_to:tools"`, `"jump_to:end"`, `"update_state"`,
`"stop_with_final"`, `"interrupt"`). `UpdateState` is queued silently (no
`ControlApplied` event) since it carries no loop-level decision.

### Returning control from a hook

Every lifecycle hook has a `_control`-suffixed counterpart
(`before_model_control`, `after_model_control`, `before_tool_control`,
`after_tool_control`, `before_agent_control`, `after_agent_control`) that the
`MiddlewareStack` actually drives. Each defaults to calling the plain hook and
returning `Continue`, so an existing `Middleware` impl that only overrides the
plain hooks keeps compiling and behaving identically — this is the source-
compatibility shim A1 was built around. Override the `_control` variant
directly (not both) when a hook needs to steer the loop:

```rust
async fn before_model_control(
    &self,
    ctx: &mut RunContext<Ctx>,
    state: &State,
    request: &mut ModelRequest,
) -> Result<MiddlewareControl> {
    if budget_exhausted() {
        return Ok(MiddlewareControl::JumpTo(LoopTarget::End));
    }
    Ok(MiddlewareControl::Continue)
}
```

A returned control is resolved into exactly the same `request_control` call a
hook could have made explicitly — returning control is sugar over the side
channel, not a second mechanism.

### Precedence within one phase, and `is_observer`

Within one phase (e.g. every registered middleware's `before_model_control`),
the **first** non-`Continue` outcome wins. Every hook *after* it in that same
phase is skipped — not called at all — unless `Middleware::is_observer`
returns `true` for it, in which case it still runs (for logging, metrics,
audit) but its own control outcome is discarded; only the first winner is ever
applied. This mirrors LangChain's `@hook_config(can_jump_to=[...])`
declaration without requiring middleware to declare targets up front.

### Turn-boundary stop: `should_stop_after_turn`

`Middleware::should_stop_after_turn(&self, ctx, run) -> bool` (default `false`)
is evaluated once at the turn boundary — after tool execution, before the loop
would otherwise continue to the next model call. It exists for a decision that
depends on the *whole turn's* tool results rather than any single call (a
tally, a cross-tool invariant); returning `true` has the same effect as
requesting `JumpTo(End)` from `after_tool_control`.

### Tools returning control

A canonical tool's own `ToolResult` (from vendored `tinytools`) carries an
optional `ToolControl { return_direct, terminate, goto, state_update }`. The
loop translates it into the same `MiddlewareControl` vocabulary after
`after_tool`/`after_tool_control` run:

- `return_direct` — the tool's own output becomes `AgentRun::final_response`
  and the loop requests `JumpTo(End)` (recording the final response directly
  rather than falling back to the last assistant text, which `JumpTo(End)`
  alone cannot target precisely). Any one call asking is enough.
- `terminate` — a *batch* decision (pi's `shouldTerminateToolBatch`): the run
  ends only when **every** call of the turn's batch asked to terminate, because
  a sibling that did not may have returned something the model still has to
  read. A mixed batch, or one with a deferred call, ignores the hint and the
  loop continues. When the batch does end the run, the final response is the
  last call's output in source order. A call that sets `terminate` does not
  also act on its own `goto`.
- `goto: Some("model" | "tools" | "end")` — mapped to the matching
  `JumpTo(LoopTarget)`; an unrecognized value is logged and ignored.
- `state_update: Some(json)` — queued as raw JSON via
  `RunContext::push_tool_state_update` / `take_tool_state_updates` (a separate
  queue from `MiddlewareControl::UpdateState`'s typed closures, since a
  canonical tool has no access to the harness's `State` type).

### Wrap outcomes: `Command`

`MiddlewareModelOutcome` and `MiddlewareToolOutcome` (both `#[non_exhaustive]`)
each gained a `Command { control: MiddlewareControl }` variant alongside their
existing `Response`/`Result` variant, for a `wrap_model`/`wrap_tool`
implementation that decides — before ever calling `next` — that the run
should stop or jump. There is no real response/result in that case;
`into_response()`/`into_result()` return an empty placeholder, and
`into_response_with_control()`/`into_result_with_control()` additionally
recover the control, which the agent loop queues via `request_control` at the
next safe checkpoint exactly like any other control request.

### Built-ins rebased on control outcomes

`BudgetMiddleware::before_model_control` requests `JumpTo(End)` when the
budget is already exhausted, instead of erroring the run out — a stop that
preserves the partial transcript rather than discarding it.
`HumanApprovalMiddleware::before_tool_control` requests `Interrupt` for a
flagged, unapproved call instead of returning `Err` directly, so the interrupt
is expressed in the shared control vocabulary (visible via
`RunContext::take_control` before it drains) rather than only as a thrown
error; the loop still surfaces the identical
`TinyAgentsError::Interrupted` once drained.

## State And Request Mutation

Middleware should prefer immutable request replacement for large changes and
small explicit mutation for local fields. It must not mutate shared registries or
global config during a run. Runtime state updates should be explicit and
observable.

## Compression Middleware

Compression is not one hook. A useful compression implementation may need to run
at several boundaries:

- `before_agent`: load previous compression state and policy.
- `before_model`: compress old messages, retrieved context, examples, or tool
  artifacts before the request is sent.
- `wrap_model`: measure full call timing, retry behavior, cache layout, and
  provider usage while preserving setup/teardown symmetry.
- `on_model_delta`: compact, redact, sample, or classify streaming output before
  it is persisted or forwarded.
- `after_model`: commit response summaries, update transcript compression state,
  and attach provenance to the final response.
- `before_tool` and `after_tool`: compress large tool arguments/results and
  decide what enters model-visible context.
- `after_agent`: persist durable summaries, compression indexes, and audit
  events.

Compression middleware must preserve enough provenance for debugging and replay:
source message ids, source artifact ids, original token estimates, compressed
token estimates, prompt segment ids, cache prefix fingerprints, policy version,
and whether the stable provider prompt-cache prefix was preserved.
