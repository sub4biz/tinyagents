//! Type definitions for the harness middleware module.
//!
//! These are the typed extension points that wrap each level of the recursive
//! harness: the [`Middleware`] trait's hooks fire identically around the parent
//! agent loop and around every nested model/tool/agent call beneath it, so
//! observation and policy compose the same way at any recursion depth.
//!
//! This file holds every public type in `crate::middleware`: the
//! [`AgentRun`] result record, the core [`Middleware`] trait, the
//! [`MiddlewareStack`] composer, and the built-in middleware implementations.
//! Behavioral code (trait default bodies, the stack runner, and built-in
//! `Middleware` impls) lives in the sibling `mod.rs`; focused tests live in
//! `test.rs`.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::middleware` directly.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::cache::CacheLayoutEvent;
use crate::context::{MiddlewareControl, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::HarnessRunStatus;
use crate::ids::{CallId, RunId};
use crate::summarization::{SummarizationPolicy, Summarizer, SummaryRecord, TrimStrategy};
use tinyinference_llm::model::{ModelDelta, ModelRequest, ModelResponse};
use tinyinference_llm::tool::{ToolCall, ToolDelta};
use tinyinference_llm::usage::UsageTotals;
use tinytools::ToolResult;

// ── AgentRun ────────────────────────────────────────────────────────────────

/// Harness-owned identity for one completed tool invocation.
///
/// [`Middleware::after_tool`] receives this separately from
/// [`ToolResult`][tinytools::ToolResult] so canonical TinyTools results remain
/// correlation-free and a tool cannot forge its own execution identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolInvocationIdentity {
    call_id: CallId,
    tool_name: String,
}

impl ToolInvocationIdentity {
    /// Creates an identity for a completed invocation.
    pub fn new(call_id: impl Into<CallId>, tool_name: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
        }
    }

    /// The provider/harness correlation id for this invocation.
    pub fn call_id(&self) -> &CallId {
        &self.call_id
    }

    /// The canonical name of the invoked tool.
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

/// The accumulated result of a single agent run.
///
/// `AgentRun` decouples middleware (and other observers) from the internals of
/// the agent loop. The loop builds and threads an `AgentRun` through the run,
/// updating its counters and message log as model and tool calls complete, and
/// hands a `&mut AgentRun` to [`Middleware::after_agent`] so middleware can
/// inspect or post-process the final result without owning loop state.
///
/// # Example
///
/// ```
/// use tinyagents_harness::middleware::AgentRun;
///
/// let mut run = AgentRun::new();
/// run.model_calls += 1;
/// run.steps += 1;
/// assert_eq!(run.text(), None);
/// ```
#[derive(Clone, Debug, Default)]
pub struct AgentRun {
    /// The full conversation transcript produced by the run, in order.
    pub messages: Vec<tinyinference_llm::message::Message>,
    /// The final model response, when the run produced one.
    pub final_response: Option<ModelResponse>,
    /// Parsed structured output, when the run requested a structured format.
    pub structured: Option<serde_json::Value>,
    /// Which schema variant matched, when [`Self::structured`] was extracted
    /// under [`crate::structured::StructuredStrategy::ToolCallUnion`] (A6).
    /// `None` for every other strategy, and whenever `structured` is `None`.
    pub structured_variant: Option<String>,
    /// Cumulative token usage across every model call in the run.
    pub usage: UsageTotals,
    /// Number of model calls dispatched during the run.
    pub model_calls: usize,
    /// Number of tool invocations executed during the run.
    pub tool_calls: usize,
    /// Names of calls that reached a tool executor, in execution order.
    ///
    /// Recovery messages for denied, unknown, or invalid calls intentionally do
    /// not appear here: they produced a transcript response but never acted.
    pub executed_tools: Vec<String>,
    /// Number of loop iterations (model/tool super-steps) executed.
    pub steps: usize,
    /// Set when the run stopped because steering latched a **pause** rather
    /// than because the model finished.
    ///
    /// A paused run has no `final_response`, exactly like a run whose model
    /// returned an empty answer — which is why the two were previously
    /// indistinguishable. Check this field (or
    /// [`HarnessRunStatus`][crate::events::HarnessRunStatus], which
    /// reports `Interrupted` for a paused run) before treating a missing final
    /// response as a completed-but-empty answer. The pause stays latched on the
    /// [`SteeringHandle`][crate::steering::SteeringHandle], so
    /// [`SteeringHandle::resume`][crate::steering::SteeringHandle::resume]
    /// lifts it and a fresh invocation continues from
    /// [`AgentRun::messages`].
    pub paused: Option<crate::steering::PauseState>,
    /// Set when the run stopped because one or more tool calls were
    /// **deferred** (A2): they need a human approval or host-side execution
    /// before the loop can continue. Like [`Self::paused`], this is not a
    /// completion — there is no `final_response`, and
    /// [`HarnessRunStatus`][crate::events::HarnessRunStatus] reports the run
    /// `Interrupted`. Persist [`Self::messages`] together with this value,
    /// resolve it into a [`crate::tool::DeferredToolResults`], and resume
    /// with [`crate::runtime::AgentHarness::resume_deferred`].
    pub deferred: Option<crate::tool::DeferredToolRequests>,
    /// Messages the host pushed onto the run queue's `Collect` lane (A4),
    /// drained once when the run ends — on every exit path, including
    /// errors. They are delivered here for the host to act on and are
    /// **never** appended to the transcript or sent to the model. Empty when
    /// the run had no queue.
    pub collected: Vec<tinyinference_llm::message::Message>,
    /// Host-only metadata tools attached to their results
    /// (`tinytools::ToolResult::metadata`, B2), one entry per answered call
    /// that carried any, in fold order. Kept beside [`Self::executed_tools`]
    /// rather than inside it so the name list stays a plain `Vec<String>`.
    /// The same value rides the call's
    /// [`AgentEvent::ToolCompleted`][crate::events::AgentEvent::ToolCompleted];
    /// neither copy is ever rendered into [`Self::messages`].
    pub tool_metadata: Vec<ToolResultMetadata>,
    /// The transcript the next model call would have seen: [`Self::messages`]
    /// with this run's context compaction applied (leading system messages, the
    /// compaction checkpoint, then the messages kept verbatim). `None` when no
    /// compaction ran.
    ///
    /// [`Self::messages`] stays the full record of the run. A host that
    /// carries history into its next turn should carry *this* one when set:
    /// the next turn then starts from the checkpoint instead of re-reading
    /// (and re-summarizing) everything the compaction already folded. Set by
    /// [`ContextCompressionMiddleware`]'s `after_agent` hook.
    pub compacted_history: Option<Vec<tinyinference_llm::message::Message>>,
    /// How the run ended, structured (see [`crate::terminal`]). Set by the
    /// agent loop on every exit path it controls — completion, a
    /// `StopWithPartial` cap, a pause, a deferral — and by the driver on
    /// failure, so a host reading a partial run (for example from
    /// [`PartialRunOutcome`][crate::agent_loop::PartialRunOutcome]) needs no
    /// string parsing. `None` only while the run is still in flight, or when a
    /// wrapping middleware replaced the loop.
    pub terminal: Option<crate::terminal::TerminalOutcome>,
}

/// Host-only metadata one tool call returned, as recorded on
/// [`AgentRun::tool_metadata`] (B2).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolResultMetadata {
    /// The call that produced it — matches the transcript row and the
    /// `ToolCompleted` event.
    pub call_id: CallId,
    /// The tool the call named (after any unknown-tool rewrite).
    pub tool_name: String,
    /// The metadata verbatim; never shown to the model.
    pub metadata: serde_json::Value,
}

// ── Middleware trait ──────────────────────────────────────────────────────────

/// A cross-cutting extension point invoked around agent, model, and tool
/// execution.
///
/// Middleware is the primary way to add behavior — tracing, guardrails,
/// trimming, caching protection, usage accounting, retries — without touching
/// the agent loop or graph internals. Every hook has a no-op default so an
/// implementor overrides only the ones it cares about.
///
/// # Ordering (onion model)
///
/// When composed in a [`MiddlewareStack`], `before_*` hooks run in registration
/// order while `after_*` hooks run in **reverse** registration order. The first
/// registered middleware is therefore the outermost layer: it sets up first and
/// tears down last, mirroring common web-middleware stacks and keeping cleanup
/// symmetrical.
///
/// # Mutation
///
/// Hooks receive mutable references to the value flowing through the run
/// (`request`, `delta`, `response`, `call`, `result`) so they can transform it
/// in place. They also receive `&mut RunContext<Ctx>` for emitting events and
/// recording limits, plus a shared `&State` for read-only application state.
///
/// All hooks are async and return [`Result`]; returning `Err` short-circuits
/// the stack (see [`MiddlewareStack`]).
#[async_trait]
pub trait Middleware<State: Send + Sync, Ctx: Send + Sync = ()>: Send + Sync {
    /// A short, stable label used in `MiddlewareStarted`/`MiddlewareCompleted`
    /// events. This is intentionally synchronous and should return a `'static`
    /// string literal.
    fn name(&self) -> &str;

    /// Runs once before the agent loop begins, before any model call.
    async fn before_agent(&self, _ctx: &mut RunContext<Ctx>, _state: &State) -> Result<()> {
        Ok(())
    }

    /// Runs once after the agent loop finishes, with the completed [`AgentRun`]
    /// available for inspection or post-processing.
    async fn after_agent(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _run: &mut AgentRun,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs before each model request is dispatched, allowing the middleware to
    /// mutate the outgoing [`ModelRequest`].
    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _request: &mut ModelRequest,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs for each streamed [`ModelDelta`] before it is forwarded or
    /// accumulated, allowing inspection or transformation of the chunk.
    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ModelDelta,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs after each model call completes, allowing the middleware to mutate
    /// the [`ModelResponse`].
    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _response: &mut ModelResponse,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs before each tool invocation, allowing the middleware to mutate the
    /// outgoing [`ToolCall`].
    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _call: &mut ToolCall,
    ) -> Result<()> {
        Ok(())
    }

    /// Observes each [`ToolDelta`] of progress a tool reported through
    /// `ToolRunContext::report_progress`.
    ///
    /// **Replayed, observe-only.** The hook needs `&mut RunContext`, which the
    /// executing tool holds, so the loop calls it for a call's deltas, in
    /// order, *after* the call settles and before `after_tool` and the terminal
    /// event. The matching `AgentEvent::ToolProgressDetail` was already emitted live,
    /// so mutating `delta` changes nothing downstream. The replay queue is
    /// bounded (newest 64 deltas, `content` capped at 4 KiB) and is not filled
    /// at all for runs without middleware. An `Err` is logged, not propagated.
    async fn on_tool_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ToolDelta,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs after each tool invocation completes, allowing the middleware to
    /// inspect harness-owned invocation identity and mutate the [`ToolResult`].
    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _invocation: &ToolInvocationIdentity,
        _result: &mut ToolResult,
    ) -> Result<()> {
        Ok(())
    }

    /// Shared-reference admission check for a **nested** tool call: a tool
    /// calling another tool through
    /// [`ToolExecutionContext::call_tool`][crate::tool::ToolExecutionContext::call_tool].
    ///
    /// `before_tool` takes `&mut RunContext`, which a running tool cannot lend,
    /// so it never runs for nested calls. A middleware whose `before_tool` is an
    /// *enforcement* (an allowlist, a deny mask, an approval gate, a plan-mode
    /// guard, a host hook) must therefore also implement this method, over the
    /// same decision, or `call_tool` becomes a way around it. The default
    /// admits the call, which is right for hooks that only observe or rewrite.
    ///
    /// Runs for every registered middleware, in registration order, after
    /// argument validation and before host authorization; the first `Err`
    /// refuses the call. A nested call can never be deferred, so a gate that
    /// would defer or interrupt must fail instead. `call.arguments` are the
    /// prepared (validated) arguments; `call.id` is the nested id
    /// (`<parent>/<n>`).
    ///
    /// A refusal is returned to the calling tool and is **not** fanned out to
    /// [`Middleware::on_error`]: that hook needs `&mut RunContext`, which a
    /// running tool cannot lend, and a refused nested call is a tool-level
    /// result, not a run failure.
    async fn check_nested_tool(
        &self,
        _ctx: &RunContext<Ctx>,
        _state: &State,
        _call: &ToolCall,
    ) -> Result<()> {
        Ok(())
    }

    /// Observes the result of a nested tool call, after it ran.
    ///
    /// `after_tool` never runs for nested calls (it takes `&mut RunContext`),
    /// so a middleware that accounts for tool results — a research budget, a
    /// repeated-failure counter, a result auditor — implements this to see
    /// them. It cannot rewrite the result. Called for every registered
    /// middleware, in registration order, once per nested call that produced a
    /// result (a tool-reported error included; a refused or raised call has no
    /// result). Interior mutability is the way to keep state: the context is
    /// shared.
    async fn observe_nested_result(
        &self,
        _ctx: &RunContext<Ctx>,
        _state: &State,
        _call: &ToolCall,
        _result: &ToolResult,
    ) {
    }

    /// Runs when any hook in the stack errors, giving every middleware a chance
    /// to log, redact, or react to the failure. The original error is still
    /// returned to the caller after this runs; errors from `on_error` itself
    /// are ignored so they cannot mask the root cause.
    async fn on_error(&self, _ctx: &mut RunContext<Ctx>, _error: &TinyAgentsError) -> Result<()> {
        Ok(())
    }

    // ── Control-outcome hooks ────────────────────────────────────────────
    //
    // Each hook above has a `_control`-suffixed counterpart the
    // [`MiddlewareStack`] actually drives. The default implementation below
    // calls the plain hook and returns [`MiddlewareControl::Continue`], so
    // every existing `Middleware` impl that only overrides the plain hooks
    // keeps compiling and behaving exactly as before (A1's source-compat
    // shim). Override a `_control` hook directly (instead of, not in
    // addition to, the plain one) when the outcome needs to steer the loop —
    // stop, jump, interrupt, or queue a state update. See
    // `docs/modules/harness/middleware.md` for the precedence rule the stack
    // applies across a phase's hooks and the checkpoints the loop honors a
    // returned control at.

    /// Control-outcome counterpart of [`Self::before_agent`].
    async fn before_agent_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
    ) -> Result<MiddlewareControl> {
        self.before_agent(ctx, state).await?;
        Ok(MiddlewareControl::Continue)
    }

    /// Control-outcome counterpart of [`Self::after_agent`].
    async fn after_agent_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        run: &mut AgentRun,
    ) -> Result<MiddlewareControl> {
        self.after_agent(ctx, state, run).await?;
        Ok(MiddlewareControl::Continue)
    }

    /// Control-outcome counterpart of [`Self::before_model`].
    async fn before_model_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: &mut ModelRequest,
    ) -> Result<MiddlewareControl> {
        self.before_model(ctx, state, request).await?;
        Ok(MiddlewareControl::Continue)
    }

    /// Control-outcome counterpart of [`Self::after_model`].
    async fn after_model_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        response: &mut ModelResponse,
    ) -> Result<MiddlewareControl> {
        self.after_model(ctx, state, response).await?;
        Ok(MiddlewareControl::Continue)
    }

    /// Control-outcome counterpart of [`Self::before_tool`].
    async fn before_tool_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        call: &mut ToolCall,
    ) -> Result<MiddlewareControl> {
        self.before_tool(ctx, state, call).await?;
        Ok(MiddlewareControl::Continue)
    }

    /// Control-outcome counterpart of [`Self::after_tool`].
    async fn after_tool_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> Result<MiddlewareControl> {
        self.after_tool(ctx, state, invocation, result).await?;
        Ok(MiddlewareControl::Continue)
    }

    /// Whether this middleware still runs (for observation) in a phase where
    /// an earlier middleware already produced a winning control outcome.
    ///
    /// The stack applies the *first* non-[`MiddlewareControl::Continue`]
    /// outcome in a phase and, by default, skips every hook after it — an
    /// early-exit tool guard or a budget stop should not pay for hooks whose
    /// work is now moot. A middleware that must still observe every call
    /// regardless (a usage accountant, an audit log) overrides this to
    /// `true`; its own control outcome is then ignored; only the first
    /// winning one is ever applied. See `docs/modules/harness/middleware.md`.
    fn is_observer(&self) -> bool {
        false
    }

    /// Whether the loop should stop after the turn currently completing,
    /// evaluated once at the turn boundary (after tool execution, before the
    /// loop would otherwise continue to the next model call).
    ///
    /// Defaults to `false`. A middleware that returns `true` here has the
    /// same effect as requesting
    /// [`MiddlewareControl::JumpTo`]`(`[`crate::context::LoopTarget::End`]`)`
    /// from `after_tool_control`, but expresses "stop once this turn settles"
    /// without needing to compute that decision inside `after_tool_control`
    /// itself (useful when the decision depends on the whole turn's tool
    /// results, not just one call).
    fn should_stop_after_turn(&self, _ctx: &RunContext<Ctx>, _run: &AgentRun) -> bool {
        false
    }
}

// ── Wrap (around-call) middleware ─────────────────────────────────────────────

/// A pinned, boxed future producing a [`ModelResponse`].
///
/// This is the return type of [`ModelBaseCall::call`] and of the futures wrap
/// middleware drive. The lifetime `'a` ties the future to the borrows it
/// captures (the run context, application state, and the base handler).
pub type BoxModelFuture<'a> = Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>>;

/// A pinned future that drives one complete agent run.
pub type BoxAgentFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

/// Mutable input to one complete agent run.
///
/// Around-agent middleware owns this value, so it may rewrite the initial
/// transcript or select the streaming path before forwarding it.
#[derive(Clone, Debug)]
pub struct AgentRequest {
    /// Initial conversation transcript.
    pub input: Vec<tinyinference_llm::message::Message>,
    /// Whether provider calls should use their streaming path.
    pub streaming: bool,
}

impl AgentRequest {
    /// Creates a complete-run request.
    pub fn new(input: Vec<tinyinference_llm::message::Message>, streaming: bool) -> Self {
        Self { input, streaming }
    }
}

/// A pinned, boxed future producing a [`ToolResult`].
///
/// The tool-wrap counterpart of [`BoxModelFuture`].
pub type BoxToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>>;

/// The innermost agent loop wrapped by [`AgentMiddleware`].
///
/// The mutable run is deliberately visible at this boundary. An outer host
/// middleware can therefore persist a partial transcript or release a
/// run-scoped resource after `next` returns an error, not only after a clean
/// completion.
pub trait AgentBaseCall<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Drives the agent loop with the possibly rewritten input.
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        request: AgentRequest,
        run: &'a mut AgentRun,
        status: &'a mut HarnessRunStatus,
    ) -> BoxAgentFuture<'a>;
}

/// The innermost model call wrapped by the [`ModelMiddleware`] onion.
///
/// This is the *real* model invocation — the cache + retry + fallback core of
/// the agent loop. The loop supplies it as the `base` of
/// [`MiddlewareStack::run_wrapped_model`]. A wrap middleware reaches it (the
/// innermost layer) by calling [`ModelHandler::run`], possibly more than once
/// (for retry) or not at all (to short-circuit).
pub trait ModelBaseCall<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Invokes the wrapped model call with the (possibly middleware-mutated)
    /// `request`.
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        request: ModelRequest,
    ) -> BoxModelFuture<'a>;
}

/// The innermost tool call wrapped by the [`ToolMiddleware`] onion.
///
/// The tool-wrap counterpart of [`ModelBaseCall`]; the loop supplies the real
/// tool invocation as the `base` of [`MiddlewareStack::run_wrapped_tool`].
pub trait ToolBaseCall<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Invokes the wrapped tool with the (possibly middleware-mutated) `call`.
    fn call<'a>(
        &'a self,
        ctx: &'a RunContext<Ctx>,
        state: &'a State,
        call: ToolCall,
    ) -> BoxToolFuture<'a>;
}

/// The outcome of a wrapped model call.
///
/// Carries the [`ModelResponse`] the wrapped call resolves to. A
/// [`ModelMiddleware`] produces one by either:
///
/// - **proceeding** — forwarding the response returned by [`ModelHandler::run`];
/// - **short-circuiting / replacing** — constructing a [`Self::Response`]
///   without ever calling `next`;
/// - **retrying** — calling `next` in a loop until it succeeds or a budget is
///   exhausted; or
/// - **falling back** — calling `next`, then substituting a response on error.
///
/// Retry and replacement are therefore expressed by *how* a middleware uses
/// `next` rather than by distinct enum variants; the enum only needs to carry
/// the resolved response. It is `#[non_exhaustive]` so future control variants
/// can be added without breaking callers.
// `Response(ModelResponse)` is large relative to `Command`'s payload; boxing
// it would ripple through every construction/destructure site across the
// crate (including the `From<ModelResponse>` impl below and every wrap
// middleware) for a value that lives only as long as one model call, so the
// size skew is accepted here rather than threaded through as indirection.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[allow(clippy::large_enum_variant)]
pub enum MiddlewareModelOutcome {
    /// The response to use as the result of the wrapped model call.
    Response(ModelResponse),
    /// Short-circuit with a [`MiddlewareControl`] instead of a response — for
    /// example a wrap middleware that decides, before ever calling `next`,
    /// that the run should stop or jump. There is no response to hand back in
    /// this case, so callers that need one (see [`Self::into_response`]) get
    /// an empty placeholder; the control itself is recovered separately, via
    /// [`Self::into_response_with_control`], and applied through the same
    /// [`RunContext::request_control`][crate::context::RunContext::request_control]
    /// path a lifecycle hook's control-outcome return uses.
    Command {
        /// The control outcome to apply.
        control: MiddlewareControl,
    },
}

impl MiddlewareModelOutcome {
    /// Unwraps the contained [`ModelResponse`], or an empty placeholder for
    /// [`Self::Command`] (see that variant's docs — prefer
    /// [`Self::into_response_with_control`] when a `Command` must not be
    /// silently discarded).
    pub fn into_response(self) -> ModelResponse {
        self.into_response_with_control().0
    }

    /// Splits this outcome into a [`ModelResponse`] (a placeholder for
    /// [`Self::Command`]) and the [`MiddlewareControl`] to apply, when this
    /// was a `Command` outcome.
    pub fn into_response_with_control(self) -> (ModelResponse, Option<MiddlewareControl>) {
        match self {
            Self::Response(response) => (response, None),
            Self::Command { control } => (ModelResponse::assistant(String::new()), Some(control)),
        }
    }
}

impl From<ModelResponse> for MiddlewareModelOutcome {
    fn from(response: ModelResponse) -> Self {
        Self::Response(response)
    }
}

/// The outcome of a wrapped tool call.
///
/// The tool-wrap counterpart of [`MiddlewareModelOutcome`]; see its docs for the
/// proceed / replace / retry / fallback patterns. `#[non_exhaustive]` for the
/// same forward-compatibility reason.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum MiddlewareToolOutcome {
    /// The result to use as the result of the wrapped tool call.
    Result(ToolResult),
    /// Short-circuit with a [`MiddlewareControl`] instead of a result. The
    /// tool-wrap counterpart of [`MiddlewareModelOutcome::Command`]; see its
    /// docs for the placeholder-result and control-recovery contract.
    Command {
        /// The control outcome to apply.
        control: MiddlewareControl,
    },
}

impl MiddlewareToolOutcome {
    /// Unwraps the contained [`ToolResult`], or an empty error placeholder for
    /// [`Self::Command`] (prefer [`Self::into_result_with_control`] when a
    /// `Command` must not be silently discarded).
    pub fn into_result(self) -> ToolResult {
        self.into_result_with_control().0
    }

    /// Splits this outcome into a [`ToolResult`] (a placeholder for
    /// [`Self::Command`]) and the [`MiddlewareControl`] to apply, when this
    /// was a `Command` outcome.
    pub fn into_result_with_control(self) -> (ToolResult, Option<MiddlewareControl>) {
        match self {
            Self::Result(result) => (result, None),
            Self::Command { control } => (ToolResult::success(String::new()), Some(control)),
        }
    }
}

impl From<ToolResult> for MiddlewareToolOutcome {
    fn from(result: ToolResult) -> Self {
        Self::Result(result)
    }
}

/// A handle to the remainder of the model-wrap onion: the inner wrap middleware
/// plus the innermost [`ModelBaseCall`].
///
/// A [`ModelMiddleware`] receives this as its `next` argument. Calling
/// [`ModelHandler::run`] **proceeds** to the next layer (eventually the base
/// model call); not calling it **short-circuits**; calling it repeatedly
/// **retries**. `run` borrows `&self`, so a middleware may invoke `next` as many
/// times as it likes.
pub struct ModelHandler<'a, State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) remaining: &'a [Arc<dyn ModelMiddleware<State, Ctx>>],
    pub(crate) base: &'a dyn ModelBaseCall<State, Ctx>,
}

/// A handle to the remainder of the around-agent middleware onion.
pub struct AgentHandler<'a, State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) remaining: &'a [Arc<dyn AgentMiddleware<State, Ctx>>],
    pub(crate) base: &'a dyn AgentBaseCall<State, Ctx>,
    pub(crate) status: &'a mut HarnessRunStatus,
}

/// A handle to the remainder of the tool-wrap onion: the inner wrap middleware
/// plus the innermost [`ToolBaseCall`].
///
/// The tool-wrap counterpart of [`ModelHandler`]; see its docs.
pub struct ToolHandler<'a, State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) remaining: &'a [Arc<dyn ToolMiddleware<State, Ctx>>],
    pub(crate) base: &'a dyn ToolBaseCall<State, Ctx>,
}

/// Around-call ("wrap") middleware for model invocations.
///
/// Unlike the lifecycle [`Middleware`] hooks (which only observe/mutate values
/// flowing past), a `ModelMiddleware` *surrounds* the inner model pipeline: it
/// receives a [`ModelHandler`] (`next`) that runs the rest of the onion plus the
/// real model call. This is the most powerful extension point — it can proceed,
/// short-circuit with a replacement response, retry `next` in a loop, or fall
/// back — all while keeping setup and teardown symmetrical around the call.
///
/// # Ordering
///
/// Wrap middleware compose as a nested onion: the first-registered middleware is
/// the **outermost** layer (it runs first and finishes last), and the innermost
/// layer is the real model call. See [`MiddlewareStack::run_wrapped_model`].
#[async_trait]
pub trait ModelMiddleware<State: Send + Sync, Ctx: Send + Sync = ()>: Send + Sync {
    /// A short, stable label used in
    /// `MiddlewareStarted`/`MiddlewareCompleted` events.
    fn name(&self) -> &str;

    /// Whether this middleware already retries the model call itself (as
    /// [`crate::middleware::library::RetryMiddleware`] does).
    ///
    /// [`MiddlewareStack::has_retry_override`] uses this to tell the loop's
    /// base call to skip its own [`crate::runtime::RunPolicy::retry`] loop
    /// when one is registered — otherwise the two retry layers compose
    /// multiplicatively (`mw.max_attempts × policy.retry.max_attempts ×
    /// |fallback|` provider calls for one logical failure) instead of
    /// replacing each other. See I-7; full unification into one engine is a
    /// later phase. Defaults to `false` so an ordinary middleware is
    /// unaffected.
    fn overrides_retry(&self) -> bool {
        false
    }

    /// Wraps the inner model pipeline. Call `next.run(ctx, state, request)` to
    /// proceed (zero or more times), or return a [`MiddlewareModelOutcome`]
    /// without calling it to short-circuit.
    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome>;
}

/// Around-call ("wrap") middleware for tool invocations.
///
/// The tool-wrap counterpart of [`ModelMiddleware`]; see its docs for the full
/// proceed / replace / retry / fallback model and onion ordering.
#[async_trait]
pub trait ToolMiddleware<State: Send + Sync, Ctx: Send + Sync = ()>: Send + Sync {
    /// A short, stable label used in
    /// `MiddlewareStarted`/`MiddlewareCompleted` events.
    fn name(&self) -> &str;

    /// Whether this wrap tolerates **overlapping** invocations: the calls of
    /// one multi-call tool batch each run the whole wrap onion at the same
    /// time, on a shared `&RunContext`.
    ///
    /// Defaults to `true`, because `wrap_tool` only receives `&RunContext` and
    /// so can do no more than read it, emit events, request control, and use
    /// interior-mutable handles it owns. Return `false` when the wrap holds
    /// state that must see one call at a time (a non-reentrant lock held
    /// across `next.run`, a strictly ordered audit log, a single-slot
    /// resource): if *any* registered wrap returns `false`, the harness runs
    /// every multi-call batch serially, in call order, exactly as before the
    /// wrap onion became concurrent.
    ///
    /// In concurrent mode every `ToolStarted` event is emitted at admission,
    /// before any wrap runs, so never infer the "current call" from event
    /// order; use the `call` argument and the `call_id` on the wrap's
    /// `MiddlewareStarted`/`MiddlewareCompleted` events.
    fn concurrent_safe(&self) -> bool {
        true
    }

    /// Wraps the inner tool pipeline. Call `next.run(ctx, state, call)` to
    /// proceed (zero or more times), or return a [`MiddlewareToolOutcome`]
    /// without calling it to short-circuit.
    ///
    /// `ctx` is shared (`&RunContext`) because the calls of a batch may run
    /// this method concurrently; see [`Self::concurrent_safe`]. Events
    /// (`ctx.emit`) and control requests (`ctx.request_control`) take `&self`.
    async fn wrap_tool(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareToolOutcome>;
}

/// Around-call middleware for a complete agent run.
///
/// This is the host-policy extension point. It can load memory and prepend it
/// to `input`, prepare a workspace in `ctx`, short-circuit a run, and perform
/// cleanup or persistence after `next` returns. Unlike paired lifecycle hooks,
/// code after `next.run(..).await` also executes when the inner loop fails.
#[async_trait]
pub trait AgentMiddleware<State: Send + Sync, Ctx: Send + Sync = ()>: Send + Sync {
    /// A short, stable label used in middleware lifecycle events.
    fn name(&self) -> &str;

    /// Wraps the complete agent loop.
    async fn wrap_agent(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: AgentRequest,
        run: &mut AgentRun,
        next: AgentHandler<'_, State, Ctx>,
    ) -> Result<()>;
}

// ── MiddlewareStack ───────────────────────────────────────────────────────────

/// An ordered collection of [`Middleware`] composed with onion semantics.
///
/// `before_*` runner methods invoke each middleware in registration order;
/// `after_*` runner methods invoke them in reverse. Every per-middleware hook
/// invocation is bracketed by `AgentEvent::MiddlewareStarted` and
/// `MiddlewareCompleted` events emitted through the [`RunContext`]. The first
/// hook that returns `Err` short-circuits the stack: every middleware's
/// [`Middleware::on_error`] is invoked, then the original error is returned.
///
/// In addition to those lifecycle hooks, the stack holds ordered lists of
/// **wrap** middleware for complete agent runs, model calls, and tool calls.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use tinyagents_harness::middleware::{LoggingMiddleware, MiddlewareStack};
///
/// let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
/// stack.push(Arc::new(LoggingMiddleware::new()));
/// assert_eq!(stack.len(), 1);
/// ```
pub struct MiddlewareStack<State: Send + Sync, Ctx: Send + Sync = ()> {
    pub(crate) middlewares: Vec<Arc<dyn Middleware<State, Ctx>>>,
    pub(crate) agent_middlewares: Vec<Arc<dyn AgentMiddleware<State, Ctx>>>,
    pub(crate) model_middlewares: Vec<Arc<dyn ModelMiddleware<State, Ctx>>>,
    pub(crate) tool_middlewares: Vec<Arc<dyn ToolMiddleware<State, Ctx>>>,
}

// ── LoggingMiddleware ─────────────────────────────────────────────────────────

/// Per-hook invocation counts captured by [`LoggingMiddleware`].
///
/// A snapshot is returned from [`LoggingMiddleware::counts`] so tests and
/// dashboards can assert which hooks fired and how often.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HookCounts {
    /// Number of `before_agent` invocations.
    pub before_agent: usize,
    /// Number of `after_agent` invocations.
    pub after_agent: usize,
    /// Number of `before_model` invocations.
    pub before_model: usize,
    /// Number of `on_model_delta` invocations.
    pub on_model_delta: usize,
    /// Number of `after_model` invocations.
    pub after_model: usize,
    /// Number of `before_tool` invocations.
    pub before_tool: usize,
    /// Number of `on_tool_delta` invocations.
    pub on_tool_delta: usize,
    /// Number of `after_tool` invocations.
    pub after_tool: usize,
    /// Number of `on_error` invocations.
    pub on_error: usize,
}

/// Observation-only middleware that records how often each hook fired.
///
/// `LoggingMiddleware` mutates nothing in the run; it only increments interior
/// counters so callers can inspect hook activity via [`LoggingMiddleware::counts`].
/// The surrounding [`MiddlewareStack`] already emits start/completed events, so
/// this type adds no events of its own.
pub struct LoggingMiddleware {
    pub(crate) label: &'static str,
    pub(crate) counts: Mutex<HookCounts>,
}

// ── MessageTrimMiddleware ─────────────────────────────────────────────────────

/// Middleware that trims the request transcript before each model call.
///
/// In `before_model` it replaces `request.messages` with the result of
/// [`crate::summarization::trim_messages`] under the configured
/// [`TrimStrategy`], bounding prompt growth across long agent loops.
pub struct MessageTrimMiddleware {
    /// The trimming strategy applied to `request.messages`.
    pub strategy: TrimStrategy,
}

// ── ContextCompressionMiddleware ──────────────────────────────────────────────

/// Default cap on the number of [`SummaryRecord`]s a
/// [`ContextCompressionMiddleware`] retains before evicting the oldest.
pub const DEFAULT_COMPRESSION_RECORD_CAP: usize = 1024;

/// How [`ContextCompressionMiddleware`] recovers when its [`Summarizer`]
/// returns an `Err`.
///
/// The default
/// [`ConcatSummarizer`][crate::summarization::ConcatSummarizer] is
/// infallible, but the [`Summarizer`] trait allows failure (for example a
/// model-backed summarizer whose provider call is rejected). A summarizer
/// failure strikes precisely on the longest, most valuable transcripts — the
/// ones that reached the compaction threshold — so propagating the error and
/// aborting the whole run is the worst outcome. This policy selects the
/// recovery behaviour; the default is [`FallbackTrim`](Self::FallbackTrim).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompressionFailurePolicy {
    /// Propagate the summarizer error, aborting the run. This is the legacy
    /// behaviour from before the policy existed.
    Abort,

    /// Deterministically front-drop the oldest messages (system messages
    /// preserved) until the transcript fits the policy's trigger budget, then
    /// continue. Loses old context but keeps the run alive. The default.
    #[default]
    FallbackTrim,

    /// Leave the transcript untouched and continue. The next model call sees the
    /// full (over-threshold) transcript; use when the caller would rather risk a
    /// provider-side context error than drop history.
    PassThrough,
}

/// The type of a `before_compaction` hook, consulted before every compaction
/// [`ContextCompressionMiddleware`] runs. Named to keep the struct field's
/// type simple (`clippy::type_complexity`).
pub type BeforeCompactionHook = std::sync::Arc<
    dyn Fn(&crate::summarization::CompactionContext) -> crate::summarization::CompactionDecision
        + Send
        + Sync,
>;

/// Middleware that summarizes/compresses the request transcript, but **only**
/// when it nears the model's context window.
///
/// In `before_model` it consults the configured [`SummarizationPolicy`]. The
/// policy is normally built with a context window (for example via
/// [`SummarizationPolicy::from_profile`] or
/// [`SummarizationPolicy::with_context_window`]) and a `threshold_fraction`
/// (default `0.9`). When the estimated transcript tokens are **below** the
/// window threshold this middleware is a complete no-op: `request.messages` is
/// left untouched and no event is emitted. When the threshold is reached, the
/// older messages are condensed by the [`Summarizer`] into a single summary
/// message, the recent window and system messages are kept verbatim, the
/// resulting [`SummaryRecord`] (with its compression provenance) is recorded,
/// and an [`AgentEvent::Compressed`][crate::events::AgentEvent::Compressed]
/// event is emitted.
///
/// [`ConcatSummarizer`][crate::summarization::ConcatSummarizer] is used by
/// default; supply any [`Summarizer`] via
/// [`ContextCompressionMiddleware::with_summarizer`]. See
/// [`ContextCompressionMiddleware::new`] for construction, and
/// [`CompressionFailurePolicy`] /
/// [`with_failure_policy`](ContextCompressionMiddleware::with_failure_policy)
/// for how a summarizer error is recovered.
pub struct ContextCompressionMiddleware {
    /// Label reported in `MiddlewareStarted`/`MiddlewareCompleted` events.
    pub(crate) label: &'static str,
    /// Threshold and context-window configuration consulted on each call.
    pub(crate) policy: SummarizationPolicy,
    /// Condenses older messages into a summary once the threshold is reached.
    pub(crate) summarizer: Box<dyn Summarizer>,
    /// Compression history, oldest first, capped at `max_records`.
    pub(crate) records: Mutex<VecDeque<SummaryRecord>>,
    /// Eviction cap for `records`.
    pub(crate) max_records: usize,
    /// Recovery behaviour when [`Summarizer::summarize`] returns `Err`.
    pub(crate) on_failure: CompressionFailurePolicy,
    /// Token budget above which a single "turn" of messages handed to the
    /// summarizer is itself split into two halves and merged (see
    /// [`crate::summarization::summarize_with_split`]). `None` disables
    /// splitting — the whole `to_summarize` slice is always summarized in one
    /// call, matching the middleware's original behaviour.
    pub(crate) max_turn_tokens: Option<u64>,
    /// Classifies a model-call error as a provider context-window overflow,
    /// consulted by [`ModelMiddleware::wrap_model`] for the
    /// overflow → compact → retry recovery path.
    pub(crate) overflow_classifier: crate::summarization::OverflowClassifier,
    /// Optional hook consulted before every compaction (proactive or
    /// overflow-triggered) that can decline it or substitute a summary. See
    /// [`crate::summarization::CompactionDecision`].
    pub(crate) before_compaction: Option<BeforeCompactionHook>,
    /// Per-run compaction state: the fold each in-flight run has made and the
    /// live transcript its last `before_model` saw. Keyed by the context's
    /// process-unique [`RunContext::instance_id`] (a `RunId` is a caller label
    /// two concurrent runs may share), so invocations sharing this middleware
    /// never read each other's fold, and dropped in `after_agent`. See [`RunCompaction`].
    pub(crate) runs: Mutex<std::collections::HashMap<u64, RunCompaction>>,
    /// Role the summary is written with. See
    /// [`crate::summarization::SummaryPlacement`].
    pub(crate) placement: crate::summarization::SummaryPlacement,
    /// Ineffective compactions in a row that engage the anti-thrash guard.
    pub(crate) thrash_strikes: u32,
    /// Model calls the guard suppresses summarization for once engaged.
    pub(crate) thrash_cooldown_calls: u32,
    /// When set, the verbatim tail is this many recent tokens instead of
    /// the policy's `keep_last` messages (see
    /// [`crate::summarization::SummarizationPolicy::plan_recent_tokens`]).
    pub(crate) keep_recent_tokens: Option<u64>,
    /// Compaction attempts one model call may make when the provider reports
    /// an overflow. See [`DEFAULT_MAX_OVERFLOW_ATTEMPTS`].
    pub(crate) max_overflow_attempts: u32,
    /// Which successful-response signals count as an overflow. See
    /// [`crate::summarization::ResponseOverflowDetection`].
    pub(crate) response_overflow: crate::summarization::ResponseOverflowDetection,
    /// Byte cap for the truncate-oversized-tool-results route; `None` (the
    /// default) never truncates. See
    /// [`ContextCompressionMiddleware::with_tool_result_truncation`].
    pub(crate) tool_result_truncation: Option<usize>,
    /// Whether a cut inside a turn gives the turn's prefix its own summary
    /// request. See
    /// [`ContextCompressionMiddleware::with_split_turn_prefix`].
    pub(crate) split_turn_prefix: bool,
    /// Derives the `<read-files>` / `<modified-files>` lists appended to each
    /// compaction summary; `None` appends nothing. See
    /// [`crate::summarization::FileOpExtractor`].
    pub(crate) file_ops: Option<std::sync::Arc<dyn crate::summarization::FileOpExtractor>>,
}

/// Default number of compaction attempts one model call may make after the
/// provider reports a context overflow. Each attempt must shrink the request,
/// so the budget bounds cost, not correctness.
pub const DEFAULT_MAX_OVERFLOW_ATTEMPTS: u32 = 3;

/// Default number of ineffective compactions in a row (the next real prompt
/// still at or above the trigger) that engage the anti-thrash guard.
pub const DEFAULT_THRASH_STRIKES: u32 = 2;

/// Default number of model calls the anti-thrash guard suppresses
/// summarization for once engaged; deterministic trim runs instead.
pub const DEFAULT_THRASH_COOLDOWN_CALLS: u32 = 10;

/// Most runs [`ContextCompressionMiddleware`] tracks at once. A run whose
/// `after_agent` never fires (an aborted invocation) would otherwise stay in
/// the map forever on a long-lived shared harness; past this many, the least
/// recently used run is evicted, which only costs it a re-compaction.
pub(crate) const MAX_TRACKED_COMPACTION_RUNS: usize = 256;

/// One run's compaction state inside [`ContextCompressionMiddleware`].
#[derive(Clone, Debug, Default)]
pub(crate) struct RunCompaction {
    /// The compaction this run already performed, re-applied to every later
    /// request. See [`CompactionFold`].
    pub(crate) fold: Option<CompactionFold>,
    /// Chained fingerprints of the live (pre-fold) non-system transcript this
    /// run's last `before_model` saw, so the overflow path can extend the fold
    /// in live-transcript coordinates.
    pub(crate) live_chain: Vec<u64>,
    /// The most recent summary this run produced, threaded into its next
    /// compaction's [`crate::summarization::SummaryRequest::previous_summary`]
    /// so an iterative [`Summarizer`] refines rather than restarts, when no
    /// fold carries it (a host that spliced the summary into its transcript).
    pub(crate) last_summary: Option<String>,
    /// A summary found in the host transcript that a subsequent fold replaces.
    pub(crate) host_applied_summary: Option<tinyinference_llm::message::Message>,
    /// Once the host has persisted a compressed transcript, subsequent
    /// boundaries are in that shortened transcript's coordinates.
    pub(crate) boundary_unaligned: bool,
    /// Set once a truncate route has run for this run: every later request has
    /// its oversized tool results cut again (a pure, idempotent rewrite), so
    /// the prompt prefix stays byte-stable and the measured size stays valid.
    pub(crate) truncating: bool,
    /// Monotonic touch stamp for least-recently-used eviction.
    pub(crate) touched: u64,
    /// Usage-based trigger and anti-thrash state. See [`CompactionPressure`].
    pub(crate) pressure: CompactionPressure,
}

/// One run's measurement state for [`ContextCompressionMiddleware`]'s trigger
/// and anti-thrash guard.
///
/// The trigger prefers what the provider actually measured: the previous
/// call's reported prompt tokens, plus an estimate of only the messages
/// appended since (and of any growth in the tool declarations). The pure
/// chars-based estimate is the fallback when no usage was reported.
///
/// The guard judges a compaction by the next *real* prompt size: still at or
/// above the trigger is a strike, and enough strikes in a row suppress
/// summarization for a cooldown during which the request is trimmed
/// deterministically instead of paying for summaries that do not help.
#[derive(Clone, Debug, Default)]
pub(crate) struct CompactionPressure {
    /// Message count and tool-schema token estimate of the last request this
    /// middleware let through, waiting for that call's usage.
    pub(crate) pending: Option<(usize, u64, u64)>,
    /// Provider-reported prompt tokens of the last answered call, with the
    /// message count and schema tokens of the request that produced it.
    pub(crate) measured: Option<MeasuredPrompt>,
    /// Set by a compaction; the next reported usage decides whether it helped.
    pub(crate) awaiting_verdict: bool,
    /// Ineffective compactions in a row.
    pub(crate) strikes: u32,
    /// Model calls left in the current suppression window.
    pub(crate) suppressed_for: u32,
}

/// A provider-measured prompt size and the request shape it was measured on.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MeasuredPrompt {
    /// Provider-reported input tokens of the call.
    pub(crate) prompt_tokens: u64,
    /// Messages the request carried (as this middleware left it).
    pub(crate) messages: usize,
    /// Estimated tokens of the tool declarations it carried.
    pub(crate) schema_tokens: u64,
    /// Chained fingerprint of those messages, so a later request is only
    /// treated as extending this one when its prefix is identical.
    pub(crate) fingerprint: u64,
}

/// A compaction this middleware already performed, remembered so it is
/// re-applied rather than recomputed.
///
/// The agent loop rebuilds every request from its own working transcript, and
/// `before_model` only rewrites that outgoing copy: the loop never sees the
/// summary. Without this record every call after the first compaction found
/// the full history over the threshold again, so it compacted on every turn
/// and handed the summarizer the whole, ever-growing history each time
/// (101 summarizer calls in one 300-call SWE task, ~85% of its cost, with the
/// agent seeing only the summary and a handful of recent messages).
#[derive(Clone, Debug)]
pub(crate) struct CompactionFold {
    /// How many leading non-system messages of the live transcript the
    /// summary stands in for.
    pub(crate) folded: usize,
    /// Chained fingerprint of those `folded` messages. A transcript whose
    /// prefix no longer matches (rewritten or replaced history) drops the
    /// fold instead of splicing a summary over the wrong messages.
    pub(crate) fingerprint: u64,
    /// The summary message spliced in place of the folded messages.
    pub(crate) summary: tinyinference_llm::message::Message,
    /// The user message the compaction kept verbatim out of the folded range
    /// (see [`crate::summarization::SummarizationPolicy::pin_turn_user_message`]).
    /// Re-applying the fold splices it in right after the summary.
    pub(crate) pinned: Option<PinnedTurnMessage>,
    /// Summary previously spliced by the host and incorporated into this one.
    /// Remove it when rebuilding requests from the host's unchanged transcript.
    pub(crate) replaces: Option<tinyinference_llm::message::Message>,
}

/// A user message a compaction pinned: kept verbatim although it lies inside
/// the folded range.
#[derive(Clone, Debug)]
pub(crate) struct PinnedTurnMessage {
    /// Its position in the live (pre-fold) non-system transcript. Always below
    /// [`CompactionFold::folded`].
    pub(crate) live_index: usize,
    /// The message as it is sent: verbatim, or size-capped with a marker.
    pub(crate) message: tinyinference_llm::message::Message,
}

// ── MicrocompactMiddleware ────────────────────────────────────────────────────

/// Middleware that clears the bodies of older **tool-result** messages while
/// keeping the `keep_recent` most recent ones verbatim.
///
/// In `before_model` it walks `request.messages`, finds the tool-result
/// messages, and — once there are more than `keep_recent` of them — replaces the
/// content of every tool result *except* the newest `keep_recent` with a fixed
/// [`placeholder`](Self::placeholder) string, preserving each message's
/// `tool_call_id`. Non-tool messages (system/user/assistant) are never touched
/// and no chat turn is dropped, so this bounds the cost of a long, tool-heavy
/// thread without the semantic loss of summarization.
///
/// This is the "micro-compaction" companion to
/// [`ContextCompressionMiddleware`]: the latter summarizes *older chat history*
/// when the transcript nears the context window, whereas this one only ever
/// blanks *stale tool payloads* that the model no longer needs verbatim. The two
/// compose cleanly.
///
/// The operation is **idempotent**: a body already equal to the placeholder is
/// left as-is, so repeated `before_model` passes converge. When there are at
/// most `keep_recent` tool results the middleware is a complete no-op.
///
/// The placeholder text is caller-supplied (via
/// [`MicrocompactMiddleware::new`]) so host applications can keep their own
/// model-facing wording stable. Event emission is **opt-in** (default off, see
/// [`MicrocompactMiddleware::with_events`]): when enabled and at least one body
/// is cleared, an
/// [`AgentEvent::Compressed`][crate::events::AgentEvent::Compressed]
/// event carrying the before/after token estimate is emitted; when disabled the
/// middleware mutates the request silently.
///
/// # Prompt-cache stability ([`token_budget`](Self::with_token_budget))
///
/// Blanking a tool body that was sent *verbatim* on an earlier iteration mutates
/// an already-transmitted prefix position, which invalidates the provider's
/// KV-cache from that point on. Because "keep the newest `keep_recent`" is a
/// *moving* boundary, the default (ungated) middleware rewrites one more
/// tool-result full→placeholder on essentially every model call once a run has
/// more than `keep_recent` tool results — churning the cache prefix even when
/// the whole transcript still fits the model's context window (paying an
/// uncached re-read to save tokens the model had room for).
///
/// [`with_token_budget`](Self::with_token_budget) gates the blanking on the
/// transcript's estimated token count: below the budget the middleware is a
/// no-op, so requests stay **append-only and fully cache-eligible**; only once
/// the (un-blanked) transcript exceeds the budget does it start reclaiming
/// tokens. The gate reads the pre-blank token estimate, which grows
/// monotonically, so it never oscillates between blanked and full. Left unset
/// (`None`, the default) the middleware behaves exactly as before.
pub struct MicrocompactMiddleware {
    pub(crate) label: &'static str,
    pub(crate) keep_recent: usize,
    pub(crate) placeholder: String,
    pub(crate) emit_events: bool,
    /// Optional estimated-token floor below which blanking is skipped so the
    /// cache prefix stays stable. `None` = always blank once past `keep_recent`
    /// (legacy behaviour). See [`MicrocompactMiddleware::with_token_budget`].
    pub(crate) token_budget: Option<u64>,
}

// ── PromptCacheGuardMiddleware ────────────────────────────────────────────────

/// Default cap on the number of [`CacheLayoutEvent`]s a
/// [`PromptCacheGuardMiddleware`] retains before evicting the oldest.
pub const DEFAULT_CACHE_GUARD_EVENT_CAP: usize = 1024;

/// Middleware that watches the prompt cache layout for accidental prefix
/// invalidations.
///
/// In `before_model` it computes the request's
/// [`crate::cache::PromptCacheLayout`]. If a layout from a previous
/// call was stored and the cacheable segment prefix changed, it records a
/// [`CacheLayoutEvent`] (retrievable via
/// [`PromptCacheGuardMiddleware::layout_events`]) so KV-cache regressions are
/// observable. This demonstrates provider prompt/KV-cache prefix protection.
pub struct PromptCacheGuardMiddleware {
    /// Label reported in `MiddlewareStarted`/`MiddlewareCompleted` events.
    pub(crate) label: &'static str,
    /// The previous pass's layout, tagged with the run it was observed in.
    ///
    /// The run id is load-bearing. A KV-cache prefix is only meaningful
    /// *within* one conversation, so comparing the last request of one run
    /// against the first request of the next compares two unrelated
    /// transcripts and reports an invalidation that never happened. Rewriting
    /// the non-cacheable history within a run also leaves a canonical stable
    /// prefix intact, so the guard ignores it while adopting the new layout.
    /// Layouts with no mapped message boundary compare the full message
    /// stream under the byte-prefix rule. A single
    /// guard instance is routinely shared across runs — a sub-agent's
    /// middleware stack is built once and its agent invoked many times — so
    /// this is the common case, not an edge case. It went unnoticed while
    /// stability was compared by segment id alone, because any two requests
    /// carrying the same segment ids compared equal regardless of content.
    pub(crate) previous: Mutex<Option<(RunId, crate::cache::PromptCacheLayout)>>,
    /// Recorded layout-change events, oldest first, capped at `max_events`.
    pub(crate) events: Mutex<VecDeque<CacheLayoutEvent>>,
    /// Eviction cap for `events`.
    pub(crate) max_events: usize,
    /// Per-conversation prompt-cache miss accounting, fed from each
    /// response's usage. See [`crate::cache::PromptCacheTracker`].
    pub(crate) cache_misses: Mutex<crate::cache::PromptCacheTracker>,
    /// Latest rewritten-prefix epoch for each durable conversation thread.
    /// Fresh run contexts start at epoch zero, so this preserves the cache key
    /// after a compacted transcript is carried into a later run.
    pub(crate) thread_epochs: Mutex<std::collections::HashMap<crate::ids::ThreadId, u64>>,
}

// ── UsageAccountingMiddleware ─────────────────────────────────────────────────

/// Middleware that folds each model response's usage into a running total.
///
/// In `after_model` it records `response.usage` into an internal
/// [`UsageTotals`]. The accumulated totals are available via
/// [`UsageAccountingMiddleware::totals`] for cost reporting and tests.
pub struct UsageAccountingMiddleware {
    pub(crate) label: &'static str,
    pub(crate) totals: Mutex<UsageTotals>,
}
