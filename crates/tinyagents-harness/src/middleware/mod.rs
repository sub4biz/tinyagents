//! Middleware stack.
//!
//! In the recursive harness, middleware is the layer that wraps
//! *every* level of the recursion identically: because a sub-agent or sub-graph
//! is just another agent loop, the same before/after hooks bracket the parent
//! run and each nested model/tool/agent call beneath it. That uniform wrapping
//! is what lets cross-cutting concerns — tracing, usage/cost roll-up, guardrails
//! — compose consistently as models call models and graphs run graphs.
//!
//! Owns the before/after hooks that wrap agent, model, and tool execution.
//! Cross-cutting behavior such as tracing, guardrails, message trimming,
//! prompt-cache protection, and usage accounting lives here as [`Middleware`]
//! implementations composed through a [`MiddlewareStack`].
//!
//! # Layout
//!
//! - `types` holds every public type (the [`Middleware`] trait, [`AgentRun`],
//!   [`MiddlewareStack`], and the built-in middleware).
//! - This file holds the impls: trait default bodies live with the trait in
//!   `types.rs`; here are the [`AgentRun`] helpers, the stack runner, and the
//!   built-in `Middleware` implementations.
//!
//! # Onion ordering
//!
//! `before_*` hooks run in registration order and `after_*` hooks run in
//! reverse, so the first-registered middleware is the outermost layer. The
//! first hook that errors short-circuits the stack: every middleware's
//! [`Middleware::on_error`] runs, then the original error is returned.

mod types;

pub use types::*;

pub mod library;
pub use library::*;

use std::sync::Arc;

use crate::context::{MiddlewareControl, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::AgentEvent;
use crate::ids::CallId;
use tinyinference_llm::model::{ModelDelta, ModelRequest, ModelResponse};
use tinyinference_llm::tool::{ToolCall, ToolDelta};
use tinytools::ToolResult;

/// Runs one per-middleware **control-outcome** hook across the whole stack,
/// bracketing each *actually invoked* call with
/// `MiddlewareStarted`/`MiddlewareCompleted` events, fanning `on_error` out to
/// every middleware on the first failure, and resolving the phase's
/// [`MiddlewareControl`] per the precedence rule documented on
/// [`Middleware::is_observer`]: the first non-[`MiddlewareControl::Continue`]
/// outcome wins; every hook after it is skipped unless
/// [`Middleware::is_observer`] returns `true` for it, in which case it still
/// runs (for observation) but its own control outcome is discarded. The
/// winning control (if any) is installed via
/// [`RunContext::request_control`], exactly as if a hook had called it
/// directly — this macro is the single place that bridges "hook returned a
/// control" and "hook called `request_control`" into one mechanism.
///
/// Factored as a macro (not an async helper) for the same reason as before
/// control outcomes existed: each hook takes different arguments and borrows
/// `ctx` mutably across its `await`, which a closure-based helper cannot
/// express without heap-boxing every call. `$iter` selects registration order
/// (`.iter()`) or reverse order (`.iter().rev()`); `$call` is the (un-awaited)
/// `_control` hook invocation on `$mw`.
macro_rules! run_stack_hook {
    ($self:ident, $ctx:ident, $iter:expr, |$mw:ident| $call:expr) => {{
        let mut winning: Option<MiddlewareControl> = None;
        for $mw in $iter {
            if winning.is_some() && !$mw.is_observer() {
                continue;
            }
            let name = $mw.name().to_string();
            $ctx.emit(AgentEvent::MiddlewareStarted {
                name: name.clone(),
                call_id: None,
            });
            let result = $call.await;
            $ctx.emit(AgentEvent::MiddlewareCompleted {
                name: name.clone(),
                call_id: None,
            });
            match result {
                Ok(control) => {
                    if winning.is_none() && !matches!(control, MiddlewareControl::Continue) {
                        winning = Some(control);
                    }
                }
                Err(e) => {
                    $ctx.emit(AgentEvent::MiddlewareFailed {
                        name,
                        error: e.to_string(),
                    });
                    $self.fan_out_on_error($ctx, &e).await;
                    return Err(e);
                }
            }
        }
        if let Some(control) = winning {
            $ctx.request_control(control);
        }
        Ok(())
    }};
}

// ── AgentRun ────────────────────────────────────────────────────────────────

impl AgentRun {
    /// Creates an empty agent-run record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the final response text, if the run produced a final response.
    pub fn text(&self) -> Option<String> {
        self.final_response.as_ref().map(|r| r.text())
    }

    /// Deserializes [`Self::structured`] into `T`, when the run produced a
    /// structured output.
    ///
    /// A typed convenience over `run.structured`, mirroring Pydantic AI's
    /// `result.output` (A3). Returns
    /// [`TinyAgentsError::StructuredOutput`][crate::error::TinyAgentsError::StructuredOutput]
    /// when the run produced no structured value, or when the value does not
    /// deserialize into `T`.
    pub fn structured_as<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        let value = self.structured.clone().ok_or_else(|| {
            TinyAgentsError::StructuredOutput("run produced no structured output".to_string())
        })?;
        serde_json::from_value(value).map_err(|error| {
            TinyAgentsError::StructuredOutput(format!("deserialization failed: {error}"))
        })
    }
}

// ── MiddlewareStack ───────────────────────────────────────────────────────────

impl<State: Send + Sync, Ctx: Send + Sync> Default for MiddlewareStack<State, Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> MiddlewareStack<State, Ctx> {
    /// Creates an empty middleware stack.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
            agent_middlewares: Vec::new(),
            model_middlewares: Vec::new(),
            tool_middlewares: Vec::new(),
        }
    }

    /// Appends a lifecycle [`Middleware`] to the stack. Registration order is the
    /// onion order: the first pushed middleware is the outermost layer.
    pub fn push(&mut self, middleware: Arc<dyn Middleware<State, Ctx>>) {
        self.middlewares.push(middleware);
    }

    /// Appends an around-agent middleware. The first registered layer is the
    /// outermost and the real agent loop is the innermost.
    pub fn push_agent_middleware(&mut self, middleware: Arc<dyn AgentMiddleware<State, Ctx>>) {
        self.agent_middlewares.push(middleware);
    }

    /// Appends a [`ModelMiddleware`] (around-model wrap hook). Registration order
    /// is the onion order: the first pushed wrap middleware is the **outermost**
    /// layer and the real model call is the innermost.
    pub fn push_model_middleware(&mut self, middleware: Arc<dyn ModelMiddleware<State, Ctx>>) {
        self.model_middlewares.push(middleware);
    }

    /// Appends a [`ToolMiddleware`] (around-tool wrap hook). Registration order
    /// is the onion order: the first pushed wrap middleware is the **outermost**
    /// layer and the real tool call is the innermost.
    pub fn push_tool_middleware(&mut self, middleware: Arc<dyn ToolMiddleware<State, Ctx>>) {
        self.tool_middlewares.push(middleware);
    }

    /// Returns the number of registered [`ModelMiddleware`] wrap hooks.
    pub fn model_middleware_len(&self) -> usize {
        self.model_middlewares.len()
    }

    /// Returns `true` when a registered [`ModelMiddleware`] already retries
    /// the model call itself (see [`ModelMiddleware::overrides_retry`]).
    ///
    /// The agent loop's base call uses this to skip its own
    /// [`crate::runtime::RunPolicy::retry`] loop, so `RetryMiddleware` and the
    /// loop's built-in retry do not multiply attempts together (I-7).
    pub fn has_retry_override(&self) -> bool {
        self.model_middlewares.iter().any(|mw| mw.overrides_retry())
    }

    /// Returns `true` when any registered lifecycle [`Middleware`] asks to
    /// stop after the turn currently completing (see
    /// [`Middleware::should_stop_after_turn`]).
    ///
    /// Called by the agent loop at the turn boundary — after tool execution,
    /// before the loop would otherwise continue — so an aggregate stop
    /// condition (a tally across the whole turn's tool results, not any
    /// single call) can end the run as cleanly as
    /// [`crate::context::MiddlewareControl::JumpTo`]`(`[`crate::context::LoopTarget::End`]`)`.
    pub fn any_should_stop_after_turn(&self, ctx: &RunContext<Ctx>, run: &AgentRun) -> bool {
        self.middlewares
            .iter()
            .any(|mw| mw.should_stop_after_turn(ctx, run))
    }

    /// Returns the number of registered around-agent middleware layers.
    pub fn agent_middleware_len(&self) -> usize {
        self.agent_middlewares.len()
    }

    /// Returns the number of registered [`ToolMiddleware`] wrap hooks.
    pub fn tool_middleware_len(&self) -> usize {
        self.tool_middlewares.len()
    }

    /// Whether every registered [`ToolMiddleware`] tolerates overlapping
    /// invocations ([`ToolMiddleware::concurrent_safe`]); vacuously `true`
    /// when none is registered.
    pub fn tool_middleware_concurrent_safe(&self) -> bool {
        self.tool_middlewares.iter().all(|mw| mw.concurrent_safe())
    }

    /// Returns the number of registered middleware.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Returns `true` if no middleware are registered.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Fans `on_error` out to every middleware, ignoring their results so the
    /// original error is never masked. No start/completed events are emitted on
    /// this internal recovery path.
    ///
    /// Marks the context so a driver that also handles the propagated error
    /// (the agent loop does) skips its own dispatch: one failure must deliver
    /// exactly one `on_error` per middleware.
    async fn fan_out_on_error(&self, ctx: &mut RunContext<Ctx>, error: &TinyAgentsError) {
        ctx.mark_on_error_dispatched();
        for mw in self.middlewares.iter() {
            let _ = mw.on_error(ctx, error).await;
        }
    }

    /// Runs every middleware's [`Middleware::before_agent`] in registration
    /// order.
    pub async fn run_before_agent(&self, ctx: &mut RunContext<Ctx>, state: &State) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter(), |mw| mw
            .before_agent_control(ctx, state))
    }

    /// Runs every middleware's [`Middleware::after_agent`] in reverse
    /// registration order.
    pub async fn run_after_agent(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        run: &mut AgentRun,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter().rev(), |mw| mw
            .after_agent_control(ctx, state, run))
    }

    /// Runs every middleware's [`Middleware::before_model`] in registration
    /// order, threading the mutable request through each.
    pub async fn run_before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter(), |mw| mw
            .before_model_control(ctx, state, request))
    }

    /// Runs `before_model` while refreshing the selected profile before each
    /// hook. Earlier hooks may change the model or required capabilities.
    pub async fn run_before_model_with_profile<P, F>(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: &mut ModelRequest,
        provider: &P,
        resolve_profile: F,
    ) -> Result<()>
    where
        F: for<'a> Fn(
            &'a P,
            &'a RunContext<Ctx>,
            &'a ModelRequest,
        ) -> futures::future::BoxFuture<
            'a,
            Result<Option<tinyinference_llm::model::ModelProfile>>,
        >,
    {
        let mut winning: Option<MiddlewareControl> = None;
        for mw in &self.middlewares {
            if winning.is_some() && !mw.is_observer() {
                continue;
            }
            ctx.model_profile = resolve_profile(provider, ctx, request).await?;
            let name = mw.name().to_string();
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: name.clone(),
                call_id: None,
            });
            let result = mw.before_model_control(ctx, state, request).await;
            ctx.emit(AgentEvent::MiddlewareCompleted {
                name: name.clone(),
                call_id: None,
            });
            match result {
                Ok(control) => {
                    if winning.is_none() && !matches!(control, MiddlewareControl::Continue) {
                        winning = Some(control);
                    }
                }
                Err(error) => {
                    ctx.emit(AgentEvent::MiddlewareFailed {
                        name,
                        error: error.to_string(),
                    });
                    self.fan_out_on_error(ctx, &error).await;
                    return Err(error);
                }
            }
        }
        if let Some(control) = winning {
            ctx.request_control(control);
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::on_model_delta`] in registration
    /// order for one streamed delta.
    ///
    /// Unlike the other stack runners, the per-delta hook is deliberately *not*
    /// bracketed by `MiddlewareStarted`/`MiddlewareCompleted` events. This runs
    /// on the streaming hot path — potentially hundreds of times per second per
    /// middleware — and emitting two events (each cloning `mw.name()` and
    /// acquiring the recorder mutex) per middleware per token dominated the
    /// stream loop's cost for zero observability value. Callers that need to
    /// observe delta-level middleware activity should instrument the hook
    /// itself.
    pub async fn run_on_model_delta(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        delta: &mut ModelDelta,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            if let Err(e) = mw.on_model_delta(ctx, state, delta).await {
                self.fan_out_on_error(ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::after_model`] in reverse
    /// registration order, threading the mutable response through each.
    pub async fn run_after_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter().rev(), |mw| mw
            .after_model_control(ctx, state, response))
    }

    /// Runs every middleware's [`Middleware::before_tool`] in registration
    /// order, threading the mutable tool call through each.
    ///
    /// Unlike the other stack runners this one recognises the per-call
    /// signals of A2/A3 — `ApprovalRequired`, `CallDeferred`, `ToolFailed`,
    /// `ModelRetry` — as *decisions about the call* rather than hook
    /// failures: they propagate to admission (which defers or answers the
    /// call) without a `MiddlewareFailed` event or an `on_error` fan-out.
    /// An `ApprovalRequired` for a call the resume path already approved
    /// ([`RunContext::is_call_approved`]) is treated as `Continue`, so a
    /// gate that cannot see the approval does not re-defer the call.
    pub async fn run_before_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        let mut winning: Option<MiddlewareControl> = None;
        for mw in self.middlewares.iter() {
            if winning.is_some() && !mw.is_observer() {
                continue;
            }
            let name = mw.name().to_string();
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: name.clone(),
                call_id: None,
            });
            let result = mw.before_tool_control(ctx, state, call).await;
            ctx.emit(AgentEvent::MiddlewareCompleted {
                name: name.clone(),
                call_id: None,
            });
            match result {
                Ok(control) => {
                    if winning.is_none() && !matches!(control, MiddlewareControl::Continue) {
                        winning = Some(control);
                    }
                }
                Err(TinyAgentsError::ApprovalRequired { .. }) if ctx.is_call_approved(&call.id) => {
                }
                Err(
                    signal @ (TinyAgentsError::ApprovalRequired { .. }
                    | TinyAgentsError::CallDeferred { .. }
                    | TinyAgentsError::ToolFailed(_)
                    | TinyAgentsError::ModelRetry(_)),
                ) => return Err(signal),
                Err(e) => {
                    ctx.emit(AgentEvent::MiddlewareFailed {
                        name,
                        error: e.to_string(),
                    });
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
            }
        }
        if let Some(control) = winning {
            ctx.request_control(control);
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::on_tool_delta`] in registration
    /// order for one streamed tool-progress delta.
    ///
    /// The agent loop calls this for every [`AgentEvent::ToolProgressDetail`] a
    /// running tool produced through
    /// [`tinytools::ToolRunContext::report_progress`]. The hook needs
    /// `&mut RunContext`, which is lent to the tool for the duration of the
    /// call, so the loop replays a call's deltas in order right after the call
    /// settles and before its terminal event — see the `crate::tool` progress
    /// gate for the full ordering contract.
    ///
    /// Like [`Self::run_on_model_delta`], and for the same reason (M-12):
    /// this is **not** bracketed by `MiddlewareStarted`/`MiddlewareCompleted`
    /// events. It used to be the one delta hook still routed through
    /// `run_stack_hook!`, so a stack of `N` middlewares produced `2*N`
    /// bookkeeping events per streamed tool-progress delta — noise a
    /// `ModelCompleted`-based exporter had to filter, for a hook that (unlike
    /// `before_tool`/`after_tool`) can fire many times per call. Both delta
    /// hooks now agree: bracket every non-delta hook, skip both delta hooks.
    /// A caller that needs to observe delta-level middleware activity should
    /// instrument the hook implementation itself.
    pub async fn run_on_tool_delta(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        delta: &mut ToolDelta,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            if let Err(e) = mw.on_tool_delta(ctx, state, delta).await {
                self.fan_out_on_error(ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::after_tool`] in reverse
    /// registration order, threading the mutable tool result through each.
    pub async fn run_after_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter().rev(), |mw| mw
            .after_tool_control(ctx, state, invocation, result))
    }

    /// Runs every middleware's [`Middleware::check_nested_tool`] in
    /// registration order; the first refusal wins.
    pub async fn run_check_nested_tool(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: &ToolCall,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            mw.check_nested_tool(ctx, state, call).await?;
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::observe_nested_result`] in
    /// registration order.
    pub async fn run_observe_nested_result(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: &ToolCall,
        result: &ToolResult,
    ) {
        for mw in self.middlewares.iter() {
            mw.observe_nested_result(ctx, state, call, result).await;
        }
    }

    /// Runs every middleware's [`Middleware::on_error`] in registration order,
    /// bracketing each with start/completed events. Inner errors are ignored so
    /// the originating error is never masked; this method always returns `Ok`.
    pub async fn run_on_error(
        &self,
        ctx: &mut RunContext<Ctx>,
        error: &TinyAgentsError,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
                call_id: None,
            });
            let _ = mw.on_error(ctx, error).await;
            ctx.emit(AgentEvent::MiddlewareCompleted {
                name: mw.name().to_string(),
                call_id: None,
            });
        }
        Ok(())
    }

    /// Runs the registered [`ModelMiddleware`] wrap hooks as a nested onion
    /// around `base` (the real model call) and returns the resolved
    /// [`MiddlewareModelOutcome`].
    ///
    /// The first-registered wrap middleware is the outermost layer; `base` is
    /// the innermost. With no wrap middleware registered this simply runs `base`
    /// and wraps its response. Each layer is bracketed by
    /// `MiddlewareStarted`/`MiddlewareCompleted` events.
    pub async fn run_wrapped_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        base: &dyn ModelBaseCall<State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        let handler = ModelHandler {
            remaining: &self.model_middlewares,
            base,
        };
        handler.run(ctx, state, request).await
    }

    /// Runs the around-agent middleware onion around the complete agent loop.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_wrapped_agent(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        input: Vec<tinyinference_llm::message::Message>,
        streaming: bool,
        run: &mut AgentRun,
        status: &mut crate::events::HarnessRunStatus,
        base: &dyn AgentBaseCall<State, Ctx>,
    ) -> Result<()> {
        AgentHandler {
            remaining: &self.agent_middlewares,
            base,
            status,
        }
        .run(ctx, state, AgentRequest::new(input, streaming), run)
        .await
    }

    /// Runs the registered [`ToolMiddleware`] wrap hooks as a nested onion around
    /// `base` (the real tool call) and returns the resolved
    /// [`MiddlewareToolOutcome`].
    ///
    /// The tool-wrap counterpart of [`Self::run_wrapped_model`].
    pub async fn run_wrapped_tool(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        base: &dyn ToolBaseCall<State, Ctx>,
    ) -> Result<MiddlewareToolOutcome> {
        let handler = ToolHandler {
            remaining: &self.tool_middlewares,
            base,
        };
        handler.run(ctx, state, call).await
    }
}

// ── Wrap onion handlers ───────────────────────────────────────────────────────

impl<State: Send + Sync, Ctx: Send + Sync> AgentHandler<'_, State, Ctx> {
    /// Advances the around-agent onion by one layer.
    pub async fn run(
        self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: AgentRequest,
        run: &mut AgentRun,
    ) -> Result<()> {
        let AgentHandler {
            remaining,
            base,
            status,
        } = self;
        match remaining.split_first() {
            Some((head, tail)) => {
                let next = AgentHandler {
                    remaining: tail,
                    base,
                    status,
                };
                let name = head.name().to_string();
                ctx.emit(AgentEvent::MiddlewareStarted {
                    name: name.clone(),
                    call_id: None,
                });
                let outcome = head.wrap_agent(ctx, state, request, run, next).await;
                ctx.emit(AgentEvent::MiddlewareCompleted {
                    name,
                    call_id: None,
                });
                outcome
            }
            None => base.call(ctx, state, request, run, status).await,
        }
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> ModelHandler<'_, State, Ctx> {
    /// Advances the model-wrap onion one layer: invokes the next
    /// [`ModelMiddleware`] (bracketed by start/completed events), or the base
    /// model call when no wrap middleware remain.
    ///
    /// Borrows `&self`, so a wrap middleware may call `run` zero times
    /// (short-circuit), once (proceed), or many times (retry).
    pub async fn run(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
    ) -> Result<MiddlewareModelOutcome> {
        match self.remaining.split_first() {
            Some((head, tail)) => {
                let next = ModelHandler {
                    remaining: tail,
                    base: self.base,
                };
                let name = head.name().to_string();
                ctx.emit(AgentEvent::MiddlewareStarted {
                    name: name.clone(),
                    call_id: None,
                });
                // Emit `Completed` whether the wrap layer succeeds or errors, so
                // a failing layer never leaves a dangling `Started` in the event
                // stream (the onion's balance invariant).
                let outcome = head.wrap_model(ctx, state, request, next).await;
                ctx.emit(AgentEvent::MiddlewareCompleted {
                    name,
                    call_id: None,
                });
                outcome
            }
            None => Ok(MiddlewareModelOutcome::Response(
                self.base.call(ctx, state, request).await?,
            )),
        }
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> ToolHandler<'_, State, Ctx> {
    /// Advances the tool-wrap onion one layer. The tool-wrap counterpart of
    /// [`ModelHandler::run`].
    pub async fn run(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: ToolCall,
    ) -> Result<MiddlewareToolOutcome> {
        match self.remaining.split_first() {
            Some((head, tail)) => {
                let next = ToolHandler {
                    remaining: tail,
                    base: self.base,
                };
                let name = head.name().to_string();
                // Tagged with the call so consumers can correlate under
                // concurrency, where events of different calls interleave.
                let call_id = Some(CallId::new(call.id.clone()));
                ctx.emit(AgentEvent::MiddlewareStarted {
                    name: name.clone(),
                    call_id: call_id.clone(),
                });
                // Balance `Started` with `Completed` even when the wrap layer
                // errors (see `ModelHandler::run`).
                let outcome = head.wrap_tool(ctx, state, call, next).await;
                ctx.emit(AgentEvent::MiddlewareCompleted { name, call_id });
                outcome
            }
            None => Ok(MiddlewareToolOutcome::Result(
                self.base.call(ctx, state, call).await?,
            )),
        }
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
