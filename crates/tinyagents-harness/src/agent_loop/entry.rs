//! Public entry points for the default agent loop: `invoke*`/
//! `invoke_streaming*` and the shared `drive` lifecycle wrapper.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

use super::*;

struct AgentLoopBase<'a, State: Send + Sync, Ctx: Send + Sync> {
    harness: &'a AgentHarness<State, Ctx>,
}

impl<State: Send + Sync, Ctx: Send + Sync> AgentBaseCall<State, Ctx>
    for AgentLoopBase<'_, State, Ctx>
{
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        request: crate::middleware::AgentRequest,
        run: &'a mut AgentRun,
        status: &'a mut HarnessRunStatus,
    ) -> BoxAgentFuture<'a> {
        Box::pin(async move {
            ctx.streaming = request.streaming;
            match self.harness.policy.execution {
                crate::runtime::LoopExecution::Graph => match self.harness.loop_driver.clone() {
                    Some(driver) => {
                        driver
                            .drive(
                                self.harness,
                                state,
                                ctx,
                                run,
                                status,
                                request.input,
                                request.streaming,
                            )
                            .await
                    }
                    None => Err(TinyAgentsError::Validation(
                        "RunPolicy::execution is LoopExecution::Graph but no LoopDriver is \
                         installed; call AgentHarness::with_loop_driver first"
                            .to_string(),
                    )),
                },
                crate::runtime::LoopExecution::Direct => {
                    self.harness
                        .run_loop(state, ctx, run, status, request.input, request.streaming)
                        .await
                }
            }
        })
    }
}

/// Owns the accumulating run until the driver reaches a terminal outcome.
///
/// If the driving future is dropped at any await point, this guard observes the
/// real partial run and invokes the host terminal hook exactly once. That is
/// what makes cancellation accounting truthful for both unary and streaming
/// hosted entry points.
struct TerminalRunGuard {
    run: AgentRun,
    observer: Option<crate::context::TerminalObserver>,
}

impl TerminalRunGuard {
    fn new(observer: Option<crate::context::TerminalObserver>) -> Self {
        Self {
            run: AgentRun::new(),
            observer,
        }
    }

    fn complete(mut self, succeeded: bool, error: Option<String>) -> AgentRun {
        if let Some(observer) = self.observer.take() {
            // A cheap summary (M-6), not a clone of the whole run: the
            // observer only ever reads text/usage/executed-tools, and cloning
            // `self.run` here duplicated the entire transcript just to throw
            // it away after the observer call — `mem::take` below is the only
            // place that needs to move the real run out.
            observer(
                crate::context::TerminalRunSummary::from_run(&self.run),
                succeeded,
                error,
            );
        }
        std::mem::take(&mut self.run)
    }
}

impl Drop for TerminalRunGuard {
    fn drop(&mut self) {
        if let Some(observer) = self.observer.take() {
            observer(
                crate::context::TerminalRunSummary::from_run(&self.run),
                false,
                Some("hosted invocation cancelled by caller".to_string()),
            );
        }
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Runs the default agent loop and returns the accumulated [`AgentRun`].
    ///
    /// `state` is shared, read-only application data passed to every model and
    /// tool call. `ctx_data` is moved into the [`RunContext`] for the run.
    /// `config` supplies the run identity and limits, and `input` seeds the
    /// working message transcript.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::LimitExceeded`] when the model- or tool-call
    /// cap is reached, [`TinyAgentsError::Timeout`] when the wall-clock deadline
    /// elapses, [`TinyAgentsError::ModelNotFound`] when no model can be
    /// resolved, [`TinyAgentsError::ToolNotFound`] when the model calls an
    /// unregistered tool, or any error surfaced by a model, tool, middleware,
    /// or structured-output extraction.
    pub async fn invoke(
        &self,
        state: &State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        self.invoke_with_status(state, ctx_data, config, input)
            .await
            .map(|result| result.run)
    }

    /// Runs the default agent loop with a generated default [`RunConfig`].
    ///
    /// Builds `RunConfig::new("run")` and a default `Ctx`. Identifiers are
    /// derived deterministically from the config (no random or time-based ids),
    /// so repeated calls with the same input behave identically.
    pub async fn invoke_default(&self, state: &State, input: Vec<Message>) -> Result<AgentRun>
    where
        Ctx: Default,
    {
        self.invoke(state, Ctx::default(), RunConfig::new("run"), input)
            .await
    }

    /// Runs the default agent loop and returns both the [`AgentRun`] and a
    /// compact [`HarnessRunStatus`] snapshot describing how the run ended.
    ///
    /// This is the underlying entry point used by [`AgentHarness::invoke`]; use
    /// it directly when you also need lifecycle/status information (phase,
    /// counters, timing, error summary). On error the returned status would have
    /// been marked failed, but the error is propagated instead so callers see
    /// the failure; use the event stream for failed-run status.
    pub async fn invoke_with_status(
        &self,
        state: &State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> Result<AgentLoopResult> {
        let ctx = RunContext::new(config, ctx_data);
        self.drive(state, ctx, input, false).await
    }

    /// Runs the default agent loop inside a caller-supplied [`RunContext`],
    /// returning the accumulated [`AgentRun`].
    ///
    /// Use this when you need to control the run's dependencies — for example
    /// to attach your own [`crate::events::EventSink`] (so an external
    /// listener or [`crate::testkit::EventRecorder`] receives every
    /// event), inject a custom [`crate::store::StoreRegistry`], or carry
    /// pre-populated `Ctx` data. The context's [`RunConfig`] supplies the run
    /// identity and limits, exactly as for [`AgentHarness::invoke`].
    ///
    /// # Errors
    ///
    /// Identical to [`AgentHarness::invoke`].
    pub async fn invoke_in_context(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        self.drive(state, ctx, input, false)
            .await
            .map(|result| result.run)
    }

    /// Like [`AgentHarness::invoke_in_context`] but also returns the compact
    /// [`HarnessRunStatus`] snapshot.
    pub async fn invoke_in_context_with_status(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentLoopResult> {
        self.drive(state, ctx, input, false).await
    }

    /// Resumes a run that stopped with [`AgentRun::deferred`] set (A2).
    ///
    /// `messages` is the deferred run's transcript (`run.messages`, which
    /// still ends with the assistant tool-call row whose deferred calls are
    /// unanswered) and `results` resolves every pending call: an
    /// [`crate::tool::ToolApprovalDecision`] runs or denies an approval-gated
    /// call, a [`crate::tool::DeferredCallResult`] injects the host's outcome
    /// for an external one. The loop answers each call — executing approved
    /// ones for real, with the model's or the approver's edited arguments —
    /// and then continues with the next model call exactly as if the batch
    /// had never paused.
    ///
    /// The only state needed to resume is the transcript plus `results`, so
    /// this works across a process restart: persist `run.messages` and
    /// `run.deferred` (both serializable), and call this from any process.
    /// Check [`crate::tool::DeferredToolRequests::remaining`] first —
    /// an incomplete `results` fails with [`TinyAgentsError::Validation`]
    /// naming the unresolved ids before anything runs.
    ///
    /// Equivalent to `invoke_in_context(state, ctx.with_deferred_results(results), messages)`,
    /// preceded by a [`AgentHarness::reconcile_tool_effects`] pass scoped to
    /// exclude the calls `results` is about to answer.
    ///
    /// A call this run's own `execution_deferral` filed (mid-execution
    /// `ApprovalRequired`/`CallDeferred`) is settled in the tool-effect
    /// ledger as [`crate::tool::ToolEffectStatus::Deferred`] the moment it
    /// pauses — not left `started` — so [`AgentHarness::reconcile_tool_effects`]
    /// (which only reconciles rows still `started`) does not treat it as a
    /// crash artifact on its own. The `excluded` set passed here is
    /// defense-in-depth on top of that: even if a call's row is unexpectedly
    /// still `started` (the `Deferred` settle write is best-effort and only
    /// logs on failure), excluding every id `results` answers guarantees this
    /// call never receives a synthesized "interrupted" answer that would
    /// pre-empt `results`'s real one.
    ///
    /// A genuinely crashed **sibling** call in the same batch — one with no
    /// entry in `results` and a ledger row still `started` because the
    /// process died before it could pause or settle — is not excluded, and
    /// is reconciled normally (re-executed or answered "interrupted before
    /// settlement" per its [`tinytools::ToolReplay`] policy) before the loop
    /// resumes.
    ///
    /// Reconciling a genuine crash with no live `results` at all (a host
    /// resuming from durable state after a real process crash, with no
    /// deferral in flight) remains a host's explicit, separate call to
    /// [`AgentHarness::reconcile_tool_effects`] — this method's own
    /// reconcile pass only ever excludes ids `results` names, so it is a
    /// strict addition, never a replacement, for that path.
    pub async fn resume_deferred(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        messages: Vec<Message>,
        results: crate::tool::DeferredToolResults,
    ) -> Result<AgentRun> {
        let mut messages = messages;
        if ctx.tool_effect_ledger.is_some() {
            let run_id = ctx.run_id().as_str().to_string();
            let excluded: std::collections::HashSet<crate::ids::CallId> = results
                .approvals
                .keys()
                .chain(results.calls.keys())
                .cloned()
                .collect();
            self.reconcile_tool_effects(&ctx, &run_id, &mut messages, &excluded)
                .await?;
        }
        self.invoke_in_context(state, ctx.with_deferred_results(results), messages)
            .await
    }

    /// Streaming counterpart of [`AgentHarness::invoke`].
    ///
    /// Behaves exactly like [`AgentHarness::invoke`] except each model call is
    /// driven through [`tinyinference_llm::model::ChatModel::stream`] rather than
    /// [`tinyinference_llm::model::ChatModel::invoke`]: incremental message deltas
    /// are emitted as [`AgentEvent::ModelDelta`] events and threaded through
    /// every middleware's
    /// [`on_model_delta`][crate::middleware::Middleware::on_model_delta]
    /// hook before the chunks are merged back into the final
    /// [`tinyinference_llm::model::ModelResponse`]. Tool execution, limits, retry,
    /// fallback, structured output, and all other lifecycle behavior are
    /// identical to the non-streaming path.
    pub async fn invoke_streaming(
        &self,
        state: &State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        let ctx = RunContext::new(config, ctx_data);
        self.drive(state, ctx, input, true)
            .await
            .map(|result| result.run)
    }

    /// Streaming counterpart of [`AgentHarness::invoke_default`].
    pub async fn invoke_streaming_default(
        &self,
        state: &State,
        input: Vec<Message>,
    ) -> Result<AgentRun>
    where
        Ctx: Default,
    {
        self.invoke_streaming(state, Ctx::default(), RunConfig::new("run"), input)
            .await
    }

    /// Streaming counterpart of [`AgentHarness::invoke_in_context`].
    pub async fn invoke_streaming_in_context(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        self.drive(state, ctx, input, true)
            .await
            .map(|result| result.run)
    }

    /// Streaming counterpart of [`AgentHarness::invoke_in_context_with_status`].
    pub async fn invoke_streaming_in_context_with_status(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentLoopResult> {
        self.drive(state, ctx, input, true).await
    }

    /// Shared driver: runs the loop inside `ctx` and owns lifecycle
    /// bookkeeping (status transitions plus `RunFailed`/`on_error` on error).
    ///
    /// `streaming` selects whether each model call is driven through
    /// [`tinyinference_llm::model::ChatModel::stream`] (firing `on_model_delta`
    /// middleware per delta) or the unary
    /// [`tinyinference_llm::model::ChatModel::invoke`] path.
    /// Runs the loop and returns the accumulated run **and** any error, instead
    /// of discarding the run when the loop fails.
    ///
    /// [`AgentHarness::invoke`] and friends return `Err` on failure, which drops
    /// the partially-populated [`AgentRun`] — every message, tool result, and
    /// usage figure the run produced before it tripped a limit or hit a tool
    /// failure. Use this when that partial work is worth keeping: to inspect
    /// what the agent had done, to repair the transcript, or to resume from it.
    ///
    /// The returned [`PartialRunOutcome::error`] is `None` exactly when the run
    /// succeeded.
    pub async fn invoke_collecting_partial(
        &self,
        state: &State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> PartialRunOutcome {
        let ctx = RunContext::new(config, ctx_data);
        self.drive_collecting(state, ctx, input, false).await
    }

    /// [`AgentHarness::invoke_collecting_partial`] against a caller-supplied
    /// [`RunContext`].
    pub async fn invoke_in_context_collecting_partial(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> PartialRunOutcome {
        self.drive_collecting(state, ctx, input, false).await
    }

    /// Streaming counterpart of [`Self::invoke_in_context_collecting_partial`].
    ///
    /// This preserves the accumulated run on a streaming provider failure so
    /// host-owned terminal sinks can report the actual usage and executed
    /// tools instead of inventing an empty failure record.
    pub async fn invoke_streaming_in_context_collecting_partial(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> PartialRunOutcome {
        self.drive_collecting(state, ctx, input, true).await
    }

    async fn drive(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
        streaming: bool,
    ) -> Result<AgentLoopResult> {
        let outcome = self.drive_collecting(state, ctx, input, streaming).await;
        match outcome.error {
            Some(error) => Err(error),
            None => Ok(AgentLoopResult {
                run: outcome.run,
                status: outcome.status,
            }),
        }
    }

    /// The shared driver both entry shapes delegate to. Never discards the
    /// run, so the failing path can hand the partial transcript back.
    async fn drive_collecting(
        &self,
        state: &State,
        mut ctx: RunContext<Ctx>,
        input: Vec<Message>,
        streaming: bool,
    ) -> PartialRunOutcome {
        let run_id = ctx.config.run_id.clone();
        let thread_id = ctx.config.thread_id.clone();
        // Record the drive mode so tool execution contexts (and the sub-agents
        // they spawn) can match it — child deltas then propagate to the parent
        // stream via the shared sink.
        ctx.streaming = streaming;

        let mut status = HarnessRunStatus::new(run_id.clone(), ComponentId::new("agent_loop"));
        if let Some(thread) = thread_id {
            status = status.with_thread(thread);
        }

        let mut terminal = TerminalRunGuard::new(ctx.terminal_observer.take());

        let base = AgentLoopBase { harness: self };
        match self
            .middleware
            .run_wrapped_agent(
                &mut ctx,
                state,
                input,
                streaming,
                &mut terminal.run,
                &mut status,
                &base,
            )
            .await
        {
            Ok(()) => {
                // A paused run is resumable, not finished: reporting it
                // `completed` is what made "paused for a human" look identical
                // to "the model produced an empty final answer".
                // A deferred run (A2) is resumable for the same reason.
                // The typed outcome is authoritative (middleware may have
                // replaced it after the loop set the legacy fields); fall back
                // to the legacy fields only when no outcome was recorded.
                let paused = match terminal.run.terminal.as_ref() {
                    Some(outcome) => outcome.class == TerminalClass::Suspended,
                    None => terminal.run.paused.is_some() || terminal.run.deferred.is_some(),
                };
                // Only `Success` is a completion and `Suspended` is handled by
                // `paused`; Failure, Timeout and Cancellation outcomes recorded
                // by middleware on an `Ok` return must not read as completed.
                let failed = terminal.run.terminal.as_ref().is_some_and(|outcome| {
                    !matches!(
                        outcome.class,
                        TerminalClass::Success | TerminalClass::Suspended
                    )
                });
                if paused {
                    status.mark_interrupted();
                } else if failed {
                    status.mark_failed("run ended with a non-success terminal outcome".to_string());
                } else {
                    status.mark_completed();
                }
                PartialRunOutcome {
                    run: terminal.complete(
                        !paused && !failed,
                        paused.then(|| "hosted turn paused before completion".to_string()),
                    ),
                    status,
                    error: None,
                }
            }
            Err(error) => {
                // Where the run stood when it failed decides the timeout phase
                // and `provider_started`: an unfinished model call leaves
                // `active_model_call` set, a completed one has bumped the
                // run's call counter.
                // A failure inside the model-call layer is `Provider` only if
                // the provider was actually dispatched; a wrap middleware that
                // rejected the call first never reached it.
                let in_model_call = ctx.active_model_call.is_some() || ctx.model_call_failed();
                let site = if in_model_call && ctx.call_provider_started() {
                    TimeoutPhase::Provider
                } else if in_model_call {
                    TimeoutPhase::BeforeProvider
                } else if terminal.run.model_calls > 0 {
                    TimeoutPhase::AfterTurn
                } else {
                    TimeoutPhase::BeforeProvider
                };
                let last_limit = matches!(error, TinyAgentsError::LimitExceeded(_))
                    .then(|| ctx.take_last_limit())
                    .flatten();
                let mut outcome =
                    TerminalOutcome::from_error(&error, site).with_limit_kind(last_limit);
                if outcome.reason == crate::terminal::TerminalReason::Timeout
                    && outcome.timeout_phase.is_none()
                {
                    // A wall-clock kind filled in after classification.
                    outcome = outcome.with_timeout_phase(site);
                }
                // `site` describes this failure; the run may still have reached
                // the provider on an earlier call.
                // A failed summarizer already received a provider response; keep the
                // match for summarizers that report usage but never marked dispatch
                // (host-defined `Summarizer` impls cannot call `mark_dispatched`).
                outcome.provider_started = ctx.provider_started()
                    || matches!(&error, TinyAgentsError::SummarizationUsage { .. });
                terminal.run.terminal = Some(outcome.clone());
                let record = ctx.emit(AgentEvent::RunFailed {
                    run_id,
                    error: error.to_string(),
                    outcome: Some(outcome),
                });
                status.set_last_event(record.id);
                status.mark_failed(error.to_string());
                // Surface the failure to every middleware. Inner errors are
                // ignored so the originating error is never masked. A failure
                // raised *inside* a lifecycle hook was already fanned out by the
                // stack before it propagated here, so skip it rather than
                // delivering the same failure twice.
                if !ctx.take_on_error_dispatched() {
                    let _ = self.middleware.run_on_error(&mut ctx, &error).await;
                }
                PartialRunOutcome {
                    run: terminal.complete(false, Some(error.to_string())),
                    status,
                    error: Some(error),
                }
            }
        }
    }
}
