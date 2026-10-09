//! Shared per-run state and the four node bodies (`plan`, `model`, `tools`,
//! `settle`) that [`super::compile_loop`] wires into a graph, and that
//! [`super::LoopIter`] steps through directly.
//!
//! See the module doc on [`super`] for the loop's documented scope.

use std::sync::Arc;

use tokio::sync::Mutex;

use tinyagents_harness::agent_loop::phases;
use tinyagents_harness::context::{LoopTarget, MiddlewareControl, RunContext};
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::events::{AgentEvent, HarnessRunStatus};
use tinyagents_harness::ids::{CallId, NodeId};
use tinyagents_harness::middleware::{AgentRun, BoxModelFuture, ModelBaseCall};
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::steering::{SteeringOutcome, apply_pending_steering};
use tinyagents_harness::structured::{StructuredExtractor, StructuredStrategy};

use crate::command::Interrupt;
use crate::{Command, NodeResult, RouteTarget};

use super::types::{LoopState, PendingStructuredPlan, node};

use tinyinference_llm::model::{ModelRequest, ModelResponse, ResponseFormat, ToolChoice};
use tinyinference_llm::tool::{ToolFormat, ToolSchema};

/// Per-run state shared by every node closure [`super::compile_loop`]
/// builds, and by [`super::LoopIter`].
///
/// Captured behind an `Arc` (with interior mutability for the pieces that
/// need `&mut` access) because [`crate::GraphBuilder::add_node`] handlers are
/// `Fn`, not `FnMut`: the graph executor may in principle invoke a node
/// concurrently with itself across forked branches, so every mutable piece
/// here is guarded by its own [`Mutex`]. In this loop's own topology no two
/// nodes ever run concurrently (it is a strictly sequential chain), so the
/// locks are never contended — they exist to satisfy `Send + Sync + 'static`
/// and the `Fn` bound, not to arbitrate real concurrency.
pub struct LoopRuntime<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) harness: Arc<AgentHarness<State, Ctx>>,
    pub(crate) app_state: Arc<State>,
    pub(crate) ctx: Mutex<RunContext<Ctx>>,
    pub(crate) run: Mutex<AgentRun>,
    pub(crate) status: Mutex<HarnessRunStatus>,
    /// Not yet consulted: `model_node`'s `DirectModelBase` always dispatches
    /// through `ChatModel::invoke`, not `ChatModel::stream` (see the module
    /// doc on `super` — streaming is out of scope for this rendition of the
    /// loop). Kept so `LoopRuntime::new`'s signature already matches what a
    /// future streaming node would need.
    #[allow(dead_code)]
    pub(crate) streaming: bool,
}

impl<State: Send + Sync, Ctx: Send + Sync> LoopRuntime<State, Ctx> {
    /// Builds a fresh, owned [`LoopRuntime`] for one run: [`super::LoopIter`]
    /// (which owns `ctx`/`input` for the run's whole lifetime) is the
    /// intended caller. [`super::GraphLoopDriver`] does **not** use this —
    /// see its module doc for why it drives the same node bodies directly
    /// over borrowed `&mut` state instead of through an owned
    /// `LoopRuntime`/`CompiledGraph`.
    pub fn new(
        harness: Arc<AgentHarness<State, Ctx>>,
        app_state: Arc<State>,
        mut ctx: RunContext<Ctx>,
        run: AgentRun,
        status: HarnessRunStatus,
        streaming: bool,
    ) -> Self {
        ctx.limits.restart();
        reconcile_call_limits(&mut ctx, harness.policy());
        Self {
            harness,
            app_state,
            ctx: Mutex::new(ctx),
            run: Mutex::new(run),
            status: Mutex::new(status),
            streaming,
        }
    }

    /// [`Self::new`] with a fresh, default [`AgentRun`]/[`HarnessRunStatus`]
    /// — the common case for starting a brand-new run (as opposed to
    /// resuming one, which would seed `run`/`status` from prior state).
    pub fn for_run(
        harness: Arc<AgentHarness<State, Ctx>>,
        app_state: Arc<State>,
        ctx: RunContext<Ctx>,
    ) -> Self {
        let run_id = ctx.run_id().clone();
        let status = HarnessRunStatus::new(
            run_id,
            tinyagents_harness::ids::ComponentId::new("agent_loop"),
        );
        Self::new(harness, app_state, ctx, AgentRun::default(), status, false)
    }
}

/// The innermost model call: a direct, single-attempt dispatch to the
/// resolved [`tinyinference_llm::model::ChatModel`].
///
/// Unlike the direct loop's `ModelCallBase` (private to
/// `tinyagents-harness::agent_loop`), this has no response-cache lookup, no
/// `RunPolicy::retry`/`RunPolicy::fallback` loop, and no host-model routing —
/// see the module doc on [`super`] for the full list of scoped-out behavior.
/// It still runs through
/// [`tinyagents_harness::middleware::MiddlewareStack::run_wrapped_model`], so
/// a registered [`tinyagents_harness::middleware::ModelMiddleware`] (for
/// example a retry-on-error wrap middleware) still applies.
struct DirectModelBase<'m, State: Send + Sync> {
    model: &'m dyn tinyinference_llm::model::ChatModel<State>,
}

impl<State: Send + Sync, Ctx: Send + Sync> ModelBaseCall<State, Ctx>
    for DirectModelBase<'_, State>
{
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        request: ModelRequest,
    ) -> BoxModelFuture<'a> {
        Box::pin(async move {
            // Reached only when the wrap onion elected to call the provider.
            ctx.mark_provider_started();
            self.model
                .invoke(state, request)
                .await
                .map_err(TinyAgentsError::from)
        })
    }
}

/// Resolves the structured-output plan for `response_format`, mirroring the
/// direct loop's `Auto`/`JsonSchema` resolution but without the
/// `Prompted`/`ToolCallUnion` overrides (see the module doc on [`super`]).
fn resolve_structured_plan(
    request: &mut ModelRequest,
    profile: Option<&tinyinference_llm::model::ModelProfile>,
) -> Option<PendingStructuredPlan> {
    match request.response_format.take() {
        Some(ResponseFormat::Auto { name, schema }) => {
            let strategy = StructuredStrategy::for_profile(profile);
            match strategy {
                StructuredStrategy::ProviderSchema => {
                    request.response_format =
                        Some(ResponseFormat::json_schema(name.clone(), schema.clone()));
                }
                StructuredStrategy::ToolCall => {
                    let fallback_schema = ToolSchema {
                        name: name.clone(),
                        description: format!("Return the result as `{name}`."),
                        parameters: schema.clone(),
                        format: ToolFormat::Json,
                    };
                    request.tools.push(fallback_schema);
                    if request.tools.len() == 1 {
                        request.tool_choice = ToolChoice::Tool(name.clone());
                    }
                }
                StructuredStrategy::Prompted { .. } | StructuredStrategy::ToolCallUnion => {
                    unreachable!("StructuredStrategy::for_profile never returns these")
                }
            }
            Some(PendingStructuredPlan {
                strategy,
                schema_name: name,
                schema,
            })
        }
        Some(ResponseFormat::JsonSchema { name, schema }) => {
            request.response_format =
                Some(ResponseFormat::json_schema(name.clone(), schema.clone()));
            Some(PendingStructuredPlan {
                strategy: StructuredStrategy::ProviderSchema,
                schema_name: name,
                schema,
            })
        }
        other => {
            request.response_format = other;
            None
        }
    }
}

/// The `plan` node body: builds the next [`ModelRequest`] from the working
/// transcript, the harness's registered tools, and the policy's response
/// format, and stashes it on [`LoopState::pending_request`].
///
/// Takes `harness`/`ctx` as plain borrows rather than the `Arc<LoopRuntime>`
/// the other node bodies (which also need `run`/`status`) use, so this exact
/// function serves two callers with different ownership shapes without
/// duplicating its logic: [`super::compile`]'s graph closures call it against
/// a locked `MutexGuard` inside an owned, `Arc`'d [`LoopRuntime`], and
/// [`super::driver::GraphLoopDriver`] calls it directly against the
/// short-lived `&mut RunContext` [`tinyagents_harness::agent_loop::phases::LoopDriver::drive`]
/// is handed — see that module's doc for why it cannot build a `LoopRuntime`
/// of its own.
pub(crate) async fn plan_node<State, Ctx>(
    harness: &AgentHarness<State, Ctx>,
    ctx: &mut RunContext<Ctx>,
    mut loop_state: LoopState,
) -> Result<NodeResult<LoopState>>
where
    State: Send + Sync,
    Ctx: Send + Sync,
{
    if ctx.cancellation.is_cancelled() {
        return Err(TinyAgentsError::Cancelled);
    }
    // Everything on the transcript when the loop is first entered is input (or
    // a resumed run's already-announced history); only later appends are
    // announced. A no-op after the first activation.
    phases::lifecycle_seed(ctx, loop_state.messages.len());
    match apply_pending_steering(ctx, &mut loop_state.messages)? {
        SteeringOutcome::Cancel => return Err(TinyAgentsError::Cancelled),
        SteeringOutcome::Pause => {
            return Ok(NodeResult::Interrupt(Interrupt {
                id: format!("{}-steering-pause", ctx.run_id()),
                node: NodeId::from(node::PLAN),
                payload: serde_json::json!({ "reason": "steering paused the run" }),
                task_id: None,
                response_schema: None,
            }));
        }
        SteeringOutcome::Continue => {}
    }
    if ctx.check_deadline().is_err() {
        return Err(TinyAgentsError::Timeout(format!(
            "run `{}` exceeded its wall-clock deadline",
            ctx.run_id()
        )));
    }

    let tool_schemas = harness.tools().schemas();
    // Mirrors `run_loop_body`'s `AgentEvent::ToolsAdvertised`, emitted once
    // per run there (right after `before_agent`) versus once per `plan`
    // activation here — a documented, harmless divergence (see the module
    // doc on `super`): the tool set does not change mid-run, so repeating
    // the event on every turn only adds extra `tool.advertised` events, it
    // never drops or reorders the one the direct loop's own listeners
    // expect.
    let advertised_record = ctx.emit(AgentEvent::ToolsAdvertised {
        direct: tool_schemas.len(),
        deferred: 0,
        schema_bytes: tinyagents_harness::token_estimation::tool_schema_bytes(&tool_schemas),
    });
    let _ = advertised_record;
    let mut request = ModelRequest {
        messages: loop_state.messages.clone(),
        tools: tool_schemas,
        ..ModelRequest::default()
    };
    if let Some(format) = &harness.policy().default_response_format {
        request.response_format = Some(format.clone());
    }
    // Same run-policy reasoning default the direct loop attaches.
    if let Some(reasoning) = harness.policy().default_reasoning.as_ref()
        && !reasoning.is_empty()
    {
        request.reasoning = Some(reasoning.clone());
    }

    // The structured plan depends on the resolved model's profile, but the
    // model is not resolved until the `model` node (mirroring the direct
    // loop's ordering). Resolve against the *default* binding here as a
    // reasonable approximation for `ResponseFormat::Auto`'s profile-based
    // strategy choice; an explicit `ResponseFormat::JsonSchema` is
    // unaffected either way. This is a documented simplification relative to
    // the direct loop, which resolves the model first.
    let profile = harness
        .models()
        .resolve_request(&request, None, None)
        .and_then(|binding| binding.model.profile().cloned());
    let structured = resolve_structured_plan(&mut request, profile.as_ref());

    loop_state.pending_request = Some(request);
    loop_state.pending_structured = structured;
    Ok(goto(loop_state, node::MODEL))
}

/// The `model` node body: dispatches the request [`plan_node`] built,
/// records usage, appends the assistant message, and routes to `tools` or
/// `settle`.
pub(crate) async fn model_node<State, Ctx>(
    harness: &AgentHarness<State, Ctx>,
    app_state: &State,
    ctx: &mut RunContext<Ctx>,
    run: &mut AgentRun,
    status: &mut HarnessRunStatus,
    mut loop_state: LoopState,
) -> Result<NodeResult<LoopState>>
where
    State: Send + Sync,
    Ctx: Send + Sync,
{
    // `ctx.record_model_call()` itself raises a bare `Validation` error on a
    // cap hit; `run_loop_body` wraps that into `LimitExceeded` (and honors
    // `LimitBehavior::StopWithPartial` by finishing cleanly instead of
    // erroring) — mirrored here so a caller sees the identical outcome
    // regardless of which engine is driving the run.
    if let Err(error) = ctx.record_model_call() {
        let record = ctx.emit(AgentEvent::LimitReached {
            kind: tinyagents_harness::events::LimitKind::ModelCalls,
        });
        status.set_last_event(record.id);
        if matches!(
            harness.policy().limits.behavior,
            tinyagents_harness::limits::LimitBehavior::StopWithPartial
        ) {
            loop_state.finished = true;
            loop_state.limit_stop = true;
            loop_state.limit_kind = Some(tinyagents_harness::events::LimitKind::ModelCalls);
            if loop_state.final_text.is_none() {
                loop_state.final_text = Some(last_assistant_text(&loop_state.messages));
            }
            return Ok(goto(loop_state, node::SETTLE));
        }
        return Err(TinyAgentsError::LimitExceeded(error.to_string()));
    }

    phases::lifecycle_seed(ctx, loop_state.messages.len());
    phases::lifecycle_resume(ctx, loop_state.turn, None);
    let entry_len = loop_state.messages.len();
    let request = loop_state
        .pending_request
        .take()
        .ok_or_else(|| TinyAgentsError::Validation("model node ran with no pending plan".into()))?;

    let binding = harness
        .models()
        .resolve_request(&request, None, None)
        .ok_or_else(|| {
            TinyAgentsError::ModelNotFound(
                request.model.clone().unwrap_or_else(|| "<default>".into()),
            )
        })?;
    let call_id = CallId::new(format!("{}-model-{}", ctx.run_id(), run.model_calls + 1));

    let mut request = request;
    let resolution_cache = Arc::new(std::sync::Mutex::new(Some((
        request.model.clone(),
        request.model_hints.clone(),
        request.required_capabilities.clone(),
        binding,
    ))));
    // Mirror the direct loop: `before_model` middleware reads the target
    // model's profile (e.g. to avoid new system messages on a model that
    // hoists them, #6962).
    ctx.model_profile = resolution_cache
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|(_, _, _, binding)| binding.model.profile().cloned());
    let profile_cache = resolution_cache.clone();
    harness
        .middleware()
        .run_before_model_with_profile(
            ctx,
            app_state,
            &mut request,
            harness,
            move |harness, _ctx, request| {
                let profile_cache = profile_cache.clone();
                Box::pin(async move {
                    let key = (
                        request.model.clone(),
                        request.model_hints.clone(),
                        request.required_capabilities.clone(),
                    );
                    if let Some((cached_model, cached_hints, cached_capabilities, binding)) =
                        profile_cache.lock().unwrap().as_ref()
                        && *cached_model == key.0
                        && *cached_hints == key.1
                        && *cached_capabilities == key.2
                    {
                        return Ok(binding.model.profile().cloned());
                    }
                    let binding = harness.models().resolve_request(request, None, None);
                    let profile = binding
                        .as_ref()
                        .and_then(|binding| binding.model.profile().cloned());
                    *profile_cache.lock().unwrap() =
                        binding.map(|binding| (key.0, key.1, key.2, binding));
                    Ok(profile)
                })
            },
        )
        .await?;

    // Middleware may select a different registered model or add required
    // capabilities. Resolve again so graph execution dispatches the request
    // that middleware actually prepared, just like the direct loop.
    let key = (
        request.model.clone(),
        request.model_hints.clone(),
        request.required_capabilities.clone(),
    );
    let binding = resolution_cache
        .lock()
        .unwrap()
        .take()
        .and_then(
            |(cached_model, cached_hints, cached_capabilities, binding)| {
                (cached_model == key.0 && cached_hints == key.1 && cached_capabilities == key.2)
                    .then_some(binding)
            },
        )
        .or_else(|| harness.models().resolve_request(&request, None, None))
        .ok_or_else(|| {
            TinyAgentsError::ModelNotFound(
                request.model.clone().unwrap_or_else(|| "<default>".into()),
            )
        })?;
    ctx.model_profile = binding.model.profile().cloned();
    tinyagents_harness::middleware::library::rehome_ephemeral_system_instructions(
        &mut request,
        ctx.model_profile.as_ref(),
    );
    let model_name = binding.resolved.name.clone();

    // Mirror the direct loop: a named effort picks up the resolved model's
    // tuned `thinking_level_map` entry, unless an explicit budget is pinned.
    if let Some(profile) = binding.model.profile()
        && let Some(reasoning) = request.reasoning.as_ref()
        && reasoning.budget_tokens.is_none()
        && let Some(effort) = reasoning.effort
        && let Some(mapped) = profile.thinking_level_map.get(effort.as_str())
    {
        request.reasoning = Some(mapped.clone());
    }

    // Same point as the direct loop: pending appends (steering) are announced,
    // the previous turn closed, and this one opened, just before `ModelStarted`.
    let turn = phases::lifecycle_start_turn(harness, ctx, &loop_state.messages);
    tracing::debug!(
        target: "tinyagents::agent_loop",
        run_id = %ctx.run_id(),
        turn,
        "[graph_loop] turn started"
    );

    let started_record = ctx.emit(AgentEvent::ModelStarted {
        call_id: call_id.clone(),
        model: model_name.clone(),
    });
    status.set_last_event(started_record.id);
    status.active_model_call = Some(call_id.clone());
    ctx.active_model_call = Some(call_id.clone());
    ctx.begin_model_call();
    let base = DirectModelBase {
        model: binding.model.as_ref(),
    };
    let wrapped = match harness
        .middleware()
        .run_wrapped_model(ctx, app_state, request, &base)
        .await
    {
        Ok(wrapped) => wrapped,
        Err(error) => {
            status.active_model_call = None;
            ctx.active_model_call = None;
            ctx.mark_model_call_failed();
            return Err(error);
        }
    };
    let (mut response, wrap_control) = wrapped.into_response_with_control();
    if let Some(control) = wrap_control {
        ctx.request_control(control);
    }

    run.model_calls += 1;
    run.steps += 1;
    status.model_calls = run.model_calls;
    status.active_model_call = None;
    ctx.active_model_call = None;
    if let Some(usage) = response.usage {
        run.usage.record(usage);
        loop_state.usage = run.usage;
        let usage_record = ctx.emit(AgentEvent::UsageRecorded { usage });
        status.set_last_event(usage_record.id);
    }

    harness
        .middleware()
        .run_after_model(ctx, app_state, &mut response)
        .await?;

    let completed_record = ctx.emit(AgentEvent::ModelCompleted {
        call_id: call_id.clone(),
        started_at_ms: None,
        usage: response.usage,
        input: None,
        output: None,
    });
    status.set_last_event(completed_record.id);

    loop_state.model_calls = run.model_calls;
    loop_state.last_call_id = Some(call_id.to_string());
    loop_state
        .messages
        .push(tinyinference_llm::message::Message::Assistant(
            response.message.clone(),
        ));
    loop_state.turn += 1;
    phases::lifecycle_flush(harness, ctx, &loop_state.messages);

    let tool_calls = response.tool_calls().to_vec();
    loop_state.pending_tool_calls = tool_calls.clone();

    let route = if tool_calls.is_empty() {
        node::SETTLE
    } else {
        node::TOOLS
    };

    // Computed above `take_control` (rather than the reverse) so a
    // `MiddlewareControl::Continue`/`UpdateState` control — which means "no
    // override, proceed with whatever the turn would have done anyway" —
    // has the real tool-routing decision to fall through to instead of an
    // arbitrary default.
    if let Some(control) = ctx.take_control() {
        let result = apply_control(ctx, &mut loop_state, control, node::MODEL, route);
        retract_on_interrupt(harness, ctx, &result, &loop_state.messages, entry_len, true);
        return result;
    }
    // Stash the response for `settle` to extract structured output from.
    // Reusing `pending_request`'s sibling field would need a new field; keep
    // it simple by re-deriving what `settle` needs from `messages` (the
    // response text/tool-calls) plus `structured` plan already on
    // `loop_state`. `response.finish_reason`/raw provider fields are not
    // needed by this reduced-scope settle (see the module doc on `super`).
    let _ = model_name;
    let _ = ModelOutcomeShadow(&response);
    Ok(goto(loop_state, route))
}

/// Zero-sized marker used only to keep `response` "used" for readability at
/// the call site above without over-cloning it into `LoopState`.
struct ModelOutcomeShadow<'a>(#[allow(dead_code)] &'a ModelResponse);

/// The `tools` node body: executes the batch [`model_node`] requested via
/// [`phases::execute_tool_batch`] (the exact same admission /
/// serial-or-concurrent execution / middleware pipeline the direct loop
/// uses — see that function's docs), then routes back to `plan` for the next
/// turn.
pub(crate) async fn tools_node<State, Ctx>(
    harness: &AgentHarness<State, Ctx>,
    app_state: &State,
    ctx: &mut RunContext<Ctx>,
    run: &mut AgentRun,
    status: &mut HarnessRunStatus,
    mut loop_state: LoopState,
) -> Result<NodeResult<LoopState>>
where
    State: Send + Sync,
    Ctx: Send + Sync,
{
    phases::lifecycle_seed(ctx, loop_state.messages.len());
    let entry_len = loop_state.messages.len();
    // The turn that issued these calls began at (or before) the assistant
    // message; a fresh runtime resuming an interrupted batch has no open turn.
    phases::lifecycle_resume(ctx, loop_state.turn, Some(entry_len.saturating_sub(1)));
    let calls = std::mem::take(&mut loop_state.pending_tool_calls);
    let outcome = phases::execute_tool_batch(
        harness,
        app_state,
        ctx,
        run,
        status,
        &mut loop_state.messages,
        calls,
    )
    .await;
    let outcome = match outcome {
        Ok(outcome) => outcome,
        Err(error)
            if matches!(
                harness.policy().limits.behavior,
                tinyagents_harness::limits::LimitBehavior::StopWithPartial
            ) && matches!(error, TinyAgentsError::LimitExceeded(_))
                && ctx.peek_last_limit()
                    == Some(tinyagents_harness::events::LimitKind::ToolCalls) =>
        {
            // Only the tool-cap admission announces `LimitReached(ToolCalls)`
            // (and `record_tool_call` cleared any earlier kind first), so a
            // `LimitExceeded` raised by middleware is not a partial stop.
            loop_state.limit_stop = true;
            loop_state.limit_kind = Some(tinyagents_harness::events::LimitKind::ToolCalls);
            loop_state.finished = true;
            if loop_state.final_text.is_none() {
                loop_state.final_text = Some(last_assistant_text(&loop_state.messages));
            }
            return Ok(goto(loop_state, node::SETTLE));
        }
        Err(error) => {
            // Results of calls that ran before the failure are on the node's
            // transcript, which the error path would otherwise drop; keep them
            // on the run, as the direct loop does, so the driver's final
            // lifecycle close announces and counts them.
            run.messages = loop_state.messages.clone();
            return Err(error);
        }
    };
    loop_state.tool_calls = run.tool_calls;
    loop_state.executed_tools = run.executed_tools.clone();
    let _ = outcome;

    if harness.middleware().any_should_stop_after_turn(ctx, run) {
        ctx.request_control(MiddlewareControl::JumpTo(LoopTarget::End));
    }

    if let Some(control) = ctx.take_control() {
        let result = apply_control(ctx, &mut loop_state, control, node::TOOLS, node::PLAN);
        // The tool turn stays open on an interrupt: the re-run closes it with the
        // results it produces (a fresh runtime re-opens it via `lifecycle_resume`).
        if !retract_on_interrupt(
            harness,
            ctx,
            &result,
            &loop_state.messages,
            entry_len,
            false,
        ) {
            // Every tool result of this batch is on the transcript: announce
            // them and close the turn, as the direct loop does after its batch.
            phases::lifecycle_close_turn(harness, ctx, &loop_state.messages);
        }
        return result;
    }
    phases::lifecycle_close_turn(harness, ctx, &loop_state.messages);

    Ok(goto(loop_state, node::PLAN))
}

/// The `settle` node body: extracts/validates structured output when the
/// turn planned one, drives the output-validation retry loop
/// (`RunPolicy::output_retry`), and finishes the run.
/// An interrupted node's state is discarded and the node re-runs from its entry
/// state on resume, so the appends it already announced are retracted: the
/// re-run announces them again, and events never name a message the kept
/// transcript lacks.
fn retract_on_interrupt<State: Send + Sync, Ctx: Send + Sync>(
    harness: &AgentHarness<State, Ctx>,
    ctx: &mut RunContext<Ctx>,
    result: &Result<NodeResult<LoopState>>,
    messages: &[tinyinference_llm::message::Message],
    entry_len: usize,
    close_turn: bool,
) -> bool {
    if !matches!(result, Ok(NodeResult::Interrupt(_))) {
        return false;
    }
    tracing::debug!(
        target: "tinyagents::agent_loop",
        run_id = %ctx.run_id(),
        entry_len,
        "[graph_loop] node interrupted; retracting its announced appends"
    );
    phases::lifecycle_retract(ctx, entry_len);
    // A model node's turn has no results to wait for, and a fresh runtime
    // cannot carry this tracker's open turn over, so close it. A tools node
    // leaves its turn open for the re-run to close with the real results.
    if close_turn {
        phases::lifecycle_close_turn(harness, ctx, &messages[..entry_len.min(messages.len())]);
    }
    true
}

pub(crate) async fn settle_node<State, Ctx>(
    harness: &AgentHarness<State, Ctx>,
    ctx: &mut RunContext<Ctx>,
    run: &mut AgentRun,
    mut loop_state: LoopState,
) -> Result<NodeResult<LoopState>>
where
    State: Send + Sync,
    Ctx: Send + Sync,
{
    phases::lifecycle_seed(ctx, loop_state.messages.len());
    // The final assistant message is the last append of its turn; close the
    // turn before any output-retry prompt is pushed (that prompt is announced
    // with the next turn's start).
    phases::lifecycle_close_turn(harness, ctx, &loop_state.messages);
    if let Some(plan) = loop_state.pending_structured.take() {
        let extractor = StructuredExtractor::new(
            plan.strategy.clone(),
            &plan.schema_name,
            plan.schema.clone(),
        );
        let last_response = last_response_from_messages(&loop_state.messages);
        let outcome = extractor.extract_outcome(&last_response);
        let variant = outcome.variant.clone();
        // Note: unlike the direct loop, this does not consult
        // `AgentHarness::with_output_validator` (A3's post-extraction
        // validator hook) — out of scope for this rendition (see the module
        // doc on `super`).
        let error = match outcome.value {
            Some(value) => {
                run.structured = Some(value.clone());
                run.structured_variant = variant.clone();
                loop_state.structured = Some(value);
                loop_state.structured_variant = variant;
                None
            }
            None => outcome.error,
        };
        if let Some(error) = error {
            let max_attempts = harness.policy().output_retry.max_attempts;
            if loop_state.output_retry_attempts < max_attempts {
                loop_state.output_retry_attempts += 1;
                let template = &harness.policy().output_retry.message_template;
                let prompt = template.replace("{error}", &error);
                loop_state
                    .messages
                    .push(tinyinference_llm::message::Message::user(prompt));
                // Back to `plan`, not `model` directly: the direct loop's
                // retry re-enters its outer loop, which rebuilds the
                // `ModelRequest` from `messages` (now including the repair
                // prompt) — `model_node` needs a fresh `pending_request`,
                // which only `plan_node` produces.
                return Ok(goto(loop_state, node::PLAN));
            }
            return Err(TinyAgentsError::StructuredOutput(error));
        }
    }

    loop_state.finished = true;
    if loop_state.final_text.is_none() {
        loop_state.final_text = Some(last_assistant_text(&loop_state.messages));
    }
    run.messages = loop_state.messages.clone();
    run.final_response = Some(ModelResponse::assistant(
        loop_state.final_text.clone().unwrap_or_default(),
    ));

    Ok(NodeResult::Command(Command {
        update: Some(loop_state),
        goto: vec![RouteTarget::Node(NodeId::from(crate::builder::END))],
        resume: None,
        resume_by_task: Default::default(),
    }))
}

/// Reconstructs the model response [`settle_node`] needs from the last
/// assistant message on the transcript. A documented simplification: the
/// full [`ModelResponse`] (usage, `finish_reason`, provider `raw`) produced by
/// [`model_node`] is not threaded through to `settle` — only the message
/// (text + tool calls) that [`tinyagents_harness::structured::StructuredExtractor`]
/// actually reads.
fn last_response_from_messages(messages: &[tinyinference_llm::message::Message]) -> ModelResponse {
    for message in messages.iter().rev() {
        if let tinyinference_llm::message::Message::Assistant(assistant) = message {
            return ModelResponse {
                message: assistant.clone(),
                usage: None,
                finish_reason: None,
                raw: None,
                resolved_model: None,
                continue_turn: None,
                served_from_cache: false,
                correlation: None,
                resolved_route: None,
            };
        }
    }
    ModelResponse::assistant(String::new())
}

fn last_assistant_text(messages: &[tinyinference_llm::message::Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| matches!(message, tinyinference_llm::message::Message::Assistant(_)))
        .map(tinyinference_llm::message::Message::text)
        .unwrap_or_default()
}

/// Appends a synthetic tool-result message for every still-unanswered tool
/// call on the last assistant message, mirroring the direct loop's
/// `close_unanswered_tool_calls` so a `JumpTo(Model)`/`JumpTo(End)`/
/// `StopWithFinal` control leaves a replayable transcript.
fn close_unanswered_tool_calls(
    messages: &mut Vec<tinyinference_llm::message::Message>,
    reason: &str,
) {
    let Some(tinyinference_llm::message::Message::Assistant(last)) = messages.last() else {
        return;
    };
    if last.tool_calls.is_empty() {
        return;
    }
    let synthetic: Vec<_> = last
        .tool_calls
        .iter()
        .map(|call| tinyinference_llm::message::Message::tool(call.id.clone(), reason))
        .collect();
    messages.extend(synthetic);
}

/// Applies a drained [`MiddlewareControl`], mirroring the direct loop's
/// `apply_pending_control` but expressed as a graph routing decision instead
/// of a `LoopExit`/`ControlEffect`.
fn apply_control<Ctx>(
    ctx: &mut RunContext<Ctx>,
    loop_state: &mut LoopState,
    control: MiddlewareControl,
    from_node: &str,
    natural_next: &str,
) -> Result<NodeResult<LoopState>>
where
    Ctx: Send + Sync,
{
    match control {
        // `Continue`/`UpdateState` request no override: route to whatever
        // the calling node had already determined the turn's natural next
        // step to be (see the call sites in `model_node`/`tools_node`).
        MiddlewareControl::Continue => Ok(goto(loop_state.clone(), natural_next)),
        MiddlewareControl::UpdateState(update) => {
            ctx.push_state_update(update);
            Ok(goto(loop_state.clone(), natural_next))
        }
        MiddlewareControl::JumpTo(LoopTarget::Tools) => {
            let route = if loop_state.pending_tool_calls.is_empty() {
                node::SETTLE
            } else {
                node::TOOLS
            };
            Ok(goto(loop_state.clone(), route))
        }
        MiddlewareControl::JumpTo(LoopTarget::Model) => {
            close_unanswered_tool_calls(
                &mut loop_state.messages,
                "run jumped back to the model before this tool call was executed",
            );
            Ok(goto(loop_state.clone(), node::PLAN))
        }
        MiddlewareControl::JumpTo(LoopTarget::End) => {
            close_unanswered_tool_calls(
                &mut loop_state.messages,
                "run stopped before this tool call was executed",
            );
            loop_state.finished = true;
            if loop_state.final_text.is_none() {
                loop_state.final_text = Some(last_assistant_text(&loop_state.messages));
            }
            Ok(goto(loop_state.clone(), node::SETTLE))
        }
        MiddlewareControl::StopWithFinal(text) => {
            close_unanswered_tool_calls(
                &mut loop_state.messages,
                "run stopped before this tool call was executed",
            );
            loop_state.finished = true;
            loop_state.final_text = Some(text);
            Ok(goto(loop_state.clone(), node::SETTLE))
        }
        MiddlewareControl::Interrupt { node, message } => Ok(NodeResult::Interrupt(Interrupt {
            id: format!("{from_node}-{node}"),
            node: NodeId::from(node.as_str()),
            payload: serde_json::json!({ "message": message }),
            task_id: None,
            response_schema: None,
        })),
    }
}

fn goto(loop_state: LoopState, target: &str) -> NodeResult<LoopState> {
    NodeResult::Command(Command {
        update: Some(loop_state),
        goto: vec![RouteTarget::Node(NodeId::from(target))],
        resume: None,
        resume_by_task: Default::default(),
    })
}

/// Reconciles `ctx.config`'s per-run call caps against `policy.limits`,
/// exactly like the direct loop's `run_loop_body` does at the top of every
/// run (`resolve_call_cap` + `LimitTracker::sync_call_limits`) — without it,
/// `ctx.limits` stays at whatever `RunContext::new` derived from `config`
/// alone, silently ignoring a `RunPolicy::limits` override, in either
/// direction. Exposed so [`super::driver::GraphLoopDriver`] (which does not
/// build a [`LoopRuntime`], see that module's docs) can apply the exact same
/// reconciliation before it starts stepping nodes.
pub(crate) fn reconcile_call_limits<Ctx: Send + Sync>(
    ctx: &mut RunContext<Ctx>,
    policy: &tinyagents_harness::runtime::RunPolicy,
) {
    let effective_model_calls = match ctx.config.max_model_calls {
        Some(explicit) => explicit.min(policy.limits.max_model_calls),
        None => policy.limits.max_model_calls,
    };
    let effective_tool_calls = match ctx.config.max_tool_calls {
        Some(explicit) => explicit.min(policy.limits.max_tool_calls),
        None => policy.limits.max_tool_calls,
    };
    ctx.limits
        .sync_call_limits(effective_model_calls, effective_tool_calls);
}
