//! [`GraphLoopDriver`]: plugs the compiled loop's node bodies into
//! [`AgentHarness::invoke`] (and friends) via
//! [`tinyagents_harness::agent_loop::phases::LoopDriver`] +
//! [`AgentHarness::with_loop_driver`], selected by
//! [`tinyagents_harness::runtime::RunPolicy::execution`]`::Graph`.
//!
//! See the module doc on [`super`] ("`GraphLoopDriver` vs.
//! `compile_loop`/`LoopIter`") for why this drives
//! [`super::runtime::plan_node`]/`model_node`/`tools_node`/`settle_node`
//! directly over borrowed `&mut` state in a hand-rolled loop instead of
//! building a [`crate::CompiledGraph`].

use async_trait::async_trait;

use tinyagents_harness::agent_loop::phases::{self, LoopDriver};
use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::events::{AgentEvent, HarnessRunStatus};
use tinyagents_harness::ids::HarnessPhase;
use tinyagents_harness::middleware::AgentRun;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::steering::PauseState;
use tinyagents_harness::terminal::{TerminalOutcome, TerminalReason};
use tinyinference_llm::message::Message;

use crate::command::{NodeResult, RouteTarget};

use super::runtime;
use super::types::{LoopState, node};

/// Drives [`AgentHarness::invoke`] through the same node bodies
/// [`super::compile_loop`] wires into a [`crate::CompiledGraph`], without
/// itself building one. Install with
/// [`AgentHarness::with_loop_driver`]`(Arc::new(GraphLoopDriver::new()))` and
/// [`tinyagents_harness::runtime::RunPolicy::execution`]`::Graph`.
///
/// Stateless — one instance can be shared (via `Arc`) across every harness
/// that wants the graph engine.
#[derive(Debug, Default, Clone, Copy)]
pub struct GraphLoopDriver;

impl GraphLoopDriver {
    /// Creates a driver. Stateless: nothing to configure.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<State, Ctx> LoopDriver<State, Ctx> for GraphLoopDriver
where
    State: Send + Sync,
    Ctx: Send + Sync,
{
    async fn drive(
        &self,
        harness: &AgentHarness<State, Ctx>,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        input: Vec<Message>,
        streaming: bool,
    ) -> Result<()> {
        // Mirrors `run_loop`'s own top-of-run bookkeeping (see that
        // function's docs on why the limit tracker restarts here rather
        // than at `RunContext::new`).
        ctx.limits.restart();
        runtime::reconcile_call_limits(ctx, harness.policy());
        ctx.streaming = streaming;

        let record = ctx.emit(AgentEvent::RunStarted {
            run_id: ctx.run_id().clone(),
            thread_id: ctx.thread_id().cloned(),
        });
        status.set_last_event(record.id);
        status.mark_running(HarnessPhase::Idle);

        harness.middleware().run_before_agent(ctx, state).await?;

        let mut loop_state = LoopState {
            messages: input,
            ..LoopState::default()
        };
        ctx.reset_turn_tracker(loop_state.messages.len());
        let mut current: &str = node::PLAN;
        let mut limit_stop = false;
        let mut limit_kind = None;

        let outcome = loop {
            // Keeps `run.messages` a running snapshot of the transcript as
            // of the start of each node call, so a node that errors or
            // interrupts mid-call (and so never hands `loop_state` back)
            // still leaves `run.messages` close to current — mirroring
            // `run_loop`'s "transcript survives every exit path" guarantee
            // as closely as this driver's by-value node contract allows.
            // The one gap: mutations a node makes to its *own* copy of
            // `loop_state.messages` before erroring/interrupting (for
            // example `plan_node`'s steering-injected message on a pause)
            // are not reflected until that node returns successfully.
            run.messages = loop_state.messages.clone();
            let result = match current {
                node::PLAN => runtime::plan_node(harness, ctx, loop_state).await,
                node::MODEL => {
                    runtime::model_node(harness, state, ctx, run, status, loop_state).await
                }
                node::TOOLS => {
                    runtime::tools_node(harness, state, ctx, run, status, loop_state).await
                }
                node::SETTLE => runtime::settle_node(harness, ctx, run, loop_state).await,
                other => {
                    break Err(TinyAgentsError::Validation(format!(
                        "GraphLoopDriver: unknown loop node `{other}`"
                    )));
                }
            };

            match result {
                Ok(NodeResult::Interrupt(interrupt))
                    if interrupt.id.ends_with("-steering-pause") =>
                {
                    // No `CompiledGraph` is in play here, so there is no
                    // checkpoint to pause against — mirror the direct loop's
                    // steering pause instead: latch `run.paused` and finish
                    // this call with `Ok(())`, exactly like
                    // `run_loop`'s `LoopExit::Paused` handling. See the
                    // module doc on `super` for why this is not a resumable
                    // graph interrupt.
                    break Ok(Some(interrupt));
                }
                Ok(NodeResult::Interrupt(interrupt)) => {
                    // A `MiddlewareControl::Interrupt` (an approval gate, not
                    // a steering pause). The direct loop surfaces this as
                    // `TinyAgentsError::Interrupted` (see
                    // `agent_loop::run_loop`'s `apply_pending_control`), not
                    // a pause — matched here so `RunPolicy::execution ==
                    // Graph` behaves identically to `Direct` for this
                    // control. Only `compile_loop`/`LoopIter`'s real
                    // `CompiledGraph` upgrades this into a resumable graph
                    // interrupt (A5's "approvals surfacing as graph
                    // interrupts").
                    let message = interrupt
                        .payload
                        .get("message")
                        .and_then(|value| value.as_str())
                        .unwrap_or("interrupted")
                        .to_string();
                    break Err(TinyAgentsError::Interrupted {
                        node: interrupt.node.to_string(),
                        message,
                    });
                }
                Ok(NodeResult::Update(updated)) => {
                    // Every node body returns `Command`/`Interrupt` (see
                    // `runtime`'s node docs); a bare `Update` would mean the
                    // loop cannot determine where to go next.
                    let _ = updated;
                    break Err(TinyAgentsError::Validation(
                        "GraphLoopDriver: loop node returned an un-routed update".to_string(),
                    ));
                }
                Ok(NodeResult::Command(command)) => {
                    loop_state = match command.update {
                        Some(update) => update,
                        None => {
                            break Err(TinyAgentsError::Validation(
                                "GraphLoopDriver: loop node's command carried no update"
                                    .to_string(),
                            ));
                        }
                    };
                    limit_stop |= loop_state.limit_stop;
                    // Latch the first kind: a later command must not erase it.
                    limit_kind = limit_kind.or(loop_state.limit_kind);
                    let Some(target) = command.goto.first() else {
                        break Err(TinyAgentsError::Validation(
                            "GraphLoopDriver: loop node's command carried no route".to_string(),
                        ));
                    };
                    let RouteTarget::Node(node_id) = target else {
                        break Err(TinyAgentsError::Validation(
                            "GraphLoopDriver: loop node routed via `Send`, which this driver \
                             does not support"
                                .to_string(),
                        ));
                    };
                    if node_id.as_str() == crate::builder::END {
                        break Ok(None);
                    }
                    current = match node_id.as_str() {
                        node::PLAN => node::PLAN,
                        node::MODEL => node::MODEL,
                        node::TOOLS => node::TOOLS,
                        node::SETTLE => node::SETTLE,
                        other => {
                            break Err(TinyAgentsError::Validation(format!(
                                "GraphLoopDriver: unknown loop node `{other}`"
                            )));
                        }
                    };
                    continue;
                }
                Err(error) => break Err(error),
            }
        };

        let terminal = match &outcome {
            Ok(None) => {
                let reason = if limit_stop {
                    TerminalOutcome::limit_reached(
                        limit_kind,
                        format!(
                            "stopped with the partial run: {} limit reached",
                            limit_kind.map_or("run", |kind| kind.as_str())
                        ),
                    )
                } else {
                    TerminalOutcome::completed()
                };
                Some(reason.with_provider_started(ctx.provider_started()))
            }
            Ok(Some(interrupt)) => {
                let reason = interrupt
                    .payload
                    .get("reason")
                    .or_else(|| interrupt.payload.get("message"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let outcome = if let Some(summary) = ctx.take_halted_by_guard() {
                    TerminalOutcome::halted(summary)
                } else {
                    TerminalOutcome::new(
                        TerminalReason::Paused,
                        reason
                            .clone()
                            .unwrap_or_else(|| format!("paused at node `{}`", interrupt.node)),
                    )
                };
                Some(outcome.with_provider_started(ctx.provider_started()))
            }
            Err(error) => {
                use tinyagents_harness::terminal::TimeoutPhase;
                let in_model_call = ctx.model_call_failed() || ctx.active_model_call.is_some();
                let site = if in_model_call && ctx.call_provider_started() {
                    TimeoutPhase::Provider
                } else if in_model_call {
                    TimeoutPhase::BeforeProvider
                } else if ctx.provider_started() {
                    TimeoutPhase::AfterTurn
                } else {
                    TimeoutPhase::BeforeProvider
                };
                // A `LimitExceeded` carries only text; the cap that tripped was
                // announced by a `LimitReached` event just before.
                let kind = matches!(error, TinyAgentsError::LimitExceeded(_))
                    .then(|| ctx.peek_last_limit())
                    .flatten();
                let mut outcome = TerminalOutcome::from_error(error, site).with_limit_kind(kind);
                // A failed summarizer already received a provider response, though
                // summarizer calls bypass the context's dispatch marker.
                outcome.provider_started = ctx.provider_started()
                    || matches!(error, TinyAgentsError::SummarizationUsage { .. });
                Some(outcome)
            }
        };
        // Announce whatever the last turn appended and close it, on every exit
        // path, as the direct loop does before the transcript moves onto the
        // run (`run.messages` is the transcript as of the last node boundary).
        phases::lifecycle_close_turn(harness, ctx, &run.messages);
        run.terminal = terminal.clone();
        status.mark_running(HarnessPhase::Middleware);
        let after_agent = harness.middleware().run_after_agent(ctx, state, run).await;
        // `after_agent` may post-process `run.messages`; announce anything it
        // appended (even if it then failed), as the direct loop does.
        phases::lifecycle_flush(harness, ctx, &run.messages);
        if let Err(hook_error) = after_agent {
            if outcome.is_err() {
                // The originating node failure stays authoritative, as in the
                // direct loop; the hook still ran for its cleanup.
                tracing::warn!(
                    target: "tinyagents::agent_loop",
                    error = %hook_error,
                    "[agent_loop] after_agent failed after a node error; keeping the node error"
                );
            } else {
                return Err(hook_error);
            }
        }

        // `status.mark_completed`/`mark_interrupted`/`mark_failed` and (on
        // error) `AgentEvent::RunFailed` are applied centrally by
        // `agent_loop::entry::drive_collecting` after this call returns,
        // identically for the direct loop and this driver — see that
        // function's doc comment. This driver only emits the terminal event
        // that (like the direct loop's `run_loop_body`) is its own
        // responsibility to raise: `RunCompleted` on a clean finish, or
        // latching `run.paused` (mirroring a steering pause) on an
        // interrupt.
        match outcome {
            Ok(None) => {
                // `after_agent` may have replaced the outcome; report the final one.
                let outcome = run
                    .terminal
                    .clone()
                    .or(terminal)
                    .expect("terminal set before after_agent");
                let record = ctx.emit(AgentEvent::RunCompleted {
                    run_id: ctx.run_id().clone(),
                    outcome: Some(outcome),
                });
                status.set_last_event(record.id);
                Ok(())
            }
            Ok(Some(interrupt)) => {
                let reason = interrupt
                    .payload
                    .get("reason")
                    .or_else(|| interrupt.payload.get("message"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: "paused".to_string(),
                    detail: reason
                        .clone()
                        .unwrap_or_else(|| format!("paused at node `{}`", interrupt.node)),
                });
                status.set_last_event(record.id);
                if !matches!(
                    run.terminal.as_ref().map(|outcome| outcome.reason),
                    Some(TerminalReason::Halted)
                ) {
                    run.paused = Some(PauseState {
                        reason,
                        paused_at_checkpoint: 0,
                    });
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}
