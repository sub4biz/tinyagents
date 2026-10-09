//! Turn-boundary control for the superstep loop: the run queue, deferred tool
//! calls, and control outcomes requested by middleware.
//!
//! Split out of `run_loop.rs`. These are the helpers the loop body calls at its
//! safe checkpoints: draining queued steering and follow-ups (A4), settling a
//! batch that deferred calls (A2), and honoring a pending
//! [`MiddlewareControl`].

use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Takes the pending items of `lane` from the run's queue (per
    /// [`RunPolicy::queue_mode`][crate::runtime::RunPolicy::queue_mode]),
    /// appends them to the working transcript, and emits
    /// [`AgentEvent::QueuedMessageApplied`] (A4). Returns whether anything
    /// was applied. A run without a queue never applies anything.
    pub(super) async fn apply_queued_lane(
        &self,
        ctx: &mut RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        lane: crate::run_queue::QueueLane,
    ) -> bool {
        let Some(queue) = ctx.run_queue.clone() else {
            return false;
        };
        let items = queue.take(lane, self.policy.queue_mode).await;
        if items.is_empty() {
            return false;
        }
        let count = items.len();
        let first_index = messages.len();
        // Payloads follow the capture policy, like every other event.
        // One slot per applied message so payloads stay aligned with
        // `first_index..first_index + count`; an uncaptured message is `null`.
        // Nothing is emitted at all when no message in the batch is captured.
        let is_captured = |message: &Message| match message {
            Message::Tool(_) => self.policy.capture.tool_io,
            _ => self.policy.capture.model_io,
        };
        let captured: Vec<serde_json::Value> = if items.iter().any(is_captured) {
            items
                .iter()
                .map(|message| {
                    if is_captured(message) {
                        super::lifecycle::to_value_logged(message)
                    } else {
                        serde_json::Value::Null
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        messages.extend(items);
        let record = ctx.emit(AgentEvent::QueuedMessageApplied {
            lane,
            count,
            first_index,
            messages: captured,
        });
        status.set_last_event(record.id);
        tracing::debug!(
            target: "tinyagents::agent_loop",
            run_id = %ctx.run_id(),
            lane = lane.as_str(),
            count,
            "[agent_loop] applied queued messages to the transcript"
        );
        true
    }

    /// The natural-finish queue boundary (A4): the model produced a final
    /// answer, so pending `Steer` items (first) or, when there are none,
    /// `Followup` items are appended and the loop runs another turn instead
    /// of returning. Returns whether the loop should continue. Only reached
    /// from the paths where the *model* finished — a middleware stop, a
    /// limit stop, a pause, or a deferral is terminal and leaves the queue
    /// untouched for the host.
    pub(super) async fn continue_from_queue_at_finish(
        &self,
        ctx: &mut RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
    ) -> bool {
        if ctx.run_queue.is_none() {
            return false;
        }
        self.apply_queued_lane(ctx, status, messages, crate::run_queue::QueueLane::Steer)
            .await
            || self
                .apply_queued_lane(ctx, status, messages, crate::run_queue::QueueLane::Followup)
                .await
    }

    /// Settles the calls a batch deferred (A2).
    ///
    /// Returns `Ok(None)` when nothing was deferred, or when a registered
    /// [`crate::tool::DeferredToolHandler`] resolved every pending call and
    /// the loop can continue. Returns `Ok(Some(LoopExit::Deferred))` when
    /// the caller must resolve the requests — no handler, or an approved
    /// call deferred a second time (surfaced rather than re-asked, so a
    /// handler and a tool that never agree cannot spin).
    pub(super) async fn settle_deferred(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        deferred: crate::tool::DeferredToolRequests,
    ) -> Result<Option<LoopExit>> {
        if deferred.is_empty() {
            return Ok(None);
        }
        let Some(handler) = &self.deferred_tool_handler else {
            return Ok(Some(LoopExit::Deferred(deferred)));
        };
        let results = handler.handle(&deferred).await?;
        let pending: Vec<ToolCall> = deferred
            .approvals
            .iter()
            .chain(deferred.calls.iter())
            .cloned()
            .collect();
        let again = self
            .apply_deferred_results(state, ctx, run, status, messages, pending, results)
            .await?;
        if again.is_empty() {
            return Ok(None);
        }
        Ok(Some(LoopExit::Deferred(again)))
    }

    /// Applies host decisions to `pending` deferred calls (A2): every call
    /// must be resolved (`Validation` error naming the missing ids
    /// otherwise). Host-supplied results and denials are answered without
    /// running a tool; approvals run the tool now through the ordinary
    /// serial pipeline, with the model's or the approver's edited
    /// arguments. Returns whatever the approved calls deferred *again*.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn apply_deferred_results(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        pending: Vec<ToolCall>,
        mut results: crate::tool::DeferredToolResults,
    ) -> Result<crate::tool::DeferredToolRequests> {
        let missing: Vec<&str> = pending
            .iter()
            .filter(|call| !results.resolves(&CallId::new(call.id.clone())))
            .map(|call| call.id.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(TinyAgentsError::Validation(format!(
                "cannot resume: deferred tool calls still unresolved: [{}]",
                missing.join(", ")
            )));
        }
        ctx.terminate_votes.clear();
        // `terminate` is a whole-batch decision, so a resume may only end the
        // run when the resumed calls *are* the original batch. Siblings that
        // answered before the pause (and whose votes are gone) may not have
        // asked to terminate, so a partial resume must not.
        let original_batch_len = messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::Assistant(assistant) => Some(assistant.tool_calls.len()),
                _ => None,
            })
            .unwrap_or(0);
        let resumes_whole_batch = pending.len() == original_batch_len;
        let mut deferred = crate::tool::DeferredToolRequests::default();
        // Follow-up user messages (B2) trail the whole resumed batch, for
        // the same provider-ordering reason as in `execute_tools`.
        let mut follow_ups = Vec::new();
        for mut call in pending {
            let call_id = CallId::new(call.id.clone());
            if let Some(outcome) = results.calls.remove(&call_id) {
                follow_ups.extend(
                    self.recover_tool_call(
                        state,
                        ctx,
                        run,
                        status,
                        messages,
                        &call,
                        outcome.into_tool_result(),
                    )
                    .await?,
                );
                continue;
            }
            let decision = results
                .approvals
                .remove(&call_id)
                .expect("every pending call was validated as resolved above");
            match decision {
                crate::tool::ToolApprovalDecision::Deny { message } => {
                    let record = ctx.emit(AgentEvent::ToolDenied {
                        call_id,
                        message: message.clone(),
                    });
                    status.set_last_event(record.id);
                    follow_ups.extend(
                        self.recover_tool_call(
                            state,
                            ctx,
                            run,
                            status,
                            messages,
                            &call,
                            tinytools::ToolResult::error(message),
                        )
                        .await?,
                    );
                }
                decision => {
                    if let crate::tool::ToolApprovalDecision::ApproveWithArgs(arguments) = decision
                    {
                        call.arguments = arguments;
                    }
                    let record = ctx.emit(AgentEvent::ToolApproved { call_id });
                    status.set_last_event(record.id);
                    ctx.mark_call_approved(call.id.clone());
                    let mut approved_promotions = std::collections::BTreeSet::new();
                    follow_ups.extend(
                        self.execute_tool_serially(
                            state,
                            ctx,
                            run,
                            status,
                            messages,
                            call,
                            &mut deferred,
                            &mut approved_promotions,
                        )
                        .await?,
                    );
                }
            }
        }
        super::tools::append_follow_ups(messages, follow_ups);
        // Siblings answered before the pause already settled (and dropped)
        // their own votes, so only a resume of the whole original batch can
        // judge whether every call asked to terminate.
        self.settle_batch_termination(ctx, run, deferred.is_empty() && resumes_whole_batch);
        Ok(deferred)
    }

    /// Drains any pending [`MiddlewareControl`] and turns it into a loop
    /// decision.
    ///
    /// Returns [`ControlEffect::None`] when nothing was requested (or the
    /// pending request needed no loop-level action, e.g.
    /// [`MiddlewareControl::UpdateState`]), [`ControlEffect::ContinueLoop`]
    /// when the current turn must be abandoned in favor of a fresh iteration
    /// (`JumpTo(Model)`), and [`ControlEffect::Exit`] when the run is done.
    /// `Err` surfaces [`MiddlewareControl::Interrupt`]. Called at every safe
    /// checkpoint — the top of an iteration, after the model call, and after
    /// tool execution — so a control raised anywhere in a turn takes effect on
    /// that turn.
    pub(super) fn apply_pending_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
    ) -> Result<ControlEffect> {
        let Some(control) = ctx.take_control() else {
            return Ok(ControlEffect::None);
        };
        // `UpdateState` is applied (queued, really — see `RunContext::
        // push_state_update`) silently: it carries no loop-level decision, so
        // audit-logging it as a `ControlApplied` event alongside jumps and
        // stops would be noise. It still shows up wherever the host inspects
        // `RunContext::take_state_updates`.
        if let MiddlewareControl::UpdateState(update) = control {
            ctx.push_state_update(update);
            return Ok(ControlEffect::None);
        }
        let record = ctx.emit(AgentEvent::ControlApplied {
            control: control.kind().to_string(),
            detail: match &control {
                MiddlewareControl::Continue => String::new(),
                MiddlewareControl::JumpTo(target) => format!("{target:?}"),
                MiddlewareControl::UpdateState(_) => unreachable!("handled above"),
                MiddlewareControl::StopWithFinal(text) => text.clone(),
                MiddlewareControl::Interrupt { node, message } => format!("{node}: {message}"),
            },
        });
        status.set_last_event(record.id);
        match control {
            MiddlewareControl::Continue => Ok(ControlEffect::None),
            MiddlewareControl::UpdateState(_) => unreachable!("handled above"),
            MiddlewareControl::JumpTo(LoopTarget::Tools) => {
                // Tool execution already runs whenever the turn produced real
                // tool calls; there is nothing else to route to when it did
                // not. Either way, this is a no-op at the loop level.
                Ok(ControlEffect::None)
            }
            MiddlewareControl::JumpTo(LoopTarget::Model) => {
                // Abandon whatever the rest of this turn would have done
                // (typically: running tools the model just requested) and go
                // straight to a fresh model call. Close out any tool calls on
                // the last assistant row first so the transcript stays
                // replayable (see the `StopWithFinal` arm below for why).
                Self::close_unanswered_tool_calls(
                    messages,
                    "run jumped back to the model before this tool call was executed",
                );
                Ok(ControlEffect::ContinueLoop)
            }
            MiddlewareControl::JumpTo(LoopTarget::End) => {
                Self::close_unanswered_tool_calls(
                    messages,
                    "run stopped before this tool call was executed",
                );
                if run.final_response.is_none() {
                    let text = Self::last_assistant_text(messages);
                    run.final_response = Some(ModelResponse::assistant(text));
                }
                Ok(ControlEffect::Exit(LoopExit::Finished))
            }
            MiddlewareControl::StopWithFinal(text) => {
                // The most recently appended assistant row may carry
                // `tool_calls` that were never answered — e.g. a middleware
                // requesting `StopWithFinal` right after the model turn that
                // requested them, before `execute_tools` ever ran. Left as
                // is, `run.messages`/`messages` end with an assistant row
                // whose tool calls have no matching tool message, which a
                // provider rejects (400) if the transcript is ever replayed
                // (M-1). Append a synthetic tool result for each unanswered
                // call so the transcript stays replayable.
                Self::close_unanswered_tool_calls(
                    messages,
                    "run stopped before this tool call was executed",
                );
                run.final_response = Some(ModelResponse::assistant(text));
                Ok(ControlEffect::Exit(LoopExit::Finished))
            }
            MiddlewareControl::Interrupt { node, message } => {
                Err(TinyAgentsError::Interrupted { node, message })
            }
        }
    }

    /// The text of the most recent assistant message, or empty when there is
    /// none. Used to synthesize a final response for
    /// [`MiddlewareControl::JumpTo`]`(`[`LoopTarget::End`]`)`, which (unlike
    /// [`MiddlewareControl::StopWithFinal`]) carries no text of its own.
    fn last_assistant_text(messages: &[Message]) -> String {
        messages
            .iter()
            .rev()
            .find(|message| matches!(message, Message::Assistant(_)))
            .map(Message::text)
            .unwrap_or_default()
    }

    /// Appends a synthetic [`Message::tool`] result for every tool call on
    /// the last message that is still unanswered, so the transcript stays
    /// replayable through a provider that requires every `tool_calls` entry
    /// on an assistant message to have a matching tool result before the next
    /// turn (M-1). A no-op when the last message is not an unanswered
    /// assistant tool-call row.
    fn close_unanswered_tool_calls(messages: &mut Vec<Message>, reason: &str) {
        let Some(Message::Assistant(last)) = messages.last() else {
            return;
        };
        if last.tool_calls.is_empty() {
            return;
        }
        let synthetic: Vec<Message> = last
            .tool_calls
            .iter()
            .map(|call| Message::tool(call.id.clone(), reason))
            .collect();
        messages.extend(synthetic);
    }
}
