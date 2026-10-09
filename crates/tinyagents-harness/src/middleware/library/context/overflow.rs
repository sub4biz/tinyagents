//! Overflow recovery for [`ContextCompressionMiddleware`]: one compaction
//! attempt over a request the provider (or its response usage) says did not
//! fit. The retry loop around it lives in `wrap_model`.
//!
//! A child module of `context` so it shares that module's private fold
//! helpers (`remember_fold`, `finish_compaction`, the fingerprint chain).

use super::*;
use crate::artifacts::{
    reducible_tool_result_bytes, truncate_older_tool_results, truncate_tool_results,
};
use crate::summarization::{OverflowInfo, detect_response_overflow};

impl ContextCompressionMiddleware {
    /// Compacts `request` once for an overflow and returns the request to
    /// retry, or `None` when no compaction can help: nothing safe to cut, the
    /// `before_compaction` hook declined, the summarizer failed, or the
    /// result would not be smaller than `request`. `None` always leaves the
    /// run's fold and transcript untouched, so the caller surfaces the
    /// original failure. A compaction that grows the request is never used, and
    /// none is persisted as a fold or boundary; `strictly_smaller` (every
    /// attempt after a call's first) also rejects one that merely ties.
    pub(super) async fn compact_for_overflow<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
        overflow: OverflowInfo,
        strictly_smaller: bool,
        truncate: Option<usize>,
    ) -> Option<ModelRequest> {
        // Sizes are compared as sent: with a truncation cap in force the
        // provider sees the cut request, so a summary replacing an already-cut
        // result must beat that, not the raw base.
        let before_tokens = sent_tokens(&request.messages, truncate);
        // Checkpoints in the request (this run's fold summary, or one carried
        // in from an earlier turn) are the previous summary, never raw history
        // to summarize again.
        let mut stripped = request.messages.clone();
        let carried = take_checkpoints(&mut stripped);
        // The provider just rejected a request `before_model` judged to fit, so
        // the estimate cannot be trusted to size the kept tail: keep at most
        // half of what the transcript estimates to (and half the provider's
        // stated limit, when it gave one), never more than the trigger budget.
        // Keeping the full trigger budget found no cut at all whenever the
        // estimate was under the trigger, which is exactly when a provider
        // overflow is a surprise.
        let keep_recent_tokens = self
            .policy
            .trigger_budget()
            .min(total_message_tokens(&stripped) / 2)
            .min(overflow.limit.map_or(u64::MAX, |limit| limit / 2))
            .max(1);
        let Some(cut) = find_cut_point(
            &stripped,
            keep_recent_tokens,
            crate::token_estimation::estimate_message_tokens,
        ) else {
            // Nothing safe to cut (e.g. the whole transcript is already
            // within budget, or is a single indivisible tool-call pair) —
            // there is no compaction that could help, so surface the
            // original provider error.
            return None;
        };

        let (system, non_system) = partition_messages_system(&stripped);
        // This run's fold and the live transcript its `before_model` saw. The
        // fold extends in live-transcript coordinates only when this request's
        // non-system messages still line up one-to-one with that live
        // remainder: a later middleware dropping messages would shift every
        // index, and persisting a shifted boundary would corrupt a resumed
        // session.
        // With no `before_model` state for this run (compression installed only
        // as model middleware), nothing rewrote the request: it is the live
        // transcript itself.
        // (Over the request's own non-system messages, checkpoint included,
        // so a carried checkpoint the fold does not know about breaks the
        // alignment instead of shifting it.)
        let (prior, live_chain) = self.run_state(ctx.instance_id()).unwrap_or_else(|| {
            let (_, live) = partition_messages_system(&request.messages);
            (None, fingerprint_chain(&live))
        });
        let folded = prior.as_ref().map_or(0, |fold| fold.folded);
        // `before_model` splices the fold's pinned message in right after the
        // summary. It is a copy of a message inside the folded range, not part
        // of the live remainder, so it is set aside for the alignment check.
        let prior_pin = prior
            .as_ref()
            .and_then(|fold| fold.pinned.clone())
            .filter(|pin| non_system.first() == Some(&pin.message));
        let remainder = &non_system[usize::from(prior_pin.is_some())..];
        // Content, not just count: a later step that rewrites messages in place
        // (microcompact blanking tool bodies) keeps the count but changes what
        // a summary of this request would describe.
        let seed = folded
            .checked_sub(1)
            .and_then(|i| live_chain.get(i))
            .copied()
            .unwrap_or(0);
        let fold_extends = live_chain.len() >= folded
            && fingerprint_chain_from(seed, remainder) == live_chain[folded..];
        // The request's checkpoint (stripped above) is superseded by the new
        // one, which is built on it: the checkpoint is taken from this very
        // request, so it describes exactly the history before the cut whether
        // or not the request still aligns with the live transcript.
        let kept_system: Vec<Message> = system;
        let plan = crate::summarization::split_at_cut(
            kept_system.clone(),
            &non_system,
            cut.index,
            self.policy.pin_turn_user_message,
        );
        if plan.to_summarize.is_empty() {
            // Only the pinned message lay before the cut, and it stays pinned:
            // nothing to compact.
            return None;
        }
        let coords = LiveCoords {
            folded,
            prior_pin: prior_pin.as_ref().map(|pin| pin.live_index),
        };
        let new_folded = coords.live_index(plan.cut);
        let pinned = pinned_from_plan(&plan, kept_system.len(), &coords);
        let boundary = LiveBoundary {
            first_kept_index: new_folded,
            pinned_user_index: pinned.as_ref().map(|pin| pin.live_index),
        };
        let crate::summarization::CompactionPlan {
            to_summarize,
            to_keep,
            ..
        } = plan;
        let from_tokens = cut.tokens_before + cut.tokens_after;
        if !fold_extends {
            tracing::debug!(
                folded,
                live = live_chain.len(),
                request = non_system.len(),
                "[context_compression] overflow request no longer aligns with the live transcript; compacting without extending the fold or persisting a boundary"
            );
        }

        // The request's own checkpoint describes its history before the cut;
        // without one, the run's summaries only do when the request still
        // lines up with the live transcript (an unaligned request's summary
        // would describe altered history). Both the summarizer and a
        // `before_compaction` hook's own summary build on it, since the
        // checkpoint it came from is dropped from `to_keep` either way.
        let previous_summary = carried.as_ref().map(summary_text).or_else(|| {
            fold_extends
                .then(|| {
                    prior
                        .as_ref()
                        .map(|fold| summary_text(&fold.summary))
                        .or_else(|| self.run_last_summary(ctx.instance_id()))
                })
                .flatten()
        });

        match self.hook_decision(
            CompactionReason::Overflow,
            from_tokens,
            &to_summarize,
            &to_keep,
        ) {
            CompactionDecision::Decline => return None,
            CompactionDecision::Proceed => {}
            CompactionDecision::UseSummary(text) => {
                let text = previous_summary
                    .as_deref()
                    .map_or(text.clone(), |previous| format!("{previous}\n{text}"));
                let text = self.hook_summary_text(&to_summarize, text);
                let record = SummaryRecord {
                    summary: checkpoint_message(self.placement, &text),
                    provenance: crate::summarization::CompressionProvenance {
                        source_ids: Vec::new(),
                        original_token_estimate: 0,
                        summary_token_estimate: 0,
                        reason: "before_compaction hook supplied the summary".to_string(),
                    },
                    usage: None,
                };
                let new_messages = splice_summary(to_keep, record.summary.clone());
                let to_tokens = total_message_tokens(&new_messages);
                let sent_to_tokens = sent_tokens(&new_messages, truncate);
                if sent_to_tokens > before_tokens
                    || (strictly_smaller && sent_to_tokens == before_tokens)
                {
                    tracing::debug!(
                        before_tokens,
                        to_tokens,
                        "[context_compression] hook summary does not shrink the request; stopping overflow recovery"
                    );
                    return None;
                }
                if fold_extends {
                    self.remember_fold(
                        ctx.instance_id(),
                        new_folded,
                        &live_chain,
                        &record.summary,
                        pinned.clone(),
                    );
                }
                self.finish_compaction(
                    ctx,
                    record,
                    self.boundary_for_run(ctx.instance_id(), fold_extends.then_some(boundary)),
                    from_tokens,
                    to_tokens,
                    CompactionReason::Overflow,
                    None,
                );
                let mut retried = request.clone();
                retried.messages = new_messages;
                self.forget_pending(ctx.instance_id());
                return Some(retried);
            }
        }

        tracing::info!(
            to_summarize = to_summarize.len(),
            to_keep = to_keep.len(),
            from_tokens,
            "[context_compression] provider reported a context overflow; compacting and retrying"
        );
        let started = std::time::Instant::now();
        let record = match self
            .summarize_batch(ctx, &to_summarize, &to_keep, previous_summary)
            .await
        {
            Ok(record) => record,
            // Compaction itself failed: nothing changed, so surface the
            // original overflow rather than a confusing summarizer error.
            Err(_) => return None,
        };
        let latency_ms = elapsed_ms(started);
        let record = self.placed(record);

        let new_messages = splice_summary(to_keep, record.summary.clone());
        let to_tokens = total_message_tokens(&new_messages);
        let sent_to_tokens = sent_tokens(&new_messages, truncate);
        // A summary that grows the request (or, after the first attempt, does
        // not shrink it) cannot help, and a further attempt would only repeat
        // it: stop here, change nothing — no fold, no boundary, no record.
        if sent_to_tokens > before_tokens || (strictly_smaller && sent_to_tokens == before_tokens) {
            tracing::info!(
                before_tokens,
                to_tokens,
                "[context_compression] compaction did not shrink the request; stopping overflow recovery"
            );
            return None;
        }
        if fold_extends {
            self.remember_fold(
                ctx.instance_id(),
                new_folded,
                &live_chain,
                &record.summary,
                pinned,
            );
        }
        self.finish_compaction(
            ctx,
            record,
            self.boundary_for_run(ctx.instance_id(), fold_extends.then_some(boundary)),
            from_tokens,
            to_tokens,
            CompactionReason::Overflow,
            Some(latency_ms),
        );

        let mut retried = request.clone();
        retried.messages = new_messages;
        self.forget_pending(ctx.instance_id());
        Some(retried)
    }
}

/// Estimated tokens of `messages` as the provider would receive them: with
/// `truncate` set, oversized tool results are cut first.
fn sent_tokens(messages: &[Message], truncate: Option<usize>) -> u64 {
    match truncate {
        None => total_message_tokens(messages),
        Some(cap) => {
            let mut cut = messages.to_vec();
            truncate_tool_results(&mut cut, cap);
            total_message_tokens(&cut)
        }
    }
}

impl ContextCompressionMiddleware {
    /// The request to send for `base`: with its oversized tool results cut when
    /// a truncation cap is in force. Pure and idempotent, so a retry loop (or
    /// a later call of the same run) rebuilds the identical request.
    pub(super) fn outgoing_request(base: &ModelRequest, truncate: Option<usize>) -> ModelRequest {
        let mut outgoing = base.clone();
        if let Some(cap) = truncate {
            truncate_tool_results(&mut outgoing.messages, cap);
        }
        outgoing
    }

    /// Whether a model call's result reports a context overflow: an error the
    /// classifier recognizes, or a successful response whose usage / stop
    /// shows the window was exceeded (see [`ResponseOverflowDetection`]).
    pub(super) fn classify_outcome<Ctx: Send + Sync>(
        &self,
        ctx: &RunContext<Ctx>,
        result: &Result<MiddlewareModelOutcome>,
        base: &ModelRequest,
    ) -> Option<OverflowInfo> {
        match result {
            Err(error) => self.overflow_classifier.classify(error),
            // A replayed cached response says nothing about this request, and
            // a streamed one has already delivered its output: discarding it
            // would stream the answer twice.
            Ok(MiddlewareModelOutcome::Response(response))
                if !response.served_from_cache && !ctx.call_streamed =>
            {
                let reported = response
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.context_window_tokens);
                let window = reported.or(self.policy.context_window);
                let info = detect_response_overflow(
                    self.response_overflow,
                    response.usage.as_ref(),
                    response.finish_reason.as_deref(),
                    window,
                    base.max_tokens,
                );
                if info.is_some() && reported.is_none() {
                    tracing::warn!(
                        window = ?window,
                        "[context_compression] response overflow judged against the policy's context \
                         window; the provider reported none"
                    );
                }
                info
            }
            Ok(_) => None,
        }
    }

    /// Hands a discarded successful response's usage to the run so the spend
    /// is still accounted, and marks the discard on the event stream.
    pub(super) fn account_discarded<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        result: &Result<MiddlewareModelOutcome>,
    ) {
        let Ok(MiddlewareModelOutcome::Response(response)) = result else {
            return;
        };
        let Some(usage) = response.usage else {
            return;
        };
        ctx.record_discarded_usage(usage);
        ctx.emit(AgentEvent::Custom {
            call_id: ctx.active_model_call.clone(),
            payload: serde_json::json!({
                "type": "overflow_discarded_response",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }),
        });
        tracing::info!(
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            "[context_compression] discarding a response that reported a context overflow; usage accounted"
        );
    }

    /// The cheapest route for an overflow `overflow` reported against `base`.
    /// Without a truncation cap (or once truncation is already in force) the
    /// only remedy is compaction. The provider's word that the request did not
    /// fit outranks our estimate, so a "fits" verdict still compacts.
    pub(super) fn overflow_route(
        &self,
        base: &ModelRequest,
        already_truncated: bool,
        overflow: &OverflowInfo,
    ) -> CompactionRoute {
        let Some(cap) = self.tool_result_truncation.filter(|_| !already_truncated) else {
            return CompactionRoute::Compact;
        };
        let prompt = overflow
            .requested
            .unwrap_or_else(|| total_message_tokens(&base.messages) + schema_tokens(&base.tools));
        let budget = overflow
            .limit
            .unwrap_or_else(|| self.policy.trigger_budget());
        let reducible = tokens_for_bytes(reducible_tool_result_bytes(&base.messages, cap));
        match CompactionPressure::route(prompt, budget, reducible) {
            CompactionRoute::Fits => CompactionRoute::Compact,
            route => route,
        }
    }

    /// Cuts `base`'s oversized tool results to `cap` bytes when that removes
    /// anything, announcing it with [`AgentEvent::Compressed`] and switching
    /// the run into truncating mode (so its later requests stay cut). Returns
    /// whether anything was cut.
    pub(super) fn announce_truncation<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        base: &ModelRequest,
        cap: usize,
    ) -> bool {
        let truncated = Self::outgoing_request(base, Some(cap));
        let from_tokens = total_message_tokens(&base.messages);
        let to_tokens = total_message_tokens(&truncated.messages);
        if to_tokens >= from_tokens {
            return false;
        }
        self.engage_truncation(ctx.instance_id());
        ctx.mark_prompt_prefix_changed();
        tracing::info!(
            from_tokens,
            to_tokens,
            cap,
            "[context_compression] truncated oversized tool results in the request"
        );
        ctx.emit(AgentEvent::Compressed {
            from_tokens,
            to_tokens,
        });
        true
    }

    /// Re-applies the run's truncation to a request rebuilt from the full
    /// transcript, once a route has engaged it, sparing the results that follow
    /// the last assistant message. Silent (the engagement was
    /// announced) and idempotent.
    pub(super) fn apply_run_truncation<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &mut ModelRequest,
    ) {
        let Some(cap) = self.tool_result_truncation else {
            return;
        };
        let engaged = self
            .runs
            .lock()
            .expect("runs mutex poisoned")
            .get(&ctx.instance_id())
            .is_some_and(|state| state.truncating);
        if engaged {
            // The newest results answer the model's latest call; they stay
            // whole until a later assistant turn has had them.
            // A cut rewrites bytes the provider cached on the previous call,
            // so the prefix epoch moves (only when something was cut) and the
            // cache guard does not report the deliberate rewrite as a miss.
            if truncate_older_tool_results(&mut request.messages, cap).truncated > 0 {
                ctx.mark_prompt_prefix_changed();
            }
        }
    }

    /// After a `before_model` compaction on the mixed route, cuts every
    /// oversized tool result again: the splice rebuilt the request from the
    /// untruncated transcript and may keep the newest oversized result.
    pub(super) fn cut_after_compaction(&self, mixed: bool, messages: &mut [Message]) {
        if let Some(cap) = self.tool_result_truncation.filter(|_| mixed) {
            truncate_tool_results(messages, cap);
        }
    }

    /// The preemptive route decision for a request `prompt_tokens` big that is
    /// over the trigger. Engages truncation when the route calls for it and
    /// reports [`OverTrigger::Done`] only when truncation alone has measurably
    /// brought the request under budget (the caller then skips compaction).
    pub(super) fn route_over_trigger<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &mut ModelRequest,
        prompt_tokens: u64,
    ) -> OverTrigger {
        let Some(cap) = self.tool_result_truncation else {
            return OverTrigger::Compact;
        };
        let reducible = tokens_for_bytes(reducible_tool_result_bytes(&request.messages, cap));
        let route =
            CompactionPressure::route(prompt_tokens, self.policy.trigger_budget(), reducible);
        tracing::debug!(
            route = route.as_str(),
            prompt_tokens,
            reducible_tokens = reducible,
            trigger = self.policy.trigger_budget(),
            "[context_compression] preemptive route"
        );
        if !route.truncates() {
            return OverTrigger::Compact;
        }
        self.engage_truncation(ctx.instance_id());
        ctx.mark_prompt_prefix_changed();
        if route != CompactionRoute::TruncateToolResults {
            // Compaction may retain the newest oversized result. Truncate the
            // current request as well so the mixed route is safe even when the
            // subsequent compaction does not get below the provider limit.
            truncate_tool_results(&mut request.messages, cap);
            return OverTrigger::CompactThenTruncate;
        }
        let from_tokens = total_message_tokens(&request.messages);
        truncate_tool_results(&mut request.messages, cap);
        let to_tokens = total_message_tokens(&request.messages);
        tracing::info!(
            from_tokens,
            to_tokens,
            "[context_compression] truncated oversized tool results instead of summarizing"
        );
        ctx.emit(AgentEvent::Compressed {
            from_tokens,
            to_tokens,
        });
        // The estimate chose this route; the measured request decides whether
        // it sufficed. Still over the trigger: compact as well, and cut the
        // results again afterwards, since the compaction rebuilds the request
        // from the untruncated transcript.
        if to_tokens < from_tokens
            && !self
                .policy
                .exceeds_trigger(to_tokens + schema_tokens(&request.tools))
        {
            OverTrigger::Done
        } else {
            OverTrigger::CompactThenTruncate
        }
    }

    /// Marks `run` as truncating. Only a run `before_model` already tracks is
    /// marked: a run seen only here has no later `before_model` to apply it.
    pub(super) fn engage_truncation(&self, run: u64) {
        if let Some(state) = self.runs.lock().expect("runs mutex poisoned").get_mut(&run) {
            state.truncating = true;
        }
    }
}

/// What the preemptive route decided for an over-trigger request.
pub(super) enum OverTrigger {
    /// Truncation alone brought the request under the trigger.
    Done,
    /// Compact; no truncation was applied for this request.
    Compact,
    /// Compact, then cut the tool results again (truncation was applied but
    /// did not suffice, or only partly covers the overflow).
    CompactThenTruncate,
}

/// Tokens the estimator charges for `bytes` of text.
pub(super) fn tokens_for_bytes(bytes: usize) -> u64 {
    (bytes as f64 / crate::token_estimation::DEFAULT_CHARS_PER_TOKEN) as u64
}
