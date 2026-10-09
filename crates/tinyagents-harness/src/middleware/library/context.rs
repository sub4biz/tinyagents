//! Context-management middleware: message trimming, summarization-based
//! compression, and prompt-cache-layout guarding.
//!
//! Split out of `middleware/mod.rs`; see that module's doc comment for the
//! full middleware pipeline overview.

use super::compaction_pressure::CompactionRoute;
use super::*;
use crate::cache::{CacheLayoutEvent, PromptCacheLayout};
use crate::middleware::AgentRun;
use crate::middleware::types::CompactionPressure;
use crate::middleware::{
    CompressionFailurePolicy, ContextCompressionMiddleware, DEFAULT_CACHE_GUARD_EVENT_CAP,
    DEFAULT_COMPRESSION_RECORD_CAP, DEFAULT_MAX_OVERFLOW_ATTEMPTS, DEFAULT_THRASH_COOLDOWN_CALLS,
    DEFAULT_THRASH_STRIKES, MessageTrimMiddleware, MicrocompactMiddleware,
    PromptCacheGuardMiddleware,
};
use crate::summarization::{
    CompactionContext, CompactionDecision, CompactionReason, CompactionRecord, ConcatSummarizer,
    DefaultFileOpExtractor, FileOpExtractor, OverflowClassifier, ResponseOverflowDetection,
    SummarizationPolicy, Summarizer, SummaryPlacement, SummaryRecord, TrimStrategy,
    checkpoint_body, checkpoint_message, find_cut_point, is_checkpoint, trim_messages,
};

// ── MessageTrimMiddleware ─────────────────────────────────────────────────────

impl MessageTrimMiddleware {
    /// Creates a trim middleware using the given [`TrimStrategy`].
    pub fn new(strategy: TrimStrategy) -> Self {
        Self { strategy }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for MessageTrimMiddleware {
    fn name(&self) -> &str {
        "message_trim"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.messages = trim_messages(&request.messages, &self.strategy);
        Ok(())
    }
}

// ── ContextCompressionMiddleware ──────────────────────────────────────────────

/// Estimate the total tokens of a message slice.
///
/// Uses the crate's shared
/// [`count_tokens_approximately`][crate::token_estimation::count_tokens_approximately]
/// estimator rather than summing `estimate_tokens(&m.text())`. `text()` returns
/// only the *textual* content blocks, so a transcript of large JSON tool
/// results or image blocks estimated to nearly zero and the micro-compaction
/// budget gate never tripped on exactly the transcripts it exists to shrink.
/// The shared estimator charges every content block, tool call, and tool-call
/// id, and calibrates against reported usage metadata when it is present.
fn total_message_tokens(messages: &[tinyinference_llm::message::Message]) -> u64 {
    crate::token_estimation::count_tokens_approximately(messages)
}

impl ContextCompressionMiddleware {
    /// Creates a compression middleware backed by the default
    /// [`ConcatSummarizer`].
    pub fn new(policy: SummarizationPolicy) -> Self {
        Self::with_summarizer(policy, Box::new(ConcatSummarizer))
    }

    /// Creates a compression middleware with a custom [`Summarizer`].
    ///
    /// The failure policy defaults to
    /// [`CompressionFailurePolicy::FallbackTrim`]; override it with
    /// [`with_failure_policy`](Self::with_failure_policy).
    pub fn with_summarizer(policy: SummarizationPolicy, summarizer: Box<dyn Summarizer>) -> Self {
        Self {
            label: "context_compression",
            policy,
            summarizer,
            records: std::sync::Mutex::new(std::collections::VecDeque::new()),
            max_records: DEFAULT_COMPRESSION_RECORD_CAP,
            on_failure: CompressionFailurePolicy::default(),
            max_turn_tokens: None,
            overflow_classifier: OverflowClassifier::default(),
            before_compaction: None,
            runs: std::sync::Mutex::new(std::collections::HashMap::new()),
            placement: SummaryPlacement::default(),
            thrash_strikes: DEFAULT_THRASH_STRIKES,
            thrash_cooldown_calls: DEFAULT_THRASH_COOLDOWN_CALLS,
            keep_recent_tokens: None,
            max_overflow_attempts: DEFAULT_MAX_OVERFLOW_ATTEMPTS,
            response_overflow: ResponseOverflowDetection::default(),
            tool_result_truncation: None,
            split_turn_prefix: true,
            file_ops: Some(std::sync::Arc::new(DefaultFileOpExtractor)),
        }
    }

    /// Replaces the extractor that derives the file lists appended to every
    /// compaction summary (`<read-files>` / `<modified-files>`, carried
    /// forward across compactions). The default,
    /// [`DefaultFileOpExtractor`], reads the `path` / `file` / `file_path` /
    /// `paths` arguments and classifies the call by tool name.
    pub fn with_file_op_extractor(mut self, extractor: impl FileOpExtractor + 'static) -> Self {
        self.file_ops = Some(std::sync::Arc::new(extractor));
        self
    }

    /// Chooses whether a compaction cut inside a turn summarizes that turn's
    /// prefix with its own request (default `true`). The prefix is an *extra*
    /// summarizer call on top of the history's (still bounded by
    /// [`with_max_turn_tokens`][Self::with_max_turn_tokens]); turn it off to
    /// summarize the folded messages in one call as before.
    pub fn with_split_turn_prefix(mut self, enabled: bool) -> Self {
        self.split_turn_prefix = enabled;
        self
    }

    /// Stops appending file lists to compaction summaries.
    pub fn without_file_operations(mut self) -> Self {
        self.file_ops = None;
        self
    }

    /// Sets how many compaction (or truncation) attempts one model call may
    /// make after the provider reports a context overflow. Each attempt must
    /// shrink the request or recovery stops. Defaults to
    /// [`DEFAULT_MAX_OVERFLOW_ATTEMPTS`]; `0` disables overflow recovery.
    pub fn with_max_overflow_attempts(mut self, attempts: u32) -> Self {
        self.max_overflow_attempts = attempts;
        self
    }

    /// Chooses which *successful-response* signals count as an overflow, in
    /// addition to the errors [`OverflowClassifier`] recognizes. Defaults to
    /// [`ResponseOverflowDetection::Off`]: successful responses are not
    /// inspected. Opt into [`ResponseOverflowDetection::Usage`] (usage above
    /// the window, and a zero-output `length` stop with the window full) or
    /// [`ResponseOverflowDetection::UsageAndShortLength`]. Both need a known
    /// context window ([`SummarizationPolicy::context_window`], or the
    /// response's own `usage.context_window_tokens`).
    pub fn with_response_overflow_detection(
        mut self,
        detection: ResponseOverflowDetection,
    ) -> Self {
        self.response_overflow = detection;
        self
    }

    /// Enables the cheap overflow route: when a request is over budget (before
    /// the call, or after the provider reports an overflow) and cutting every
    /// tool result longer than `max_bytes` would cover the overflow with
    /// margin, the results are cut and no summary is bought. When cutting
    /// would only partly cover it, compaction runs first and the cut follows.
    ///
    /// Opt-in: unset (the default) never truncates. The cut rewrites only the
    /// outgoing request; the transcript keeps the full results, and a run that
    /// has truncated once keeps truncating, so its prompt prefix stays
    /// byte-stable within a run, apart from each result being cut once when a
    /// later assistant turn follows it (the newest results are spared). Results flagged `trusted_verbatim` are never cut.
    pub fn with_tool_result_truncation(mut self, max_bytes: usize) -> Self {
        self.tool_result_truncation = Some(max_bytes).filter(|bytes| *bytes > 0);
        self
    }

    /// Keeps the most recent `tokens` of history verbatim at each compaction
    /// (cut at a user or assistant message), instead of the policy's
    /// `keep_last` message count. A fixed message count keeps almost nothing
    /// when recent turns are large tool results and too much when they are
    /// short; a token budget keeps a predictable working set. 20k is the
    /// measured default for coding agents with large windows; scale it to
    /// about a fifth of a small model's window.
    pub fn with_keep_recent_tokens(mut self, tokens: u64) -> Self {
        self.keep_recent_tokens = Some(tokens);
        self
    }

    /// Sets the role the summary is written with. Defaults to
    /// [`SummaryPlacement::User`]: a reference-only checkpoint after the
    /// system prompt that leaves the system prompt and tool declarations
    /// byte-stable. [`SummaryPlacement::System`] restores the original
    /// system-role summary.
    pub fn with_summary_placement(mut self, placement: SummaryPlacement) -> Self {
        self.placement = placement;
        self
    }

    /// Configures the anti-thrash guard: after `strikes` compactions in a row
    /// whose next provider-reported prompt is still at or above the trigger,
    /// summarization is suppressed for `cooldown_calls` model calls and the
    /// request is trimmed deterministically instead. Defaults to
    /// [`DEFAULT_THRASH_STRIKES`] and [`DEFAULT_THRASH_COOLDOWN_CALLS`];
    /// `strikes == 0` disables the guard.
    pub fn with_thrash_guard(mut self, strikes: u32, cooldown_calls: u32) -> Self {
        self.thrash_strikes = strikes;
        self.thrash_cooldown_calls = cooldown_calls;
        self
    }

    /// Returns the configured [`SummaryPlacement`].
    pub fn summary_placement(&self) -> SummaryPlacement {
        self.placement
    }

    /// Sets the token budget above which a single turn handed to the
    /// summarizer is split into two halves and merged (see
    /// [`summarize_with_split`][crate::summarization::summarize_with_split]). Unset (the default) never splits.
    pub fn with_max_turn_tokens(mut self, max_turn_tokens: u64) -> Self {
        self.max_turn_tokens = Some(max_turn_tokens);
        self
    }

    /// Replaces the [`OverflowClassifier`] consulted by the
    /// overflow → compact → retry recovery path (this middleware's
    /// [`ModelMiddleware::wrap_model`] implementation). Defaults to
    /// [`OverflowClassifier::default`]'s built-in provider patterns.
    pub fn with_overflow_classifier(mut self, classifier: OverflowClassifier) -> Self {
        self.overflow_classifier = classifier;
        self
    }

    /// Installs a `before_compaction` hook consulted before every compaction
    /// this middleware runs (proactive threshold or reactive overflow
    /// recovery). See [`CompactionDecision`].
    pub fn with_before_compaction(
        mut self,
        hook: impl Fn(&CompactionContext) -> CompactionDecision + Send + Sync + 'static,
    ) -> Self {
        self.before_compaction = Some(std::sync::Arc::new(hook));
        self
    }

    /// Sets the [`CompressionFailurePolicy`] applied when the [`Summarizer`]
    /// returns an `Err`. Defaults to
    /// [`CompressionFailurePolicy::FallbackTrim`].
    pub fn with_failure_policy(mut self, on_failure: CompressionFailurePolicy) -> Self {
        self.on_failure = on_failure;
        self
    }

    /// Returns the configured [`CompressionFailurePolicy`].
    pub fn failure_policy(&self) -> CompressionFailurePolicy {
        self.on_failure
    }

    /// Sets the maximum number of [`SummaryRecord`]s retained before the
    /// oldest is evicted. `0` disables recording entirely.
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = max_records;
        let mut records = self.records.lock().expect("records mutex poisoned");
        while records.len() > max_records {
            records.pop_front();
        }
        drop(records);
        self
    }

    /// Returns the configured [`SummarizationPolicy`].
    pub fn policy(&self) -> &SummarizationPolicy {
        &self.policy
    }

    /// Returns the [`SummaryRecord`]s produced so far, in order. Bounded to
    /// at most [`ContextCompressionMiddleware::with_max_records`] entries
    /// (default [`DEFAULT_COMPRESSION_RECORD_CAP`]); older records are
    /// evicted first.
    pub fn records(&self) -> Vec<SummaryRecord> {
        self.records
            .lock()
            .expect("records mutex poisoned")
            .iter()
            .cloned()
            .collect()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ContextCompressionMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    /// Hands the finished run's compaction to the host, then drops the run's
    /// state: the fold describes that run's transcript only.
    ///
    /// When the run compacted, [`AgentRun::compacted_history`] is set to the
    /// transcript the next call would have seen, so a host that carries it
    /// into its next turn starts from the checkpoint instead of re-reading
    /// (and re-summarizing) everything this run already folded.
    async fn after_agent(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        run: &mut AgentRun,
    ) -> Result<()> {
        let state = self
            .runs
            .lock()
            .expect("runs mutex poisoned")
            .remove(&ctx.instance_id());
        if let Some(fold) = state.and_then(|state| state.fold)
            && let Some(history) = compacted_history(&fold, &run.messages)
        {
            tracing::info!(
                run_id = %ctx.run_id(),
                full = run.messages.len(),
                compacted = history.len(),
                "[context_compression] run ends compacted; exposing the folded transcript \
                 as AgentRun::compacted_history"
            );
            run.compacted_history = Some(history);
        }
        Ok(())
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        let suppressed = touch_run(
            &mut self.runs.lock().expect("runs mutex poisoned"),
            ctx.instance_id(),
        )
        .pressure
        .begin_call();
        let result = self.compress_request(ctx, request, suppressed).await;
        self.apply_run_truncation(ctx, request);
        // Whatever the outcome, this is the request whose provider usage the
        // next `after_model` attributes.
        let schema_tokens = schema_tokens(&request.tools);
        if let Some(state) = self
            .runs
            .lock()
            .expect("runs mutex poisoned")
            .get_mut(&ctx.instance_id())
        {
            state
                .pressure
                .note_request(&request.messages, schema_tokens);
        }
        result
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        if let Some(state) = self
            .runs
            .lock()
            .expect("runs mutex poisoned")
            .get_mut(&ctx.instance_id())
        {
            state.pressure.observe(
                response.usage.as_ref(),
                &self.policy,
                self.thrash_strikes,
                self.thrash_cooldown_calls,
            );
        }
        Ok(())
    }
}

impl ContextCompressionMiddleware {
    /// The `before_model` body: re-apply this run's fold, then compact further
    /// when the request is still over the trigger.
    async fn compress_request<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &mut ModelRequest,
        suppressed: bool,
    ) -> Result<()> {
        // The loop rebuilds every request from its full working transcript, so
        // re-apply the compaction this instance already made before deciding
        // anything (see `CompactionFold`). Without this, every call after the
        // first compaction re-crossed the threshold and re-summarized the
        // whole history.
        let (mut system, live) = partition_messages_system(&request.messages);
        let chain = fingerprint_chain(&live);
        let check = self.check_fold(ctx.instance_id(), &chain, &mut system);
        // A system-role checkpoint (`SummaryPlacement::System`) the host
        // carried in from an earlier turn: it is the previous summary, never an
        // instruction to keep beside a newer one.
        let inherited_system = take_checkpoints(&mut system);
        let mut prior = match &check {
            FoldCheck::Applies(fold) => Some((**fold).clone()),
            _ => None,
        };
        // A user-role checkpoint opening the live transcript was written by an
        // earlier compaction whose result the host persisted (see
        // `AgentRun::compacted_history`), or spliced in itself. Adopt it as a
        // fold over itself: it is refined as the previous summary rather than
        // summarized as raw history, and later folds extend it in live
        // coordinates.
        let mut adopted = false;
        if prior.is_none()
            && let (Some(first), Some(_)) = (live.first(), chain.first())
            && is_checkpoint(first)
        {
            tracing::debug!(
                live = live.len(),
                "[context_compression] adopting the transcript's checkpoint as the prior summary"
            );
            self.remember_fold(ctx.instance_id(), 1, &chain, first, None);
            prior = Some(crate::middleware::types::CompactionFold {
                folded: 1,
                fingerprint: chain[0],
                summary: first.clone(),
                pinned: None,
                replaces: None,
            });
            adopted = true;
        }
        let folded = prior.as_ref().map_or(0, |fold| fold.folded);
        // What stands in for everything before the unfolded remainder: this
        // run's fold, or the same summary when the host spliced it into its
        // transcript itself (it was lifted out of `system` above, so the new
        // summary replaces it instead of sitting beside it). An adopted
        // checkpoint (persisted by the host from an earlier turn, or spliced
        // in by it) is that summary in user-role form. A transcript that no
        // longer matches the fold gets no previous summary from it: that
        // summary describes some other history. Otherwise fall back to a
        // system checkpoint the host carried in, then to the last summary
        // this run produced. Both the summarizer and a `before_compaction`
        // hook's own summary build on it.
        let previous_summary = match &check {
            _ if adopted => prior.as_ref().map(|fold| summary_text(&fold.summary)),
            FoldCheck::Applies(fold) => Some(summary_text(&fold.summary)),
            FoldCheck::HostApplied(text) => Some(strip_checkpoint_marker(text)),
            FoldCheck::Stale => inherited_system.as_ref().map(summary_text),
            FoldCheck::None => inherited_system
                .as_ref()
                .map(summary_text)
                .or_else(|| self.run_last_summary(ctx.instance_id())),
        };
        // A user message the fold pinned rides right after its summary: it is
        // a verbatim copy of a message inside the folded range.
        let prior_pin = prior.as_ref().and_then(|fold| fold.pinned.clone());
        if let Some(fold) = &prior {
            request.messages = splice_summary(
                system
                    .iter()
                    .cloned()
                    .chain(prior_pin.iter().map(|pin| pin.message.clone()))
                    .chain(live[folded..].iter().cloned())
                    .collect(),
                fold.summary.clone(),
            );
        }

        // A run that already truncated keeps its tool results cut, before the
        // size is measured, so the measurement matches what was sent.
        self.apply_run_truncation(ctx, request);

        // Below the threshold: pass through (no new compaction, no event).
        // Prefer the provider's own count of the previous request plus an
        // estimate of what was appended since; the tool declarations count
        // either way, since they ride along on every request.
        let schema_tokens = schema_tokens(&request.tools);
        let (prompt_tokens, source) = touch_run(
            &mut self.runs.lock().expect("runs mutex poisoned"),
            ctx.instance_id(),
        )
        .pressure
        .prompt_tokens(&request.messages, schema_tokens);
        if !self.policy.exceeds_trigger(prompt_tokens) {
            return Ok(());
        }
        tracing::debug!(
            prompt_tokens,
            source = source.as_str(),
            trigger = self.policy.trigger_budget(),
            "[context_compression] request over the trigger"
        );

        // The cheaper route first: cutting oversized tool results can cover the
        // overflow without a summary.
        let mixed = match self.route_over_trigger(ctx, request, prompt_tokens) {
            OverTrigger::Done => return Ok(()),
            OverTrigger::Compact => false,
            OverTrigger::CompactThenTruncate => true,
        };

        // Anti-thrash: summaries that did not bring the prompt under the
        // trigger are not bought again during the cooldown.
        if suppressed {
            tracing::info!(
                prompt_tokens,
                "[context_compression] summarization suppressed by the anti-thrash guard; \
                 trimming deterministically"
            );
            let from_tokens = total_message_tokens(&request.messages);
            self.trim_to_trigger(ctx, request, from_tokens);
            return Ok(());
        }

        // Plan over the unfolded remainder only: the folded prefix is already
        // represented by the prior summary, which reaches the summarizer as
        // `previous_summary` instead of as messages to re-read. A message the
        // fold pinned is planned again: it stays pinned while no newer user
        // message takes over, and is summarized once one does.
        let unfolded: Vec<Message> = system
            .iter()
            .cloned()
            .chain(prior_pin.iter().map(|pin| pin.message.clone()))
            .chain(live[folded..].iter().cloned())
            .collect();
        let plan = match self.keep_recent_tokens {
            Some(tokens) => self.policy.plan_split_recent_tokens(&unfolded, tokens),
            None => self.policy.plan_split(&unfolded),
        };
        // Nothing old enough to compress (e.g. keep_last covers everything):
        // keep the request as it stands rather than summarizing an empty set.
        if plan.to_summarize.is_empty() {
            return Ok(());
        }

        let from_tokens = total_message_tokens(&request.messages);
        // The record's `first_kept_index` is in live-transcript coordinates —
        // what a session-backed `CompactionSink` maps to an entry id — see
        // `compaction::CompactionRecord::first_kept_index`. The plan's indices
        // are into `[prior pin?, live[folded..]...]`; a pinned message out of
        // the middle of the head means the tail does not start at
        // `folded + to_summarize.len()`.
        let coords = LiveCoords {
            folded,
            prior_pin: prior_pin.as_ref().map(|pin| pin.live_index),
        };
        let first_kept_index = coords.live_index(plan.cut);
        let pinned = pinned_from_plan(&plan, system.len(), &coords);
        let pinned_index = pinned.as_ref().map(|pin| pin.live_index);
        if let Some(pin) = &pinned {
            tracing::debug!(
                folded,
                first_kept_index,
                pinned = pin.live_index,
                "[context_compression] pinning the turn's user message across the compaction"
            );
        }
        let crate::summarization::CompactionPlan {
            to_summarize,
            to_keep,
            ..
        } = plan;

        match self.hook_decision(
            CompactionReason::Threshold,
            from_tokens,
            &to_summarize,
            &to_keep,
        ) {
            CompactionDecision::Decline => return Ok(()),
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
                self.remember_fold(
                    ctx.instance_id(),
                    first_kept_index,
                    &chain,
                    &record.summary,
                    pinned.clone(),
                );
                let mut new_messages = splice_summary(to_keep, record.summary.clone());
                self.cut_after_compaction(mixed, &mut new_messages);
                let to_tokens = total_message_tokens(&new_messages);
                self.finish_compaction(
                    ctx,
                    record,
                    self.boundary_for_run(
                        ctx.instance_id(),
                        Some(LiveBoundary {
                            first_kept_index,
                            pinned_user_index: pinned_index,
                        }),
                    ),
                    from_tokens,
                    to_tokens,
                    CompactionReason::Threshold,
                    None,
                );
                request.messages = new_messages;
                ctx.emit(AgentEvent::Compressed {
                    from_tokens,
                    to_tokens,
                });
                return Ok(());
            }
            CompactionDecision::Proceed => {}
        }

        tracing::debug!(
            folded,
            to_summarize = to_summarize.len(),
            to_keep = to_keep.len(),
            from_tokens,
            incremental = previous_summary.is_some(),
            "[context_compression] compacting"
        );
        let started = std::time::Instant::now();
        let record = match self
            .summarize_batch(ctx, &to_summarize, &to_keep, previous_summary)
            .await
        {
            Ok(record) => record,
            Err(err) => {
                // A summarizer failure hits precisely the longest, most valuable
                // transcripts (the ones that reached the compaction threshold).
                // Emit a diagnostic and recover per the configured
                // `CompressionFailurePolicy` instead of aborting the whole run.
                // The prior fold (if any) stays applied to the request: it is
                // still a valid summary of the prefix it covers.
                ctx.emit(AgentEvent::MiddlewareFailed {
                    name: self.label.to_string(),
                    error: err.to_string(),
                });
                match self.on_failure {
                    // Legacy behaviour: propagate and let the run fail.
                    CompressionFailurePolicy::Abort => return Err(err),
                    // Keep the (over-threshold) transcript verbatim and continue.
                    CompressionFailurePolicy::PassThrough => return Ok(()),
                    // Deterministic front-drop to the policy's trigger budget,
                    // preserving system messages (see `TrimStrategy::MaxTokens`).
                    // The trigger itself now charges tool schemas
                    // (`should_summarize_with_tools` above), so the message
                    // budget here must reserve that same schema cost first —
                    // otherwise a request whose schemas already consume a
                    // meaningful share of `trigger_budget` (or all of it)
                    // would still trim messages to the *full* budget and can
                    // remain above the threshold after trimming, with no
                    // further recovery possible.
                    CompressionFailurePolicy::FallbackTrim => {
                        self.trim_to_trigger(ctx, request, from_tokens);
                        return Ok(());
                    }
                }
            }
        };
        let latency_ms = elapsed_ms(started);
        let record = self.placed(record);

        // `plan` returns `to_keep` as `[system prompts..., recent turns...]`.
        // `splice_summary` inserts the summary *after* the leading system
        // prompts, not at index 0: a system prompt must stay first so its
        // persistent instructions keep priority and the cacheable prefix is
        // not churned. The summary of the elided older turns then sits
        // between the system prompt and the kept recent turns, in
        // chronological position. It replaces the prior fold's summary, which
        // it was built on.
        self.remember_fold(
            ctx.instance_id(),
            first_kept_index,
            &chain,
            &record.summary,
            pinned,
        );
        let mut new_messages = splice_summary(to_keep, record.summary.clone());
        self.cut_after_compaction(mixed, &mut new_messages);
        let to_tokens = total_message_tokens(&new_messages);

        self.finish_compaction(
            ctx,
            record,
            self.boundary_for_run(
                ctx.instance_id(),
                Some(LiveBoundary {
                    first_kept_index,
                    pinned_user_index: pinned_index,
                }),
            ),
            from_tokens,
            to_tokens,
            CompactionReason::Threshold,
            Some(latency_ms),
        );
        request.messages = new_messages;

        ctx.emit(AgentEvent::Compressed {
            from_tokens,
            to_tokens,
        });
        Ok(())
    }
}

mod overflow;
use overflow::OverTrigger;
mod summary;

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx>
    for ContextCompressionMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    /// Overflow recovery v2 (pi's overflow → compact → retry, extended): runs
    /// the wrapped model call and, when it reports a context overflow, shrinks
    /// the request and retries the *same* turn, up to
    /// [`with_max_overflow_attempts`][ContextCompressionMiddleware::with_max_overflow_attempts]
    /// times (default [`DEFAULT_MAX_OVERFLOW_ATTEMPTS`][crate::middleware::DEFAULT_MAX_OVERFLOW_ATTEMPTS]).
    ///
    /// An overflow is either an error
    /// [`Self::overflow_classifier`][ContextCompressionMiddleware] classifies,
    /// or a *successful* response whose usage / stop shows the window was
    /// exceeded (see [`ResponseOverflowDetection`]).
    ///
    /// Each attempt takes the cheapest step that can help: with
    /// [`with_tool_result_truncation`][ContextCompressionMiddleware::with_tool_result_truncation]
    /// configured, oversized tool results are cut first (no model call spent
    /// on a summary); otherwise, or when that cannot cover the overflow, the
    /// transcript is compacted ([`CompactionReason::Overflow`]). Every attempt
    /// after the first must produce a smaller request than the one before;
    /// one that cannot (nothing safe to cut, the `before_compaction` hook
    /// declined, the summary is no smaller) ends recovery and the original error — or
    /// response — is returned, so a transcript that cannot be shrunk under the
    /// window cannot loop. The transcript itself is never rewritten: the
    /// shrunk request is a rewrite, and compactions extend the run's
    /// fingerprint-chained fold exactly as `before_model` compactions do.
    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        // `base` holds the compactions so far; this loop never truncates it, so
        // a later compaction summarizes the results as it received them.
        // `truncate` is applied on top of it when sending. (A request that
        // `before_model` already cut in truncating mode arrives cut.)
        let mut base = request;
        let mut truncate: Option<usize> = None;
        let mut attempts = 0u32;
        loop {
            let outgoing = Self::outgoing_request(&base, truncate);
            let result = next.run(ctx, state, outgoing).await;
            let Some(overflow) = self.classify_outcome(ctx, &result, &base) else {
                return result;
            };
            if attempts >= self.max_overflow_attempts {
                return result;
            }
            attempts += 1;
            let route = self.overflow_route(&base, truncate.is_some(), &overflow);
            tracing::info!(
                attempt = attempts,
                max = self.max_overflow_attempts,
                route = route.as_str(),
                requested = ?overflow.requested,
                limit = ?overflow.limit,
                "[context_compression] context overflow reported; recovering"
            );
            let cap = self.tool_result_truncation;
            if route == CompactionRoute::TruncateToolResults
                && let Some(cap) = cap
                && self.announce_truncation(ctx, &base, cap)
            {
                self.account_discarded(ctx, &result);
                truncate = Some(cap);
                continue;
            }
            match self
                .compact_for_overflow(ctx, &base, overflow, attempts > 1, truncate)
                .await
            {
                Some(shrunk) => {
                    self.account_discarded(ctx, &result);
                    base = shrunk;
                    if route.truncates()
                        && let Some(cap) = cap
                        && truncate.is_none()
                        && self.announce_truncation(ctx, &base, cap)
                    {
                        truncate = Some(cap);
                    }
                }
                None => {
                    // Nothing to compact: cutting the tool results may still
                    // be enough, so it is the last resort for a mixed route.
                    if route.truncates()
                        && truncate.is_none()
                        && let Some(cap) = cap
                        && self.announce_truncation(ctx, &base, cap)
                    {
                        self.account_discarded(ctx, &result);
                        truncate = Some(cap);
                        continue;
                    }
                    return result;
                }
            }
        }
    }
}

/// Inserts `summary` into `to_keep` right after any leading system messages,
/// so a system prompt stays first (preserving both its instruction priority
/// and the cacheable prefix) and the summary sits chronologically between it
/// and the kept recent turns.
fn splice_summary(mut to_keep: Vec<Message>, summary: Message) -> Vec<Message> {
    let system_prefix = to_keep
        .iter()
        .take_while(|m| matches!(m, Message::System(_)))
        .count();
    let recent = to_keep.split_off(system_prefix);
    let mut new_messages = Vec::with_capacity(to_keep.len() + recent.len() + 1);
    new_messages.append(&mut to_keep);
    new_messages.push(summary);
    new_messages.extend(recent);
    new_messages
}

/// Maps an index into the slice a compaction planned over —
/// `[prior pinned message?, live[folded..]...]` — back to the live (pre-fold)
/// non-system transcript, the coordinates every persisted boundary uses.
struct LiveCoords {
    /// Live messages the prior fold already stands in for.
    folded: usize,
    /// Live index of the prior fold's pinned message, which heads the planned
    /// slice when present.
    prior_pin: Option<usize>,
}

impl LiveCoords {
    fn live_index(&self, planned: usize) -> usize {
        match self.prior_pin {
            Some(pin) if planned == 0 => pin,
            Some(_) => self.folded + planned - 1,
            None => self.folded + planned,
        }
    }
}

/// Where a compaction's verbatim tail starts in the live (pre-fold) non-system
/// transcript, and where the user message it pinned out of the folded range
/// sits, if it pinned one. What a [`CompactionRecord`] persists.
#[derive(Clone, Copy, Debug)]
struct LiveBoundary {
    first_kept_index: usize,
    pinned_user_index: Option<usize>,
}

/// The message `plan` pinned, as the fold remembers it. `system_len` is how
/// many system messages lead `plan.to_keep`; the pinned message follows them.
fn pinned_from_plan(
    plan: &crate::summarization::CompactionPlan,
    system_len: usize,
    coords: &LiveCoords,
) -> Option<crate::middleware::types::PinnedTurnMessage> {
    let index = plan.pinned?;
    let message = plan.to_keep.get(system_len)?.clone();
    debug_assert!(matches!(message, Message::User(_)));
    Some(crate::middleware::types::PinnedTurnMessage {
        live_index: coords.live_index(index),
        message,
    })
}

/// [`crate::summarization::pairing`] partitions operate on non-system
/// slices; this mirrors that split for callers outside the `summarization`
/// module (`compaction::find_cut_point` already partitions internally, but
/// its caller here also needs the same partition to rebuild `to_keep`).
fn partition_messages_system(messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
    let system = messages
        .iter()
        .filter(|m| matches!(m, Message::System(_)))
        .cloned()
        .collect();
    let non_system = messages
        .iter()
        .filter(|m| !matches!(m, Message::System(_)))
        .cloned()
        .collect();
    (system, non_system)
}

/// How a run's stored fold relates to the transcript of the current request.
enum FoldCheck {
    /// The run has no fold.
    None,
    /// The transcript still starts with the folded messages: re-apply it.
    Applies(Box<crate::middleware::types::CompactionFold>),
    /// The host replaced the folded messages with this summary itself.
    HostApplied(String),
    /// The transcript is some other history; the fold was dropped.
    Stale,
}

/// `run`'s state, created if needed and stamped as most recently used. Evicts
/// the least recently used run once [`MAX_TRACKED_COMPACTION_RUNS`] are held.
fn touch_run(
    runs: &mut std::collections::HashMap<u64, crate::middleware::types::RunCompaction>,
    run: u64,
) -> &mut crate::middleware::types::RunCompaction {
    let stamp = runs.values().map(|state| state.touched).max().unwrap_or(0) + 1;
    if !runs.contains_key(&run)
        && runs.len() >= crate::middleware::types::MAX_TRACKED_COMPACTION_RUNS
        && let Some(oldest) = runs
            .iter()
            .min_by_key(|(_, state)| state.touched)
            .map(|(id, _)| *id)
    {
        runs.remove(&oldest);
    }
    let state = runs.entry(run).or_default();
    state.touched = stamp;
    state
}

/// Chained fingerprints of `messages`: entry `i` identifies the whole prefix
/// `messages[..=i]`, so one comparison checks that a remembered prefix is still
/// intact. Hashes each message's serialized form, so any edit to an earlier
/// message changes every later entry.
pub(super) fn fingerprint_chain(messages: &[Message]) -> Vec<u64> {
    fingerprint_chain_from(0, messages)
}

/// [`fingerprint_chain`] continued from `seed`, the chain value of whatever
/// precedes `messages`. `fingerprint_chain_from(chain[k - 1], &m[k..])` equals
/// `chain[k..]` exactly when `m[k..]` is unchanged.
fn fingerprint_chain_from(seed: u64, messages: &[Message]) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    let mut prev = seed;
    messages
        .iter()
        .map(|message| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prev.hash(&mut hasher);
            match serde_json::to_vec(message) {
                Ok(bytes) => bytes.hash(&mut hasher),
                // Unserializable content still has to move the chain; its
                // debug form is stable for the life of the process.
                Err(_) => format!("{message:?}").hash(&mut hasher),
            }
            prev = hasher.finish();
            prev
        })
        .collect()
}

impl ContextCompressionMiddleware {
    /// Consults the `before_compaction` hook, when one is installed;
    /// defaults to [`CompactionDecision::Proceed`] otherwise.
    fn hook_decision(
        &self,
        reason: CompactionReason,
        tokens_before: u64,
        to_summarize: &[Message],
        to_keep: &[Message],
    ) -> CompactionDecision {
        match &self.before_compaction {
            Some(hook) => hook(&CompactionContext {
                reason,
                tokens_before,
                to_summarize_count: to_summarize.len(),
                to_keep_count: to_keep.len(),
            }),
            None => CompactionDecision::Proceed,
        }
    }

    /// Records `chain` as `run`'s live transcript and classifies its fold.
    ///
    /// When the fold no longer matches but `system` carries its summary, the
    /// host spliced the summary into its own transcript: the summary is lifted
    /// out of `system` and returned so the next compaction replaces it. A
    /// user-role checkpoint the host spliced in as its first live message is
    /// reported the same way; the caller adopts it from the live transcript.
    fn check_fold(&self, run: u64, chain: &[u64], system: &mut Vec<Message>) -> FoldCheck {
        let mut runs = self.runs.lock().expect("runs mutex poisoned");
        let state = touch_run(&mut runs, run);
        state.live_chain = chain.to_vec();
        let Some(fold) = state.fold.clone() else {
            return FoldCheck::None;
        };
        if fold.folded > 0 && chain.get(fold.folded - 1) == Some(&fold.fingerprint) {
            if let Some(replaced) = &fold.replaces {
                system.retain(|message| message != replaced);
            }
            return FoldCheck::Applies(Box::new(fold));
        }
        if let Some(at) = system.iter().position(|m| *m == fold.summary) {
            // Keep the fold: every later call carrying this summary must be
            // recognized too, until a compaction replaces it, or a below-
            // threshold call here would forget it and a later compaction would
            // send the old summary beside the new one built on it.
            system.remove(at);
            // The host's transcript has its own coordinate space. Keep the
            // summary available for replacement, but stop claiming that it
            // covers a prefix of the new live transcript.
            // The host's transcript carries the pinned message itself, too.
            if let Some(host_fold) = state.fold.as_mut() {
                host_fold.folded = 0;
                host_fold.pinned = None;
            }
            state.host_applied_summary = Some(fold.summary.clone());
            state.boundary_unaligned = true;
            tracing::debug!("[context_compression] host carries the fold summary itself");
            return FoldCheck::HostApplied(fold.summary.text());
        }
        // The same, for a user-role checkpoint: the host spliced it in as the
        // first live message, so it is part of the live transcript rather
        // than of `system`. The caller adopts it as a fold over itself (which
        // supersedes this one); its boundaries are in the host's shortened
        // transcript's coordinates from here on.
        if is_checkpoint(&fold.summary)
            && chain.first() == fingerprint_chain(std::slice::from_ref(&fold.summary)).first()
        {
            state.fold = None;
            state.host_applied_summary = None;
            state.boundary_unaligned = true;
            tracing::debug!(
                "[context_compression] host carries the fold's checkpoint as its first live message"
            );
            return FoldCheck::HostApplied(fold.summary.text());
        }
        state.fold = None;
        state.host_applied_summary = None;
        state.last_summary = None;
        tracing::debug!(
            folded = fold.folded,
            live = chain.len(),
            "[context_compression] transcript no longer matches the fold; dropping it"
        );
        FoldCheck::Stale
    }

    /// `run`'s fold and the live transcript its last `before_model` saw.
    /// `None` when `before_model` never ran for `run` (compression installed
    /// only as model middleware).
    fn run_state(
        &self,
        run: u64,
    ) -> Option<(Option<crate::middleware::types::CompactionFold>, Vec<u64>)> {
        let runs = self.runs.lock().expect("runs mutex poisoned");
        runs.get(&run)
            .map(|state| (state.fold.clone(), state.live_chain.clone()))
    }

    /// The last summary `run` produced, if any.
    fn run_last_summary(&self, run: u64) -> Option<String> {
        let runs = self.runs.lock().expect("runs mutex poisoned");
        runs.get(&run).and_then(|state| state.last_summary.clone())
    }

    fn boundary_for_run(&self, run: u64, boundary: Option<LiveBoundary>) -> Option<LiveBoundary> {
        let runs = self.runs.lock().expect("runs mutex poisoned");
        runs.get(&run)
            .filter(|state| !state.boundary_unaligned)
            .and(boundary)
    }

    /// Records that `summary` now stands in for `run`'s first `folded` live
    /// non-system messages, whose chained fingerprints are `chain`.
    fn remember_fold(
        &self,
        run: u64,
        folded: usize,
        chain: &[u64],
        summary: &Message,
        pinned: Option<crate::middleware::types::PinnedTurnMessage>,
    ) {
        let Some(fingerprint) = folded.checked_sub(1).and_then(|i| chain.get(i)).copied() else {
            return;
        };
        let mut runs = self.runs.lock().expect("runs mutex poisoned");
        let state = touch_run(&mut runs, run);
        let replaces = state
            .host_applied_summary
            .take()
            .or_else(|| state.fold.as_ref().and_then(|fold| fold.replaces.clone()));
        state.fold = Some(crate::middleware::types::CompactionFold {
            folded,
            fingerprint,
            summary: summary.clone(),
            pinned,
            replaces,
        });
        // Only a summary attached to a valid fold is worth building on: one
        // from an unaligned overflow request describes altered history.
        state.last_summary = Some(summary_text(summary));
    }

    /// Finalizes a successful compaction: records `record` in the in-process
    /// history, builds a [`CompactionRecord`], persists it through
    /// [`RunContext::compaction_sink`] when attached, and emits
    /// [`AgentEvent::Compacted`]. Does **not** emit `Compressed` — callers
    /// that also want the legacy event emit it themselves.
    #[allow(clippy::too_many_arguments)]
    fn finish_compaction<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        record: SummaryRecord,
        boundary: Option<LiveBoundary>,
        tokens_before: u64,
        tokens_after: u64,
        reason: CompactionReason,
        latency_ms: Option<u64>,
    ) {
        // The prompt prefix is rewritten on purpose: the provider's next cache
        // read is expected to be cold.
        ctx.mark_prompt_prefix_changed();
        // The run's last summary is set by `remember_fold` only: a summary
        // with no valid fold (an unaligned overflow) is not built on later.
        touch_run(
            &mut self.runs.lock().expect("runs mutex poisoned"),
            ctx.instance_id(),
        )
        .pressure
        .note_compaction();
        let usage = record.usage;
        tracing::info!(
            run_id = %ctx.run_id(),
            reason = reason.as_str(),
            boundary = boundary.map(|b| b.first_kept_index),
            pinned_user_index = boundary.and_then(|b| b.pinned_user_index),
            tokens_before,
            tokens_after,
            latency_ms,
            summarizer_input_tokens = usage.map(|u| u.input_tokens),
            summarizer_output_tokens = usage.map(|u| u.output_tokens),
            placement = ?self.placement,
            "[context_compression] compacted"
        );

        // `boundary` is the first kept live-transcript position, or `None`
        // when it cannot be trusted as one (see the overflow path): a durable
        // boundary there would restore or duplicate the wrong messages on
        // resume, so nothing is persisted.
        if let Some(boundary) = boundary
            && let Some(sink) = &ctx.compaction_sink
        {
            let compaction_record = CompactionRecord {
                summary: record.summary.text(),
                placement: self.placement,
                first_kept_index: boundary.first_kept_index,
                tokens_before,
                tokens_after,
                usage,
                details: compaction_details(
                    &record.provenance.source_ids,
                    boundary.pinned_user_index,
                ),
                reason,
            };
            if let Err(err) = sink.persist(&compaction_record) {
                tracing::debug!("[context_compression] compaction sink persist failed: {err}");
            }
        }

        {
            let mut records = self.records.lock().expect("records mutex poisoned");
            if self.max_records > 0 {
                if records.len() >= self.max_records {
                    records.pop_front();
                }
                records.push_back(record);
            }
        }

        ctx.emit(AgentEvent::Compacted {
            reason,
            tokens_before,
            tokens_after,
            usage,
            latency_ms,
        });
    }

    /// `record` with its summary rewritten as this middleware's checkpoint
    /// message (see [`SummaryPlacement`]).
    fn placed(&self, mut record: SummaryRecord) -> SummaryRecord {
        record.summary = checkpoint_message(self.placement, &record.summary.text());
        record
    }

    /// Deterministic front-drop of `request` to the policy's trigger budget,
    /// preserving system messages (see `TrimStrategy::MaxTokens`).
    ///
    /// The trigger charges tool schemas, so the message budget reserves that
    /// same schema cost first: otherwise a request whose schemas already
    /// consume a meaningful share of `trigger_budget` (or all of it) would
    /// still trim messages to the *full* budget and stay over the threshold.
    pub(super) fn trim_to_trigger<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &mut ModelRequest,
        from_tokens: u64,
    ) {
        ctx.mark_prompt_prefix_changed();
        let message_budget = self
            .policy
            .trigger_budget()
            .saturating_sub(schema_tokens(&request.tools));
        // Preserve the folded history independently of checkpoint role while
        // trimming the live tail to the remaining budget.
        let checkpoint_at = request.messages.iter().position(is_checkpoint);
        let trim_tail = |messages: &[Message], budget| {
            if self.policy.pin_turn_user_message {
                crate::summarization::trim_keeping_turn_user_message(messages, budget)
            } else {
                crate::summarization::enforce_approximate_budget(
                    trim_messages(messages, &TrimStrategy::MaxTokens(budget)),
                    budget,
                    None,
                )
            }
        };
        let trimmed = match checkpoint_at {
            Some(at) => {
                let mut rest = request.messages.clone();
                let checkpoint = rest.remove(at);
                let system_tokens = crate::token_estimation::count_tokens_approximately(
                    &rest
                        .iter()
                        .filter(|message| matches!(message, Message::System(_)))
                        .cloned()
                        .collect::<Vec<_>>(),
                );
                let checkpoint_budget = message_budget.saturating_sub(system_tokens);
                let checkpoint = if crate::token_estimation::count_tokens_approximately(
                    std::slice::from_ref(&checkpoint),
                ) >= checkpoint_budget
                {
                    shrink_checkpoint(&checkpoint, checkpoint_budget / 2, checkpoint_budget)
                } else {
                    Some(checkpoint)
                };
                match checkpoint {
                    Some(checkpoint) => {
                        let budget = message_budget.saturating_sub(
                            crate::token_estimation::count_tokens_approximately(
                                std::slice::from_ref(&checkpoint),
                            ),
                        );
                        let mut trimmed = trim_tail(&rest, budget);
                        let insert = trimmed
                            .iter()
                            .take_while(|m| matches!(m, Message::System(_)))
                            .count();
                        trimmed.insert(insert, checkpoint);
                        trimmed
                    }
                    None => trim_tail(&rest, message_budget),
                }
            }
            None => trim_tail(&request.messages, message_budget),
        };
        let to_tokens = total_message_tokens(&trimmed);
        request.messages = trimmed;
        ctx.emit(AgentEvent::Compressed {
            from_tokens,
            to_tokens,
        });
    }

    /// Drops `run`'s pending usage attribution: an overflow retry sends a
    /// request whose shape `before_model` never saw, so its usage cannot
    /// anchor the next call's measurement.
    fn forget_pending(&self, run: u64) {
        if let Some(state) = self.runs.lock().expect("runs mutex poisoned").get_mut(&run) {
            state.pressure.pending = None;
        }
    }
}

/// Shrinks a checkpoint to `preferred_tokens`, including marker and
/// truncation notice. If the framing alone exceeds that target, it may use
/// up to `max_tokens`; a checkpoint that cannot fit even then is dropped.
fn shrink_checkpoint(
    checkpoint: &Message,
    preferred_tokens: u64,
    max_tokens: u64,
) -> Option<Message> {
    let body = checkpoint_body(checkpoint).unwrap_or_default();
    let placement = if matches!(checkpoint, Message::System(_)) {
        crate::summarization::SummaryPlacement::System
    } else {
        crate::summarization::SummaryPlacement::User
    };
    let render = |keep: usize| {
        let cut: String = body.chars().take(keep).collect();
        checkpoint_message(
            placement,
            &format!("{cut}\n[checkpoint truncated to fit the context budget]"),
        )
    };
    let minimum = render(0);
    let framing_tokens =
        crate::token_estimation::count_tokens_approximately(std::slice::from_ref(&minimum));
    if framing_tokens > max_tokens {
        tracing::warn!(
            framing_tokens,
            max_tokens,
            "[context_compression] checkpoint framing exceeds trim budget; dropping it"
        );
        return None;
    }
    let target = preferred_tokens.max(framing_tokens).min(max_tokens);
    let mut low = 0;
    let mut high = body
        .chars()
        .count()
        .min(usize::try_from(target.saturating_mul(4)).unwrap_or(usize::MAX));
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if crate::token_estimation::count_tokens_approximately(std::slice::from_ref(&render(mid)))
            <= target
        {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    tracing::warn!(
        from_chars = body.chars().count(),
        to_chars = low,
        "[context_compression] checkpoint exceeds trim budget; truncating it"
    );
    Some(render(low))
}

/// `messages` with `fold` applied, when the fold still matches them and covers
/// more than a checkpoint the transcript already opened with.
///
/// Leading system messages stay first, then the fold's summary, then every
/// message after the folded prefix; mid-conversation system messages are
/// kept, and system-role checkpoints the fold superseded are dropped.
fn compacted_history(
    fold: &crate::middleware::types::CompactionFold,
    messages: &[Message],
) -> Option<Vec<Message>> {
    let (_, live) = partition_messages_system(messages);
    let chain = fingerprint_chain(&live);
    if fold.folded == 0 || chain.get(fold.folded - 1) != Some(&fold.fingerprint) {
        return None;
    }
    if fold.folded == 1 && live.first() == Some(&fold.summary) {
        // Only an adopted checkpoint: nothing new was folded.
        return None;
    }
    let leading = messages
        .iter()
        .take_while(|m| matches!(m, Message::System(_)) && !is_checkpoint(m))
        .count();
    let mut history = Vec::with_capacity(messages.len());
    history.extend(messages[..leading].iter().cloned());
    history.push(fold.summary.clone());
    // The message the fold pinned out of the folded range follows its summary.
    history.extend(fold.pinned.iter().map(|pin| pin.message.clone()));
    let mut skipped = 0usize;
    for message in &messages[leading..] {
        let system = matches!(message, Message::System(_));
        if system && is_checkpoint(message) {
            continue;
        }
        if !system && skipped < fold.folded {
            skipped += 1;
            continue;
        }
        history.push(message.clone());
    }
    Some(history)
}

/// Estimated tokens of a request's tool declarations.
fn schema_tokens(tools: &[tinyinference_llm::tool::ToolSchema]) -> u64 {
    crate::token_estimation::count_tool_schema_tokens(
        tools,
        &crate::token_estimation::TokenCountOptions::default(),
    )
}

/// Milliseconds since `started`, saturating.
fn elapsed_ms(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The summary a message carries: a checkpoint's body without its marker, or
/// the message text for any other summary message.
fn summary_text(message: &Message) -> String {
    checkpoint_body(message).unwrap_or_else(|| message.text())
}

/// `text` without a leading checkpoint marker.
fn strip_checkpoint_marker(text: &str) -> String {
    text.strip_prefix(crate::summarization::CHECKPOINT_PREFIX)
        .unwrap_or(text)
        .trim()
        .to_string()
}

/// Removes every compaction checkpoint from `messages`, returning the last
/// one (the most recent summary).
fn take_checkpoints(messages: &mut Vec<Message>) -> Option<Message> {
    let mut last = None;
    messages.retain(|m| {
        if is_checkpoint(m) {
            last = Some(m.clone());
            false
        } else {
            true
        }
    });
    last
}

/// [`CompactionRecord::details`] for a compaction: the summarized source ids
/// and, when a user message was pinned out of the folded range, its live index
/// as `pinned_user_index`. Everything before `first_kept_index` was folded
/// into the summary *except* that message, so a sink restoring the compacted
/// transcript must keep it after the summary.
fn compaction_details(
    source_ids: &[String],
    pinned_user_index: Option<usize>,
) -> serde_json::Value {
    let mut details = serde_json::json!({ "source_ids": source_ids });
    if let Some(index) = pinned_user_index {
        details["pinned_user_index"] = serde_json::json!(index);
    }
    details
}

// ── MicrocompactMiddleware ────────────────────────────────────────────────────

impl MicrocompactMiddleware {
    /// Creates a micro-compaction middleware that keeps the newest `keep_recent`
    /// tool-result bodies verbatim and blanks older ones with `placeholder`.
    /// Event emission is off by default; enable it with
    /// [`MicrocompactMiddleware::with_events`].
    pub fn new(keep_recent: usize, placeholder: impl Into<String>) -> Self {
        Self {
            label: "microcompact",
            keep_recent,
            placeholder: placeholder.into(),
            emit_events: false,
            token_budget: None,
        }
    }

    /// Enable or disable emitting an
    /// [`AgentEvent::Compressed`][crate::events::AgentEvent::Compressed]
    /// event whenever at least one tool body is cleared. Off by default so the
    /// middleware can be a silent transcript rewrite.
    pub fn with_events(mut self, emit_events: bool) -> Self {
        self.emit_events = emit_events;
        self
    }

    /// Only blank stale tool bodies once the transcript's estimated tokens
    /// exceed `budget`; below it the middleware is a no-op so the request stays
    /// append-only and the provider KV-cache prefix is preserved (issue
    /// tinyhumansai/openhuman#4755).
    ///
    /// Set this below the model's context window (leaving headroom for the reply
    /// and the `keep_recent` verbatim results) so a run that fits the window is
    /// never compacted — compaction only kicks in when it is actually needed to
    /// stay under the window, which is the only time the cache-invalidation cost
    /// of blanking an already-sent tool body pays for itself. `budget == 0`
    /// disables the gate (equivalent to leaving it unset).
    pub fn with_token_budget(mut self, budget: u64) -> Self {
        self.token_budget = (budget > 0).then_some(budget);
        self
    }

    /// The number of most-recent tool-result bodies kept verbatim.
    pub fn keep_recent(&self) -> usize {
        self.keep_recent
    }

    /// The placeholder text swapped in for cleared tool-result bodies.
    pub fn placeholder(&self) -> &str {
        &self.placeholder
    }

    /// The estimated-token floor below which blanking is skipped, if configured.
    pub fn token_budget(&self) -> Option<u64> {
        self.token_budget
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for MicrocompactMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        // Indices of every tool-result message, oldest → newest.
        let tool_idxs: Vec<usize> = request
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| matches!(m, Message::Tool(_)))
            .map(|(i, _)| i)
            .collect();
        if tool_idxs.len() <= self.keep_recent {
            return Ok(());
        }

        // Prompt-cache preservation (issue tinyhumansai/openhuman#4755): when a
        // token budget is configured, skip blanking while the transcript still
        // fits within it. Blanking a tool body that was sent verbatim on an
        // earlier iteration mutates an already-transmitted prefix position and
        // invalidates the provider KV-cache from there on; doing that every call
        // (the boundary moves by one each turn) churns the cache to reclaim
        // tokens the model still has room for. Gating on the *pre-blank* estimate
        // — which grows monotonically as the run appends — keeps the request
        // append-only (fully cache-eligible) below budget and can't oscillate.
        // Reuse `from_tokens` when events are on so we never estimate twice.
        let from_tokens = if self.emit_events {
            total_message_tokens(&request.messages)
        } else {
            0
        };
        if let Some(budget) = self.token_budget {
            let tokens = if self.emit_events {
                from_tokens
            } else {
                total_message_tokens(&request.messages)
            };
            // The schemas are part of what the model has to fit, so they are
            // part of what is measured against the budget.
            let schema_tokens = crate::token_estimation::count_tool_schema_tokens(
                &request.tools,
                &crate::token_estimation::TokenCountOptions::default(),
            );
            if tokens + schema_tokens <= budget {
                return Ok(());
            }
        }

        let cut = tool_idxs.len() - self.keep_recent;
        let mut cleared = 0usize;
        for &i in &tool_idxs[..cut] {
            // Skip messages already reduced to the placeholder; otherwise swap the
            // body for it (idempotent, preserves the tool_call_id).
            if request.messages[i].text() == self.placeholder {
                continue;
            }
            if let Message::Tool(t) = &request.messages[i] {
                // `ToolMessage::trusted_verbatim` means the producing tool
                // asked for its content to reach the model byte-for-byte, and
                // its doc names blanking as exactly the rewrite a host must not
                // perform. Blanking one produced content that reads fine and is
                // wrong — an input schema the model copies argument names out
                // of, a signature, a diff — so leave it intact and reclaim
                // tokens elsewhere.
                if t.trusted_verbatim {
                    tracing::debug!(
                        target: "tinyagents::middleware",
                        tool_call_id = %t.tool_call_id,
                        "[microcompact] skipping a trusted_verbatim tool result"
                    );
                    continue;
                }
                let id = t.tool_call_id.clone();
                request.messages[i] = Message::tool(id, self.placeholder.clone());
                cleared += 1;
            }
        }

        if self.emit_events && cleared > 0 {
            let to_tokens = total_message_tokens(&request.messages);
            ctx.emit(AgentEvent::Compressed {
                from_tokens,
                to_tokens,
            });
        }
        Ok(())
    }
}

// ── PromptCacheGuardMiddleware ────────────────────────────────────────────────

impl PromptCacheGuardMiddleware {
    /// Creates a cache-guard middleware with the default label
    /// `"prompt_cache_guard"`.
    pub fn new() -> Self {
        Self {
            label: "prompt_cache_guard",
            previous: std::sync::Mutex::new(None),
            events: std::sync::Mutex::new(std::collections::VecDeque::new()),
            max_events: DEFAULT_CACHE_GUARD_EVENT_CAP,
            cache_misses: std::sync::Mutex::new(crate::cache::PromptCacheTracker::default()),
            thread_epochs: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Sets the noise floor of the prompt-cache miss accounting: a call whose
    /// cache read falls short of the previous prompt by this many tokens or
    /// fewer is not reported. Defaults to
    /// [`DEFAULT_CACHE_MISS_NOISE_FLOOR_TOKENS`][crate::cache::DEFAULT_CACHE_MISS_NOISE_FLOOR_TOKENS].
    pub fn with_cache_miss_noise_floor(self, tokens: u64) -> Self {
        *self
            .cache_misses
            .lock()
            .expect("cache misses mutex poisoned") = crate::cache::PromptCacheTracker::new(tokens);
        self
    }

    /// The conversation key cache accounting groups calls by the thread when
    /// the run has one (a provider cache spans runs of a thread), else the run
    /// instance, plus the prompt-prefix epoch. Thread epochs are retained in
    /// the guard so a fresh context for a resumed thread does not revert to the
    /// pre-compaction epoch zero.
    fn cache_key<Ctx: Send + Sync>(&self, ctx: &RunContext<Ctx>) -> String {
        let conversation = ctx
            .thread_id()
            .map(|thread| format!("thread:{thread}"))
            // A run id is a caller's label that two runs may share; the
            // instance id is unique.
            .unwrap_or_else(|| format!("run:{}", ctx.instance_id()));
        format!("{conversation}@{}", self.prefix_epoch(ctx))
    }

    fn prefix_epoch<Ctx: Send + Sync>(&self, ctx: &RunContext<Ctx>) -> u64 {
        let Some(thread) = ctx.thread_id() else {
            return ctx.prompt_prefix_epoch();
        };
        let mut epochs = self
            .thread_epochs
            .lock()
            .expect("thread epochs mutex poisoned");
        if ctx.prompt_prefix_epoch() != 0 {
            const MAX_THREAD_EPOCHS: usize = 256;
            if !epochs.contains_key(thread)
                && epochs.len() >= MAX_THREAD_EPOCHS
                && let Some(evicted) = epochs.keys().next().cloned()
            {
                epochs.remove(&evicted);
            }
            epochs.insert(thread.clone(), ctx.prompt_prefix_epoch());
        }
        epochs.get(thread).copied().unwrap_or(0)
    }

    /// Sets the maximum number of [`CacheLayoutEvent`]s retained before the
    /// oldest is evicted. `0` disables recording entirely.
    pub fn with_max_events(mut self, max_events: usize) -> Self {
        self.max_events = max_events;
        let mut events = self.events.lock().expect("events mutex poisoned");
        while events.len() > max_events {
            events.pop_front();
        }
        drop(events);
        self
    }

    /// Returns the cache-layout change events recorded so far, in order.
    /// Bounded to at most [`PromptCacheGuardMiddleware::with_max_events`]
    /// entries (default [`DEFAULT_CACHE_GUARD_EVENT_CAP`]); older events are
    /// evicted first.
    pub fn layout_events(&self) -> Vec<CacheLayoutEvent> {
        self.events
            .lock()
            .expect("events mutex poisoned")
            .iter()
            .cloned()
            .collect()
    }
}

impl Default for PromptCacheGuardMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for PromptCacheGuardMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        let layout = PromptCacheLayout::from_request(request);
        let run_id = ctx.run_id().clone();
        let mut previous = self.previous.lock().expect("previous mutex poisoned");
        // Only compare within one run. See the field docs on
        // `PromptCacheGuardMiddleware::previous`: a prefix cache is scoped to a
        // single conversation, so a baseline carried over from a previous run
        // would report an invalidation that never happened.
        if let Some((prev_run, prev)) = previous.as_ref()
            && prev_run == &run_id
        {
            // Only an explicit canonical layout maps segment ids to message
            // boundaries. Every other request checks the whole message stream
            // with the byte-prefix rule: a rewritten message invalidates it,
            // while a pure tail append can still reuse the earlier KV prefix.
            let full_request_fallback =
                !prev.canonical_message_boundary || !layout.canonical_message_boundary;
            let changed = if full_request_fallback {
                !prev.is_prefix_stable_against(&layout)
            } else {
                !prev.has_same_stable_prefix_as(&layout)
            };
            if changed {
                // The prefix changed on purpose (or by accident the layout
                // event below records): the next uncached tokens are new
                // content, not a silent miss.
                self.cache_misses
                    .lock()
                    .expect("cache misses mutex poisoned")
                    .reset(&self.cache_key(ctx));
                tracing::debug!(
                    "[cache] prompt_cache_guard: stable prefix changed run={run_id} \
                     before={} after={}",
                    prev.fingerprint(),
                    layout.fingerprint()
                );
                let event = CacheLayoutEvent::new(prev, &layout);
                let mut events = self.events.lock().expect("events mutex poisoned");
                if self.max_events > 0 {
                    if events.len() >= self.max_events {
                        events.pop_front();
                    }
                    events.push_back(event);
                }
            } else if !prev.is_prefix_stable_against(&layout) {
                tracing::debug!(
                    run = %run_id,
                    "[cache] history changed while stable prefix remained reusable"
                );
            }
        }
        *previous = Some((run_id, layout));
        Ok(())
    }

    /// Compares the call's `cache_read_tokens` with the previous call's prompt
    /// for the same conversation and emits [`AgentEvent::PromptCacheMiss`]
    /// when the cache read fell short beyond the noise floor.
    async fn after_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        // A replayed response consumed no provider cache.
        let Some(usage) = response
            .usage
            .as_ref()
            .filter(|_| !response.served_from_cache)
        else {
            return Ok(());
        };
        let key = self.cache_key(ctx);
        let model = response
            .resolved_model
            .as_ref()
            .map_or("", |resolved| resolved.name.as_str());
        let miss = self
            .cache_misses
            .lock()
            .expect("cache misses mutex poisoned")
            .observe(&key, model, usage);
        if let Some(miss) = miss {
            let call_id = ctx
                .active_model_call
                .clone()
                .unwrap_or_else(|| crate::ids::CallId::new(format!("{}-model", ctx.run_id())));
            tracing::warn!(
                run = %ctx.run_id(),
                expected_cached = miss.expected_cached_tokens,
                cached = miss.cached_tokens,
                wasted = miss.wasted_input_tokens,
                "[cache] prompt cache read back less than the previous prompt"
            );
            ctx.emit(AgentEvent::PromptCacheMiss {
                call_id,
                expected_cached_tokens: miss.expected_cached_tokens,
                cached_tokens: miss.cached_tokens,
                wasted_input_tokens: miss.wasted_input_tokens,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "context_overflow_tests.rs"]
mod context_overflow_tests;
#[cfg(test)]
#[path = "context_prompt_cache_miss_tests.rs"]
mod context_prompt_cache_miss_tests;
#[cfg(test)]
#[path = "context_summary_tests.rs"]
mod context_summary_tests;
