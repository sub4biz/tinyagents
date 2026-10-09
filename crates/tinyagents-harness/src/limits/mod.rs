//! Run-scoped limit enforcement.
//!
//! Limits are what keep recursion bounded: because agents can call agents and
//! graphs can run graphs, an unbounded run tree could fan out forever or burn a
//! provider budget. [`RunLimits::max_depth`] caps how deep the sub-agent /
//! sub-graph recursion may go, while the call and wall-clock caps bound the
//! work within each run.
//!
//! [`RunLimits`] holds the policy; [`LimitTracker`] tracks live counters and
//! checks them against the policy.  Every model call and tool call must go
//! through the tracker so limits are fail-closed.
//!
//! # Example
//!
//! ```
//! use tinyagents_harness::limits::{RunLimits, LimitTracker};
//!
//! let limits = RunLimits::default();
//! let mut tracker = LimitTracker::new(limits);
//! tracker.record_model_call().expect("within limit");
//! assert_eq!(tracker.model_calls(), 1);
//! assert_eq!(tracker.remaining_model_calls(), 24);
//! ```

mod types;

pub use types::*;

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::error::{Result, TinyAgentsError};

impl RunLimits {
    /// Sets the maximum number of model calls allowed per run.
    pub fn with_max_model_calls(mut self, n: usize) -> Self {
        self.max_model_calls = n;
        self
    }

    /// Sets the maximum number of tool calls allowed per run.
    pub fn with_max_tool_calls(mut self, n: usize) -> Self {
        self.max_tool_calls = n;
        self
    }

    /// Sets a wall-clock deadline in milliseconds. `None` removes the limit.
    pub fn with_max_wall_clock_ms(mut self, ms: Option<u64>) -> Self {
        self.max_wall_clock_ms = ms;
        self
    }

    /// Sets a per-model-call wall-clock ceiling in milliseconds. `None`
    /// removes the ceiling. See [`RunLimits::max_model_call_ms`].
    pub fn with_max_model_call_ms(mut self, ms: Option<u64>) -> Self {
        self.max_model_call_ms = ms;
        self
    }

    /// Sets the per-call retry cap (a retry *count*, not counting the first
    /// attempt). See [`RunLimits::max_retries_per_call`].
    pub fn with_max_retries_per_call(mut self, n: usize) -> Self {
        self.max_retries_per_call = n;
        self
    }

    /// Sets the maximum sub-agent / recursion depth for the run tree.
    pub fn with_max_depth(mut self, n: usize) -> Self {
        self.max_depth = n;
        self
    }

    /// Sets what the run does when a call cap is reached. See
    /// [`LimitBehavior`].
    pub fn with_behavior(mut self, behavior: LimitBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// Caps how many tool calls a concurrently-executed batch may run at
    /// once. `None` removes the cap. See
    /// [`RunLimits::max_tool_concurrency`].
    pub fn with_max_tool_concurrency(mut self, n: Option<usize>) -> Self {
        self.max_tool_concurrency = n;
        self
    }

    /// Sets the maximum nested tool-call depth. See
    /// [`RunLimits::max_nested_depth`].
    pub fn with_max_nested_depth(mut self, n: usize) -> Self {
        self.max_nested_depth = n;
        self
    }

    /// Sets the maximum silence between streaming-model output events, after
    /// the first one. `None` disables the inactivity timeout. See
    /// [`RunLimits::stream_idle_timeout_ms`].
    pub fn with_stream_idle_timeout_ms(mut self, ms: Option<u64>) -> Self {
        self.stream_idle_timeout_ms = ms;
        self
    }

    /// Sets the opt-in maximum wait for a streaming model call's first output
    /// event. `None` (the default) means no separate bound. See
    /// [`RunLimits::stream_first_event_timeout_ms`].
    pub fn with_stream_first_event_timeout_ms(mut self, ms: Option<u64>) -> Self {
        self.stream_first_event_timeout_ms = ms;
        self
    }

    /// Sets how many consecutive stream idle timeouts on one model trip the
    /// breaker. `None` disables it. See
    /// [`RunLimits::max_consecutive_stream_idle_timeouts`].
    pub fn with_max_consecutive_stream_idle_timeouts(mut self, n: Option<usize>) -> Self {
        self.max_consecutive_stream_idle_timeouts = n;
        self
    }
}

/// Tracks live counters for a single harness run and enforces [`RunLimits`].
///
/// The tracker records the wall-clock start time when it is created and
/// computes elapsed time on demand via [`check_wall_clock`].
///
/// [`check_wall_clock`]: LimitTracker::check_wall_clock
pub struct LimitTracker {
    limits: RunLimits,
    model_calls: usize,
    tool_calls: usize,
    /// Tool calls tools made through `ToolExecutionContext::call_tool`.
    /// Shared (atomic) because nested calls run from inside a tool future that
    /// holds only `&RunContext`; they count against `max_tool_calls` together
    /// with `tool_calls`.
    nested_tool_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Streaming model calls that ended in an idle timeout since the last
    /// output event arrived (the agent loop also clears it when it switches to
    /// a fallback model, making it a per-model count). See [`LimitTracker::record_stream_idle_timeout`].
    consecutive_stream_idle_timeouts: usize,
    /// Idle-timeout strikes keyed by model name. The public scalar helpers
    /// remain for compatibility with callers that track one stream at a time.
    stream_idle_timeouts_by_model: HashMap<String, usize>,
    /// Models the failover policy has written off for the rest of the run
    /// (see [`crate::retry::FailoverReason::skips_model_for_run`]).
    skipped_models: HashSet<String>,
    started_at: Instant,
}

impl LimitTracker {
    /// Creates a new tracker with zeroed counters and the current time as the
    /// run start.
    pub fn new(limits: RunLimits) -> Self {
        Self {
            limits,
            model_calls: 0,
            tool_calls: 0,
            nested_tool_calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            consecutive_stream_idle_timeouts: 0,
            stream_idle_timeouts_by_model: HashMap::new(),
            skipped_models: HashSet::new(),
            started_at: Instant::now(),
        }
    }

    /// Records that a streaming model call went idle past its timeout and
    /// returns the number of *consecutive* idle timeouts so far.
    ///
    /// The count survives the retry loop; the agent loop clears it on any
    /// output event and when it moves to a fallback model. The caller compares
    /// it with
    /// [`RunLimits::max_consecutive_stream_idle_timeouts`].
    pub fn record_stream_idle_timeout(&mut self) -> usize {
        self.consecutive_stream_idle_timeouts += 1;
        self.consecutive_stream_idle_timeouts
    }

    /// Clears the consecutive stream-idle-timeout count: a stream delivered
    /// output, or the loop moved to a different model.
    pub fn reset_stream_idle_timeouts(&mut self) {
        self.consecutive_stream_idle_timeouts = 0;
    }

    /// Returns the current consecutive stream-idle-timeout count.
    pub fn consecutive_stream_idle_timeouts(&self) -> usize {
        self.consecutive_stream_idle_timeouts
    }

    /// Records an idle timeout for `model` and returns that model's strikes.
    pub fn record_stream_idle_timeout_for(&mut self, model: &str) -> usize {
        let strikes = self
            .stream_idle_timeouts_by_model
            .entry(model.to_owned())
            .or_default();
        *strikes += 1;
        *strikes
    }

    /// Clears the idle-timeout strikes for `model` after visible output.
    pub fn reset_stream_idle_timeouts_for(&mut self, model: &str) {
        self.stream_idle_timeouts_by_model.remove(model);
    }

    /// Returns the idle-timeout strikes for `model`.
    pub fn consecutive_stream_idle_timeouts_for(&self, model: &str) -> usize {
        self.stream_idle_timeouts_by_model
            .get(model)
            .copied()
            .unwrap_or_default()
    }

    /// Marks `model` as unusable for the remainder of this run: later model
    /// calls skip it in favour of the next fallback (the cross-call skip hint
    /// recorded for [`crate::retry::FailoverReason::AuthPermanent`]).
    pub fn skip_model_for_run(&mut self, model: &str) {
        self.skipped_models.insert(model.to_owned());
    }

    /// Whether [`LimitTracker::skip_model_for_run`] was called for `model`.
    pub fn is_model_skipped(&self, model: &str) -> bool {
        self.skipped_models.contains(model)
    }

    /// Resets the wall-clock start to now, leaving the call counters and
    /// limits untouched.
    ///
    /// [`RunContext::new`][crate::context::RunContext::new] constructs the
    /// tracker (and therefore stamps `started_at`) at context-construction
    /// time, which is not always the same moment the run actually starts
    /// doing work — a context built ahead of time and queued, or reused
    /// across a retry of the *surrounding* host operation, would otherwise
    /// have its wall-clock deadline silently burn down before the agent loop
    /// issues its first model call (M-8). The agent loop calls this at the
    /// top of the run so the deadline is always measured from when the run
    /// actually began.
    pub fn restart(&mut self) {
        self.started_at = Instant::now();
    }

    /// Records one model call and returns an error if the cap is exceeded.
    ///
    /// The counter is incremented **before** the check so the limit is
    /// inclusive (a cap of `N` allows exactly `N` calls).
    pub fn record_model_call(&mut self) -> Result<()> {
        self.try_record_model_call()?;
        Ok(())
    }

    /// Records one tool call and returns an error if the cap is exceeded.
    pub fn record_tool_call(&mut self) -> Result<()> {
        self.try_record_tool_call()?;
        Ok(())
    }

    /// Records one model call and reports the cap decision as a
    /// [`LimitOutcome`], honoring [`RunLimits::behavior`].
    ///
    /// - Within the cap → `Ok(LimitOutcome::Proceed)`.
    /// - Cap exhausted under [`LimitBehavior::Error`] → `Err(LimitExceeded)`,
    ///   exactly as [`LimitTracker::record_model_call`] has always behaved.
    /// - Cap exhausted under [`LimitBehavior::StopWithPartial`] →
    ///   `Ok(LimitOutcome::Stop(LimitKind::ModelCalls))`, so the loop can stop
    ///   cleanly and return the partial run instead of discarding it.
    pub fn try_record_model_call(&mut self) -> Result<LimitOutcome> {
        self.model_calls += 1;
        if self.model_calls > self.limits.max_model_calls {
            return self.exhausted(LimitKind::ModelCalls, self.limits.max_model_calls);
        }
        Ok(LimitOutcome::Proceed)
    }

    /// Records one tool call and reports the cap decision as a
    /// [`LimitOutcome`]. See [`LimitTracker::try_record_model_call`].
    pub fn try_record_tool_call(&mut self) -> Result<LimitOutcome> {
        self.tool_calls += 1;
        if self.tool_calls + self.nested_tool_calls() > self.limits.max_tool_calls {
            return self.exhausted(LimitKind::ToolCalls, self.limits.max_tool_calls);
        }
        Ok(LimitOutcome::Proceed)
    }

    /// Reserves one slot of the tool-call cap for a nested call (a tool
    /// calling another tool), counted together with the model-issued calls.
    ///
    /// Takes `&self` because nested calls run while a tool future holds a
    /// shared `&RunContext`. Always fails closed with `LimitExceeded` when the
    /// cap is spent, whatever [`RunLimits::behavior`] says: there is no loop
    /// boundary at which a nested call could be answered with a partial result.
    /// Pair a successful reservation with
    /// [`LimitTracker::release_nested_tool_call`] when the call never runs.
    pub fn try_reserve_nested_tool_call(&self) -> Result<()> {
        // One compare-and-swap loop: the slot is taken only if the combined
        // count is still under the cap, so a rejected reservation never
        // touches the counter and concurrent reservations cannot overspend it.
        let cap = self.limits.max_tool_calls;
        let issued = self.tool_calls;
        if self.update_nested(|nested| (issued + nested < cap).then_some(nested + 1)) {
            return Ok(());
        }
        Err(TinyAgentsError::LimitExceeded(format!(
            "max tool calls ({cap}) exceeded by a nested tool call"
        )))
    }

    /// Releases a slot taken by [`LimitTracker::try_reserve_nested_tool_call`]
    /// for a call that never ran. Saturates at zero.
    pub fn release_nested_tool_call(&self) {
        self.update_nested(|nested| nested.checked_sub(1));
    }

    /// Applies `step` to the nested counter atomically; `false` when `step`
    /// declined (returned `None`).
    fn update_nested(&self, step: impl Fn(usize) -> Option<usize>) -> bool {
        use std::sync::atomic::Ordering;
        let mut current = self.nested_tool_calls.load(Ordering::SeqCst);
        loop {
            let Some(next) = step(current) else {
                return false;
            };
            match self.nested_tool_calls.compare_exchange_weak(
                current,
                next,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Returns the number of nested tool calls counted against the cap so far.
    pub fn nested_tool_calls(&self) -> usize {
        self.nested_tool_calls
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Shared exhaustion branch: error or clean stop, per [`RunLimits::behavior`].
    fn exhausted(&self, kind: LimitKind, cap: usize) -> Result<LimitOutcome> {
        match self.limits.behavior {
            LimitBehavior::Error => {
                tracing::debug!(
                    target: "tinyagents::limits",
                    limit_kind = kind.as_str(),
                    cap,
                    "[limits] cap exhausted; failing the run"
                );
                Err(TinyAgentsError::Validation(format!(
                    "max {} ({cap}) exceeded",
                    match kind {
                        LimitKind::ModelCalls => "model calls",
                        LimitKind::ToolCalls => "tool calls",
                    }
                )))
            }
            LimitBehavior::StopWithPartial => {
                tracing::debug!(
                    target: "tinyagents::limits",
                    limit_kind = kind.as_str(),
                    cap,
                    "[limits] cap exhausted; stopping with the partial result"
                );
                Ok(LimitOutcome::Stop(kind))
            }
        }
    }

    /// Un-counts `n` tool calls that were requested but never executed.
    ///
    /// Needed by the [`LimitBehavior::StopWithPartial`] tool path: when the cap
    /// trips mid-batch the loop answers the remaining `tool_call_id`s with a
    /// "stopped before this could run" result rather than executing them, so
    /// leaving them counted would over-report the work done. LangChain's
    /// `ToolCallLimitMiddleware` does the same rollback under `"end"`.
    ///
    /// Saturates at zero.
    pub fn rollback_tool_calls(&mut self, n: usize) {
        self.tool_calls = self.tool_calls.saturating_sub(n);
    }

    /// Checks whether the run has exceeded the configured wall-clock deadline.
    ///
    /// Returns `Ok(())` when no deadline is configured or the deadline has not
    /// been reached. Returns a [`Validation`][crate::error::TinyAgentsError::Validation]
    /// error otherwise.
    pub fn check_wall_clock(&self) -> Result<()> {
        if let Some(max_ms) = self.limits.max_wall_clock_ms {
            let elapsed_ms = self.started_at.elapsed().as_millis() as u64;
            if elapsed_ms > max_ms {
                return Err(TinyAgentsError::Validation(format!(
                    "wall-clock limit ({max_ms} ms) exceeded after {elapsed_ms} ms"
                )));
            }
        }
        Ok(())
    }

    /// Returns the wall-clock time elapsed since this tracker was created.
    ///
    /// Exposed so callers (the agent loop) can compute a remaining budget
    /// against a deadline sourced from somewhere other than the run config —
    /// for example the harness-level [`RunLimits::max_wall_clock_ms`].
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns the wall-clock budget still remaining before the configured
    /// deadline, measured from this tracker's start instant.
    ///
    /// Returns `None` when no wall-clock deadline is configured (so callers
    /// should not bound work by time). When a deadline is configured the
    /// returned [`Duration`] is the remaining budget, saturating at
    /// [`Duration::ZERO`] once the deadline has already elapsed.
    ///
    /// This is the budget the agent loop uses to bound an individual model call
    /// (via `tokio::time::timeout`) so a hung or slow provider call is
    /// interrupted rather than only being detected by the between-call
    /// [`check_wall_clock`] check.
    ///
    /// [`check_wall_clock`]: LimitTracker::check_wall_clock
    pub fn remaining_wall_clock(&self) -> Option<Duration> {
        self.limits.max_wall_clock_ms.map(|max_ms| {
            let max = Duration::from_millis(max_ms);
            max.checked_sub(self.started_at.elapsed())
                .unwrap_or(Duration::ZERO)
        })
    }

    /// Returns the number of model calls recorded so far.
    pub fn model_calls(&self) -> usize {
        self.model_calls
    }

    /// Returns the number of **model-issued** tool calls recorded so far.
    ///
    /// Nested calls are counted separately
    /// ([`LimitTracker::nested_tool_calls`]); both count against the cap, so
    /// the slots left are `max_tool_calls - tool_calls - nested_tool_calls`.
    pub fn tool_calls(&self) -> usize {
        self.tool_calls
    }

    /// Returns the number of model calls remaining before the cap is hit.
    ///
    /// Returns `0` rather than wrapping if the counter has somehow already
    /// exceeded the limit.
    pub fn remaining_model_calls(&self) -> usize {
        self.limits.max_model_calls.saturating_sub(self.model_calls)
    }

    /// Returns a reference to the active [`RunLimits`] policy.
    pub fn limits(&self) -> &RunLimits {
        &self.limits
    }

    /// **Fail-open** override of the model-call and tool-call caps in place,
    /// preserving already-recorded counts and the wall-clock start time.
    ///
    /// A `RunContext` derives its tracker's initial limits from its
    /// `RunConfig`, which always carries a concrete default. That can silently
    /// disagree with a harness-wide `RunPolicy` configured with a different
    /// cap, so the *reported* limit (the policy's) and the limit that actually
    /// trips (the tracker's) diverge. The agent loop calls this once per run to
    /// reconcile the two into a single enforced source of truth.
    ///
    /// # This is a plain assignment, in **both** directions
    ///
    /// It raises a cap as readily as it lowers one. A caller writing
    /// `RunConfig::new("r").with_max_model_calls(2)` against the default policy
    /// therefore gets **25** model calls, not 2 — their explicit ceiling is
    /// silently widened by a policy default they never set.
    ///
    /// Prefer [`LimitTracker::tighten_call_limits`], which takes the stricter
    /// of the two and cannot widen anything. This method remains for the one
    /// case that genuinely needs widening: a harness-wide `RunPolicy` that
    /// deliberately configures a *higher* cap than the `RunConfig` **default**
    /// (not than an explicitly-set `RunConfig` value).
    ///
    /// Distinguishing those two cases needs information this module does not
    /// have — whether a `RunConfig` cap was set by the caller or merely
    /// defaulted. See the note on [`LimitTracker::tighten_call_limits`] for what
    /// the agent loop has to do about it.
    pub fn sync_call_limits(&mut self, max_model_calls: usize, max_tool_calls: usize) {
        tracing::debug!(
            target: "tinyagents::limits",
            from_model_calls = self.limits.max_model_calls,
            from_tool_calls = self.limits.max_tool_calls,
            to_model_calls = max_model_calls,
            to_tool_calls = max_tool_calls,
            "[limits] fail-open sync_call_limits override"
        );
        self.limits.max_model_calls = max_model_calls;
        self.limits.max_tool_calls = max_tool_calls;
    }

    /// **Fail-closed** reconciliation: keeps whichever of the tracker's current
    /// cap and the supplied cap is *stricter*, so a second limit source can
    /// only ever tighten the run, never loosen it.
    ///
    /// This is the semantics a "hard limit" needs, and the same rule
    /// [`RetryPolicy::max_attempts_capped_at`][crate::retry::RetryPolicy::max_attempts_capped_at]
    /// already applies to the retry cap. Counts and the wall-clock start are
    /// preserved.
    ///
    /// # How the agent loop actually uses this
    ///
    /// `RunConfig`'s call caps are `Option<usize>` precisely so the loop can
    /// tell "explicitly set" apart from "merely defaulted" before reconciling
    /// with the harness `RunPolicy`. Per axis (model calls, tool calls) it
    /// resolves an *effective* cap itself — the stricter of the two when the
    /// `RunConfig` cap was explicitly set, otherwise the policy's cap
    /// outright — and then calls [`LimitTracker::sync_call_limits`] once with
    /// the already-reconciled values. `tighten_call_limits` is not used there:
    /// applying it on top of an effective cap already derived from the
    /// config's *default* would additionally min against that default and so
    /// could not honor a policy that legitimately raises an unset cap. See
    /// `run_loop.rs`'s `resolve_call_cap` in `harness::agent_loop`.
    pub fn tighten_call_limits(&mut self, max_model_calls: usize, max_tool_calls: usize) {
        let model = self.limits.max_model_calls.min(max_model_calls);
        let tool = self.limits.max_tool_calls.min(max_tool_calls);
        tracing::debug!(
            target: "tinyagents::limits",
            from_model_calls = self.limits.max_model_calls,
            from_tool_calls = self.limits.max_tool_calls,
            candidate_model_calls = max_model_calls,
            candidate_tool_calls = max_tool_calls,
            to_model_calls = model,
            to_tool_calls = tool,
            "[limits] fail-closed tighten_call_limits"
        );
        self.limits.max_model_calls = model;
        self.limits.max_tool_calls = tool;
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
