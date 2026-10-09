//! Configuration and lifecycle hooks for final-answer verification.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ModelRequest, ModelResponse};

use super::types::{
    DEFAULT_MIN_REMAINING_WALL_CLOCK, FinishActivity, FinishCheckTrigger,
    MIN_REMAINING_MODEL_CALLS, RunState, VerifyBeforeFinishMiddleware,
};
use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::events::AgentEvent;
use crate::middleware::{AgentRun, Middleware};

impl VerifyBeforeFinishMiddleware {
    /// A middleware that appends `check` as a user turn. By default it fires
    /// for any run that used at least one tool round, so a plain chat answer is
    /// never second-guessed.
    pub fn new(check: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            trigger: min_rounds_trigger(1),
            min_remaining_wall_clock: DEFAULT_MIN_REMAINING_WALL_CLOCK,
            wall_clock_limit: None,
            runs: Mutex::default(),
            wrap_up: None,
        }
    }

    /// Stay quiet in any run where `wrap_up` has already announced a budget
    /// notice, so the check ("re-read and verify") cannot contradict a notice
    /// telling the model to stop gathering and finish. Pass the same `Arc`
    /// that is installed as middleware.
    pub fn with_wrap_up(
        mut self,
        wrap_up: Arc<crate::middleware::library::FinalCallWrapUpMiddleware>,
    ) -> Self {
        self.wrap_up = Some(wrap_up);
        self
    }

    /// Fire only for runs with at least `rounds` tool rounds. Replaces any
    /// trigger set before.
    pub fn with_min_tool_rounds(mut self, rounds: usize) -> Self {
        self.trigger = min_rounds_trigger(rounds);
        self
    }

    /// Fire only when `trigger` accepts the run's activity. Replaces any
    /// trigger set before.
    pub fn with_trigger(
        mut self,
        trigger: impl Fn(&FinishActivity) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.trigger = Arc::new(trigger);
        self
    }

    /// Skip the check when the run's wall-clock deadline is closer than `min`.
    /// A run with no deadline is never skipped for time.
    pub fn with_min_remaining_wall_clock(mut self, min: Duration) -> Self {
        self.min_remaining_wall_clock = min;
        self
    }

    /// The run's policy-level wall-clock cap
    /// ([`RunLimits::max_wall_clock_ms`](crate::limits::RunLimits::max_wall_clock_ms)),
    /// measured from the run's start. A `RunPolicy` cap is enforced by the loop
    /// but is not on the [`RunContext`] a middleware sees, so a host that sets
    /// one passes it here too; the context's own deadline
    /// ([`RunContext::remaining_wall_clock`]) is always honoured.
    pub fn with_wall_clock_limit(mut self, limit: Duration) -> Self {
        self.wall_clock_limit = Some(limit);
        self
    }

    /// The tighter of the context deadline and the declared policy cap.
    fn remaining_wall_clock<C>(&self, ctx: &RunContext<C>) -> Option<Duration> {
        let policy = self
            .wall_clock_limit
            .map(|limit| limit.saturating_sub(ctx.limits.elapsed()));
        match (ctx.remaining_wall_clock(), policy) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Why this response must not be held for the check, or `None` when it may.
    fn skip_reason<C>(
        &self,
        ctx: &RunContext<C>,
        response: &ModelResponse,
    ) -> Option<&'static str> {
        if !response.tool_calls().is_empty() {
            return Some("not_final");
        }
        // Truncation first: a reasoning model that spends its whole output
        // budget on the hidden channel returns `length` with no text at all.
        // Reported as `empty_answer` that reads like the model declining to
        // answer; it is a call that produced nothing after running out of
        // room, and the two call for different responses from whoever reads
        // the log.
        if crate::finish_reason::is_length_stop(response.finish_reason.as_deref()) {
            return Some(if response.text().trim().is_empty() {
                "truncated_before_any_output"
            } else {
                "truncated"
            });
        }
        if response.text().trim().is_empty() {
            return Some("empty_answer");
        }
        if response.continue_turn.is_some() {
            return Some("already_continued");
        }
        if self
            .wrap_up
            .as_ref()
            .is_some_and(|wrap_up| wrap_up.budget_notice_announced(ctx))
        {
            return Some("budget_notice_announced");
        }
        if ctx.limits.remaining_model_calls() < MIN_REMAINING_MODEL_CALLS {
            return Some("model_call_budget");
        }
        if self
            .remaining_wall_clock(ctx)
            .is_some_and(|left| left < self.min_remaining_wall_clock)
        {
            return Some("wall_clock");
        }
        None
    }
}

const MAX_RETAINED_RUNS: usize = 1_024;
const RESUME_KEY: &str = "tinyagents.verify_before_finish.v1";

fn state_for<'a, C>(runs: &'a mut HashMap<u64, RunState>, ctx: &RunContext<C>) -> &'a mut RunState {
    // A cancelled run can bypass both terminal hooks. Its context marker is
    // gone, so prune it without discarding any still-active run's fired flag.
    if runs.len() >= MAX_RETAINED_RUNS {
        runs.retain(|_, run| run.lifecycle.upgrade().is_some());
    }
    let run = runs.entry(ctx.instance_id()).or_default();
    run.lifecycle = Arc::downgrade(&ctx.lifecycle);
    run
}

fn min_rounds_trigger(rounds: usize) -> FinishCheckTrigger {
    Arc::new(move |activity: &FinishActivity| activity.tool_rounds >= rounds)
}

#[async_trait]
impl<S: Send + Sync, C: Send + Sync> Middleware<S, C> for VerifyBeforeFinishMiddleware {
    fn name(&self) -> &str {
        "verify_before_finish"
    }

    async fn before_agent(&self, ctx: &mut RunContext<C>, _state: &S) -> Result<()> {
        if let Some(results) = ctx.deferred_results.as_ref()
            && let Ok(mut runs) = self.runs.lock()
        {
            let run = state_for(&mut runs, ctx);
            if let Some(saved) = results.resume_metadata.get(RESUME_KEY)
                && let Ok((activity, fired)) =
                    serde_json::from_value::<(FinishActivity, bool)>(saved.clone())
            {
                run.activity = activity;
                run.fired = fired;
            } else {
                // Legacy result sets did not carry middleware state. Recover
                // only what the surviving transcript can prove.
                run.activity.tool_rounds = 1;
                run.restore_deferred = true;
            }
        }
        Ok(())
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<C>,
        _state: &S,
        request: &mut ModelRequest,
    ) -> Result<()> {
        if let Ok(mut runs) = self.runs.lock()
            && let Some(run) = runs.get_mut(&ctx.instance_id())
            && run.restore_deferred
        {
            run.restore_deferred = false;
            let mut rounds = 0;
            for message in &request.messages {
                if let Message::Assistant(assistant) = message
                    && !assistant.tool_calls.is_empty()
                {
                    rounds += 1;
                    run.activity
                        .tools_called
                        .extend(assistant.tool_calls.iter().map(|call| call.name.clone()));
                }
            }
            run.activity.tool_rounds = run.activity.tool_rounds.max(rounds);
        }
        Ok(())
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<C>,
        _state: &S,
        response: &mut ModelResponse,
    ) -> Result<()> {
        let Ok(mut runs) = self.runs.lock() else {
            tracing::warn!("[tinyagents::mw] verify_before_finish state poisoned; not checking");
            return Ok(());
        };
        let run = state_for(&mut runs, ctx);

        let calls = response.tool_calls();
        if !calls.is_empty() {
            run.activity.tool_rounds += 1;
            run.activity
                .tools_called
                .extend(calls.iter().map(|call| call.name.clone()));
            return Ok(());
        }
        if run.fired {
            return Ok(());
        }
        if let Some(reason) = self.skip_reason(ctx, response) {
            tracing::debug!(
                reason,
                tool_rounds = run.activity.tool_rounds,
                remaining_model_calls = ctx.limits.remaining_model_calls(),
                "[tinyagents::mw] verify_before_finish skipped"
            );
            return Ok(());
        }
        if !(self.trigger)(&run.activity) {
            tracing::debug!(
                tool_rounds = run.activity.tool_rounds,
                "[tinyagents::mw] verify_before_finish not triggered by this run's activity"
            );
            return Ok(());
        }

        run.fired = true;
        let tool_rounds = run.activity.tool_rounds;
        drop(runs);
        response.continue_turn = Some(self.check.clone());
        // The check is the one call in a run where thinking is worth the
        // bounded risk of a dead call: a result fitted on the wrong axis is
        // caught by asking what the request implied, which a model running
        // without reasoning (the fallback after dead calls) does not do.
        ctx.request_reasoning();
        tracing::info!(
            tool_rounds,
            remaining_model_calls = ctx.limits.remaining_model_calls(),
            "[tinyagents::mw] verify_before_finish holding the first final answer for one check"
        );
        ctx.emit(AgentEvent::ControlApplied {
            control: "verify_before_finish".to_string(),
            detail: format!(
                "final answer after {tool_rounds} tool round(s) held for one check against the \
                 request"
            ),
        });
        Ok(())
    }

    async fn on_error(&self, ctx: &mut RunContext<C>, _error: &TinyAgentsError) -> Result<()> {
        if let Ok(mut runs) = self.runs.lock() {
            runs.remove(&ctx.instance_id());
        }
        Ok(())
    }

    async fn after_agent(
        &self,
        ctx: &mut RunContext<C>,
        _state: &S,
        run: &mut AgentRun,
    ) -> Result<()> {
        let state = self
            .runs
            .lock()
            .ok()
            .and_then(|mut runs| runs.remove(&ctx.instance_id()));
        if let (Some(state), Some(deferred)) = (state, run.deferred.as_mut()) {
            deferred.resume_metadata.insert(
                RESUME_KEY.to_string(),
                serde_json::to_value((&state.activity, state.fired))
                    .expect("finish activity is serializable"),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "middleware_tests.rs"]
mod tests;
