//! Types for run-scoped limit enforcement.
//!
//! [`RunLimits`] carries the configured policy — including the
//! [`RunLimits::max_depth`] recursion cap that bounds how far the sub-agent /
//! sub-graph run tree may nest; [`LimitTracker`] (in the sibling `mod.rs`)
//! holds the live counters and checks them against the policy.

/// Configures the hard limits applied across a single harness run.
///
/// All limits are checked fail-closed: the first call that exceeds a cap
/// returns an error and the run should be stopped.
///
/// # Examples
///
/// ```
/// use tinyagents_harness::limits::RunLimits;
///
/// let limits = RunLimits::default()
///     .with_max_model_calls(10)
///     .with_max_tool_calls(20);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RunLimits {
    /// Maximum number of model API calls permitted for this run.
    pub max_model_calls: usize,
    /// Maximum number of tool invocations permitted for this run.
    pub max_tool_calls: usize,
    /// Maximum elapsed wall-clock time in milliseconds. `None` means no limit.
    pub max_wall_clock_ms: Option<u64>,
    /// Maximum wall-clock time in milliseconds for a **single model call**,
    /// applied afresh to every call (and every retry attempt). `None` means no
    /// per-call ceiling.
    ///
    /// This bounds an individual hung or wedged provider call independently of
    /// [`max_wall_clock_ms`](Self::max_wall_clock_ms), which measures the whole
    /// run. Without it the only per-call bound is the run's *remaining*
    /// wall-clock budget, which conflates two different guards: hang detection
    /// (a wedged call should die fast) and runaway-run bounding (a run should
    /// not live forever). With only the run deadline, a host must choose one
    /// number for both — a generous run ceiling means a hung call can hold the
    /// run for that whole ceiling, and a tight one kills late calls in long,
    /// *productive* runs even though every earlier call succeeded.
    ///
    /// The effective budget for a model call is the tighter of this ceiling and
    /// the run's remaining wall-clock budget, so a per-call cap can never
    /// extend a run past its deadline. Deliberately **not** applied to tool
    /// calls: a sub-agent delegation is a tool call wrapping an entire child
    /// run and must not inherit a model-call-sized cap; tools carry their own
    /// [`ToolTimeoutSettings`][crate::tool::ToolTimeoutSettings]
    /// deadlines and remain bounded by the run's remaining budget.
    ///
    /// Size it generously: a hidden-reasoning model call can be legitimately
    /// silent for minutes, so this is a backstop for calls that will never
    /// return, not a latency target.
    pub max_model_call_ms: Option<u64>,
    /// Maximum number of retry *attempts* (not counting the first try)
    /// permitted for an individual model call. Reconciled with
    /// [`crate::retry::RetryPolicy::max_attempts`] by the agent loop
    /// (see [`crate::retry::RetryPolicy::max_attempts_capped_at`]):
    /// whichever of the two is stricter wins, so this is a hard ceiling a
    /// looser `RetryPolicy` cannot exceed.
    pub max_retries_per_call: usize,
    /// Maximum sub-agent / recursion depth allowed for the run tree rooted at
    /// this run. A top-level run is depth `0`; each nested child run increments
    /// the depth. A sub-agent invocation whose child depth would exceed this cap
    /// fails fast (see the `tinyagents-orchestration` sub-agent invoker). Defaults to
    /// [`RunLimits::DEFAULT_MAX_DEPTH`].
    pub max_depth: usize,
    /// What the run should do when a call cap is reached. Defaults to
    /// [`LimitBehavior::Error`], which is the historical behaviour.
    pub behavior: LimitBehavior,
    /// Caps how many tool calls in one concurrently-executed batch (see
    /// [`should_execute_tools_concurrently`][crate::agent_loop] and its
    /// module docs) may be in flight at once. `None` (the default) leaves the
    /// batch unbounded — every eligible call in the turn starts together, as
    /// before this field existed.
    ///
    /// Only applies to the concurrent tool path; the serial path always runs
    /// one call at a time regardless of this setting. A `Some(0)` behaves the
    /// same as `Some(1)`: at least one call must be in flight to make
    /// progress.
    pub max_tool_concurrency: Option<usize>,
    /// Maximum depth of tool calls a tool makes through
    /// [`ToolExecutionContext::call_tool`][crate::tool::ToolExecutionContext::call_tool].
    /// A call the model issues is level `0`; what that tool calls is level
    /// `1`, and so on. A nested call whose level would exceed this cap fails
    /// with a clear error. Defaults to [`RunLimits::DEFAULT_MAX_NESTED_DEPTH`],
    /// `0`: **nested calls are off** and `call_tool` fails with "nested tool
    /// calls are disabled (max_nested_depth = 0)".
    ///
    /// Opt in (`with_max_nested_depth`) only once every `before_tool`
    /// enforcement the host registers also implements
    /// [`Middleware::check_nested_tool`][crate::middleware::Middleware::check_nested_tool];
    /// `before_tool` never runs for nested calls, so an enforcement that lacks
    /// it is bypassed by `call_tool`.
    ///
    /// Distinct from [`Self::max_depth`], which bounds sub-*agent* recursion.
    pub max_nested_depth: usize,
    /// Maximum silence, in milliseconds, between output events of a
    /// **streaming** model call, measured from the previous output event.
    /// `None` (or `Some(0)`) disables the inactivity timeout. Defaults to
    /// [`RunLimits::DEFAULT_STREAM_IDLE_TIMEOUT_MS`].
    ///
    /// Applies only **after** the first output event; before it, see
    /// [`stream_first_event_timeout_ms`](Self::stream_first_event_timeout_ms).
    /// The deadline is advanced only by output events (text, reasoning, tool
    /// fragments, block events). The stream-opened marker and usage updates
    /// never extend it, so a provider that keeps sending those cannot keep a
    /// stalled call alive.
    ///
    /// [`max_model_call_ms`](Self::max_model_call_ms) bounds a call's *total*
    /// duration, so a provider that starts answering and then goes quiet holds
    /// the call for that whole ceiling. This timer is re-armed on every output
    /// event, so a long answer that keeps producing tokens is never cut off
    /// while a wedged stream dies fast. When it fires the call fails with the
    /// retryable
    /// [`TinyAgentsError::CallTimeout`][crate::error::TinyAgentsError::CallTimeout],
    /// so the normal retry/fallback path takes over. Not applied to
    /// non-streaming calls, which produce no intermediate events to measure
    /// (those are bounded by `max_model_call_ms`).
    pub stream_idle_timeout_ms: Option<u64>,
    /// Opt-in maximum wait, in milliseconds, for the **first output event** of
    /// a streaming model call. `None` (the default, and `Some(0)`) means no
    /// separate bound: the wait is limited only by
    /// [`max_model_call_ms`](Self::max_model_call_ms) and the run's deadline.
    ///
    /// Off by default and independent of
    /// [`stream_idle_timeout_ms`](Self::stream_idle_timeout_ms), because
    /// providers that hide their reasoning and local models doing a long CPU
    /// prefill are legitimately silent for many minutes before the first
    /// token. Set it only when the provider is known to answer promptly. The
    /// stream-opened marker and usage updates do not end this phase.
    pub stream_first_event_timeout_ms: Option<u64>,
    /// Circuit breaker: after this many *consecutive* stream idle timeouts on
    /// one model (idle timeouts since the last output event), retrying that
    /// model stops. The fallback chain is still consulted, with a fresh count
    /// for each model; when the chain is exhausted the run fails with
    /// [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded].
    /// Any output event resets the count. `None` (or `Some(0)`) disables the
    /// breaker. Defaults to
    /// [`RunLimits::DEFAULT_MAX_CONSECUTIVE_STREAM_IDLE_TIMEOUTS`].
    ///
    /// Without it a stalled provider is hit once per retry attempt, each
    /// paying the full window. Because any output event resets the count, it
    /// trips on streams that stall *before* producing output, so in practice
    /// it pairs with [`stream_first_event_timeout_ms`](Self::stream_first_event_timeout_ms);
    /// a stream that emits a token and then stalls on every attempt keeps
    /// making progress and is bounded by the retry cap instead.
    pub max_consecutive_stream_idle_timeouts: Option<usize>,
}

/// What a run does when it reaches a configured call cap.
///
/// Ported from LangChain's `exit_behavior` on `ModelCallLimitMiddleware` /
/// `ToolCallLimitMiddleware`. Every cap here used to be a hard error, which
/// throws away the whole run *and everything it already accomplished* — for a
/// long research run that is often the worst possible outcome, since the
/// partial answer was the valuable part.
///
/// The variant only names the *policy*; the agent loop is what acts on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LimitBehavior {
    /// Fail the run with
    /// [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded].
    /// The historical (and still default) behaviour.
    #[default]
    Error,

    /// Stop the loop cleanly and return whatever the run has produced so far,
    /// as if the model had finished its turn.
    ///
    /// # Contract for the agent loop (wave 2)
    ///
    /// When [`LimitTracker::record_model_call`][crate::limits::LimitTracker::record_model_call]
    /// or [`record_tool_call`][crate::limits::LimitTracker::record_tool_call]
    /// reports exhaustion under this behaviour they return
    /// [`LimitOutcome::Stop`] instead of `Err`, and the loop must:
    ///
    /// 1. Emit the existing
    ///    [`AgentEvent::LimitReached`][crate::events::AgentEvent::LimitReached]
    ///    so the stop is still observable and still distinguishable from a
    ///    model that simply finished.
    /// 2. Break out of the loop and finalize normally, keeping the transcript
    ///    accumulated so far as the run result.
    /// 3. For the **tool** cap specifically, mirror LangChain's `"end"` path:
    ///    append a tool result for every remaining requested call saying it was
    ///    stopped before it could run (the provider APIs require every
    ///    `tool_call_id` to be answered, so skipping them corrupts the
    ///    transcript), and **roll the counter back** by the number of calls that
    ///    never executed via
    ///    [`LimitTracker::rollback_tool_calls`][crate::limits::LimitTracker::rollback_tool_calls],
    ///    so the reported count reflects work actually done.
    StopWithPartial,
}

impl LimitBehavior {
    /// Stable, snake_case label for logs and telemetry dimensions.
    pub fn as_str(self) -> &'static str {
        match self {
            LimitBehavior::Error => "error",
            LimitBehavior::StopWithPartial => "stop_with_partial",
        }
    }
}

/// The result of recording a call against a [`RunLimits`] cap.
///
/// Returned by the `try_record_*` methods so the caller sees the cap decision
/// as data rather than only as a `Result`. The `record_*` methods remain for
/// callers that always want the hard-error form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitOutcome {
    /// The call is within the cap; carry on.
    Proceed,
    /// The cap is exhausted and [`RunLimits::behavior`] is
    /// [`LimitBehavior::StopWithPartial`]: stop the loop cleanly and return
    /// what the run has so far. Carries which cap tripped.
    Stop(LimitKind),
}

/// Names which cap a [`LimitOutcome::Stop`] refers to.
///
/// Deliberately mirrors
/// [`crate::events::LimitKind`] rather than reusing it, so the limits
/// module (a leaf with no event dependency) does not have to import the
/// observability layer. Convert with [`LimitKind::as_str`] when emitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitKind {
    /// The per-run model-call cap.
    ModelCalls,
    /// The per-run tool-call cap.
    ToolCalls,
}

impl LimitKind {
    /// Stable, snake_case label matching
    /// [`crate::events::LimitKind::as_str`].
    pub fn as_str(self) -> &'static str {
        match self {
            LimitKind::ModelCalls => "model_calls",
            LimitKind::ToolCalls => "tool_calls",
        }
    }
}

impl RunLimits {
    /// Default sub-agent / recursion depth cap when none is configured.
    pub const DEFAULT_MAX_DEPTH: usize = 8;
    /// Default [`RunLimits::max_nested_depth`]: `0`, nested calls disabled.
    pub const DEFAULT_MAX_NESTED_DEPTH: usize = 0;
    /// Default [`RunLimits::stream_idle_timeout_ms`]: two minutes of silence.
    ///
    /// Generous on purpose: providers that hide reasoning can be quiet for a
    /// long time between events, and this is a wedge detector, not a latency
    /// target.
    pub const DEFAULT_STREAM_IDLE_TIMEOUT_MS: u64 = 120_000;
    /// Default [`RunLimits::max_consecutive_stream_idle_timeouts`].
    pub const DEFAULT_MAX_CONSECUTIVE_STREAM_IDLE_TIMEOUTS: usize = 5;
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            max_model_calls: 25,
            max_tool_calls: 50,
            max_wall_clock_ms: None,
            max_model_call_ms: None,
            max_retries_per_call: 3,
            max_depth: Self::DEFAULT_MAX_DEPTH,
            behavior: LimitBehavior::Error,
            max_tool_concurrency: None,
            max_nested_depth: Self::DEFAULT_MAX_NESTED_DEPTH,
            stream_idle_timeout_ms: Some(Self::DEFAULT_STREAM_IDLE_TIMEOUT_MS),
            stream_first_event_timeout_ms: None,
            max_consecutive_stream_idle_timeouts: Some(
                Self::DEFAULT_MAX_CONSECUTIVE_STREAM_IDLE_TIMEOUTS,
            ),
        }
    }
}
