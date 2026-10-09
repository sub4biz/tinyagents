//! Types for the compiled-graph rendition of the agent loop (A5).
//!
//! See the module doc on [`super`] for the full design and its documented
//! scope relative to the harness's direct loop.

use serde::{Deserialize, Serialize};

use tinyagents_harness::structured::StructuredStrategy;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ModelRequest;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::UsageTotals;

/// The committed graph state driven around the `plan -> model -> tools ->
/// settle` loop.
///
/// This is a whole-state graph (`Update == State`, see
/// [`crate::GraphBuilder::overwrite`]): every node returns the complete next
/// `LoopState`, not a partial patch, so [`LoopState`] doubles as its own
/// `Update` type (aliased as [`LoopUpdate`]).
///
/// Everything here is `Serialize`/`Deserialize` so a graph-driven loop can be
/// checkpointed mid-run (including at an [`crate::Interrupt`]) and resumed —
/// unlike the harness's direct-loop [`RunContext`][tinyagents_harness::context::RunContext],
/// which is deliberately non-serializable.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LoopState {
    /// The working transcript, in order.
    pub messages: Vec<Message>,
    /// Cumulative token usage across every model call so far.
    pub usage: UsageTotals,
    /// Extracted structured output, once a final turn produced one.
    pub structured: Option<serde_json::Value>,
    /// Which schema variant matched (see
    /// [`tinyagents_harness::middleware::AgentRun::structured_variant`]).
    pub structured_variant: Option<String>,
    /// Number of loop turns (model-call + optional tool-batch pairs) executed
    /// so far.
    pub turn: u32,
    /// Number of model calls dispatched so far.
    pub model_calls: usize,
    /// Number of tool invocations executed so far.
    pub tool_calls: usize,
    /// Names of calls that reached a tool executor, in execution order.
    pub executed_tools: Vec<String>,
    /// Set once the loop has produced a terminal outcome — finished, not
    /// necessarily successfully. A run that ended in an error reports it
    /// through the driver's `Result`, not through this struct.
    pub finished: bool,
    /// The final assistant text, once [`Self::finished`] is set by a normal
    /// completion, [`MiddlewareControl::StopWithFinal`][mc], or
    /// [`MiddlewareControl::JumpTo`][mc]`(`[`LoopTarget::End`][lt]`)`.
    ///
    /// [mc]: tinyagents_harness::context::MiddlewareControl
    /// [lt]: tinyagents_harness::context::LoopTarget
    pub final_text: Option<String>,
    /// The plan built by the `plan` node for the `model` node to dispatch.
    /// `None` before the first `plan` activation of a turn.
    pub(crate) pending_request: Option<ModelRequest>,
    /// The structured-output plan resolved alongside `pending_request`, when
    /// the run requested structured output.
    pub(crate) pending_structured: Option<PendingStructuredPlan>,
    /// The tool calls the `model` node's response requested, for the `tools`
    /// node to execute. Empty when the last model response requested none.
    pub(crate) pending_tool_calls: Vec<ToolCall>,
    /// The harness-assigned id of the most recent model call, for
    /// correlation.
    pub(crate) last_call_id: Option<String>,
    /// How many output-validation retries have been spent so far (bounds
    /// [`tinyagents_harness::runtime::RunPolicy::output_retry`]).
    pub(crate) output_retry_attempts: u8,
    /// Set when the run finished because a call cap tripped under
    /// `LimitBehavior::StopWithPartial`, so the driver can report
    /// `TerminalReason::LimitReached` instead of a plain completion.
    #[serde(default)]
    pub(crate) limit_stop: bool,
    /// Which cap tripped when `limit_stop` is set (`None` when unknown).
    #[serde(default)]
    pub(crate) limit_kind: Option<tinyagents_harness::events::LimitKind>,
}

impl LoopState {
    /// Seeds a fresh [`LoopState`] with `messages` as the starting
    /// transcript, everything else at its `Default`. The public constructor
    /// for callers outside this crate (this type's remaining fields are
    /// crate-private, so a struct-literal `LoopState { messages, ..Default::default() }`
    /// is not otherwise expressible from `tinyagents-integration-tests`).
    pub fn seed(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Self::default()
        }
    }
}

/// The resolved structured-output plan for the in-flight turn. Kept
/// crate-private and distinct from [`tinyagents_harness::agent_loop::phases::StructuredPlan`]
/// only in that it carries the real [`StructuredStrategy`] (needed to build a
/// [`tinyagents_harness::structured::StructuredExtractor`] without
/// re-deriving it from a string tag).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PendingStructuredPlan {
    pub(crate) strategy: StructuredStrategy,
    pub(crate) schema_name: String,
    pub(crate) schema: serde_json::Value,
}

/// [`LoopState`] doubles as its own partial-update type: every node in
/// [`super::compile_loop`]'s graph returns the whole next state (see
/// [`crate::GraphBuilder::overwrite`]).
pub type LoopUpdate = LoopState;

/// Node ids used by [`super::compile_loop`]'s compiled graph. Exposed so a
/// caller can name a node for [`super::LoopIter::override_next`] or interpret
/// a [`super::LoopStep::node`].
pub mod node {
    /// Builds the next turn's [`super::TurnPlan`][tinyagents_harness::agent_loop::phases::TurnPlan]-shaped request.
    pub const PLAN: &str = "plan";
    /// Dispatches the model call built by [`PLAN`].
    pub const MODEL: &str = "model";
    /// Executes the tool calls the last model response requested.
    pub const TOOLS: &str = "tools";
    /// Settles the turn: structured extraction/validation and the
    /// finish/continue decision.
    pub const SETTLE: &str = "settle";
}
