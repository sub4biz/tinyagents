//! Typed phase contracts and the [`LoopDriver`] seam for driving the agent
//! loop as discrete steps instead of the monolithic [`super::run_loop`] body.
//!
//! # Why this module exists
//!
//! `tinyagents-graph` depends on `tinyagents-harness` (never the reverse), so
//! a compiled-graph rendition of the agent loop (`tinyagents_graph::agent_loop`,
//! A5 in `docs/runtime-comparison/feature-gaps.md`) must live in the graph
//! crate. This module is the harness-side half of that boundary: it defines
//! the data that crosses a node edge in the graph rendition (a turn's plan,
//! a model call's outcome, a tool batch's outcome, and how a turn settled),
//! and the [`LoopDriver`] trait that lets an [`AgentHarness`] delegate
//! `invoke`/`invoke_with_status` to an alternate engine — the graph crate's
//! `GraphLoopDriver` — instead of [`super::run_loop`].
//!
//! # Scope
//!
//! [`super::run_loop`]'s body (`run_loop_body` in `run_loop.rs`) is a single
//! ~1,100-line function that interleaves request building, host-model
//! routing, cross-provider handoff transforms, truncated-empty-response
//! recovery, every structured-output strategy, host budget admission, and
//! the A6 end-strategy resolution — all behaviorally load-bearing and
//! covered by the harness's ~1,174-test baseline. Splitting that function
//! into four clean phase functions **in place** (rewriting `run_loop_body`
//! itself to call them) was judged too risky to attempt as part of this
//! change: it would touch the single highest-blast-radius function in the
//! crate with no incremental way to verify each extracted phase preserves
//! every one of those behaviors.
//!
//! Instead, this module defines the phase *contracts* — the types a graph
//! node reads and writes — and the [`LoopDriver`] hook that lets
//! `tinyagents-graph` supply a complete alternate implementation of those
//! phases (`tinyagents_graph::agent_loop`). That implementation intentionally
//! covers a **subset** of `run_loop_body`'s behavior (documented on
//! `tinyagents_graph::agent_loop::compile_loop`): the common tool-calling /
//! structured-output / limit / interrupt / steering paths, without host-model
//! routing, cross-provider handoff transforms, the deferred-tool discovery
//! bridge, or truncated-empty-response recovery. [`RunPolicy::execution`]
//! defaults to [`LoopExecution::Direct`], so every existing caller keeps
//! running the full-fidelity `run_loop_body` unless it explicitly opts in.
//!
//! [`AgentHarness`]: crate::runtime::AgentHarness
//! [`RunPolicy::execution`]: crate::runtime::RunPolicy::execution
//! [`LoopExecution::Direct`]: crate::runtime::LoopExecution::Direct

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::RunContext;
use crate::error::Result;
use crate::events::HarnessRunStatus;
use crate::middleware::AgentRun;
use crate::runtime::AgentHarness;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;

/// The structured-output plan resolved for one turn.
///
/// Mirrors the `(StructuredStrategy, name, schema)` tuple `run_loop_body`
/// computes internally, using a plain string tag for the strategy instead of
/// the private [`crate::structured::StructuredStrategy`] enum so this type
/// can be `Serialize`/`Deserialize` and cross a graph node boundary (and,
/// eventually, a checkpoint) without exposing that internal type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuredPlan {
    /// `"provider_schema"` (native `response_format`) or `"tool_call"` (a
    /// synthetic tool the model is asked to call with the schema's shape).
    pub strategy: String,
    /// The schema's name, also used as the synthetic tool's name under the
    /// `"tool_call"` strategy.
    pub schema_name: String,
    /// The JSON Schema describing the desired output shape.
    pub schema: serde_json::Value,
}

/// The strategy tag for [`StructuredPlan::strategy`] under provider-native
/// schema mode.
pub const STRATEGY_PROVIDER_SCHEMA: &str = "provider_schema";
/// The strategy tag for [`StructuredPlan::strategy`] under the tool-call
/// fallback.
pub const STRATEGY_TOOL_CALL: &str = "tool_call";

/// What one turn intends to send: the built [`ModelRequest`] plus the
/// resolved structured-output plan (if any).
///
/// Produced by a `plan_turn`-shaped step (`tinyagents_graph::agent_loop`'s
/// `plan` node) and consumed by a `call_model`-shaped step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnPlan {
    /// The request ready to dispatch to the resolved model.
    pub request: ModelRequest,
    /// The structured-output plan for this turn, when the run requested one.
    pub structured: Option<StructuredPlan>,
}

/// The outcome of dispatching one model call.
///
/// Produced by a `call_model`-shaped step (`tinyagents_graph::agent_loop`'s
/// `model` node) and consumed by the routing decision that picks `tools` or
/// `settle` next.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelOutcome {
    /// The harness-assigned correlation id for this call.
    pub call_id: String,
    /// The provider's response.
    pub response: ModelResponse,
}

/// The outcome of executing one turn's batch of tool calls.
///
/// Produced by an `execute_tool_batch`-shaped step
/// (`tinyagents_graph::agent_loop`'s `tools` node).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolBatchOutcome {
    /// The tool-result messages to append to the transcript, in call order.
    pub results: Vec<Message>,
    /// Names of calls that reached a tool executor, in execution order (see
    /// [`AgentRun::executed_tools`]).
    pub executed_tools: Vec<String>,
}

/// How a turn settled: whether the run is finished, and any structured
/// output extracted.
///
/// Produced by a `settle_turn`-shaped step (`tinyagents_graph::agent_loop`'s
/// `settle` node).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Settlement {
    /// Whether the run should end after this turn.
    pub finished: bool,
    /// The extracted structured output, when one was requested and
    /// successfully extracted/validated.
    pub structured: Option<serde_json::Value>,
    /// Which schema variant matched, under
    /// [`crate::structured::StructuredStrategy::ToolCallUnion`]. `None` for
    /// every other strategy.
    pub structured_variant: Option<String>,
}

/// A seam that lets an [`AgentHarness`] delegate its loop execution to an
/// alternate engine instead of the built-in [`super::run_loop`].
///
/// `tinyagents-graph` is the only intended implementor
/// (`GraphLoopDriver`, see `tinyagents_graph::agent_loop`): harness cannot
/// depend on the graph crate (dependency direction is graph -> harness), so
/// this trait — plus [`AgentHarness::with_loop_driver`] — is the hook the
/// graph crate uses to plug itself in without an inverted dependency.
///
/// A driver's [`Self::drive`] has the exact same contract as
/// [`super::run_loop`] (which it replaces at the call site in
/// `agent_loop::entry::drive_collecting`): it owns the whole run from
/// `RunStarted` through `before_agent`/`after_agent` middleware and the
/// terminal `RunCompleted`/`RunFailed`/pause event, writing every message it
/// produces onto `run.messages` before returning (so the transcript is
/// preserved on every exit path, including an error, exactly as the direct
/// loop preserves it — see `agent_loop`'s module docs on "Exits").
#[async_trait]
pub trait LoopDriver<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Drives one run to completion (or a deliberate pause/error), mirroring
    /// [`super::run_loop`]'s contract.
    #[allow(clippy::too_many_arguments)]
    async fn drive(
        &self,
        harness: &AgentHarness<State, Ctx>,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        input: Vec<Message>,
        streaming: bool,
    ) -> Result<()>;
}

/// Executes one turn's batch of tool calls and reports what changed.
///
/// This is a thin, behavior-preserving wrapper over the same
/// [`AgentHarness::execute_tools_with_promotions`][super::AgentHarness::execute_tools_with_promotions]
/// serial-admission / serial-or-concurrent-execution / ordered-fold pipeline
/// the direct loop uses (see `agent_loop::tools`) — unlike [`TurnPlan`] and
/// friends, this phase's *implementation*, not just its data contract, is
/// reused as-is, so a graph-driven tool batch preserves the exact ordering,
/// concurrency-eligibility, budget/limit, and middleware semantics the direct
/// loop guarantees. `messages` and `run` are mutated in place, exactly as
/// `execute_tools_with_promotions` does; the returned [`ToolBatchOutcome`] additionally
/// reports just the slice each produced, for a caller (a graph node) that
/// wants the batch's own delta rather than diffing the whole transcript
/// itself.
pub async fn execute_tool_batch<State: Send + Sync, Ctx: Send + Sync>(
    harness: &AgentHarness<State, Ctx>,
    state: &State,
    ctx: &mut RunContext<Ctx>,
    run: &mut AgentRun,
    status: &mut HarnessRunStatus,
    messages: &mut Vec<Message>,
    tool_calls: Vec<ToolCall>,
) -> Result<ToolBatchOutcome> {
    let messages_before = messages.len();
    let executed_before = run.executed_tools.len();
    harness
        .execute_tools_with_promotions(
            state,
            ctx,
            run,
            status,
            messages,
            tool_calls,
            &mut std::collections::BTreeSet::new(),
        )
        .await?;
    Ok(ToolBatchOutcome {
        results: messages[messages_before..].to_vec(),
        executed_tools: run.executed_tools[executed_before..].to_vec(),
    })
}

// ── Lifecycle events for alternate drivers ─────────────────────────────────

/// Fixes the lifecycle seed for a driver that enters a run mid-flight (a node
/// activation of the compiled-graph loop, possibly resumed from a checkpoint):
/// the first `transcript_len` messages are input and are never announced.
/// A no-op once a seed exists, so it is safe to call on every node entry.
pub fn lifecycle_seed<Ctx>(ctx: &mut RunContext<Ctx>, transcript_len: usize) {
    ctx.ensure_turn_tracker_seeded(transcript_len);
}

/// Announces transcript appends not yet announced (`MessageAppended`), using
/// the harness's payload-capture policy. Mirrors the direct loop's flush
/// points; tool calls nested inside a tool never reach the transcript, so they
/// are never announced.
pub fn lifecycle_flush<State: Send + Sync, Ctx: Send + Sync>(
    harness: &AgentHarness<State, Ctx>,
    ctx: &mut RunContext<Ctx>,
    messages: &[Message],
) {
    ctx.flush_transcript(harness.policy().capture, messages);
}

/// Opens the next turn (`TurnStarted`), first announcing pending appends and
/// closing any turn still open. Call right before the model call.
pub fn lifecycle_start_turn<State: Send + Sync, Ctx: Send + Sync>(
    harness: &AgentHarness<State, Ctx>,
    ctx: &mut RunContext<Ctx>,
    messages: &[Message],
) -> u32 {
    ctx.start_turn(harness.policy().capture, messages)
}

/// Announces pending appends and closes the open turn (`TurnCompleted`), if
/// any. Idempotent: closing with no open turn only flushes.
pub fn lifecycle_close_turn<State: Send + Sync, Ctx: Send + Sync>(
    harness: &AgentHarness<State, Ctx>,
    ctx: &mut RunContext<Ctx>,
    messages: &[Message],
) {
    ctx.close_turn(harness.policy().capture, messages);
}

/// Reports that the transcript was truncated to `new_len` messages, emitting
/// `MessageRetracted` for each announced message removed (highest index first).
/// A driver whose node discards its state (an interrupt re-runs the node from
/// its entry state) calls this so a mirror stays consistent with the transcript
/// that will actually be kept.
pub fn lifecycle_retract<Ctx>(ctx: &mut RunContext<Ctx>, new_len: usize) {
    ctx.retract_transcript(new_len);
}
