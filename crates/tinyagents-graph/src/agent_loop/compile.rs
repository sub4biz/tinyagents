//! [`compile_loop`]: assembles the `plan -> model -> tools -> settle` graph
//! over a [`LoopRuntime`].
//!
//! See the module doc on [`super`] for the loop's documented scope and the
//! shape of [`LoopState`]/[`LoopUpdate`].

use std::sync::Arc;

use crate::builder::{GraphBuilder, NodeContext};
use crate::compiled::CompiledGraph;
use tinyagents_harness::error::Result;
use tinyagents_harness::runtime::AgentHarness;

use super::runtime::{self, LoopRuntime};
use super::types::{LoopState, LoopUpdate, node};

/// Compiles the `plan -> model -> tools -> settle` agent loop into a
/// [`CompiledGraph<LoopState, LoopUpdate>`] bound to `rt`.
///
/// # Shape
///
/// [`LoopState`] doubles as its own [`LoopUpdate`] (`GraphBuilder::overwrite`,
/// see that type's docs): every node returns the whole next state rather than
/// a partial patch, so the reducer is a plain overwrite and there is no
/// separate merge step to keep in sync with the state shape.
///
/// Every node routes with an explicit [`crate::Command::goto`]
/// (`mark_command_routing`) rather than static/conditional edges — the
/// `plan`/`model`/`tools` nodes can each jump to more than one destination
/// depending on the turn's outcome (a tool-free response skips `tools`; a
/// [`tinyagents_harness::context::MiddlewareControl::JumpTo`] can loop back
/// to `plan` from `tools`, or straight to `settle` from `model`/`tools`), so a
/// fixed edge table cannot express the routing — see `runtime::apply_control`
/// for the full mapping from [`tinyagents_harness::context::MiddlewareControl`]
/// to a `goto`. `plan`, `model`, and `tools` are also marked as interrupt
/// points for the export (see this function's body, below): a steering
/// pause (from `plan`) or a
/// [`tinyagents_harness::context::MiddlewareControl::Interrupt`] (from any of
/// the three) surfaces as a real [`crate::Interrupt`], checkpointed by the
/// graph executor exactly like any other durable interrupt — this is how
/// A5's "approvals surfacing as graph interrupts" requirement is met: an
/// approval gate is just a middleware that requests
/// `MiddlewareControl::Interrupt`, and the graph rendition pauses/resumes it
/// through the same `CompiledGraph::resume` path a hand-authored
/// human-in-the-loop node would use.
///
/// # Tool batch execution
///
/// The `tools` node calls
/// [`tinyagents_harness::agent_loop::phases::execute_tool_batch`] once per
/// activation, for the *whole* turn's tool calls in one node run rather than
/// one graph node per tool call. This preserves the direct loop's ordering,
/// concurrency-eligibility, and budget/limit semantics exactly (see that
/// function's docs) without re-deriving them as a `Send`-fanout over
/// per-call nodes, which would have to reimplement the direct loop's
/// serial-admission / serial-or-concurrent-dispatch decision as graph
/// topology instead of reusing it.
///
/// # Recursion / limits
///
/// [`tinyagents_harness::limits::RunLimits`] is enforced the same way the
/// direct loop enforces it — via `ctx.record_model_call()`/tool budget
/// checks inside the node bodies, fed from the same
/// [`tinyagents_harness::context::RunContext::limits`] — so a caller who also
/// wants the *graph's own* recursion guard (independent of the harness's
/// limits) can additionally call
/// [`crate::compiled::CompiledGraph::with_recursion_policy`]/
/// [`crate::compiled::CompiledGraph::with_run_deadline`] on the returned
/// graph, exactly as any other compiled graph.
pub fn compile_loop<State, Ctx>(
    rt: Arc<LoopRuntime<State, Ctx>>,
) -> Result<CompiledGraph<LoopState, LoopUpdate>>
where
    State: Send + Sync + 'static,
    Ctx: Send + Sync + 'static,
{
    let mut builder = GraphBuilder::<LoopState, LoopUpdate>::overwrite()
        .with_name("tinyagents.agent_loop")
        .add_node(node::PLAN, {
            let rt = rt.clone();
            move |loop_state: LoopState, _ctx: NodeContext| {
                let rt = rt.clone();
                async move {
                    let harness: Arc<AgentHarness<State, Ctx>> = rt.harness.clone();
                    let mut ctx_guard = rt.ctx.lock().await;
                    runtime::plan_node(&harness, &mut ctx_guard, loop_state).await
                }
            }
        })
        .add_node(node::MODEL, {
            let rt = rt.clone();
            move |loop_state: LoopState, _ctx: NodeContext| {
                let rt = rt.clone();
                async move {
                    let harness = rt.harness.clone();
                    let app_state = rt.app_state.clone();
                    let mut ctx_guard = rt.ctx.lock().await;
                    let mut run_guard = rt.run.lock().await;
                    let mut status_guard = rt.status.lock().await;
                    runtime::model_node(
                        &harness,
                        &app_state,
                        &mut ctx_guard,
                        &mut run_guard,
                        &mut status_guard,
                        loop_state,
                    )
                    .await
                }
            }
        })
        .add_node(node::TOOLS, {
            let rt = rt.clone();
            move |loop_state: LoopState, _ctx: NodeContext| {
                let rt = rt.clone();
                async move {
                    let harness = rt.harness.clone();
                    let app_state = rt.app_state.clone();
                    let mut ctx_guard = rt.ctx.lock().await;
                    let mut run_guard = rt.run.lock().await;
                    let mut status_guard = rt.status.lock().await;
                    runtime::tools_node(
                        &harness,
                        &app_state,
                        &mut ctx_guard,
                        &mut run_guard,
                        &mut status_guard,
                        loop_state,
                    )
                    .await
                }
            }
        })
        .add_node(node::SETTLE, {
            let rt = rt.clone();
            move |loop_state: LoopState, _ctx: NodeContext| {
                let rt = rt.clone();
                async move {
                    let harness = rt.harness.clone();
                    let mut ctx_guard = rt.ctx.lock().await;
                    let mut run_guard = rt.run.lock().await;
                    runtime::settle_node(&harness, &mut ctx_guard, &mut run_guard, loop_state)
                        .await
                }
            }
        })
        .set_entry(node::PLAN)
        .mark_command_routing(node::PLAN)
        .mark_command_routing(node::MODEL)
        .mark_command_routing(node::TOOLS)
        .mark_command_routing(node::SETTLE);

    // `plan`/`model`/`tools` each already return a real, node-emitted
    // `NodeResult::Interrupt` when they need to pause (a steering pause from
    // `plan`, a `MiddlewareControl::Interrupt` from any of the three — see
    // the module doc above); that alone is a genuine, checkpointed executor
    // pause, with no help from `GraphBuilder::mark_interrupt` needed.
    // `mark_interrupt` is *not* used here because — unlike when this graph
    // was first written — it is no longer a behavior-free export marker: it
    // now aliases `GraphBuilder::interrupt_before`, which would make the
    // executor pause *every* activation of these nodes on its own, before
    // the node (and its middleware) ever runs, double-pausing on top of the
    // node's own interrupt and desyncing resume's interrupt-acknowledgement
    // bookkeeping across turns. Setting the `NodeMeta` interrupt marker
    // directly (`pub(crate)`, reachable from this sibling module) restores
    // the original export-only annotation without opting into that runtime
    // pause.
    for node in [node::PLAN, node::MODEL, node::TOOLS] {
        builder.node_meta.entry(node.into()).or_default().interrupt = true;
    }

    builder.compile()
}
