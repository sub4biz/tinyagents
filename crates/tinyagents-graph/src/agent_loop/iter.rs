//! [`AgentLoopGraphExt::iter`]/[`LoopIter`]: steps the compiled loop one node
//! at a time.
//!
//! See the module doc on [`super`] for how this relates to [`super::compile_loop`]
//! and [`super::GraphLoopDriver`].

use std::sync::Arc;

use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::events::HarnessRunStatus;
use tinyagents_harness::ids::ComponentId;
use tinyagents_harness::middleware::AgentRun;
use tinyagents_harness::runtime::AgentHarness;
use tinyinference_llm::message::Message;

use crate::command::NodeResult;

use super::runtime::{self, LoopRuntime};
use super::types::{LoopState, node};

/// One completed activation reported by [`LoopIter::next`].
#[derive(Clone, Debug)]
pub struct LoopStep {
    /// The node that just ran.
    pub node: String,
    /// The node [`LoopIter::next`] will run next, unless
    /// [`LoopIter::override_next`] changes it first. `None` once the run has
    /// finished (reached `END`).
    pub next: Option<String>,
    /// Whether this activation interrupted the run (a steering pause or a
    /// [`tinyagents_harness::context::MiddlewareControl::Interrupt`]) rather
    /// than completing normally. A caller that wants to resume calls
    /// [`LoopIter::next`] again — the interrupted node re-runs, exactly like
    /// [`crate::compiled::CompiledGraph::resume`] re-running an interrupted
    /// node — after applying whatever unblocks it (for example draining a
    /// [`tinyagents_harness::steering::SteeringHandle`]).
    pub interrupted: bool,
}

/// Extension trait adding [`Self::iter`] to `Arc<`[`AgentHarness`]`<State,
/// Ctx>>`, mirroring pydantic-ai's `Agent.iter` (`docs/runtime-comparison/pydantic-ai.md`
/// §4).
///
/// Implemented for `Arc<AgentHarness<State, Ctx>>` rather than
/// `AgentHarness<State, Ctx>` directly because the returned [`LoopIter`]
/// keeps driving the loop across many `.await` points spanning its own
/// lifetime (unlike a single [`AgentHarness::invoke`] call), so it needs to
/// own a durable handle to the harness — see the module doc on [`super`]
/// ("`GraphLoopDriver` vs. `compile_loop`/`LoopIter`") for why that rules out
/// a borrowed `&AgentHarness` the way [`super::GraphLoopDriver`] gets one.
pub trait AgentLoopGraphExt<State: Send + Sync + 'static, Ctx: Send + Sync + 'static> {
    /// Starts a steppable run: builds the compiled `plan -> model -> tools ->
    /// settle` graph (see [`super::compile_loop`]) and returns a
    /// [`LoopIter`] positioned at its entry (`plan`), seeded with `input` as
    /// the starting transcript. `app_state` is the harness's shared,
    /// read-only application state (the same value an
    /// [`AgentHarness::invoke`] caller would pass as `state`); `ctx` supplies
    /// the run identity and dependencies exactly as for `invoke`.
    fn iter(
        self,
        app_state: Arc<State>,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<LoopIter<State, Ctx>>;
}

impl<State, Ctx> AgentLoopGraphExt<State, Ctx> for Arc<AgentHarness<State, Ctx>>
where
    State: Send + Sync + 'static,
    Ctx: Send + Sync + 'static,
{
    fn iter(
        self,
        app_state: Arc<State>,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<LoopIter<State, Ctx>> {
        LoopIter::new(self, app_state, ctx, input)
    }
}

/// Steps the compiled agent loop one node activation at a time.
///
/// Built via [`AgentLoopGraphExt::iter`]. Each [`Self::next`] call runs
/// exactly one node body (the same [`super::runtime::plan_node`]/`model_node`/
/// `tools_node`/`settle_node`  [`super::compile_loop`] wires into a
/// [`crate::CompiledGraph`]) against this iterator's own [`LoopRuntime`], so
/// stepping through a `LoopIter` observes the identical transcript, usage,
/// and routing decisions a full [`crate::CompiledGraph::run`] over the same
/// graph would — just one activation at a time, with [`Self::override_next`]
/// able to redirect the very next activation (for tests, debugging, or a
/// host that wants to splice in extra bookkeeping between turns).
pub struct LoopIter<State: Send + Sync + 'static, Ctx: Send + Sync + 'static> {
    rt: Arc<LoopRuntime<State, Ctx>>,
    state: LoopState,
    next: Option<String>,
    overridden: Option<String>,
    finished: bool,
}

impl<State, Ctx> LoopIter<State, Ctx>
where
    State: Send + Sync + 'static,
    Ctx: Send + Sync + 'static,
{
    pub(crate) fn new(
        harness: Arc<AgentHarness<State, Ctx>>,
        app_state: Arc<State>,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<Self> {
        let run_id = ctx.run_id().clone();
        let status = HarnessRunStatus::new(run_id, ComponentId::new("agent_loop.iter"));
        let rt = Arc::new(LoopRuntime::new(
            harness,
            app_state,
            ctx,
            AgentRun::default(),
            status,
            false,
        ));
        Ok(Self {
            rt,
            state: LoopState {
                messages: input,
                ..LoopState::default()
            },
            next: Some(node::PLAN.to_string()),
            overridden: None,
            finished: false,
        })
    }

    /// The committed [`LoopState`] as of the last completed [`Self::next`]
    /// call (or the seeded input, before the first call).
    pub fn state(&self) -> &LoopState {
        &self.state
    }

    /// The accumulated [`AgentRun`] mirrored alongside the loop state —
    /// useful for reading `usage`/`model_calls`/`tool_calls` without waiting
    /// for `settle` to populate [`LoopState::final_text`].
    pub async fn run(&self) -> AgentRun {
        self.rt.run.lock().await.clone()
    }

    /// The node [`Self::next`] will run on its next call, or `None` once the
    /// run has finished.
    pub fn next_node(&self) -> Option<&str> {
        self.next.as_deref()
    }

    /// Redirects the very next [`Self::next`] call to `node` instead of
    /// wherever the last activation routed to. Consumed by that one call —
    /// subsequent calls follow the graph's normal routing again unless
    /// overridden again. A no-op once the run has finished.
    pub fn override_next(&mut self, node: impl Into<String>) {
        if !self.finished {
            self.overridden = Some(node.into());
        }
    }

    /// Runs exactly one node activation and reports what happened, or `Ok(None)`
    /// if the run had already finished.
    pub async fn next(&mut self) -> Result<Option<LoopStep>> {
        if self.finished {
            return Ok(None);
        }
        let current = self
            .overridden
            .take()
            .or_else(|| self.next.clone())
            .ok_or_else(|| {
                TinyAgentsError::Validation("LoopIter::next: no node to run next".to_string())
            })?;

        let loop_state = std::mem::take(&mut self.state);
        let harness = self.rt.harness.clone();
        let app_state = self.rt.app_state.clone();
        let mut ctx_guard = self.rt.ctx.lock().await;
        let mut run_guard = self.rt.run.lock().await;
        let mut status_guard = self.rt.status.lock().await;

        let result = match current.as_str() {
            node::PLAN => runtime::plan_node(&harness, &mut ctx_guard, loop_state).await?,
            node::MODEL => {
                runtime::model_node(
                    &harness,
                    &app_state,
                    &mut ctx_guard,
                    &mut run_guard,
                    &mut status_guard,
                    loop_state,
                )
                .await?
            }
            node::TOOLS => {
                runtime::tools_node(
                    &harness,
                    &app_state,
                    &mut ctx_guard,
                    &mut run_guard,
                    &mut status_guard,
                    loop_state,
                )
                .await?
            }
            node::SETTLE => {
                runtime::settle_node(&harness, &mut ctx_guard, &mut run_guard, loop_state).await?
            }
            other => {
                return Err(TinyAgentsError::Validation(format!(
                    "LoopIter::next: unknown loop node `{other}`"
                )));
            }
        };
        drop((ctx_guard, run_guard, status_guard));

        match result {
            NodeResult::Interrupt(_interrupt) => Ok(Some(LoopStep {
                node: current.clone(),
                // The interrupted node is the natural resume target: calling
                // `next()` again re-runs it, mirroring
                // `CompiledGraph::resume`.
                next: Some(current),
                interrupted: true,
            })),
            NodeResult::Command(command) => {
                self.state = command.update.ok_or_else(|| {
                    TinyAgentsError::Validation(
                        "LoopIter::next: loop node's command carried no update".to_string(),
                    )
                })?;
                let target = command.goto.first().ok_or_else(|| {
                    TinyAgentsError::Validation(
                        "LoopIter::next: loop node's command carried no route".to_string(),
                    )
                })?;
                let crate::command::RouteTarget::Node(node_id) = target else {
                    return Err(TinyAgentsError::Validation(
                        "LoopIter::next: loop node routed via `Send`, which `LoopIter` does not \
                         support"
                            .to_string(),
                    ));
                };
                let next_str = node_id.as_str().to_string();
                if next_str == crate::builder::END {
                    self.finished = true;
                    self.next = None;
                    Ok(Some(LoopStep {
                        node: current,
                        next: None,
                        interrupted: false,
                    }))
                } else {
                    self.next = Some(next_str.clone());
                    Ok(Some(LoopStep {
                        node: current,
                        next: Some(next_str),
                        interrupted: false,
                    }))
                }
            }
            NodeResult::Update(_) => Err(TinyAgentsError::Validation(
                "LoopIter::next: loop node returned an un-routed update".to_string(),
            )),
        }
    }

    /// Steps to completion (or the first unresolved interrupt), returning the
    /// final [`LoopState`].
    ///
    /// Stops — without erroring — the moment [`Self::next`] reports an
    /// interrupted step, leaving [`Self::next_node`] pointing at the
    /// interrupted node so a caller can resolve whatever paused it (drain
    /// steering, record an approval) and call [`Self::run_to_end`] again to
    /// continue.
    pub async fn run_to_end(&mut self) -> Result<&LoopState> {
        while let Some(step) = self.next().await? {
            if step.interrupted {
                break;
            }
        }
        Ok(&self.state)
    }
}
