//! Member worker graph execution: a generic three-step DAG that invokes a host
//! worker, routes on success/failure, and calls back to the host.
//!
//! This module owns the structure of member execution as seen by the graph
//! layer (entry → execute → complete/fail → done), while letting the host
//! supply the actual work (`run_worker`) and post-work effects (`on_complete`
//! and `on_failed`). It is the bridge between durable team coordination
//! (membership, tasks, event delivery) and the graph's lifecycle and event
//! streaming.

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;
use tinyagents_graph::export::GraphTopology;
use tinyagents_harness::CancellationToken;
use tinyagents_graph::stream::GraphEventSink;
use tinyagents_graph::{
    ClosureStateReducer, Command, CompiledGraph, GraphBuilder, NodeContext, NodeResult,
};

use crate::subagent::{
    AgentStepConfig, AgentStepError, AgentStepIdentity, StepSuccess, SubagentOutcomeKind,
    run_agent_step,
};

const LOG_PREFIX: &str = "[team-member-step]";

/// The driver-side policy and identity of one member step.
///
/// [`run_member_graph`] uses [`MemberStep::default`], which is inert (unlimited
/// admission, no timeout, one attempt, no result trimming), so the worker runs
/// exactly as it did before the step went through
/// [`SubagentDriver`](crate::subagent::SubagentDriver). Hosts that want the
/// driver's limits call [`run_member_graph_with`].
#[derive(Clone)]
pub struct MemberStep {
    /// Spawn admission, timeout/retry/budget, result policy and role.
    pub config: AgentStepConfig,
    /// Team (admission scope) and member (task id, allowlist target).
    pub identity: AgentStepIdentity,
}

impl MemberStep {
    /// A step for `member_id` of `team_id` under `config`; the member id is
    /// also the allowlist target.
    pub fn new(
        config: AgentStepConfig,
        team_id: impl Into<String>,
        member_id: impl Into<String>,
    ) -> Self {
        let member_id = member_id.into();
        Self {
            config,
            identity: AgentStepIdentity::new(team_id, member_id.clone()).with_target(member_id),
        }
    }
}

impl Default for MemberStep {
    fn default() -> Self {
        Self {
            config: AgentStepConfig::default(),
            identity: AgentStepIdentity::new("team", "member"),
        }
    }
}

/// Terminal classification of a host worker run.
///
/// Returned by a host's `run_worker` callback to signal whether the member
/// completed its work successfully or failed. The member graph routes on this
/// outcome and calls the appropriate host callback (`on_complete` or
/// `on_failed`).
pub enum MemberOutcome {
    /// Member completed successfully; carries the output to be recorded.
    Completed { output: String },
    /// Member failed; carries the failure reason to be recorded.
    Failed { reason: String },
}

#[derive(Clone, Default)]
struct MemberState {
    payload: Option<String>,
}

enum MemberUpdate {
    Payload(String),
    Noop,
}

fn graph_err(error: anyhow::Error) -> tinyagents_harness::TinyAgentsError {
    tinyagents_harness::TinyAgentsError::Graph(error.to_string())
}

/// Run the generic complete-or-fail member graph with host supplied effects.
///
/// `event_sink` is optional because observability belongs to the embedding host;
/// when supplied it receives the graph executor's lifecycle events unchanged.
pub async fn run_member_graph<W, WF, C, CF, F, FF>(
    event_sink: Option<Arc<dyn GraphEventSink>>,
    run_worker: W,
    on_complete: C,
    on_failed: F,
) -> Result<()>
where
    W: Fn() -> WF + Clone + Send + Sync + 'static,
    WF: Future<Output = Result<MemberOutcome>> + Send + 'static,
    C: Fn(String) -> CF + Clone + Send + Sync + 'static,
    CF: Future<Output = Result<()>> + Send + 'static,
    F: Fn(String) -> FF + Clone + Send + Sync + 'static,
    FF: Future<Output = Result<()>> + Send + 'static,
{
    run_member_graph_with(
        event_sink,
        MemberStep::default(),
        run_worker,
        on_complete,
        on_failed,
    )
    .await
}

/// [`run_member_graph`] with explicit driver policy.
///
/// The `execute` node runs `run_worker` through
/// [`SubagentDriver`](crate::subagent::SubagentDriver) (via
/// [`run_agent_step`]) under `step`. Outcomes map onto the existing routing:
/// a completed run goes to `on_complete` with its (result-policy trimmed)
/// output; a worker-reported failure, spawn rejection, timeout or exceeded
/// budget goes to `on_failed` with the reason; a worker `Err` still fails the
/// graph run, unchanged.
pub async fn run_member_graph_with<W, WF, C, CF, F, FF>(
    event_sink: Option<Arc<dyn GraphEventSink>>,
    step: MemberStep,
    run_worker: W,
    on_complete: C,
    on_failed: F,
) -> Result<()>
where
    W: Fn() -> WF + Clone + Send + Sync + 'static,
    WF: Future<Output = Result<MemberOutcome>> + Send + 'static,
    C: Fn(String) -> CF + Clone + Send + Sync + 'static,
    CF: Future<Output = Result<()>> + Send + 'static,
    F: Fn(String) -> FF + Clone + Send + Sync + 'static,
    FF: Future<Output = Result<()>> + Send + 'static,
{
    let step = Arc::new(step);
    let mut graph = build_member_graph(
        move || {
            let step = step.clone();
            let run_worker = run_worker.clone();
            async move { drive_member(&step, run_worker).await }
        },
        on_complete,
        on_failed,
    )?;
    if let Some(event_sink) = event_sink {
        graph = graph.with_event_sink(event_sink);
    }
    graph
        .run(MemberState::default())
        .await
        .map_err(|error| anyhow::anyhow!("member graph run failed: {error}"))?;
    Ok(())
}

/// Runs the host worker as one driver lifecycle and projects the typed
/// outcome back onto [`MemberOutcome`].
async fn drive_member<W, WF>(step: &MemberStep, run_worker: W) -> Result<MemberOutcome>
where
    W: Fn() -> WF + Clone + Send + Sync + 'static,
    WF: Future<Output = Result<MemberOutcome>> + Send + 'static,
{
    let result = run_agent_step(
        &step.config,
        step.identity.clone(),
        CancellationToken::new(),
        move |_token| {
            let fut = run_worker();
            async move {
                Ok(match fut.await? {
                    MemberOutcome::Completed { output } => StepSuccess::new(output, ()),
                    MemberOutcome::Failed { reason } => StepSuccess::incomplete(reason, ()),
                })
            }
        },
    )
    .await;
    match result {
        Ok(result) => Ok(match result.outcome.status {
            SubagentOutcomeKind::Completed => MemberOutcome::Completed {
                output: result.outcome.output,
            },
            SubagentOutcomeKind::Incomplete(incomplete) => MemberOutcome::Failed {
                reason: incomplete.reason,
            },
            SubagentOutcomeKind::Cancelled | SubagentOutcomeKind::AwaitingInput(_) => {
                MemberOutcome::Failed {
                    reason: "member step did not complete".to_owned(),
                }
            }
        }),
        Err(AgentStepError::Worker(error)) => Err(error),
        Err(error @ AgentStepError::Rejected(_)) => {
            tracing::debug!("{LOG_PREFIX} rejected member={} {error}", step.identity.task_id);
            Ok(MemberOutcome::Failed {
                reason: error.to_string(),
            })
        }
        Err(error) => Err(anyhow::anyhow!("{error}")),
    }
}

fn build_member_graph<W, WF, C, CF, F, FF>(
    run_worker: W,
    on_complete: C,
    on_failed: F,
) -> Result<CompiledGraph<MemberState, MemberUpdate>>
where
    W: Fn() -> WF + Clone + Send + Sync + 'static,
    WF: Future<Output = Result<MemberOutcome>> + Send + 'static,
    C: Fn(String) -> CF + Clone + Send + Sync + 'static,
    CF: Future<Output = Result<()>> + Send + 'static,
    F: Fn(String) -> FF + Clone + Send + Sync + 'static,
    FF: Future<Output = Result<()>> + Send + 'static,
{
    let mut builder = GraphBuilder::<MemberState, MemberUpdate>::new().set_reducer(
        ClosureStateReducer::new(|mut state: MemberState, update: MemberUpdate| {
            if let MemberUpdate::Payload(payload) = update {
                state.payload = Some(payload);
            }
            Ok(state)
        }),
    );
    builder = builder.add_node(
        "execute",
        move |_state: MemberState, _context: NodeContext| {
            let run_worker = run_worker.clone();
            async move {
                match run_worker().await.map_err(graph_err)? {
                    MemberOutcome::Completed { output } => Ok(NodeResult::Command(
                        Command::default()
                            .with_update(MemberUpdate::Payload(output))
                            .with_goto(["complete"]),
                    )),
                    MemberOutcome::Failed { reason } => Ok(NodeResult::Command(
                        Command::default()
                            .with_update(MemberUpdate::Payload(reason))
                            .with_goto(["fail"]),
                    )),
                }
            }
        },
    );
    builder = builder.add_node(
        "complete",
        move |state: MemberState, _context: NodeContext| {
            let on_complete = on_complete.clone();
            async move {
                on_complete(state.payload.unwrap_or_default())
                    .await
                    .map_err(graph_err)?;
                Ok(NodeResult::Update(MemberUpdate::Noop))
            }
        },
    );
    builder = builder.add_node("fail", move |state: MemberState, _context: NodeContext| {
        let on_failed = on_failed.clone();
        async move {
            on_failed(state.payload.unwrap_or_default())
                .await
                .map_err(graph_err)?;
            Ok(NodeResult::Update(MemberUpdate::Noop))
        }
    });
    builder
        .add_node(
            "done",
            |_state: MemberState, _context: NodeContext| async move {
                Ok(NodeResult::Update(MemberUpdate::Noop))
            },
        )
        .add_edge("complete", "done")
        .add_edge("fail", "done")
        .set_entry("execute")
        .mark_command_routing("execute")
        .set_finish("done")
        .compile()
        .map_err(|error| anyhow::anyhow!("member graph compile failed: {error}"))
}

/// Structure-only view of the generic member execution graph.
///
/// Returns a topology that reflects the nodes and edges (entry, execute,
/// complete, fail, done, routing rules) without running any real worker or
/// calling effects. Used for introspection and documentation.
pub fn member_graph_topology() -> Result<GraphTopology> {
    Ok(build_member_graph(
        || async {
            Ok(MemberOutcome::Completed {
                output: String::new(),
            })
        },
        |_| async { Ok(()) },
        |_| async { Ok(()) },
    )?
    .topology())
}

#[cfg(test)]
mod tests;
