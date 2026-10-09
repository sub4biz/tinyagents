//! Run one host-supplied unit of agent work through [`SubagentDriver`].
//!
//! Team members and workflow phase agents are not driven by a harness agent
//! loop the crate controls: the host hands over an opaque async worker. This
//! module adapts such a worker to the driver's planner / executor /
//! persistence seams so those steps get the same lifecycle features as every
//! other subagent path: spawn admission ([`SpawnPolicy`](super::SpawnPolicy)),
//! timeout / retry / budget ([`SubAgentPolicy`]), the [`ResultPolicy`], the
//! [`SubagentRole`], and the typed [`SubagentOutcomeKind`].
//!
//! [`AgentStepConfig::default`] is deliberately inert (unlimited admission, no
//! timeout, a single attempt, no result trimming), so a step run with the
//! default config behaves like calling the worker directly.

mod seams;
mod types;

use std::future::Future;
use std::sync::{Arc, Mutex};

use tinyagents_harness::CancellationToken;
use tinyagents_harness::context::{RunConfig, RunContext};

use self::seams::{Slot, StepExecutor, StepPersistence, StepPlanner, Work};
pub use self::types::{
    AgentStepConfig, AgentStepError, AgentStepIdentity, AgentStepResult, StepContext, StepSuccess,
    StepWorkError,
};
use super::{
    SubagentCapabilities, SubagentDriver, SubagentError, SubagentOutcomeKind, SubagentRequest,
};

const LOG_PREFIX: &str = "[agent-step]";

/// Runs `work` as one subagent lifecycle on a [`SubagentDriver`] configured
/// from `config`.
///
/// `work` receives a [`StepContext`] (the child cancellation token, cancelled on
/// lifecycle cancellation or policy timeout, and the role it must enforce) and may be invoked more than once when the
/// retry policy retries a [`StepWorkError::Transient`] failure.
pub async fn run_agent_step<T, F, Fut>(
    config: &AgentStepConfig,
    identity: AgentStepIdentity,
    cancellation: CancellationToken,
    work: F,
) -> Result<AgentStepResult<T>, AgentStepError>
where
    T: Send + 'static,
    F: Fn(StepContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<StepSuccess<T>, StepWorkError>> + Send + 'static,
{
    let target = identity
        .target
        .clone()
        .unwrap_or_else(|| "agent-step".to_owned());
    tracing::debug!(
        "{LOG_PREFIX} start parent={} task={} target={target}",
        identity.parent_run_id,
        identity.task_id
    );
    let slot = Arc::new(Mutex::new(Slot {
        value: None,
        error: None,
    }));
    let work: Work<T> = Arc::new(move |token| Box::pin(work(token)));
    let driver = SubagentDriver::new(SubagentCapabilities {
        planner: Some(Arc::new(StepPlanner {
            config: config.clone(),
            target: target.clone(),
        })),
        executor: Some(Arc::new(StepExecutor {
            work,
            slot: slot.clone(),
        })),
        persistence: Some(Arc::new(StepPersistence::default())),
    })
    .map_err(AgentStepError::Driver)?
    .with_spawn_admission(config.admission.clone());

    let parent = RunContext::new(RunConfig::new(identity.parent_run_id.as_str()), ());
    let child = parent
        .child(
            RunConfig::new(format!("{}:{}", identity.parent_run_id, identity.task_id)),
            (),
        )
        .map_err(|e| AgentStepError::Driver(SubagentError::InvalidRequest(e.to_string())))?;
    let mut request = SubagentRequest::fresh_from_parent(
        &parent,
        child,
        identity.task_id.as_str(),
        (),
        identity.task_id.as_str(),
        None,
    )
    .map_err(AgentStepError::Driver)?;
    if identity.target.is_some() {
        request = request.with_target(target);
    }

    let run = driver.run(request, cancellation).await;
    let mut slot = slot.lock().expect("agent-step slot poisoned");
    match run {
        Ok(result) => {
            tracing::debug!(
                "{LOG_PREFIX} done parent={} task={} status={}",
                identity.parent_run_id,
                identity.task_id,
                status_label(&result.outcome.status)
            );
            if matches!(result.outcome.status, SubagentOutcomeKind::Cancelled)
                && slot.value.is_none()
            {
                return Err(AgentStepError::Cancelled);
            }
            Ok(AgentStepResult {
                outcome: result.outcome,
                value: slot.value.take(),
            })
        }
        Err(SubagentError::SpawnRejected(rejection)) => {
            tracing::debug!(
                "{LOG_PREFIX} rejected parent={} task={} reason={rejection}",
                identity.parent_run_id,
                identity.task_id
            );
            Err(AgentStepError::Rejected(rejection))
        }
        Err(SubagentError::Cancelled) => Err(AgentStepError::Cancelled),
        Err(SubagentError::Execution(_) | SubagentError::Transient { .. })
            if slot.error.is_some() =>
        {
            Err(AgentStepError::Worker(slot.error.take().expect("checked")))
        }
        Err(error) => Err(AgentStepError::Driver(error)),
    }
}

fn status_label(status: &SubagentOutcomeKind) -> &'static str {
    match status {
        SubagentOutcomeKind::Completed => "completed",
        SubagentOutcomeKind::AwaitingInput(_) => "awaiting_input",
        SubagentOutcomeKind::Incomplete(_) => "incomplete",
        SubagentOutcomeKind::Cancelled => "cancelled",
    }
}
