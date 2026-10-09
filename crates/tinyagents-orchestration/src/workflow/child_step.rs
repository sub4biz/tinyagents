//! One workflow child invocation, run as a [`SubagentDriver`](crate::subagent::SubagentDriver)
//! lifecycle so workflow agent phases share the spawn-admission, timeout /
//! retry / budget, result-policy and typed-outcome behaviour of every other
//! subagent path.

use std::sync::Arc;

use serde_json::Value;
use tinyagents_harness::CancellationToken;

use super::engine::{
    OrchestrationError, WorkflowChildRegistration, WorkflowChildRequest, WorkflowChildResult,
    WorkflowExecutor, render_compat_output,
};
use crate::subagent::{
    AgentStepConfig, AgentStepIdentity, StepSuccess, SubagentOutcomeKind, run_agent_step,
};

const LOG_PREFIX: &str = "[workflow-child-step]";

pub(super) async fn run_child_step<E: WorkflowExecutor + 'static>(
    config: &AgentStepConfig,
    executor: Arc<E>,
    request: WorkflowChildRequest,
    cancel: CancellationToken,
    registration: Arc<dyn WorkflowChildRegistration>,
) -> Result<WorkflowChildResult, OrchestrationError> {
    let identity = AgentStepIdentity::new(
        request.run_id.clone(),
        format!("{}:{}", request.phase, request.index_in_phase),
    )
    .with_target(request.agent_id.clone());
    tracing::debug!(
        "{LOG_PREFIX} start run={} phase={} index={} agent={}",
        request.run_id,
        request.phase,
        request.index_in_phase,
        request.agent_id
    );
    let work_request = request.clone();
    // `cancel` is the lifecycle token: cancelling the run reaches the child,
    // whose own token the driver derives from it.
    let result = run_agent_step(config, identity, cancel, move |token| {
        let executor = executor.clone();
        let request = work_request.clone();
        let registration = registration.clone();
        async move {
            let result = executor
                .execute(request, token, registration)
                .await
                .map_err(|error| anyhow::anyhow!(error.0))?;
            let rendered = render_compat_output(&result.output);
            Ok(StepSuccess::new(rendered, result))
        }
    })
    .await;
    match result {
        Ok(step) => match step.outcome.status {
            SubagentOutcomeKind::Completed => {
                let mut child = step.value.ok_or_else(|| {
                    OrchestrationError("workflow child completed without a result".to_owned())
                })?;
                // Keep the executor's structured output unless the result
                // policy actually changed the text.
                if step.outcome.output != render_compat_output(&child.output) {
                    child.output = Value::String(step.outcome.output);
                }
                Ok(child)
            }
            SubagentOutcomeKind::Incomplete(incomplete) => {
                Err(OrchestrationError(incomplete.reason))
            }
            // The engine re-checks its own token after the fan-out and treats
            // the phase as interrupted; hand it the real result when there is
            // one so the registered child id is not lost.
            SubagentOutcomeKind::Cancelled => step
                .value
                .ok_or_else(|| OrchestrationError("workflow child cancelled".to_owned())),
            SubagentOutcomeKind::AwaitingInput(_) => Err(OrchestrationError(
                "workflow child paused for input".to_owned(),
            )),
        },
        Err(error) => Err(OrchestrationError(error.to_string())),
    }
}
