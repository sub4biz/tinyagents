//! The driver seams ([`SubagentPlanner`], [`SubagentExecutor`],
//! [`SubagentPersistence`]) an agent step plugs an opaque worker into.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_runtime::ToolSnapshot;
use tinyinference_llm::message::Message;

use super::types::{AgentStepConfig, StepContext, StepSuccess, StepWorkError};
use crate::subagent::{
    PersistedSubagentPause, PreparedSubagent, SubagentError, SubagentExecution, SubagentExecutor,
    SubagentIncomplete, SubagentOutcome, SubagentPausePersistenceDisposition, SubagentPersistence,
    SubagentPlanner, SubagentRequest, SubagentResume, SubagentTaskKey,
    SubagentTerminalPersistenceDisposition,
};

pub(super) type Work<T> = Arc<
    dyn Fn(
            StepContext,
        ) -> Pin<Box<dyn Future<Output = Result<StepSuccess<T>, StepWorkError>> + Send>>
        + Send
        + Sync,
>;

pub(super) struct Slot<T> {
    pub(super) value: Option<T>,
    pub(super) error: Option<anyhow::Error>,
}

pub(super) struct StepPlanner {
    pub(super) config: AgentStepConfig,
    pub(super) target: String,
}

#[async_trait]
impl SubagentPlanner<(), ()> for StepPlanner {
    async fn prepare(
        &self,
        request: SubagentRequest<(), ()>,
    ) -> Result<PreparedSubagent<()>, SubagentError> {
        let parts = request.into_parts();
        let task_id = parts.task_key.task_id.clone();
        let factory_task = task_id.clone();
        Ok(PreparedSubagent::new(
            task_id,
            self.target.clone(),
            vec![Message::user(parts.input)],
            ToolSnapshot::new(vec![]).map_err(|e| SubagentError::Planning(e.to_string()))?,
            parts.run_context,
        )
        .with_role(self.config.role)
        .with_policy(self.config.policy.clone())
        .with_result_policy(self.config.result_policy.clone())
        .with_retry_context(Arc::new(move |attempt| {
            Ok(RunContext::new(
                RunConfig::new(format!("{factory_task}-attempt-{attempt}")),
                (),
            ))
        })))
    }
}

pub(super) struct StepExecutor<T> {
    pub(super) work: Work<T>,
    pub(super) slot: Arc<Mutex<Slot<T>>>,
}

#[async_trait]
impl<T: Send + 'static> SubagentExecutor<()> for StepExecutor<T> {
    async fn execute(
        &self,
        execution: SubagentExecution<()>,
    ) -> Result<SubagentOutcome, SubagentError> {
        let task_id = execution.prepared.task_id.clone();
        if execution.cancellation.is_cancelled() {
            return Err(SubagentError::Cancelled);
        }
        let result = (self.work)(StepContext {
            cancellation: execution.cancellation.clone(),
            role: execution.prepared.role,
            max_model_calls: execution.prepared.run_context.config.max_model_calls,
            max_tool_calls: execution.prepared.run_context.config.max_tool_calls,
        })
        .await;
        let mut slot = self.slot.lock().expect("agent-step slot poisoned");
        match result {
            Ok(success) => {
                slot.value = Some(success.value);
                slot.error = None;
                let mut outcome = match success.incomplete_reason {
                    Some(reason) => {
                        SubagentOutcome::incomplete(task_id, SubagentIncomplete::new(reason))
                    }
                    None => SubagentOutcome::completed(task_id, success.output),
                };
                outcome.usage = success.usage;
                Ok(outcome)
            }
            Err(StepWorkError::Fatal(error)) => {
                let message = error.to_string();
                slot.error = Some(error);
                Err(SubagentError::Execution(message))
            }
            Err(StepWorkError::Transient { error, tools_ran }) => {
                let message = error.to_string();
                slot.error = Some(error);
                Err(SubagentError::Transient { message, tools_ran })
            }
        }
    }
}

/// In-memory persistence: a step is a single-process lifecycle with no pause.
#[derive(Default)]
pub(super) struct StepPersistence(Mutex<HashMap<SubagentTaskKey, SubagentOutcome>>);

#[async_trait]
impl SubagentPersistence for StepPersistence {
    async fn load_terminal(
        &self,
        key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(self
            .0
            .lock()
            .expect("step persistence poisoned")
            .get(key)
            .cloned())
    }
    async fn load(&self, _: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError> {
        Ok(None)
    }
    async fn load_pause(
        &self,
        _: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(None)
    }
    async fn save_pause(
        &self,
        _: PersistedSubagentPause,
    ) -> Result<SubagentPausePersistenceDisposition, SubagentError> {
        Err(SubagentError::Persistence(
            "agent steps never pause".to_owned(),
        ))
    }
    async fn record_terminal(
        &self,
        key: &SubagentTaskKey,
        outcome: &SubagentOutcome,
        _: Option<&SubagentResume>,
    ) -> Result<SubagentTerminalPersistenceDisposition, SubagentError> {
        let mut map = self.0.lock().expect("step persistence poisoned");
        if map.contains_key(key) {
            return Ok(SubagentTerminalPersistenceDisposition::Existing);
        }
        map.insert(key.clone(), outcome.clone());
        Ok(SubagentTerminalPersistenceDisposition::Inserted)
    }
}
