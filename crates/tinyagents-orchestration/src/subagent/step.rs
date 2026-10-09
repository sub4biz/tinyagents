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

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tinyagents_harness::CancellationToken;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_runtime::ToolSnapshot;
use tinyinference_llm::message::Message;
use tinyinference_llm::usage::UsageTotals;

use super::{
    PersistedSubagentPause, PreparedSubagent, ResultPolicy, SpawnAdmission, SpawnRejection,
    SubAgentPolicy, SubagentCapabilities, SubagentDriver, SubagentError, SubagentExecution,
    SubagentExecutor, SubagentIncomplete, SubagentOutcome, SubagentOutcomeKind,
    SubagentPausePersistenceDisposition, SubagentPersistence, SubagentPlanner, SubagentRequest,
    SubagentResume, SubagentRole, SubagentTaskKey, SubagentTerminalPersistenceDisposition,
};

const LOG_PREFIX: &str = "[agent-step]";

/// Policies applied around every step run with this config.
///
/// Cloning shares the [`SpawnAdmission`] ledger, so one config reused across a
/// team's members (or a workflow's children) enforces its limits across all
/// of them. The ledger scope is the parent run id passed in
/// [`AgentStepIdentity::parent_run_id`] (the team id / workflow run id).
#[derive(Clone, Default)]
pub struct AgentStepConfig {
    /// Spawn limits (D2). Default: unlimited.
    pub admission: SpawnAdmission,
    /// Timeout, retry and budget (D6). Default: none.
    pub policy: SubAgentPolicy,
    /// Output trimming / schema check (D5). Default: inactive.
    pub result_policy: ResultPolicy,
    /// Delegation role (D9). Default: orchestrator.
    pub role: SubagentRole,
}

impl AgentStepConfig {
    /// Replaces the spawn admission ledger.
    pub fn with_admission(mut self, admission: SpawnAdmission) -> Self {
        self.admission = admission;
        self
    }

    /// Replaces the timeout/retry/budget policy.
    pub fn with_policy(mut self, policy: SubAgentPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Replaces the result policy.
    pub fn with_result_policy(mut self, result_policy: ResultPolicy) -> Self {
        self.result_policy = result_policy;
        self
    }

    /// Replaces the delegation role.
    pub fn with_role(mut self, role: SubagentRole) -> Self {
        self.role = role;
        self
    }
}

/// Durable-style identity of one step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentStepIdentity {
    /// The enclosing run (team id, workflow run id). Spawn admission is scoped
    /// to it and it becomes the step's parent/root run id.
    pub parent_run_id: String,
    /// Id of this step inside the parent run.
    pub task_id: String,
    /// Name checked against [`SpawnPolicy::allowed_targets`](super::SpawnPolicy).
    pub target: Option<String>,
}

impl AgentStepIdentity {
    /// An identity with no target.
    pub fn new(parent_run_id: impl Into<String>, task_id: impl Into<String>) -> Self {
        Self {
            parent_run_id: parent_run_id.into(),
            task_id: task_id.into(),
            target: None,
        }
    }

    /// Names the step's target for allowlist checks.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }
}

/// A worker's successful result: the final text the result policy governs,
/// the usage the budget is checked against, and a host value carried back
/// untouched.
pub struct StepSuccess<T> {
    /// Final visible output.
    pub output: String,
    /// Model usage the worker consumed.
    pub usage: UsageTotals,
    /// Host payload returned alongside the outcome.
    pub value: T,
    /// When set, the worker finished without a usable answer (a failure it
    /// reports itself, not an error): the outcome is `Incomplete` with this
    /// reason instead of `Completed`.
    pub incomplete_reason: Option<String>,
}

impl<T> StepSuccess<T> {
    /// A success with no reported usage.
    pub fn new(output: impl Into<String>, value: T) -> Self {
        Self {
            output: output.into(),
            usage: UsageTotals::default(),
            value,
            incomplete_reason: None,
        }
    }

    /// A worker-reported failure: the outcome is `Incomplete(reason)`.
    pub fn incomplete(reason: impl Into<String>, value: T) -> Self {
        Self {
            output: String::new(),
            usage: UsageTotals::default(),
            value,
            incomplete_reason: Some(reason.into()),
        }
    }

    /// Sets the reported usage.
    pub fn with_usage(mut self, usage: UsageTotals) -> Self {
        self.usage = usage;
        self
    }
}

/// How a worker failed.
#[derive(Debug)]
pub enum StepWorkError {
    /// Not retried.
    Fatal(anyhow::Error),
    /// Retried per [`SubAgentPolicy::retry`] (after tool calls only when
    /// `tools_ran` is false or the policy allows it).
    Transient {
        /// The underlying failure.
        error: anyhow::Error,
        /// Whether the attempt had already executed tools.
        tools_ran: bool,
    },
}

impl From<anyhow::Error> for StepWorkError {
    fn from(error: anyhow::Error) -> Self {
        Self::Fatal(error)
    }
}

/// Why a step did not produce an outcome.
#[derive(Debug)]
pub enum AgentStepError {
    /// The spawn policy refused the step; the worker never ran.
    Rejected(SpawnRejection),
    /// The worker failed; carries its original error.
    Worker(anyhow::Error),
    /// The step was cancelled before or while running, with no worker result.
    Cancelled,
    /// A lifecycle failure unrelated to the worker.
    Driver(SubagentError),
}

impl std::fmt::Display for AgentStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(rejection) => write!(f, "agent step was not admitted: {rejection}"),
            Self::Worker(error) => error.fmt(f),
            Self::Cancelled => f.write_str("agent step was cancelled"),
            Self::Driver(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for AgentStepError {}

/// The driver's outcome for a step plus the worker's payload, when it ran to
/// a result.
pub struct AgentStepResult<T> {
    /// The post-policy outcome (status, trimmed output, usage).
    pub outcome: SubagentOutcome,
    /// The worker's payload from its last successful attempt. Present for
    /// every status the worker produced a result for, including an outcome the
    /// policies later downgraded to incomplete or cancelled.
    pub value: Option<T>,
}

type Work<T> = Arc<
    dyn Fn(
            CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Result<StepSuccess<T>, StepWorkError>> + Send>>
        + Send
        + Sync,
>;

struct Slot<T> {
    value: Option<T>,
    error: Option<anyhow::Error>,
}

struct StepPlanner {
    config: AgentStepConfig,
    target: String,
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

struct StepExecutor<T> {
    work: Work<T>,
    slot: Arc<Mutex<Slot<T>>>,
}

#[async_trait]
impl<T: Send + 'static> SubagentExecutor<()> for StepExecutor<T> {
    async fn execute(
        &self,
        execution: SubagentExecution<()>,
    ) -> Result<SubagentOutcome, SubagentError> {
        let task_id = execution.prepared.task_id.clone();
        let result = (self.work)(execution.cancellation.clone()).await;
        let mut slot = self.slot.lock().expect("agent-step slot poisoned");
        match result {
            Ok(success) => {
                slot.value = Some(success.value);
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
struct StepPersistence(Mutex<HashMap<SubagentTaskKey, SubagentOutcome>>);

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
        self.0
            .lock()
            .expect("step persistence poisoned")
            .insert(key.clone(), outcome.clone());
        Ok(SubagentTerminalPersistenceDisposition::Inserted)
    }
}

/// Runs `work` as one subagent lifecycle on a [`SubagentDriver`] configured
/// from `config`.
///
/// `work` receives the child cancellation token (cancelled on lifecycle
/// cancellation or policy timeout) and may be invoked more than once when the
/// retry policy retries a [`StepWorkError::Transient`] failure.
pub async fn run_agent_step<T, F, Fut>(
    config: &AgentStepConfig,
    identity: AgentStepIdentity,
    cancellation: CancellationToken,
    work: F,
) -> Result<AgentStepResult<T>, AgentStepError>
where
    T: Send + 'static,
    F: Fn(CancellationToken) -> Fut + Send + Sync + 'static,
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

#[cfg(test)]
#[path = "step_tests.rs"]
mod tests;
