//! Public types of the agent-step adapter: config, identity, worker
//! contract, and errors.

use tinyagents_harness::CancellationToken;
use tinyinference_llm::usage::UsageTotals;

use crate::subagent::{
    ResultPolicy, SpawnAdmission, SpawnRejection, SubAgentPolicy, SubagentError, SubagentOutcome,
    SubagentRole,
};

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
    /// Delegation role (D9). Default: orchestrator. Passed to the worker in
    /// [`StepContext::role`]; the worker enforces it.
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

/// What the driver hands the worker for one attempt.
///
/// The crate cannot see the tools an opaque worker holds, so
/// [`AgentStepConfig::role`] cannot filter them itself: a worker that can
/// delegate must honour [`Self::role`] (drop its delegation tools when it is
/// [`SubagentRole::Leaf`]).
#[derive(Clone, Debug)]
pub struct StepContext {
    /// Child token: cancelled on lifecycle cancellation or policy timeout.
    pub cancellation: CancellationToken,
    /// The configured delegation role the worker must enforce.
    pub role: SubagentRole,
    /// Model-call cap from [`SubAgentBudget`](super::SubAgentBudget) call caps
    /// (already tightened onto the child's `RunConfig`); the worker must
    /// enforce it, as the crate cannot count an opaque worker's calls.
    pub max_model_calls: Option<usize>,
    /// Tool-call cap, as for [`Self::max_model_calls`].
    pub max_tool_calls: Option<usize>,
}
