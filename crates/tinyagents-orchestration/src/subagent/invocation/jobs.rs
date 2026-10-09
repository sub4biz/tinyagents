//! Asynchronous subagent job registry and host-facing control tools.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyagents_harness::cancel::CancellationToken;
use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::ids::{TaskId, next_seq};
use tinyagents_harness::steering::{
    SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringPolicy,
};
use tinyagents_harness::tool::{ToolDispatch, ToolRegistry};
use tinyinference_llm::message::Message;
use tinytools::{Tool, ToolResult};

use crate::subagent::{AppliedResult, IncompleteKind};

use tinyagents_tasks::{DetachedTaskRegistry, DetachedTaskRegistryError};
use tokio::sync::watch;

use super::types::{JobControl, JobMeta};
use super::{
    JobLink, SubAgentJob, SubAgentJobError, SubAgentJobId, SubAgentJobRegistry, SubAgentJobStatus,
};

const LOG_PREFIX: &str = "[subagent-jobs]";

/// Settles an inline job if its tool future is dropped (tool timeout, parent
/// stream drop) or unwinds from a panic before the result is recorded.
/// Call [`Self::disarm`] once the result has been written.
pub(crate) struct InlineJobGuard {
    jobs: SubAgentJobRegistry,
    id: SubAgentJobId,
    armed: bool,
}

impl InlineJobGuard {
    pub(crate) fn new(jobs: SubAgentJobRegistry, id: SubAgentJobId) -> Self {
        Self {
            jobs,
            id,
            armed: true,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for InlineJobGuard {
    fn drop(&mut self) {
        if self.armed {
            self.jobs.mark_aborted(&self.id, std::thread::panicking());
        }
    }
}

impl SubAgentJobRegistry {
    /// Creates an empty asynchronous job registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The detached-task registry that owns this registry's live state.
    pub(crate) fn tasks(&self) -> &DetachedTaskRegistry<JobMeta, SubAgentJob> {
        &self.tasks
    }

    /// Whether the job still holds a live (unreleased) cancellation token.
    #[allow(dead_code)]
    pub(crate) fn holds_live_cancellation(&self, id: &SubAgentJobId) -> bool {
        self.tasks
            .holds_cancellation(&TaskId::new(id.as_str()))
            .unwrap_or(false)
    }

    #[allow(dead_code)]
    pub(crate) fn create(&self, agent: &str, owner: u64) -> (SubAgentJobId, SteeringHandle) {
        self.create_with_cancellation(agent, owner, CancellationToken::new(), JobLink::default())
    }

    /// Registers a job whose child run observes `cancellation`, so
    /// [`Self::cancel_owned`] can stop exactly this job.
    pub(crate) fn create_with_cancellation(
        &self,
        agent: &str,
        owner: u64,
        cancellation: CancellationToken,
        link: JobLink,
    ) -> (SubAgentJobId, SteeringHandle) {
        let id = SubAgentJobId(format!("subagent-job-{}", next_seq()));
        let task_id = TaskId::new(id.as_str());
        let steering =
            SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::InjectMessage));
        let job = SubAgentJob {
            id: id.clone(),
            agent: agent.to_owned(),
            status: SubAgentJobStatus::Queued,
            output: None,
            error: None,
            subagent_run_id: link.subagent_run_id,
            parent_tool_call_id: link.parent_tool_call_id,
            incomplete_kind: None,
            artifacts: Vec::new(),
            schema_error: None,
            artifact_error: None,
        };
        let (status, watcher) = watch::channel(job);
        let mut controls = self.controls();
        self.steering.register(task_id.clone(), steering.clone());
        if let Err(error) = self.tasks.register_cooperative(
            task_id,
            owner.to_string(),
            JobMeta,
            watcher,
            cancellation,
        ) {
            tracing::error!("{LOG_PREFIX} register.failed job_id={id} error={error}");
        }
        controls.insert(
            id.clone(),
            JobControl {
                status,
                cancellation_requested: false,
            },
        );
        (id, steering)
    }

    pub(crate) fn mark_running(&self, id: &SubAgentJobId) {
        if let Some(control) = self.controls().get(id) {
            control.status.send_if_modified(|job| {
                if job.status == SubAgentJobStatus::Queued {
                    job.status = SubAgentJobStatus::Running;
                    true
                } else {
                    false
                }
            });
        }
    }

    /// Applies `settle` to a job that has not yet reached a terminal state,
    /// releasing its cancellation token and steering handle in the same step.
    /// `settle` receives whether cancellation was requested. The first
    /// terminal state wins: a settled job is left untouched and `false` is
    /// returned.
    fn settle(&self, id: &SubAgentJobId, settle: impl FnOnce(&mut SubAgentJob, bool)) -> bool {
        let controls = self.controls();
        let Some(control) = controls.get(id) else {
            return false;
        };
        if control.status.borrow().status.is_terminal() {
            return false;
        }
        let task_id = TaskId::new(id.as_str());
        let _ = self.tasks.release_cancellation(&task_id);
        let cancellation_requested = control.cancellation_requested;
        control
            .status
            .send_modify(|job| settle(job, cancellation_requested));
        self.steering.deregister(&task_id);
        true
    }

    pub(crate) fn mark_result(
        &self,
        id: &SubAgentJobId,
        result: Result<tinyagents_harness::middleware::AgentRun, TinyAgentsError>,
    ) {
        self.mark_result_applied(id, result, None);
    }

    /// Settles a job like [`Self::mark_result`], publishing the
    /// result-policy-applied output in the same registry write so no reader
    /// ever sees a terminal job with the raw, unpolicied output.
    pub(crate) fn mark_result_applied(
        &self,
        id: &SubAgentJobId,
        result: Result<tinyagents_harness::middleware::AgentRun, TinyAgentsError>,
        applied: Option<AppliedResult>,
    ) {
        let settled = self.settle(id, |job, cancellation_requested| match result {
            Ok(run) => {
                if cancellation_requested {
                    job.status = SubAgentJobStatus::Cancelled;
                    job.error = Some(TinyAgentsError::Cancelled.to_string());
                } else {
                    job.status = SubAgentJobStatus::Completed;
                    job.output = run.text();
                    if let Some(applied) = applied {
                        job.output = Some(applied.text);
                        job.artifacts.extend(applied.artifact);
                        job.schema_error = applied.schema_error;
                        job.artifact_error = applied.artifact_error;
                    }
                }
            }
            Err(TinyAgentsError::Cancelled) => {
                job.status = SubAgentJobStatus::Cancelled;
                job.error = Some(TinyAgentsError::Cancelled.to_string());
            }
            Err(error @ TinyAgentsError::LimitExceeded(_)) => {
                job.status = SubAgentJobStatus::Incomplete;
                job.incomplete_kind = Some(IncompleteKind::BudgetExceeded);
                job.error = Some(error.to_string());
            }
            Err(error @ TinyAgentsError::Timeout(_)) => {
                job.status = SubAgentJobStatus::Incomplete;
                job.incomplete_kind = Some(IncompleteKind::Timeout);
                job.error = Some(error.to_string());
            }
            Err(error) => {
                job.status = SubAgentJobStatus::Failed;
                job.error = Some(error.to_string());
            }
        });
        if !settled {
            // Already settled (e.g. cancelled by the owner): the first
            // terminal state wins.
            tracing::debug!("{LOG_PREFIX} mark_result.ignored job_id={id}");
        }
    }

    /// Points the job link at the attempt that is now running.
    pub(crate) fn set_attempt_run_id(&self, id: &SubAgentJobId, run_id: &str) {
        if let Some(control) = self.controls().get(id) {
            control.status.send_if_modified(|job| {
                if job.status.is_terminal() {
                    return false;
                }
                job.subagent_run_id = Some(run_id.to_owned());
                true
            });
        }
    }

    /// Settles a job whose run finished but overshot a budget: it keeps the
    /// (policy-applied) output and ends `Incomplete(BudgetExceeded)`.
    pub(crate) fn mark_budget_overrun(
        &self,
        id: &SubAgentJobId,
        applied: AppliedResult,
        reason: String,
    ) {
        self.settle(id, |job, cancellation_requested| {
            if cancellation_requested {
                // An owner cancellation that raced the finish wins, as in
                // `mark_result`.
                job.status = SubAgentJobStatus::Cancelled;
                job.error = Some(TinyAgentsError::Cancelled.to_string());
                return;
            }
            job.status = SubAgentJobStatus::Incomplete;
            job.incomplete_kind = Some(IncompleteKind::BudgetExceeded);
            job.error = Some(reason);
            job.output = Some(applied.text);
            job.artifacts.extend(applied.artifact);
            job.schema_error = applied.schema_error;
            job.artifact_error = applied.artifact_error;
        });
    }

    /// Marks a job `Failed` because its child task panicked or was aborted
    /// before it could report a result.
    pub(crate) fn mark_aborted(&self, id: &SubAgentJobId, panicked: bool) {
        self.settle(id, |job, _| {
            tracing::warn!("{LOG_PREFIX} child_task.aborted job_id={id} panicked={panicked}");
            if panicked {
                job.status = SubAgentJobStatus::Failed;
                job.error = Some("subagent job panicked before completing".to_owned());
            } else {
                job.status = SubAgentJobStatus::Cancelled;
                job.error = Some(TinyAgentsError::Cancelled.to_string());
            }
        });
    }

    /// Cancels one queued or running job owned by `owner` and marks it
    /// `Cancelled`. The job's own cancellation token is tripped, so the parent
    /// run and sibling jobs are unaffected.
    pub(crate) fn cancel_owned(
        &self,
        job_id: &str,
        owner: u64,
    ) -> Result<SubAgentJob, SubAgentJobError> {
        let id = SubAgentJobId(job_id.to_owned());
        let mut controls = self.controls();
        let control = controls
            .get_mut(&id)
            .ok_or_else(|| SubAgentJobError::NotFound(job_id.to_owned()))?;
        let task_id = TaskId::new(job_id);
        let mut snapshot = match self.tasks.cancel_cooperative(&task_id, &owner.to_string()) {
            Ok(snapshot) => snapshot.status,
            Err(DetachedTaskRegistryError::AlreadyDone) => {
                return Err(SubAgentJobError::Terminal {
                    job_id: job_id.to_owned(),
                    status: control.status.borrow().status,
                });
            }
            Err(_) => return Err(SubAgentJobError::NotFound(job_id.to_owned())),
        };
        tracing::debug!("{LOG_PREFIX} cancel_owned job_id={job_id}");
        control.cancellation_requested = true;
        snapshot.error =
            Some("cancellation requested; job will be cancelled when the child unwinds".to_owned());
        Ok(snapshot)
    }

    /// Returns a snapshot for `job_id` when it belongs to `owner`.
    pub(crate) fn get_owned(&self, job_id: &str, owner: u64) -> Option<SubAgentJob> {
        self.tasks
            .snapshot(&TaskId::new(job_id), &owner.to_string())
            .ok()
            .map(|snapshot| snapshot.status)
    }

    /// Returns a job snapshot for trusted host-side supervision.
    ///
    /// Model-visible tools must use the run-scoped dispatch path instead.
    pub fn get(&self, job_id: &str) -> Option<SubAgentJob> {
        self.tasks
            .snapshot_trusted(&TaskId::new(job_id))
            .ok()
            .map(|snapshot| snapshot.status)
    }

    /// Returns this run's jobs in stable id order.
    fn list_owned(&self, owner: u64) -> Vec<SubAgentJob> {
        self.tasks
            .snapshots(Some(&owner.to_string()))
            .unwrap_or_default()
            .into_iter()
            .map(|snapshot| snapshot.status)
            .collect()
    }

    /// Returns every job for trusted host-side supervision.
    ///
    /// Model-visible tools must use the run-scoped dispatch path instead.
    pub fn list(&self) -> Vec<SubAgentJob> {
        self.tasks
            .snapshots(None)
            .unwrap_or_default()
            .into_iter()
            .map(|snapshot| snapshot.status)
            .collect()
    }

    /// Queues a user message for delivery at the running child's next safe
    /// steering checkpoint.
    #[allow(dead_code)]
    pub(crate) fn send_message_owned(
        &self,
        job_id: &str,
        owner: u64,
        message: impl Into<String>,
    ) -> Result<(), SubAgentJobError> {
        self.send_message_with_request_id(job_id, owner, message, None)
            .map(|_| ())
    }

    /// Idempotent [`Self::send_message_owned`]: a `request_id` already applied
    /// to this job is acknowledged (`Ok(true)`, "duplicate") without queueing
    /// the message again. Only the last
    /// [`RecentRequestIds::DEFAULT_CAPACITY`] ids per job are remembered, and a
    /// rejected send (unknown, foreign or terminal job) never consumes its id.
    pub(crate) fn send_message_with_request_id(
        &self,
        job_id: &str,
        owner: u64,
        message: impl Into<String>,
        request_id: Option<&str>,
    ) -> Result<bool, SubAgentJobError> {
        let id = SubAgentJobId(job_id.to_owned());
        let controls = self.controls();
        let not_found = || SubAgentJobError::NotFound(job_id.to_owned());
        let control = controls.get(&id).ok_or_else(not_found)?;
        let task_id = TaskId::new(job_id);
        let owner = owner.to_string();
        let job = self
            .tasks
            .snapshot(&task_id, &owner)
            .map_err(|_| not_found())?
            .status;
        if job.status.is_terminal() {
            return Err(SubAgentJobError::Terminal {
                job_id: job_id.to_owned(),
                status: job.status,
            });
        }
        if control.cancellation_requested {
            return Err(SubAgentJobError::Cancelling(job_id.to_owned()));
        }
        if let Some(request_id) = request_id {
            match self.tasks.claim_steer_request(&task_id, request_id) {
                Ok(false) => {
                    tracing::debug!("{LOG_PREFIX} send_message.duplicate job_id={job_id}");
                    return Ok(true);
                }
                Ok(true) => {}
                Err(DetachedTaskRegistryError::RequestIdTooLong) => {
                    return Err(SubAgentJobError::RequestIdTooLong);
                }
                Err(_) => return Err(not_found()),
            }
        }
        self.tasks
            .steering_handle(&task_id, &owner)
            .map_err(|_| not_found())?
            .send(SteeringCommand::InjectMessage(Message::user(
                message.into(),
            )));
        Ok(false)
    }

    fn controls(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::HashMap<SubAgentJobId, JobControl>> {
        self.controls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Typed host tool that queries a job or lists jobs owned by the requesting run.
#[derive(Clone)]
pub struct SubAgentJobsTool {
    jobs: SubAgentJobRegistry,
}

impl SubAgentJobsTool {
    /// Creates the query tool over `jobs`.
    pub fn new(jobs: SubAgentJobRegistry) -> Self {
        Self { jobs }
    }
}

#[async_trait]
impl Tool for SubAgentJobsTool {
    fn name(&self) -> &str {
        "subagent_jobs"
    }

    fn description(&self) -> &str {
        "Query an asynchronous subagent job by id, list all subagent jobs, or cancel one job with action \"cancel\"."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" },
                "action": {
                    "type": "string",
                    "enum": ["query", "cancel"],
                    "description": "`query` (default) reads a job or lists jobs; `cancel` stops the job named by job_id."
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let _ = args;
        anyhow::bail!("subagent_jobs requires typed-parent dispatch")
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolDispatch<State, Ctx> for SubAgentJobsTool {
    fn tool(&self) -> Arc<dyn Tool> {
        Arc::new(self.clone())
    }

    async fn execute(
        &self,
        _state: &State,
        _call_id: tinyagents_harness::ids::CallId,
        args: Value,
        _options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<ToolResult> {
        let object = args
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("arguments must be an object"))?;
        match object
            .get("action")
            .filter(|value| !value.is_null())
            .map(Value::as_str)
        {
            None | Some(Some("query")) => {}
            Some(Some("cancel")) => {
                let job_id = object
                    .get("job_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        anyhow::anyhow!("job_id must be a string for action `cancel`")
                    })?;
                let job = self.jobs.cancel_owned(job_id, parent.instance_id())?;
                return Ok(ToolResult::json(serde_json::to_value(job)?));
            }
            Some(_) => anyhow::bail!("action must be `query` or `cancel`"),
        }
        if let Some(value) = object.get("job_id") {
            let job_id = value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("job_id must be a string when provided"))?;
            let job = self
                .jobs
                .get_owned(job_id, parent.instance_id())
                .ok_or_else(|| SubAgentJobError::NotFound(job_id.to_owned()))?;
            Ok(ToolResult::json(serde_json::to_value(job)?))
        } else {
            Ok(ToolResult::json(serde_json::to_value(
                self.jobs.list_owned(parent.instance_id()),
            )?))
        }
    }
}

/// Host tool that sends a message to a queued or running subagent job.
#[derive(Clone)]
pub struct SubAgentMessageTool {
    jobs: SubAgentJobRegistry,
}

impl SubAgentMessageTool {
    /// Creates the message tool over `jobs`.
    pub fn new(jobs: SubAgentJobRegistry) -> Self {
        Self { jobs }
    }
}

#[async_trait]
impl Tool for SubAgentMessageTool {
    fn name(&self) -> &str {
        "subagent_message"
    }

    fn description(&self) -> &str {
        "Send a message to a queued or running asynchronous subagent job."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" },
                "message": { "type": "string" },
                "request_id": {
                    "type": "string",
                    "description": "Optional idempotency key: resending the same request_id to the same job does not queue the message again while it is among the most recent 64 request ids remembered for that job (older ids are evicted); ids over 128 bytes are rejected."
                }
            },
            "required": ["job_id", "message"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let _ = args;
        anyhow::bail!("subagent_message requires typed-parent dispatch")
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolDispatch<State, Ctx> for SubAgentMessageTool {
    fn tool(&self) -> Arc<dyn Tool> {
        Arc::new(self.clone())
    }

    async fn execute(
        &self,
        _state: &State,
        _call_id: tinyagents_harness::ids::CallId,
        args: Value,
        _options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<ToolResult> {
        let job_id = args
            .get("job_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("job_id must be a string"))?;
        let message = args
            .get("message")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("message must be a string"))?;
        let request_id = match args.get("request_id") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let id = value
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("request_id must be a string when provided"))?;
                Some(id)
            }
        };
        let duplicate = self.jobs.send_message_with_request_id(
            job_id,
            parent.instance_id(),
            message,
            request_id,
        )?;
        let mut payload = json!({
            "job_id": job_id,
            "status": "message_queued"
        });
        if duplicate {
            payload["duplicate"] = Value::Bool(true);
        }
        Ok(ToolResult::json(payload))
    }
}

/// Registers the standard run-scoped query and message tools in a harness registry.
pub fn register_subagent_job_tools<State: Send + Sync, Ctx: Send + Sync>(
    registry: &mut ToolRegistry<State, Ctx>,
    jobs: SubAgentJobRegistry,
) -> &mut ToolRegistry<State, Ctx> {
    registry
        .register_dispatch(Arc::new(SubAgentJobsTool::new(jobs.clone())))
        .register_dispatch(Arc::new(SubAgentMessageTool::new(jobs)))
}
