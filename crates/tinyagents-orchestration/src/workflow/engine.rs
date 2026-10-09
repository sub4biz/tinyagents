//! Workflow execution engine: orchestrates phase scheduling, task spawning,
//! and result collection.
//!
//! [`WorkflowEngine`] drives the scheduler, manages child task creation and
//! claim-based assignment, handles phase state persistence and transitions,
//! and enforces bounded concurrency. It is generic over a host-supplied
//! [`WorkflowExecutor`] that creates and monitors actual work.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tinyagents_graph::GraphEventSink;
use tinyagents_graph::parallel::{FailurePolicy, ParallelOptions, map_reduce};
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::{
    WorkflowLeaseClaim, WorkflowRun, WorkflowRunStatus, WorkflowRunUpsert,
    compare_and_swap_workflow_run, get_workflow_run, renew_workflow_run_lease,
    try_claim_workflow_run, upsert_workflow_run,
};

use super::child_step::run_child_step;
use crate::subagent::AgentStepConfig;
use super::state::{
    PhaseStatus, all_phases_completed, init_phase_states, next_runnable_phase, phase_prompt,
    reset_running_phases, set_phase_reason, set_phase_status, synthesize_summary, upstream_outputs,
};
use super::{WorkflowDefinition, WorkflowPhase};

/// Error returned by a host child executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrchestrationError(pub String);

impl fmt::Display for OrchestrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for OrchestrationError {}

impl From<anyhow::Error> for OrchestrationError {
    fn from(error: anyhow::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<tinyagents_harness::TinyAgentsError> for OrchestrationError {
    fn from(error: tinyagents_harness::TinyAgentsError) -> Self {
        Self(error.to_string())
    }
}

/// Durable workflow rows supplied by the session owner.
pub trait WorkflowStore: Send + Sync {
    fn load(&self, id: &str) -> Result<Option<WorkflowRun>, OrchestrationError>;
    fn upsert(&self, row: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError>;
    fn claim(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<WorkflowLeaseClaim, OrchestrationError>;
    fn compare_and_swap(
        &self,
        row: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: Duration,
    ) -> Result<Option<WorkflowRun>, OrchestrationError>;
    fn renew(&self, id: &str, owner: &str, lease_for: Duration)
    -> Result<bool, OrchestrationError>;
}

/// `tinyagents-session` run-ledger adapter with a caller-selected workspace.
#[derive(Debug, Clone)]
pub struct SessionWorkflowStore {
    workspace_dir: PathBuf,
}

impl SessionWorkflowStore {
    pub fn new(workspace_dir: impl Into<PathBuf>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
        }
    }

    pub fn workspace_dir(&self) -> &Path {
        &self.workspace_dir
    }
}

impl WorkflowStore for SessionWorkflowStore {
    fn load(&self, id: &str) -> Result<Option<WorkflowRun>, OrchestrationError> {
        get_workflow_run(&self.workspace_dir, id).map_err(OrchestrationError::from)
    }

    fn upsert(&self, row: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError> {
        upsert_workflow_run(&self.workspace_dir, row).map_err(OrchestrationError::from)
    }

    fn claim(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<WorkflowLeaseClaim, OrchestrationError> {
        try_claim_workflow_run(
            &self.workspace_dir,
            id,
            owner,
            chrono::Duration::from_std(lease_for)
                .map_err(|error| OrchestrationError(error.to_string()))?,
        )
        .map_err(OrchestrationError::from)
    }

    fn compare_and_swap(
        &self,
        row: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: Duration,
    ) -> Result<Option<WorkflowRun>, OrchestrationError> {
        compare_and_swap_workflow_run(
            &self.workspace_dir,
            row,
            expected_revision,
            owner,
            chrono::Duration::from_std(lease_for)
                .map_err(|error| OrchestrationError(error.to_string()))?,
        )
        .map_err(OrchestrationError::from)
    }

    fn renew(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<bool, OrchestrationError> {
        renew_workflow_run_lease(
            &self.workspace_dir,
            id,
            owner,
            chrono::Duration::from_std(lease_for)
                .map_err(|error| OrchestrationError(error.to_string()))?,
        )
        .map_err(OrchestrationError::from)
    }
}

/// One host-authorized child invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowChildRequest {
    pub run_id: String,
    pub phase: String,
    pub agent_id: String,
    pub index_in_phase: usize,
    pub prompt: String,
}

/// One child's terminal result. `output` is retained verbatim in phase state.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowChildResult {
    pub child_id: String,
    pub output: Value,
}

/// Called by a host immediately after it has created a real child.  Registering
/// before waiting makes the child visible to a concurrent cancellation request
/// even when the worker is still in flight.
pub trait WorkflowChildRegistration: Send + Sync {
    fn register(&self, child_id: String) -> Result<(), OrchestrationError>;
}

/// Host-owned execution and cancellation mechanism.
#[async_trait]
pub trait WorkflowExecutor: Send + Sync {
    async fn execute(
        &self,
        request: WorkflowChildRequest,
        cancel: CancellationToken,
        registration: Arc<dyn WorkflowChildRegistration>,
    ) -> Result<WorkflowChildResult, OrchestrationError>;

    async fn cancel_children(&self, child_ids: &[String]);
}

/// Generic durable workflow engine. It never creates tasks: hosts decide where
/// work runs and which authorization context is in force via [`WorkflowExecutor`].
pub struct WorkflowEngine<S, E> {
    store: Arc<S>,
    executor: Arc<E>,
    event_sink: Option<Arc<dyn GraphEventSink>>,
    event_seq: AtomicU64,
    lease_for: Duration,
    step_config: AgentStepConfig,
}

const WORKFLOW_LEASE: Duration = Duration::from_secs(10 * 60);

struct PersistRequest {
    phase_states: Value,
    child_run_ids: Vec<String>,
    status: WorkflowRunStatus,
    summary: Option<String>,
    terminal: bool,
}

struct PhaseRegistration<S: WorkflowStore> {
    store: Arc<S>,
    owner: String,
    run: parking_lot::Mutex<WorkflowRun>,
    phase_states: Value,
    lease_for: Duration,
}

impl<S: WorkflowStore> PhaseRegistration<S> {
    fn current(&self) -> WorkflowRun {
        self.run.lock().clone()
    }
}

impl<S: WorkflowStore + 'static> WorkflowChildRegistration for PhaseRegistration<S> {
    fn register(&self, child_id: String) -> Result<(), OrchestrationError> {
        let mut run = self.run.lock();
        if run.child_run_ids.iter().any(|known| known == &child_id) {
            return Ok(());
        }
        let mut children = run.child_run_ids.clone();
        children.push(child_id);
        let Some(updated) = self.store.compare_and_swap(
            WorkflowRunUpsert {
                id: run.id.clone(),
                definition_id: run.definition_id.clone(),
                parent_thread_id: run.parent_thread_id.clone(),
                input: run.input.clone(),
                phase_states: self.phase_states.clone(),
                child_run_ids: children,
                status: WorkflowRunStatus::Running,
                summary: None,
                started_at: Some(run.started_at),
                completed_at: None,
            },
            run.revision,
            &self.owner,
            self.lease_for,
        )?
        else {
            return Err(OrchestrationError(
                "workflow lease lost while registering child".into(),
            ));
        };
        *run = updated;
        Ok(())
    }
}

impl<S, E> WorkflowEngine<S, E>
where
    S: WorkflowStore + 'static,
    E: WorkflowExecutor + 'static,
{
    pub fn new(store: Arc<S>, executor: Arc<E>) -> Self {
        Self {
            store,
            executor,
            event_sink: None,
            event_seq: AtomicU64::new(0),
            lease_for: WORKFLOW_LEASE,
            step_config: AgentStepConfig::default(),
        }
    }

    /// Attach an optional host sink for ordinary graph lifecycle tracing.
    pub fn with_event_sink(mut self, sink: Arc<dyn GraphEventSink>) -> Self {
        self.event_sink = Some(sink);
        self
    }

    /// Apply driver policy to every agent child: spawn admission (scoped per
    /// workflow run), timeout/retry/budget, result policy and role.
    ///
    /// Without this call the config is inert and children run exactly as
    /// before. A child refused by the spawn policy, timed out, or over budget
    /// fails its phase with that reason.
    pub fn with_step_config(mut self, config: AgentStepConfig) -> Self {
        self.step_config = config;
        self
    }

    /// Override the driver lease for deterministic tests or hosts with a
    /// deliberately shorter failure-detection window.
    pub fn with_lease_duration(mut self, lease_for: Duration) -> Self {
        self.lease_for = lease_for.max(Duration::from_millis(3));
        self
    }

    /// Initialise a durable run before the host schedules [`Self::drive`].
    pub fn initialise(
        &self,
        id: String,
        definition: &WorkflowDefinition,
        input: Value,
        parent_thread_id: Option<String>,
    ) -> Result<WorkflowRun, OrchestrationError> {
        self.store.upsert(WorkflowRunUpsert {
            id,
            definition_id: definition.id.clone(),
            parent_thread_id,
            input,
            phase_states: init_phase_states(definition),
            child_run_ids: Vec::new(),
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: None,
            completed_at: None,
        })
    }

    /// Drive a run to a terminal state. Completed phases are never executed
    /// again, so a host may safely call this after process restart or resume.
    pub async fn drive(
        &self,
        run_id: &str,
        definition: &WorkflowDefinition,
        cancel: CancellationToken,
    ) -> Result<(), OrchestrationError> {
        // A driver lease is acquired before looking for runnable work.  This
        // is deliberately separate from the in-process cancellation token:
        // resume can race in another process, and only the durable lease
        // prevents both drivers from spawning the same phase.
        let owner = uuid::Uuid::new_v4().to_string();
        let mut run = match self.store.claim(run_id, &owner, self.lease_for)? {
            WorkflowLeaseClaim::Acquired(run) => run,
            WorkflowLeaseClaim::Busy(_) => return Ok(()),
            WorkflowLeaseClaim::Missing => {
                return Err(OrchestrationError(format!(
                    "workflow run {run_id} vanished before start"
                )));
            }
        };
        // A crashed owner can leave the durable phase marked `running`. That
        // marker is intentionally not runnable, so reclaiming the lease must
        // turn it back into retryable work before scheduling. The claim above
        // fences every prior owner; this CAS is the new owner's durable
        // recovery transition rather than a read/modify/write race.
        if run.phase_states.as_object().is_some_and(|phases| {
            phases
                .values()
                .any(|phase| phase.get("status").and_then(Value::as_str) == Some("running"))
        }) {
            // The previous owner may have crashed after registering remote
            // children. Fence side effects before making its phase runnable.
            self.executor.cancel_children(&run.child_run_ids).await;
            let mut phase_states = run.phase_states.clone();
            reset_running_phases(
                &mut phase_states,
                "workflow owner expired; phase will retry after lease takeover",
            );
            run = self.persist(
                &run,
                PersistRequest {
                    phase_states,
                    child_run_ids: Vec::new(),
                    status: WorkflowRunStatus::Running,
                    summary: None,
                    terminal: false,
                },
                &owner,
            )?;
        }
        self.emit(
            run_id,
            tinyagents_graph::GraphEvent::RunStarted {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
            },
        );
        // Registered children from an interrupted attempt are historical, not
        // part of the retry's spawn budget.
        let mut total_spawned = 0;

        loop {
            if cancel.is_cancelled() {
                self.executor.cancel_children(&run.child_run_ids).await;
                let mut phase_states = run.phase_states.clone();
                reset_running_phases(
                    &mut phase_states,
                    "workflow interrupted; phase will retry on resume",
                );
                if let Err(error) = self.persist(
                    &run,
                    PersistRequest {
                        phase_states,
                        child_run_ids: run.child_run_ids.clone(),
                        status: WorkflowRunStatus::Interrupted,
                        summary: None,
                        terminal: false,
                    },
                    &owner,
                ) {
                    if self.owner_lost(run_id, &owner)
                        || self.emit_recorded_terminal(run_id, total_spawned as usize)
                    {
                        return Ok(());
                    }
                    self.finish_failed(run_id, error.to_string());
                    return Err(error);
                }
                self.finish_cancelled(run_id);
                return Ok(());
            }
            let Some(phase) = next_runnable_phase(definition, &run.phase_states).cloned() else {
                if all_phases_completed(definition, &run.phase_states) {
                    if let Err(error) = self.persist(
                        &run,
                        PersistRequest {
                            phase_states: run.phase_states.clone(),
                            child_run_ids: run.child_run_ids.clone(),
                            status: WorkflowRunStatus::Completed,
                            summary: synthesize_summary(definition, &run.phase_states),
                            terminal: true,
                        },
                        &owner,
                    ) {
                        if self.owner_lost(run_id, &owner) {
                            return Ok(());
                        }
                        self.finish_failed(run_id, error.to_string());
                        return Err(error);
                    }
                    self.finish_completed(run_id, total_spawned as usize);
                } else {
                    let reason = "no runnable phase (dependency deadlock)".to_owned();
                    if let Err(error) = self.persist(
                        &run,
                        PersistRequest {
                            phase_states: run.phase_states.clone(),
                            child_run_ids: run.child_run_ids.clone(),
                            status: WorkflowRunStatus::Failed,
                            summary: Some(reason.clone()),
                            terminal: true,
                        },
                        &owner,
                    ) {
                        if self.owner_lost(run_id, &owner) {
                            return Ok(());
                        }
                        self.finish_failed(run_id, error.to_string());
                        return Err(error);
                    }
                    self.finish_failed(run_id, reason);
                }
                return Ok(());
            };
            self.emit(
                run_id,
                tinyagents_graph::GraphEvent::NodeStarted {
                    node: tinyagents_harness::ids::NodeId::new("run_phase"),
                    step: total_spawned as usize + 1,
                },
            );
            let phase_result = self
                .run_phase(
                    &run,
                    definition,
                    &phase,
                    total_spawned,
                    cancel.clone(),
                    &owner,
                )
                .await;
            let (updated, spawned) = match phase_result {
                Ok(result) => result,
                Err(error) => {
                    // A host stop/resume fences this owner with a revision CAS.
                    // Do not turn that intentional hand-off into a stale
                    // failure event or overwrite the newer durable state.
                    if self.owner_lost(run_id, &owner)
                        || self.emit_recorded_terminal(run_id, total_spawned as usize)
                    {
                        return Ok(());
                    }
                    self.finish_failed(run_id, error.to_string());
                    return Err(error);
                }
            };
            run = updated;
            self.emit(
                run_id,
                tinyagents_graph::GraphEvent::NodeCompleted {
                    node: tinyagents_harness::ids::NodeId::new("run_phase"),
                    step: total_spawned as usize + 1,
                },
            );
            total_spawned += spawned;
            if run.status != WorkflowRunStatus::Running {
                match run.status {
                    WorkflowRunStatus::Completed => {
                        self.finish_completed(run_id, total_spawned as usize)
                    }
                    WorkflowRunStatus::Interrupted | WorkflowRunStatus::Cancelled => {
                        self.finish_cancelled(run_id)
                    }
                    WorkflowRunStatus::Failed => self.finish_failed(
                        run_id,
                        run.summary
                            .clone()
                            .unwrap_or_else(|| "workflow phase failed".to_owned()),
                    ),
                    WorkflowRunStatus::Pending | WorkflowRunStatus::Running => {}
                }
                return Ok(());
            }
        }
    }

    async fn run_phase(
        &self,
        run: &WorkflowRun,
        definition: &WorkflowDefinition,
        phase: &WorkflowPhase,
        total_spawned: u32,
        cancel: CancellationToken,
        owner: &str,
    ) -> Result<(WorkflowRun, u32), OrchestrationError> {
        let mut phase_states = run.phase_states.clone();
        let mut child_ids = run.child_run_ids.clone();
        set_phase_status(&mut phase_states, &phase.name, PhaseStatus::Running, None);
        let running = self.persist(
            run,
            PersistRequest {
                phase_states: phase_states.clone(),
                child_run_ids: child_ids.clone(),
                status: WorkflowRunStatus::Running,
                summary: None,
                terminal: false,
            },
            owner,
        )?;

        let budget = definition.max_children.saturating_sub(total_spawned) as usize;
        if budget == 0 {
            return self.fail_phase(
                &running,
                &mut phase_states,
                child_ids,
                phase,
                format!(
                    "max_children cap ({}) reached before phase '{}' completed",
                    definition.max_children, phase.name
                ),
                owner,
            );
        }
        let capacity = phase.agent_ids.len().min(budget);
        let capped = capacity != phase.agent_ids.len();
        let upstream = upstream_outputs(phase, &phase_states);
        let requests = phase.agent_ids[..capacity]
            .iter()
            .enumerate()
            .map(|(index_in_phase, agent_id)| WorkflowChildRequest {
                run_id: run.id.clone(),
                phase: phase.name.clone(),
                agent_id: agent_id.clone(),
                index_in_phase,
                prompt: phase_prompt(&run.input, phase, index_in_phase, &upstream),
            })
            .collect::<Vec<_>>();
        let registration = Arc::new(PhaseRegistration {
            store: self.store.clone(),
            owner: owner.to_owned(),
            run: parking_lot::Mutex::new(running.clone()),
            phase_states: phase_states.clone(),
            lease_for: self.lease_for,
        });
        let executor = self.executor.clone();
        let step_config = self.step_config.clone();
        let worker_cancel = cancel.clone();
        let worker_registration = registration.clone();
        let outcomes = map_reduce(
            requests,
            ParallelOptions::default()
                .with_max_concurrency(definition.default_concurrency as usize)
                .with_failure_policy(FailurePolicy::CollectAll)
                .with_cancellation(cancel.clone()),
            move |_index, request| {
                let executor = executor.clone();
                let cancel = worker_cancel.clone();
                let registration = worker_registration.clone();
                let step_config = step_config.clone();
                async move {
                    run_child_step(&step_config, executor, request, cancel, registration)
                        .await
                        .map_err(|error| {
                            tinyagents_harness::TinyAgentsError::Graph(error.to_string())
                        })
                }
            },
        );
        tokio::pin!(outcomes);
        let renew_every = self.lease_for.div_f32(3.0).max(Duration::from_millis(1));
        let mut heartbeat = tokio::time::interval(renew_every);
        // Ignore interval's eager first tick: `claim` has just installed this
        // lease, so renewals begin only while a child can be in flight.
        heartbeat.tick().await;
        let outcomes = loop {
            tokio::select! {
                outcomes = &mut outcomes => break outcomes,
                _ = heartbeat.tick() => {
                    let renewed = self.store.renew(&run.id, owner, self.lease_for);
                    if !matches!(renewed, Ok(true)) {
                        cancel.cancel();
                        let children = registration.current().child_run_ids;
                        self.executor.cancel_children(&children).await;
                        if let Err(error) = renewed {
                            return Err(OrchestrationError(format!(
                                "workflow lease renewal errored; cancelled registered children: {error}"
                            )));
                        }
                        return Err(OrchestrationError(
                            "workflow lease renewal failed; cancelled registered children".to_owned(),
                        ));
                    }
                }
            }
        };
        let outcomes = match outcomes {
            Ok(outcomes) => outcomes,
            Err(tinyagents_harness::TinyAgentsError::Cancelled) => {
                let children = registration.current().child_run_ids;
                self.executor.cancel_children(&children).await;
                reset_running_phases(
                    &mut phase_states,
                    "workflow interrupted; phase will retry on resume",
                );
                let updated = self.persist(
                    &registration.current(),
                    PersistRequest {
                        phase_states,
                        child_run_ids: children,
                        status: WorkflowRunStatus::Interrupted,
                        summary: None,
                        terminal: false,
                    },
                    owner,
                )?;
                return Ok((updated, 0));
            }
            Err(error) => return Err(OrchestrationError(error.to_string())),
        };
        child_ids = registration.current().child_run_ids;
        let mut outputs = Vec::new();
        let mut failure = None;
        let mut spawned = 0_u32;
        for outcome in outcomes.outcomes {
            match outcome.result {
                Ok(result) => {
                    spawned += 1;
                    // The executor registered the real id before it could
                    // await completion. Keep older executors harmlessly
                    // compatible by accepting an already-present id only.
                    if !child_ids.iter().any(|id| id == &result.child_id) {
                        child_ids.push(result.child_id.clone());
                    }
                    outputs.push(json!({
                        // Preserve the persisted/RPC v1 projection while the
                        // v2 metadata remains lossless for engine consumers.
                        "agentId": phase.agent_ids[outcome.index],
                        "output": render_compat_output(&result.output),
                        "metadata": { "version": 2, "rawOutput": result.output },
                    }));
                }
                Err(error) if failure.is_none() => failure = Some(error),
                Err(_) => {}
            }
        }
        if cancel.is_cancelled() {
            let children = registration.current().child_run_ids;
            self.executor.cancel_children(&children).await;
            reset_running_phases(
                &mut phase_states,
                "workflow interrupted; phase will retry on resume",
            );
            let updated = self.persist(
                &registration.current(),
                PersistRequest {
                    phase_states,
                    child_run_ids: children,
                    status: WorkflowRunStatus::Interrupted,
                    summary: None,
                    terminal: false,
                },
                owner,
            )?;
            return Ok((updated, 0));
        }
        if let Some(reason) = failure.or_else(|| {
            capped.then(|| {
                format!(
                    "max_children cap ({}) reached before phase '{}' completed",
                    definition.max_children, phase.name
                )
            })
        }) {
            return self.fail_phase(
                &registration.current(),
                &mut phase_states,
                child_ids,
                phase,
                reason,
                owner,
            );
        }
        set_phase_status(
            &mut phase_states,
            &phase.name,
            PhaseStatus::Completed,
            Some(Value::Array(outputs)),
        );
        let updated = self.persist(
            &registration.current(),
            PersistRequest {
                phase_states,
                child_run_ids: child_ids,
                status: WorkflowRunStatus::Running,
                summary: None,
                terminal: false,
            },
            owner,
        )?;
        Ok((updated, spawned))
    }

    fn fail_phase(
        &self,
        run: &WorkflowRun,
        phase_states: &mut Value,
        child_ids: Vec<String>,
        phase: &WorkflowPhase,
        reason: String,
        owner: &str,
    ) -> Result<(WorkflowRun, u32), OrchestrationError> {
        set_phase_status(
            phase_states,
            &phase.name,
            PhaseStatus::Failed,
            Some(json!([])),
        );
        set_phase_reason(phase_states, &phase.name, &reason);
        let updated = self.persist(
            run,
            PersistRequest {
                phase_states: phase_states.clone(),
                child_run_ids: child_ids,
                status: WorkflowRunStatus::Failed,
                summary: Some(reason),
                terminal: true,
            },
            owner,
        )?;
        Ok((updated, 0))
    }

    fn persist(
        &self,
        run: &WorkflowRun,
        request: PersistRequest,
        owner: &str,
    ) -> Result<WorkflowRun, OrchestrationError> {
        self.store
            .compare_and_swap(
                WorkflowRunUpsert {
                    id: run.id.clone(),
                    definition_id: run.definition_id.clone(),
                    parent_thread_id: run.parent_thread_id.clone(),
                    input: run.input.clone(),
                    phase_states: request.phase_states,
                    child_run_ids: request.child_run_ids,
                    status: request.status,
                    summary: request.summary,
                    started_at: Some(run.started_at),
                    completed_at: request.terminal.then(Utc::now),
                },
                run.revision,
                owner,
                self.lease_for,
            )?
            .ok_or_else(|| {
                OrchestrationError("workflow lease lost before durable state transition".to_owned())
            })
    }

    fn emit(&self, run_id: &str, event: tinyagents_graph::GraphEvent) {
        if let Some(sink) = &self.event_sink {
            sink.emit(tinyagents_graph::GraphEventEnvelope {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
                task_id: None,
                ns: Vec::new(),
                seq: self.event_seq.fetch_add(1, Ordering::Relaxed),
                event,
            });
        }
    }

    fn finish_completed(&self, run_id: &str, steps: usize) {
        self.emit(
            run_id,
            tinyagents_graph::GraphEvent::RunCompleted {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
                steps,
            },
        );
        self.flush_terminal_events();
    }

    fn finish_failed(&self, run_id: &str, error: String) {
        self.emit(
            run_id,
            tinyagents_graph::GraphEvent::RunFailed {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
                error,
            },
        );
        self.flush_terminal_events();
    }

    fn finish_cancelled(&self, run_id: &str) {
        self.emit(
            run_id,
            tinyagents_graph::GraphEvent::RunCancelled {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
            },
        );
        self.flush_terminal_events();
    }

    fn flush_terminal_events(&self) {
        if let Some(sink) = &self.event_sink {
            sink.flush();
        }
    }

    /// A lifecycle hand-off or lease takeover has fenced this driver. It must
    /// not manufacture a terminal graph event for the replacement owner.
    fn owner_lost(&self, run_id: &str, owner: &str) -> bool {
        self.store
            .load(run_id)
            .ok()
            .flatten()
            .is_some_and(|current| {
                current.lease_owner.as_deref() != Some(owner)
                    || current
                        .lease_expires_at
                        .is_none_or(|expires| expires <= Utc::now())
            })
    }

    /// Returns true after emitting the terminal event already committed by a
    /// newer lifecycle owner. This is the stale-driver escape hatch: it never
    /// writes, so a stop/resume hand-off cannot be overwritten by its loser.
    fn emit_recorded_terminal(&self, run_id: &str, steps: usize) -> bool {
        let Ok(Some(current)) = self.store.load(run_id) else {
            return false;
        };
        if !current.status.is_terminal() {
            return false;
        }
        match current.status {
            WorkflowRunStatus::Completed => self.finish_completed(run_id, steps),
            WorkflowRunStatus::Interrupted | WorkflowRunStatus::Cancelled => {
                self.finish_cancelled(run_id)
            }
            WorkflowRunStatus::Failed => self.finish_failed(
                run_id,
                current
                    .summary
                    .unwrap_or_else(|| "workflow phase failed".to_owned()),
            ),
            WorkflowRunStatus::Pending | WorkflowRunStatus::Running => return false,
        }
        true
    }
}

pub(super) fn render_compat_output(output: &Value) -> String {
    match output {
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(output)
            .unwrap_or_else(|_| "<unserializable workflow output>".to_owned()),
    }
}
