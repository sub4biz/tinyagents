//! Host-neutral subagent invocation and lifecycle orchestration.
//!
//! [`SubAgent`], [`SubAgentTool`], and [`SubAgentSession`] compose harness
//! agents as direct child runs. The `Subagent*` lifecycle traits and driver
//! separately coordinate resume loading, preparation, execution, and one
//! mutually exclusive persistence action. Hosts still resolve policy, agent
//! definitions, prompts, tool allowlists, and persistence implementations.
//!
//! Dependency direction remains `orchestration -> {harness, runtime}`. Lower
//! TinyAgents layers must not depend on this module.

mod admission;
mod completion;
mod detached;
mod driver;
mod executor;
mod invocation;
mod persistence;
mod planner;
mod policy;
mod result_policy;
mod role;
mod types;

pub use admission::{SpawnAdmission, SpawnPolicy, SpawnRejection, SpawnReservation};
pub use detached::{
    DETACHED_LEDGER_TIMEOUT_MS, DetachedCompletionTarget, DetachedSubagentStatus, FinishedOutcome,
    SpawnedSubagent, SteerAccess, SteerError, SteerReceipt, SteerRoute, SubagentIdentity,
    SubagentResumeRef, SubagentSnapshot, WaitError, WaitOutcome, cancel_for_thread,
    distinct_parent_threads, list_subagent_records, orphaned_subagent_reason, queue_lane_name,
    record_agent_id, record_cancelled, record_detached_completion, record_parent_session,
    record_spawned, record_status, record_subagent_session_id, record_to_wait_outcome,
    resume_ref_for_task, resume_ref_from_record, snapshot_for_owner, spawn_status_watcher,
    spawn_status_watcher_with_completions, steer_detached, steer_detached_with_request_id,
    steering_command_for_lane, subagent_record_for_task, task_id_for_session,
    task_id_for_session_in_records, task_status_label, wait_detached, wait_error_from_registry,
};
pub use driver::{SubagentCapabilities, SubagentDriver};
pub use executor::SubagentExecutor;
pub use invocation::{
    ChildDataPolicy, SUBAGENT_MODE_FIELD, SubAgent, SubAgentJob, SubAgentJobError, SubAgentJobId,
    SubAgentJobRegistry, SubAgentJobStatus, SubAgentJobsTool, SubAgentMessageTool, SubAgentMode,
    SubAgentSession, SubAgentTool, register_subagent_job_tools,
};
pub use persistence::SubagentPersistence;
pub use planner::SubagentPlanner;
pub use policy::{SubAgentBudget, SubAgentPolicy};
pub use result_policy::{
    AppliedResult, ArtifactStore, ResultOverflow, ResultPolicy, truncate_head_tail,
};
pub use role::{
    SUBAGENT_JOBS_TOOL, SUBAGENT_MESSAGE_TOOL, SubagentRole, is_delegation_tool, restrict_tools,
    subagent_framing,
};
#[allow(deprecated)]
pub use types::SubagentStatus;
pub use types::{
    ArtifactReference, AttemptContextFactory, IncompleteKind, PersistedSubagentPause,
    PreparedSubagent, SubagentError, SubagentExecution, SubagentIncomplete, SubagentOutcome,
    SubagentOutcomeKind, SubagentPause, SubagentPausePersistenceDisposition,
    SubagentPersistenceDisposition, SubagentRequest, SubagentRequestParts, SubagentResume,
    SubagentRunResult, SubagentTaskKey, SubagentTerminalPersistenceDisposition,
};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;

#[cfg(test)]
#[path = "driver_completion_tests.rs"]
mod driver_completion_test;

#[cfg(test)]
#[path = "driver_policy_tests.rs"]
mod driver_policy_test;

#[cfg(test)]
#[path = "mod_outcome_kind_tests.rs"]
mod outcome_kind_test;
