//! Durable run ledger for agent and workflow execution state.
//!
//! Extends [`super`] with a queryable, restart-survivable ledger for background
//! agent and workflow runs. Conversation transcripts remain in the session
//! store; this ledger holds compact run metadata, child lineage, events,
//! telemetry, and checkpoint references — enough for a host to reconstruct
//! what was in flight after a crash and resume or interrupt it.
//!
//! Shares the session database and connection helper with `super::store`, so
//! a run and the session that produced it are queryable together.

pub mod command_center;
pub mod ops;
pub mod recovery;
mod status_map;
pub mod store;
pub mod tool_effects;
pub mod types;

pub use ops::{
    RunTransition, append_run_event, claim_agent_team_task, compare_and_swap_workflow_run,
    compare_and_swap_workflow_run_lifecycle, complete_agent_team_task, get_agent_run,
    get_agent_team, get_agent_team_member, get_agent_team_task, get_workflow_run,
    interrupt_orphaned_agent_runs, list_agent_runs, list_agent_team_members, list_agent_team_tasks,
    list_agent_teams, list_recent_run_events, list_workflow_runs, mark_agent_team_member_idle,
    mark_agent_team_member_running, release_agent_team_task, renew_workflow_run_lease,
    shutdown_agent_team_member, transition_agent_run_status, transition_agent_run_status_from,
    try_claim_workflow_run, upsert_agent_run, upsert_agent_team, upsert_agent_team_member,
    upsert_agent_team_task, upsert_run_telemetry, upsert_workflow_run,
};
pub use recovery::{
    CallRecovery, DanglingToolCall, MissingEffectRow, RecoveryClass, classify_recovery,
    classify_recovery_with, overall_recovery,
};
pub use tool_effects::{
    RunLedgerToolEffects, ToolEffectRow, ToolEffectSettle, ToolEffectStart, ToolEffectStatus,
    list_unresolved_tool_effects, mark_interrupted as mark_tool_effect_interrupted,
    record_tool_started, settle_tool_effect,
};
pub use types::{
    AgentRun, AgentRunKind, AgentRunListRequest, AgentRunListResponse, AgentRunStatus,
    AgentRunUpsert, AgentTeam, AgentTeamListRequest, AgentTeamListResponse, AgentTeamMember,
    AgentTeamMemberStatus, AgentTeamMemberUpsert, AgentTeamStatus, AgentTeamTask,
    AgentTeamTaskStatus, AgentTeamTaskUpsert, AgentTeamUpsert, ClaimOutcome, CompletionOutcome,
    RunEvent, RunEventAppend, RunEventListRequest, RunEventListResponse, RunTelemetry,
    RunTelemetryUpsert, WorkflowLeaseClaim, WorkflowRun, WorkflowRunListRequest,
    WorkflowRunListResponse, WorkflowRunStatus, WorkflowRunUpsert,
};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
