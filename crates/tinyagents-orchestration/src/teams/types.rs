//! Data types for team coordination: members, tasks, shutdown results, and
//! validation errors.

use serde::Serialize;
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::{AgentTeam, AgentTeamMember, AgentTeamTask};

use crate::subagent::{AgentStepConfig, AgentStepIdentity};

/// Sentinel sender for a lead or user message rather than a member row.
///
/// Used in the event log to distinguish lead/user messages from member-to-member
/// messages. All events with this sender should be treated as external input
/// rather than team member output.
pub const LEAD_SENDER: &str = "lead";

/// One member supplied when a team is created.
///
/// Carries only the identity and (optional) agent-id hint needed for
/// initialization; the full persistent [`AgentTeamMember`] row is created
/// by the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewMember {
    /// Human-readable member name for prompts and display.
    pub name: String,
    /// Optional reference to an agent definition in the host registry. When
    /// supplied, the host may use it to resolve member prompts and tool access.
    pub agent_id: Option<String>,
}

/// A durable team and the member/task rows needed to render it.
///
/// Projects a complete team state at one point in time: the team metadata,
/// its members, and its tasks. Used for snapshots, UI display, and audit logs.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamView {
    /// The team metadata and status.
    pub team: AgentTeam,
    /// All members of the team, in creation order.
    pub members: Vec<AgentTeamMember>,
    /// All tasks the team has been assigned, in creation order.
    pub tasks: Vec<AgentTeamTask>,
}

/// Result of stopping a member and releasing its active claims.
///
/// Returned when a running member is shut down. Records the stopped member
/// and the task ids it had claimed (now released and available for
/// reassignment).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemberShutdown {
    /// The member that was shut down.
    pub member: AgentTeamMember,
    /// Task ids that member had claimed, now released.
    pub released_task_ids: Vec<String>,
}

/// Coordination validation errors that are independent of a host or storage backend.
///
/// These errors detect structural issues in team definition or task dependency
/// graphs that do not require access to a host's agent registry or persistence
/// layer. They are deterministic and stable across runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind", content = "detail")]
pub enum TeamError {
    /// A team member name appears more than once.
    DuplicateMemberName { name: String },
    /// A task references a member id that is not registered.
    UnknownMember { member_id: String },
    /// A task depends on itself.
    SelfDependency { task_id: String },
    /// Task dependencies form a cycle.
    CyclicDependency,
    /// A task depends on another task that is not registered.
    UnknownDependency { depends_on: String },
}

impl std::fmt::Display for TeamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateMemberName { name } => write!(f, "duplicate member name: {name}"),
            Self::UnknownMember { member_id } => write!(f, "unknown member: {member_id}"),
            Self::SelfDependency { task_id } => write!(f, "task {task_id} cannot depend on itself"),
            Self::CyclicDependency => write!(f, "dependency cycle detected"),
            Self::UnknownDependency { depends_on } => {
                write!(f, "unknown dependency: {depends_on}")
            }
        }
    }
}

impl std::error::Error for TeamError {}

/// The driver-side policy and identity of one member step.
///
/// [`run_member_graph`](super::run_member_graph) uses [`MemberStep::default`], which is inert (unlimited
/// admission, no timeout, one attempt, no result trimming), so the worker runs
/// exactly as it did before the step went through
/// [`SubagentDriver`](crate::subagent::SubagentDriver). Hosts that want the
/// driver's limits call [`run_member_graph_with`](super::run_member_graph_with).
#[derive(Clone)]
pub struct MemberStep {
    /// Spawn admission, timeout/retry/budget, result policy and role.
    pub config: AgentStepConfig,
    /// Team (admission scope) and member (task id, allowlist target).
    pub identity: AgentStepIdentity,
    /// Lifecycle cancellation of the enclosing run. A token cancelled before
    /// the step starts keeps the worker from running (the member is routed to
    /// `on_failed`); cancelling it later cancels the worker's child token.
    pub cancellation: CancellationToken,
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
            cancellation: CancellationToken::new(),
        }
    }

    /// Ties the step to the enclosing run's cancellation token.
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }
}

impl Default for MemberStep {
    fn default() -> Self {
        Self {
            config: AgentStepConfig::default(),
            identity: AgentStepIdentity::new("team", "member"),
            cancellation: CancellationToken::new(),
        }
    }
}
