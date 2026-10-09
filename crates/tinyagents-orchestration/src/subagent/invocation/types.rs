//! Type definitions for first-class sub-agents.
//!
//! A [`SubAgent`] wraps an [`AgentHarness`] so it can be invoked as a *child
//! run*: a fully independent agent loop that runs one level deeper in the
//! recursion tree than its caller. [`SubAgentTool`] adapts a sub-agent into a
//! typed [`tinyagents_harness::tool::ToolDispatch`] so a parent agent can call another agent
//! through its live run context — the key agent-calling-agent compositional
//! pattern.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `tinyagents_orchestration::subagent` directly. Implementations and tests live in the
//! sibling `mod.rs` and its `*_tests.rs` files.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use tinyagents_harness::events::EventSink;
use tinyagents_harness::runtime::AgentHarness;
use tokio::sync::watch;

use tinyagents_tasks::{DetachedTaskRegistry, SteeringRegistry};
use tinyinference_llm::message::Message;

/// The argument key a [`SubAgentTool`] reads the child input from.
///
/// When a parent model calls the tool, the harness passes the model-supplied
/// JSON arguments. The tool reads the string field named by this constant as
/// the child run's user prompt; if the arguments are a bare JSON string the
/// whole string is used instead.
pub const SUBAGENT_INPUT_FIELD: &str = "input";

/// The argument key a [`SubAgentTool`] reads the delegation mode from.
pub const SUBAGENT_MODE_FIELD: &str = "mode";

/// How a [`SubAgentTool`] call runs its child.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SubAgentMode {
    /// Spawn the child as a background job and return its job id immediately.
    #[default]
    Background,
    /// Await the child inside the tool call and return its final result.
    Inline,
}

/// Typed policy that constructs a child's user data from its parent data.
///
/// Recursive capabilities are inherited by [`RunContext::child`](tinyagents_harness::context::RunContext::child); this policy
/// makes the separate application-data decision explicit instead of silently
/// substituting `Default` or reaching for task-local state.
#[derive(Clone)]
pub struct ChildDataPolicy<Ctx> {
    pub(crate) transform: Arc<dyn Fn(&Ctx) -> Ctx + Send + Sync>,
}

impl<Ctx> ChildDataPolicy<Ctx> {
    /// Creates a policy from a pure parent-to-child transformation.
    pub fn new(transform: impl Fn(&Ctx) -> Ctx + Send + Sync + 'static) -> Self {
        Self {
            transform: Arc::new(transform),
        }
    }

    /// Produces one child's data value from its parent.
    pub fn child_data(&self, parent: &Ctx) -> Ctx {
        (self.transform)(parent)
    }
}

/// A reusable, named child agent built on top of an [`AgentHarness`].
///
/// A `SubAgent` bundles:
/// - an `Arc<AgentHarness<State, Ctx>>` that drives the child agent loop,
/// - a stable `name` and `description` (used for tool schemas and observability),
/// - an optional `system_prompt` prepended to every child run as a system
///   message (the "fixed prompt template").
///
/// Invoking a sub-agent always produces a *child run* one level deeper than the
/// caller's depth. The harness's [`tinyagents_harness::limits::RunLimits::max_depth`]
/// cap bounds how deep nested sub-agents may recurse; an invocation whose child
/// depth would exceed the cap fails with
/// [`tinyagents_harness::error::TinyAgentsError::SubAgentDepth`].
///
/// `SubAgent` is cheap to clone-share via `Arc`; wrap it in an `Arc` to expose
/// the same child agent through several [`SubAgentTool`]s.
pub struct SubAgent<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// The harness that drives the child agent loop.
    pub(crate) harness: Arc<AgentHarness<State, Ctx>>,
    /// Stable identifier for the sub-agent (used as the default tool name).
    pub(crate) name: String,
    /// Human/model readable description of what the sub-agent does.
    pub(crate) description: String,
    /// Optional system prompt prepended to every child run.
    pub(crate) system_prompt: Option<String>,
}

/// A persistent, *reusable* conversation with a single [`SubAgent`].
///
/// Where [`SubAgentTool`] runs a fresh, stateless child run per tool call, a
/// `SubAgentSession` keeps the **same** underlying [`SubAgent`] (and therefore
/// the same [`AgentHarness`]) alive across multiple turns and retains the full
/// conversation transcript between them. This is *post-completion reuse*: the
/// child run finishes normally, the orchestrator inspects/awaits human input,
/// then calls the same sub-agent again — distinct from *steering*, which
/// interrupts a still-running agent.
///
/// # Human-in-the-loop reuse flow
///
/// 1. `send` the first input (e.g. a user question). The session appends it to
///    the retained transcript, runs the sub-agent over the full transcript, and
///    folds the resulting assistant (and any tool) messages back in.
/// 2. Inspect the returned [`AgentRun`](tinyagents_harness::middleware::AgentRun) and obtain human input out-of-band.
/// 3. Wrap that human input as a [`Message::user`] and `send` it again. Because
///    the prior turn's messages are still in the transcript, the sub-agent
///    answers *with full context* — without being killed and restarted.
///
/// Each send after the first emits [`AgentEvent::SubAgentReused`][reused]
/// (alongside the usual [`SubAgentStarted`][started]/[`SubAgentCompleted`][completed]
/// bracket) so reuse is observable in the event stream.
///
/// [reused]: tinyagents_harness::events::AgentEvent::SubAgentReused
/// [started]: tinyagents_harness::events::AgentEvent::SubAgentStarted
/// [completed]: tinyagents_harness::events::AgentEvent::SubAgentCompleted
pub struct SubAgentSession<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// The reused child agent. The same `Arc` is shared across every send, so
    /// the underlying harness is never reconstructed.
    pub(crate) subagent: Arc<SubAgent<State, Ctx>>,
    /// The accumulating conversation transcript carried across sends.
    pub(crate) transcript: Vec<Message>,
    /// Number of completed sends (turns) so far.
    pub(crate) turn: usize,
    /// Caller depth the child runs at; the child run is created at
    /// `parent_depth + 1` (default `0`, so the child runs at depth `1`).
    pub(crate) parent_depth: usize,
    /// Event sink the reuse lifecycle (and the child run's own events) are
    /// emitted on. Defaults to a fresh, unsubscribed sink.
    pub(crate) events: EventSink,
    /// Whether the fixed system prompt has been seeded into the transcript yet
    /// (it is prepended once, on the first send).
    pub(crate) seeded: bool,
}

/// A typed-parent dispatcher that exposes a [`SubAgent`] to a parent agent —
/// the surface that turns "agents calling agents" into an ordinary tool call.
///
/// When the parent model calls this tool, [`SubAgentTool`] spawns the wrapped
/// sub-agent as a background child run and immediately returns a
/// [`SubAgentJobId`]. By default it never waits for the child's final answer;
/// the optional `mode: "inline"` argument ([`SubAgentMode::Inline`]) instead
/// awaits the child and returns its final result in the same call. Hosts share
/// the tool's [`SubAgentJobRegistry`] with [`super::SubAgentJobsTool`] and
/// [`super::SubAgentMessageTool`] so callers can query completion or inject a message
/// at the child's next steering checkpoint.
///
/// Register this with [`tinyagents_harness::tool::ToolRegistry::register_dispatch`]. The
/// agent loop calls it with the live parent [`tinyagents_harness::context::RunContext`], so
/// its depth, cancellation, events, stores, workspace, steering, and streaming
/// state are inherited by the child. [`ChildDataPolicy`] makes the separate
/// application-data decision explicit.
pub struct SubAgentTool<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// The wrapped child agent.
    pub(crate) subagent: Arc<SubAgent<State, Ctx>>,
    /// Tool name exposed to the model (defaults to the sub-agent name).
    pub(crate) tool_name: String,
    /// Explicit parent-to-child application-data policy.
    pub(crate) child_data: ChildDataPolicy<Ctx>,
    /// JSON Schema describing the tool's model-visible arguments.
    pub(crate) parameters: Value,
    /// Cached [`tinyagents_harness::tool::ToolDispatch::tool`] declaration, built once from
    /// `tool_name`/`parameters` on first access rather than allocated fresh
    /// (a new `Arc` with cloned schema `Value`) on every call — `tool()` is
    /// invoked several times per admitted call plus once per tool per run for
    /// `schemas()` (M-4). Safe to cache lazily: the `with_tool_name`/
    /// `with_parameters` builders consume `self` and are only meant to run
    /// before the tool is registered, never after.
    pub(crate) declaration: std::sync::OnceLock<Arc<dyn tinytools::Tool>>,
    /// Shared registry that owns asynchronous child-job state and controls.
    pub(crate) jobs: SubAgentJobRegistry,
    /// Spawn admission ledger; unlimited unless the host injects a policy.
    pub(crate) admission: crate::subagent::SpawnAdmission,
    /// Timeout, retry and budget applied around each spawned child.
    pub(crate) policy: crate::subagent::SubAgentPolicy,
    /// Whether the child may delegate; a leaf must not expose delegation tools.
    pub(crate) role: crate::subagent::SubagentRole,
    /// Names of the host's own delegation tools a leaf child must not expose.
    pub(crate) delegation_tools: Vec<String>,
    /// Trim/check applied to the child's final output.
    pub(crate) result_policy: crate::subagent::ResultPolicy,
}

/// Stable identifier returned immediately when a subagent job is spawned.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SubAgentJobId(pub(crate) String);

impl SubAgentJobId {
    /// Returns the host-safe identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SubAgentJobId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Observable lifecycle state of an asynchronous subagent job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubAgentJobStatus {
    /// Accepted and waiting for its Tokio task to begin polling.
    Queued,
    /// The child agent loop is executing.
    Running,
    /// The child completed successfully.
    Completed,
    /// The child failed.
    Failed,
    /// The child stopped without a complete answer (timeout or budget); see
    /// [`SubAgentJob::incomplete_kind`].
    Incomplete,
    /// The child observed cooperative cancellation.
    Cancelled,
}

impl SubAgentJobStatus {
    /// Whether no further execution transition can occur.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Incomplete | Self::Cancelled
        )
    }
}

/// Host-queryable snapshot of one asynchronous subagent job.
///
/// The registry is the source of job snapshots. The link fields are populated
/// by the registry when a job is created; callers should obtain snapshots from
/// it rather than constructing this record directly.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SubAgentJob {
    /// Stable job identifier returned by the spawning tool.
    pub id: SubAgentJobId,
    /// Named subagent executing the work.
    pub agent: String,
    /// Current lifecycle status.
    pub status: SubAgentJobStatus,
    /// Latest successful assistant text, once completed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Host-safe failure text, once failed or cancelled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Run id of the child run executing this job; the same id the child's
    /// own events and transcript use. Absent on jobs created before the link
    /// existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_run_id: Option<String>,
    /// Id of the parent tool call that spawned this job, when the dispatcher
    /// supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_tool_call_id: Option<String>,
    /// Typed cause when [`Self::status`] is `Incomplete`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_kind: Option<crate::subagent::IncompleteKind>,
    /// Artifacts the result policy stored for an oversized output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<crate::subagent::ArtifactReference>,
    /// Why the output failed the result policy's schema, if one was set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_error: Option<String>,
    /// Why an oversized output could not be stored as an artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_error: Option<String>,
}

/// Explicit link from a spawned job back to the parent call and child run.
#[derive(Clone, Debug, Default)]
pub(crate) struct JobLink {
    pub(crate) subagent_run_id: Option<String>,
    pub(crate) parent_tool_call_id: Option<String>,
}

/// Shared registry behind asynchronous subagent spawning and host controls.
///
/// This is a thin adapter over [`tinyagents_tasks::DetachedTaskRegistry`], the
/// one live detached-task registry implementation. The registry holds each
/// job's [`SubAgentJob`] snapshot as its watched status (so ownership,
/// snapshots, steering lookup and request-id dedupe are the detached
/// registry's), registered cooperatively because a job is stopped through its
/// own cancellation token rather than hard-aborted. The adapter adds only what
/// is specific to subagent jobs: the lifecycle transitions, the
/// cancel-then-settle protocol, and retention of settled jobs.
#[derive(Clone)]
pub struct SubAgentJobRegistry {
    pub(crate) tasks: DetachedTaskRegistry<JobMeta, SubAgentJob>,
    pub(crate) steering: SteeringRegistry,
    /// Status senders and cancel flags. Also the transition gate: every
    /// mutation holds this lock, so a settle and a cancel never interleave.
    pub(crate) controls: Arc<Mutex<HashMap<SubAgentJobId, JobControl>>>,
}

impl Default for SubAgentJobRegistry {
    fn default() -> Self {
        let steering = SteeringRegistry::new();
        Self {
            // Settled jobs must stay queryable, so the soft cap that makes the
            // detached registry sweep terminal entries is never reached.
            tasks: DetachedTaskRegistry::new(steering.clone(), usize::MAX, |job: &SubAgentJob| {
                job.status.is_terminal()
            }),
            steering,
            controls: Arc::default(),
        }
    }
}

/// Application metadata kept with each job in the detached registry.
#[derive(Clone, Debug, Default)]
pub(crate) struct JobMeta;

/// The adapter-owned half of a job: the status publisher and cancel flag.
pub(crate) struct JobControl {
    pub(crate) status: watch::Sender<SubAgentJob>,
    /// Whether cancellation was requested while the child was still running.
    /// The job remains non-terminal until the child reports its result.
    pub(crate) cancellation_requested: bool,
}

/// Error returned by job lookup or live-message delivery.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SubAgentJobError {
    /// No job exists for the supplied id.
    #[error("unknown subagent job `{0}`")]
    NotFound(String),
    /// Messages and cancellation only apply while a job is queued or running.
    #[error("subagent job `{job_id}` is already {status:?}")]
    Terminal {
        /// Target job id.
        job_id: String,
        /// Terminal status observed by the registry.
        status: SubAgentJobStatus,
    },
    /// Cancellation has been requested and queued messages will not be read.
    #[error("subagent job `{0}` is cancelling")]
    Cancelling(String),
    /// The request identifier exceeds the bounded registry size.
    #[error("subagent request id is too long")]
    RequestIdTooLong,
}
