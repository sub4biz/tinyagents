//! First-class sub-agents with recursion-depth tracking.
//!
//! This is the harness's flagship recursion surface: it lets one agent run
//! another agent as a child of itself, so a language model orchestrating tools
//! is, transparently, a language model orchestrating *other models*. It is the
//! concrete "agents calling agents" mechanism behind the crate's
//! recursive-language-model framing — the in-harness analogue of the graph-side
//! `subgraph` recursion (in `tinyagents-graph`).
//!
//! This module provides the agent-calling-agent compositional primitive:
//!
//! - [`SubAgent`] wraps an [`AgentHarness`] and runs it as a *child run* one
//!   level deeper in the recursion tree than its caller.
//! - [`SubAgentTool`] adapts a [`SubAgent`] into a typed [`ToolDispatch`] that
//!   runs in the background by default, returning a job id immediately; callers
//!   can request `mode: "inline"` to await the child and receive its final
//!   result in the same call.
//! - [`SubAgentJobsTool`] and [`SubAgentMessageTool`] let the host or parent
//!   query those jobs and steer live children by job id.
//! - [`SubAgentSession`] keeps a single [`SubAgent`] alive across multiple
//!   turns, *reusing* the same harness while accumulating the conversation
//!   transcript — the post-completion, human-in-the-loop reuse primitive.
//!
//! # Reuse vs. steering
//!
//! There are two ways an orchestrator keeps a sub-agent "in play" across human
//! input:
//!
//! - **Reuse** ([`SubAgentSession`]): the child run *completes*, the
//!   orchestrator obtains human input, then calls the **same** sub-agent again
//!   carrying the prior transcript. Nothing is killed or restarted.
//! - **Steering** ([`SubAgentMessageTool`]): an orchestrator or human sends a
//!   message to a **still-running** job. Delivery occurs at a safe checkpoint.
//!
//! `SubAgentSession` implements the first. The flow is:
//!
//! 1. `session.send(state, ctx, vec![Message::user("…")])` — runs the sub-agent
//!    over the retained transcript and folds its reply back in.
//! 2. Inspect the returned [`AgentRun`]; obtain human input out-of-band.
//! 3. `session.send(state, ctx, vec![Message::user(human_reply)])` — the same
//!    sub-agent answers with the full prior context still in the transcript.
//!
//! Every send after the first emits [`AgentEvent::SubAgentReused`] so the reuse
//! is visible alongside the per-send
//! [`SubAgentStarted`][AgentEvent::SubAgentStarted]/[`SubAgentCompleted`][AgentEvent::SubAgentCompleted]
//! bracket.
//!
//! # Depth tracking
//!
//! Every run carries a `depth` in its [`RunConfig`] (top-level runs are depth
//! `0`). When a sub-agent is invoked at `parent_depth`, its child run is created
//! at `parent_depth + 1`. The depth cap is
//! [`RunLimits::max_depth`][tinyagents_harness::limits::RunLimits::max_depth]
//! (default [`RunLimits::DEFAULT_MAX_DEPTH`][tinyagents_harness::limits::RunLimits::DEFAULT_MAX_DEPTH],
//! i.e. `8`), read from the child harness's [`RunPolicy`][tinyagents_harness::runtime::RunPolicy].
//! If the child depth would exceed the cap, the invocation fails fast with
//! [`TinyAgentsError::SubAgentDepth`] *before* any model call — a deterministic,
//! cheap guard against unbounded recursion.
//!
//! # Observability
//!
//! Each invocation emits [`AgentEvent::SubAgentStarted`] and
//! [`AgentEvent::SubAgentCompleted`] (carrying the sub-agent name and child
//! depth). When invoked with a shared [`EventSink`] — via
//! [`SubAgent::invoke_with_events`] or [`SubAgent::invoke_in_parent`] — the child
//! run's own events also flow onto the parent sink, so a parent observer sees
//! the full nested run tree.
//!
//! # Layout
//!
//! - `types` holds the public type definitions.
//! - This file holds the impls (constructors, the invoke methods, and the
//!   typed-parent dispatcher).
//! - `*_tests.rs` files hold the focused tests.

mod jobs;
mod policy_run;
mod status_map;
mod types;

const LOG_PREFIX: &str = "[subagent-tool]";

pub use jobs::{SubAgentJobsTool, SubAgentMessageTool, register_subagent_job_tools};
pub use types::*;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::events::{AgentEvent, EventSink};
use tinyagents_harness::ids::{RunId, ThreadId, next_seq};
use tinyagents_harness::middleware::AgentRun;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::tool::ToolDispatch;
use tinyinference_llm::message::Message;

use super::SpawnAdmission;

impl<State: Send + Sync, Ctx: Send + Sync + 'static> SubAgent<State, Ctx> {
    /// Creates a sub-agent wrapping `harness` with a stable `name` and
    /// `description`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        harness: Arc<AgentHarness<State, Ctx>>,
    ) -> Self {
        Self {
            harness,
            name: name.into(),
            description: description.into(),
            system_prompt: None,
        }
    }

    /// Sets a fixed system prompt prepended to every child run as a leading
    /// system message. Returns `self` for chaining.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Returns the sub-agent's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the sub-agent's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the wrapped harness.
    pub fn harness(&self) -> &Arc<AgentHarness<State, Ctx>> {
        &self.harness
    }

    /// Builds the child run's seed messages from the optional system prompt and
    /// the caller `input`.
    fn seed_messages(&self, input: String) -> Vec<Message> {
        let mut messages = Vec::with_capacity(2);
        if let Some(prompt) = &self.system_prompt {
            messages.push(Message::system(prompt.clone()));
        }
        messages.push(Message::user(input));
        messages
    }

    /// Builds the child [`RunConfig`] for an invocation at `parent_depth`,
    /// enforcing the depth cap and deriving an isolated child thread from a
    /// parent thread when one is available.
    ///
    /// `parent` is `Some((parent_run_id, ordinal))` for every entry point that
    /// has a live parent [`RunContext`] to derive from — `ordinal` comes from
    /// [`RunContext::next_child_ordinal`], a counter scoped to that one
    /// context instance (not process-global). The child run id is then a pure
    /// function of the parent's run id and that ordinal
    /// (`{name}-d{depth}-{parent_run_id}-{ordinal}`), so two processes
    /// replaying the identical sequence of calls against the identical parent
    /// run derive the identical child run ids (M-2) — unlike the historical
    /// `ids::next_seq()` suffix, which restarts at a different value every
    /// process and made replayed journals of nested runs diverge across
    /// processes.
    ///
    /// `parent` is `None` only for the standalone entry points
    /// ([`Self::invoke`]/[`Self::invoke_with_events`]) that are not called
    /// with a live parent context at all; those fall back to
    /// [`tinyagents_harness::ids::next_seq`] since there is no parent run to derive
    /// determinism from.
    ///
    /// Returns [`TinyAgentsError::SubAgentDepth`] when the child depth
    /// (`parent_depth + 1`) would exceed the harness policy's `max_depth`.
    fn child_config(
        &self,
        parent_depth: usize,
        thread_id: Option<&ThreadId>,
        max_turn_output_tokens: Option<u32>,
        parent: Option<(&str, u64)>,
    ) -> Result<RunConfig> {
        let max_depth = self.harness.policy().limits.max_depth;
        let child_depth = RunConfig::checked_child_depth(parent_depth, max_depth)?;
        let child_run_id = match parent {
            Some((parent_run_id, ordinal)) => {
                format!("{}-d{child_depth}-{parent_run_id}-{ordinal}", self.name)
            }
            // No parent context to derive determinism from: suffix a
            // process-unique sequence so each invocation still gets its own
            // run id (a bare `{name}-d{depth}` was reused across invocations,
            // which interleaved journals and status stores keyed by run id).
            None => format!("{}-d{child_depth}-{}", self.name, next_seq()),
        };
        let mut config = RunConfig::new(child_run_id.clone())
            .with_depth(child_depth)
            .with_max_depth(max_depth);
        config.thread_id = thread_id.map(|parent| child_thread_id(parent, &child_run_id));
        config.max_turn_output_tokens = max_turn_output_tokens;
        Ok(config)
    }

    /// Runs the sub-agent as a child run at `parent_depth`, returning the
    /// child's [`AgentRun`].
    ///
    /// The child run is created at `parent_depth + 1`. `ctx_data` seeds the
    /// child [`RunContext`]. Sub-agent lifecycle events are emitted on the
    /// child's own (fresh) event sink; use [`Self::invoke_with_events`] to fan
    /// them out to a shared parent sink.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::SubAgentDepth`] if the child depth would
    /// exceed the configured `max_depth`, or any error surfaced by the child
    /// agent loop.
    pub async fn invoke(
        &self,
        state: &State,
        ctx_data: Ctx,
        parent_depth: usize,
        input: impl Into<String>,
    ) -> Result<AgentRun> {
        let config = self.child_config(parent_depth, None, None, None)?;
        let ctx = RunContext::new(config, ctx_data);
        self.run_child(state, ctx, input.into(), false).await
    }

    /// Like [`Self::invoke`] but routes the child run's events (and the
    /// sub-agent lifecycle events) onto the shared `events` sink so a parent
    /// observer sees the nested run.
    pub async fn invoke_with_events(
        &self,
        state: &State,
        ctx_data: Ctx,
        parent_depth: usize,
        input: impl Into<String>,
        events: &EventSink,
    ) -> Result<AgentRun> {
        let config = self.child_config(parent_depth, None, None, None)?;
        let ctx = RunContext::new(config, ctx_data).with_events(events.clone());
        self.run_child(state, ctx, input.into(), false).await
    }

    /// Runs the sub-agent as a child of the live `parent` context.
    ///
    /// This is the fully context-threaded entry point: the child depth is
    /// derived from `parent.depth()` and the child inherits the parent's event
    /// sink so all nested events share one stream. `ctx_data` seeds the child
    /// context's user data.
    ///
    /// # Errors
    ///
    /// Identical to [`Self::invoke`].
    pub async fn invoke_in_parent(
        &self,
        state: &State,
        ctx_data: Ctx,
        parent: &RunContext<Ctx>,
        input: impl Into<String>,
    ) -> Result<AgentRun> {
        // This is the generic explicit-model entry point, intentionally kept
        // available to borrowed `State` callers. A hosted parent must enter
        // through `invoke_hosted_in_parent` below, where the `State: 'static`
        // bound makes the invocation authority type-safe. Falling through to
        // the explicit loop here would discard the parent's definition and
        // approval authority, so reject it before constructing a child.
        if parent.is_hosted() {
            return Err(TinyAgentsError::Validation(
                "hosted parent delegation requires invoke_hosted_in_parent".into(),
            ));
        }
        // The child harness may tighten the tree cap, but it may never widen
        // the explicit parent lineage cap. `RunContext::child` is the one
        // place that copies the live recursive capabilities and creates the
        // isolated counters/control slot for this invocation.
        let config = self.child_config(
            parent.depth(),
            parent.thread_id(),
            parent.config.max_turn_output_tokens,
            Some((parent.run_id().as_str(), parent.next_child_ordinal())),
        )?;
        let ctx = parent.child(config, ctx_data)?;
        self.run_child(state, ctx, input.into(), parent.streaming)
            .await
    }

    /// Shared driver: emits the sub-agent lifecycle events around the child
    /// agent loop.
    ///
    /// When `streaming` is `true` the child runs through the streaming loop
    /// path, so its per-token model/reasoning deltas are emitted onto the
    /// shared [`EventSink`] and reach the parent's stream (stamped with the
    /// child's own `run_id` and `depth`). When `false` the child runs the
    /// unary path, leaving the parent's event stream unchanged.
    async fn run_child(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: String,
        streaming: bool,
    ) -> Result<AgentRun> {
        let depth = ctx.depth();
        let messages = self.seed_messages(input.clone());
        // Clone the sink (it shares listeners and the offset counter with the
        // context) so the completion event can be emitted after `ctx` is moved
        // into the child agent loop.
        let events = ctx.events.clone();

        events.emit(AgentEvent::SubAgentStarted {
            name: self.name.clone(),
            depth,
        });

        let run = if streaming {
            self.harness
                .invoke_streaming_in_context(state, ctx, messages)
                .await?
        } else {
            self.harness.invoke_in_context(state, ctx, messages).await?
        };

        events.emit(AgentEvent::SubAgentCompleted {
            name: self.name.clone(),
            depth,
        });

        Ok(run)
    }
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync + 'static> SubAgent<State, Ctx> {
    /// Runs this child under the exact host authority installed on `parent`.
    ///
    /// Unlike [`Self::invoke_in_parent`], this path resolves the parent's
    /// delegate allowlist and re-enters the child through the parent's shared
    /// invocation bundle. A context that is not hosted fails closed rather
    /// than silently acquiring an unrelated harness configuration.
    pub async fn invoke_hosted_in_parent(
        &self,
        state: &State,
        ctx_data: Ctx,
        parent: &RunContext<Ctx>,
        input: impl Into<String>,
    ) -> Result<AgentRun> {
        if !parent.is_hosted() {
            return Err(TinyAgentsError::Validation(
                "hosted subagent invocation requires parent host authority".into(),
            ));
        }
        let config = self.child_config(
            parent.depth(),
            parent.thread_id(),
            parent.config.max_turn_output_tokens,
            Some((parent.run_id().as_str(), parent.next_child_ordinal())),
        )?;
        let child = parent.child(config, ctx_data)?;
        self.run_hosted_child(state, child, input.into(), parent.streaming)
            .await
    }

    /// Hosted recursive driver. Kept separate from the generic explicit-model
    /// path so a borrowed `State` never has to interact with live host
    /// authority stored on a `RunContext`.
    async fn run_hosted_child(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: String,
        streaming: bool,
    ) -> Result<AgentRun> {
        if !ctx.is_hosted() {
            return self.run_child(state, ctx, input, streaming).await;
        }
        let depth = ctx.depth();
        let messages = self.seed_messages(input.clone());
        let events = ctx.events.clone();
        events.emit(AgentEvent::SubAgentStarted {
            name: self.name.clone(),
            depth,
        });
        let run = match self
            .harness
            .invoke_authorized_child(state, ctx, self.name.clone(), messages, streaming)
            .await?
        {
            Some(run) => run,
            None => {
                return Err(TinyAgentsError::Validation(
                    "hosted child invocation lost its parent authority".into(),
                ));
            }
        };
        events.emit(AgentEvent::SubAgentCompleted {
            name: self.name.clone(),
            depth,
        });
        Ok(run)
    }
}

/// Stamps the explicit parent-call -> child-run link onto a child run's
/// metadata, preserving whatever metadata the child already carries.
fn stamp_link_metadata(
    metadata: &mut Value,
    subagent_run_id: &str,
    job_id: &str,
    tool_call_id: Option<&str>,
) {
    if !metadata.is_object() {
        let original = std::mem::take(metadata);
        *metadata = json!({"value": original});
    }
    if let Value::Object(map) = metadata {
        map.insert("subagent_run_id".into(), json!(subagent_run_id));
        map.insert("subagent_job_id".into(), json!(job_id));
        if let Some(tool_call_id) = tool_call_id {
            map.insert("parent_tool_call_id".into(), json!(tool_call_id));
        } else {
            map.remove("parent_tool_call_id");
        }
    }
}

/// Derives an isolated child thread id from the parent thread and the child's
/// run id. The run id already carries a process-unique sequence, so the thread
/// id inherits its uniqueness.
fn child_thread_id(parent: &ThreadId, child_run_id: &str) -> ThreadId {
    ThreadId::new(format!("{}-subagent-{child_run_id}", parent.as_str()))
}

impl<State: Send + Sync, Ctx: Send + Sync> SubAgentSession<State, Ctx> {
    /// Creates a session that reuses `subagent` across turns.
    ///
    /// The child runs at depth `1` by default (caller `parent_depth = 0`); use
    /// [`Self::with_parent_depth`] to express deeper nesting.
    pub fn new(subagent: Arc<SubAgent<State, Ctx>>) -> Self {
        Self {
            subagent,
            transcript: Vec::new(),
            turn: 0,
            parent_depth: 0,
            events: EventSink::new(),
            seeded: false,
        }
    }

    /// Creates a session from an owned [`SubAgent`], wrapping it in an `Arc`.
    pub fn from_subagent(subagent: SubAgent<State, Ctx>) -> Self {
        Self::new(Arc::new(subagent))
    }

    /// Routes the reuse lifecycle and the child run's own events onto `events`
    /// so an external observer (or testkit recorder) sees every send. Returns
    /// `self` for chaining.
    pub fn with_events(mut self, events: EventSink) -> Self {
        self.events = events;
        self
    }

    /// Sets the caller depth the child runs at; the child run is created at
    /// `parent_depth + 1`. Returns `self` for chaining.
    pub fn with_parent_depth(mut self, parent_depth: usize) -> Self {
        self.parent_depth = parent_depth;
        self
    }

    /// Returns the reused sub-agent. The same `Arc` is shared across every
    /// send, so this is how callers confirm the harness was never rebuilt.
    pub fn subagent(&self) -> &Arc<SubAgent<State, Ctx>> {
        &self.subagent
    }

    /// Returns the accumulated conversation transcript carried across sends.
    pub fn transcript(&self) -> &[Message] {
        &self.transcript
    }

    /// Returns the number of completed sends (turns).
    pub fn turns(&self) -> usize {
        self.turn
    }

    /// Clears the retained transcript and turn counter, so the next [`Self::send`]
    /// starts a fresh conversation (re-seeding the fixed system prompt). The
    /// underlying [`SubAgent`]/harness is left untouched and still reused.
    pub fn reset(&mut self) {
        self.transcript.clear();
        self.turn = 0;
        self.seeded = false;
    }

    /// Builds the child [`RunConfig`] for the current turn, enforcing the depth
    /// cap exactly as [`SubAgent::invoke`] does.
    fn child_config(&self) -> Result<RunConfig> {
        let max_depth = self.subagent.harness.policy().limits.max_depth;
        let child_depth = RunConfig::checked_child_depth(self.parent_depth, max_depth)?;
        // As in `SubAgent::child_config`, suffix a process-unique sequence so
        // two sessions reusing the same sub-agent never share run ids for the
        // same turn number.
        Ok(RunConfig::new(format!(
            "{}-t{}-d{child_depth}-{}",
            self.subagent.name,
            self.turn,
            next_seq()
        ))
        .with_depth(child_depth)
        .with_max_depth(max_depth))
    }

    /// Runs the reused sub-agent for one turn over the FULL accumulated
    /// transcript, then folds the produced assistant/tool messages back into
    /// the transcript so the next send continues with full context.
    ///
    /// `input` (typically a single [`Message::user`] carrying human input) is
    /// appended to the retained transcript before the run. On the first send
    /// the sub-agent's fixed [`SubAgent::with_system_prompt`] is prepended once.
    /// The same underlying harness is reused on every call — nothing is
    /// reconstructed.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::SubAgentDepth`] if the child depth would
    /// exceed the configured `max_depth`, or any error surfaced by the child
    /// agent loop.
    pub async fn send(
        &mut self,
        state: &State,
        ctx_data: Ctx,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        // Seed the fixed system prompt once, on the first send.
        if !self.seeded {
            if let Some(prompt) = &self.subagent.system_prompt {
                self.transcript.push(Message::system(prompt.clone()));
            }
            self.seeded = true;
        }

        // Append the new (e.g. human/user) input to the retained transcript.
        self.transcript.extend(input);

        let config = self.child_config()?;
        let depth = config.depth();
        let ctx = RunContext::new(config, ctx_data).with_events(self.events.clone());

        // Clone the sink so we can emit the completion event after `ctx` is
        // moved into the child agent loop.
        let events = self.events.clone();
        events.emit(AgentEvent::SubAgentStarted {
            name: self.subagent.name.clone(),
            depth,
        });
        if self.turn > 0 {
            events.emit(AgentEvent::SubAgentReused {
                name: self.subagent.name.clone(),
                turn: self.turn,
            });
        }

        // REUSE the same underlying harness/SubAgent (no reconstruction),
        // running it over the full accumulated transcript.
        let run = self
            .subagent
            .harness
            .invoke_in_context(state, ctx, self.transcript.clone())
            .await?;

        events.emit(AgentEvent::SubAgentCompleted {
            name: self.subagent.name.clone(),
            depth,
        });

        // Carry the produced assistant/tool messages forward. `run.messages` is
        // the working transcript the loop ended with (everything we passed plus
        // the new assistant/tool messages), so the next send continues with the
        // full context.
        self.transcript = run.messages.clone();
        self.turn += 1;

        Ok(run)
    }
}

impl<State: Clone + Send + Sync + 'static, Ctx: Send + Sync + 'static> SubAgentTool<State, Ctx> {
    /// Default JSON Schema for a sub-agent tool: an object with one required
    /// string field named [`SUBAGENT_INPUT_FIELD`].
    fn default_parameters() -> Value {
        json!({
            "type": "object",
            "properties": {
                SUBAGENT_INPUT_FIELD: {
                    "type": "string",
                    "description": "The task or question to delegate to the sub-agent."
                },
                SUBAGENT_MODE_FIELD: {
                    "type": "string",
                    "enum": ["background", "inline"],
                    "description": "`background` (default) returns a job id immediately while the sub-agent keeps running; `inline` waits for the sub-agent and returns its final result in this call."
                }
            },
            "required": [SUBAGENT_INPUT_FIELD]
        })
    }

    /// Wraps `subagent` as a typed-parent tool.
    ///
    /// `child_data` is required: it makes application-data inheritance explicit
    /// for every recursive invocation.
    pub fn new(subagent: Arc<SubAgent<State, Ctx>>, child_data: ChildDataPolicy<Ctx>) -> Self {
        let tool_name = subagent.name().to_owned();
        Self {
            subagent,
            tool_name,
            child_data,
            parameters: Self::default_parameters(),
            declaration: std::sync::OnceLock::new(),
            jobs: SubAgentJobRegistry::new(),
            admission: SpawnAdmission::default(),
            policy: crate::subagent::SubAgentPolicy::default(),
            role: crate::subagent::SubagentRole::default(),
            delegation_tools: Vec::new(),
            result_policy: crate::subagent::ResultPolicy::default(),
        }
    }

    /// Applies a timeout/retry/budget policy to every spawned child.
    ///
    /// Call caps tighten the child's run limits; token caps are checked when it
    /// finishes; a retry needs `retry.max_attempts > 1` and re-runs a fresh
    /// child only after a retryable failure that ran no tools (unless
    /// [`SubAgentPolicy::retry_after_tool_calls`](crate::subagent::SubAgentPolicy)).
    pub fn with_policy(mut self, policy: crate::subagent::SubAgentPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Declares the child's role. A [`SubagentRole::Leaf`](crate::subagent::SubagentRole)
    /// child's harness must not expose delegation tools: the spawn is refused
    /// otherwise (the shared harness cannot be filtered per call).
    pub fn with_role(mut self, role: crate::subagent::SubagentRole) -> Self {
        self.role = role;
        self.warn_if_leaf_misconfigured();
        self
    }

    /// Names the host's own delegation tools, which a leaf must not expose.
    pub fn with_delegation_tools(mut self, names: Vec<String>) -> Self {
        self.delegation_tools = names;
        self.warn_if_leaf_misconfigured();
        self
    }

    /// Why a leaf cannot spawn: its harness exposes a delegation tool (the job
    /// and message tools, a host-named tool, or a tool with this tool's own
    /// name). `None` for an orchestrator or a clean leaf.
    fn leaf_violation(&self) -> Option<String> {
        if self.role.can_delegate() {
            return None;
        }
        let exposed = policy_run::delegation_tools_exposed(
            &self.subagent,
            &self.delegation_tools,
            &self.tool_name,
        );
        (!exposed.is_empty()).then(|| {
            format!(
                "You are a leaf agent: do this work yourself; do not call this tool again. (Its harness exposes delegation tools {exposed:?}, so `{}` was not started.)",
                self.tool_name
            )
        })
    }

    /// Construction-time visibility for a misconfigured leaf: the spawn is
    /// refused per call, and this logs the cause once up front.
    fn warn_if_leaf_misconfigured(&self) {
        if let Some(message) = self.leaf_violation() {
            tracing::warn!(
                "{LOG_PREFIX} leaf_misconfigured tool={} {message}",
                self.tool_name
            );
        }
    }

    /// Trims and schema-checks each child's final output.
    pub fn with_result_policy(mut self, policy: crate::subagent::ResultPolicy) -> Self {
        self.result_policy = policy;
        self
    }

    /// Enforces spawn limits through `admission` (see [`super::SpawnPolicy`]).
    ///
    /// Share one [`SpawnAdmission`] across every tool whose spawns should
    /// count against the same limits. Without this call spawning is unlimited.
    pub fn with_spawn_admission(mut self, admission: SpawnAdmission) -> Self {
        self.admission = admission;
        self
    }

    /// Returns the admission ledger this tool reserves spawn slots from.
    pub fn spawn_admission(&self) -> &SpawnAdmission {
        &self.admission
    }

    /// Uses a host-shared registry for spawned jobs and control tools.
    pub fn with_job_registry(mut self, jobs: SubAgentJobRegistry) -> Self {
        self.jobs = jobs;
        self
    }

    /// Returns the registry that owns jobs spawned by this tool.
    pub fn job_registry(&self) -> &SubAgentJobRegistry {
        &self.jobs
    }

    /// Overrides the model-visible tool name.
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    /// Overrides the model-visible JSON Schema for the tool arguments.
    pub fn with_parameters(mut self, parameters: Value) -> Self {
        self.parameters = parameters;
        self
    }

    /// Extracts the child input string from model-supplied `arguments`.
    ///
    /// Accepts either an object carrying a string [`SUBAGENT_INPUT_FIELD`] field
    /// or a bare JSON string; anything else yields the empty string.
    fn extract_input(arguments: &Value) -> String {
        match arguments {
            Value::String(s) => s.clone(),
            Value::Object(map) => map
                .get(SUBAGENT_INPUT_FIELD)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            _ => String::new(),
        }
    }

    /// Reads the optional `mode` argument; absent means [`SubAgentMode::Background`].
    fn extract_mode(arguments: &Value) -> std::result::Result<SubAgentMode, String> {
        match arguments.get(SUBAGENT_MODE_FIELD) {
            None | Some(Value::Null) => Ok(SubAgentMode::Background),
            Some(Value::String(mode)) if mode == "background" => Ok(SubAgentMode::Background),
            Some(Value::String(mode)) if mode == "inline" => Ok(SubAgentMode::Inline),
            Some(_) => Err(format!(
                "`{SUBAGENT_MODE_FIELD}` must be \"background\" or \"inline\""
            )),
        }
    }

    /// Spawns this sub-agent from the actual parent [`RunContext`].
    ///
    /// This is the agent-native recursive-tool boundary.  It is intentionally
    /// separate from `tinytools::Tool`: TinyTools only receives the narrow
    /// workspace/thread/output vocabulary it needs for ordinary tools, while
    /// a child agent must inherit the parent run's live lineage, cancellation,
    /// stores, events, workspace, steering, and streaming state.  The harness
    /// tool dispatcher registers this typed entry point explicitly; it does
    /// not downcast a generic tool or recover a parent from a global map.
    pub async fn invoke_in_parent_context(
        &self,
        state: &State,
        args: Value,
        options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> Result<tinytools::ToolResult> {
        self.invoke_in_parent_context_for_call(state, args, options, parent, None)
            .await
    }

    /// Like [`Self::invoke_in_parent_context`], additionally recording the
    /// parent's `call_id` as the explicit parent-call -> child-run link.
    ///
    /// The link is carried by the queued result (`subagent_run_id`,
    /// `parent_tool_call_id`, `job_id`), the job snapshot, and the child run's
    /// metadata (`subagent_run_id`, `subagent_job_id`, `parent_tool_call_id`).
    pub async fn invoke_in_parent_context_for_call(
        &self,
        state: &State,
        args: Value,
        _options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
        call_id: Option<&tinyagents_harness::ids::CallId>,
    ) -> Result<tinytools::ToolResult> {
        let input = Self::extract_input(&args);
        let mode = match Self::extract_mode(&args) {
            Ok(mode) => mode,
            Err(message) => {
                tracing::debug!("{LOG_PREFIX} invalid_mode tool={}", self.tool_name);
                return Ok(tinytools::ToolResult::error(message));
            }
        };
        if let Some(message) = self.leaf_violation() {
            tracing::debug!("{LOG_PREFIX} leaf_violation tool={}", self.tool_name);
            return Ok(tinytools::ToolResult::error(message));
        }
        // Reserve the slot atomically before anything is spawned. The guard
        // refunds on every early return below; it is committed once the child
        // is registered and then lives exactly as long as the child runs.
        let mut reservation = match self
            .admission
            .try_reserve(&parent.config, self.subagent.name())
        {
            Ok(reservation) => reservation,
            Err(rejection) => {
                tracing::debug!(
                    "{LOG_PREFIX} spawn_rejected tool={} reason={rejection}",
                    self.tool_name
                );
                return Ok(tinytools::ToolResult::error(format!(
                    "Sub-agent `{}` was not started because a spawn limit was reached: {rejection}. The parent orchestrator should treat this as a delegated-agent limit signal, not a completed answer.",
                    self.tool_name
                )));
            }
        };
        let mut config = match self.subagent.child_config(
            parent.depth(),
            parent.thread_id(),
            parent.config.max_turn_output_tokens,
            Some((parent.run_id().as_str(), parent.next_child_ordinal())),
        ) {
            Ok(config) => config,
            Err(error) => {
                return Ok(tinytools::ToolResult::error(format!(
                    "Sub-agent `{}` stopped before completing because it hit its recursion depth limit: {error}. The parent orchestrator should treat this as a delegated-agent limit signal, not a completed answer.",
                    self.tool_name
                )));
            }
        };
        self.policy.budget.apply_call_caps(&mut config);
        let retry_base = (self.policy.retry.max_attempts > 1).then(|| config.clone());
        let child_data = self.child_data.child_data(&parent.data);
        let child = match parent.child(config, child_data) {
            Ok(child) => child,
            Err(TinyAgentsError::SubAgentDepth(_)) => {
                return Ok(tinytools::ToolResult::error(format!(
                    "Sub-agent `{}` stopped before completing because it hit its recursion depth limit. The parent orchestrator should treat this as a delegated-agent limit signal, not a completed answer.",
                    self.tool_name
                )));
            }
            Err(error) => return Err(error),
        };
        let subagent_run_id = child.run_id().as_str().to_owned();
        let tool_call_id = call_id.map(|id| id.as_str().to_owned());
        let (job_id, steering) = self.jobs.create_with_cancellation(
            &self.tool_name,
            parent.instance_id(),
            child.cancellation.clone(),
            JobLink {
                subagent_run_id: Some(subagent_run_id.clone()),
                parent_tool_call_id: tool_call_id.clone(),
            },
        );
        tracing::debug!(
            "{LOG_PREFIX} spawn job_id={job_id} subagent_run_id={subagent_run_id} tool_call_id={tool_call_id:?}"
        );
        let mut child = child.with_steering(steering.clone());
        stamp_link_metadata(
            &mut child.config.metadata,
            &subagent_run_id,
            job_id.as_str(),
            tool_call_id.as_deref(),
        );
        // Retries need fresh child contexts, and a background child outlives the
        // borrow of `parent`, so the (cheap) contexts are built up front. Their
        // ids derive from the first child's (`{first}-a{n}`), so no extra child
        // ordinals are consumed and sibling ids do not depend on retries.
        let attempt_count = self.policy.retry.max_attempts.max(1);
        let watch = attempt_count > 1;
        let job_token = child.cancellation.clone();
        let mut attempts = vec![policy_run::Attempt::new(child, watch)];
        if let Some(base) = retry_base {
            for n in 1..attempt_count {
                let spare_id = format!("{subagent_run_id}-a{n}");
                let mut config = base.clone();
                config.run_id = RunId::new(spare_id.clone());
                config.thread_id = parent
                    .thread_id()
                    .map(|thread| child_thread_id(thread, &spare_id));
                match parent.child(config, self.child_data.child_data(&parent.data)) {
                    Ok(spare) => {
                        let mut spare = spare
                            .with_cancellation(job_token.clone())
                            .with_steering(steering.clone());
                        stamp_link_metadata(
                            &mut spare.config.metadata,
                            &spare_id,
                            job_id.as_str(),
                            tool_call_id.as_deref(),
                        );
                        attempts.push(policy_run::Attempt::new(spare, watch));
                    }
                    Err(error) => {
                        tracing::debug!("{LOG_PREFIX} retry_context_unavailable error={error}");
                        break;
                    }
                }
            }
        }
        reservation.commit();
        let streaming = parent.streaming;
        if mode == SubAgentMode::Inline {
            tracing::debug!("{LOG_PREFIX} inline.start job_id={job_id}");
            self.jobs.mark_running(&job_id);
            let mut guard = jobs::InlineJobGuard::new(self.jobs.clone(), job_id.clone());
            let finished = policy_run::run_attempts(
                &self.subagent,
                &self.policy,
                state,
                attempts,
                input,
                streaming,
                &job_token,
                &self.jobs,
                &job_id,
            )
            .await;
            policy_run::settle(&self.jobs, &job_id, finished, &self.result_policy).await;
            guard.disarm();
            drop(reservation);
            let Some(job) = self.jobs.get(job_id.as_str()) else {
                return Ok(tinytools::ToolResult::error(format!(
                    "Sub-agent job `{job_id}` was removed before its result could be read."
                )));
            };
            tracing::debug!(
                "{LOG_PREFIX} inline.done job_id={job_id} status={:?}",
                job.status
            );
            // The job snapshot, plus the `job_id` key the queued result uses
            // so both modes name the job the same way.
            let mut payload = serde_json::to_value(&job)?;
            payload["job_id"] = json!(job_id);
            return Ok(match job.status {
                SubAgentJobStatus::Completed => tinytools::ToolResult::json(payload),
                _ => tinytools::ToolResult::error(payload.to_string()),
            });
        }
        let jobs = self.jobs.clone();
        let task_job_id = job_id.clone();
        let subagent = self.subagent.clone();
        let owned_state = state.clone();
        let policy = self.policy.clone();
        let result_policy = self.result_policy.clone();
        // The child runs in its own task and a supervisor awaits its
        // `JoinHandle`: a panic inside the child surfaces as a `JoinError`
        // there, so the job can never stay `Running` forever.
        let child_job_id = task_job_id.clone();
        let child_jobs = jobs.clone();
        let child_task = tokio::spawn(async move {
            // The slot is held for the child's whole lifetime and released
            // when this task ends, however it ends (result, panic, abort).
            let _reservation = reservation;
            child_jobs.mark_running(&child_job_id);
            let finished = policy_run::run_attempts(
                &subagent,
                &policy,
                &owned_state,
                attempts,
                input,
                streaming,
                &job_token,
                &child_jobs,
                &child_job_id,
            )
            .await;
            policy_run::settle(&child_jobs, &child_job_id, finished, &result_policy).await;
        });
        tokio::spawn(async move {
            if let Err(join_error) = child_task.await {
                jobs.mark_aborted(&task_job_id, join_error.is_panic());
            }
        });

        let mut queued = json!({
            "job_id": job_id,
            "status": "queued",
            "subagent_run_id": subagent_run_id,
        });
        if let Some(tool_call_id) = tool_call_id {
            queued["parent_tool_call_id"] = Value::String(tool_call_id);
        }
        Ok(tinytools::ToolResult::json(queued))
    }
}

#[async_trait]
impl<State, Ctx> ToolDispatch<State, Ctx> for SubAgentTool<State, Ctx>
where
    State: Clone + Send + Sync + 'static,
    Ctx: Send + Sync + 'static,
{
    fn tool(&self) -> Arc<dyn tinytools::Tool> {
        // Built once and cached (M-4): `tool()` is called several times per
        // admitted call and once per tool per run for `schemas()`, and a
        // fresh `Arc<SubAgentToolDeclaration>` with a cloned `parameters`
        // `Value` on every call is unnecessary allocation for a declaration
        // that never changes after registration.
        Arc::clone(self.declaration.get_or_init(|| {
            Arc::new(SubAgentToolDeclaration {
                name: self.tool_name.clone(),
                description: self.subagent.description().to_owned(),
                parameters: self.parameters.clone(),
            })
        }))
    }

    fn output_origin(&self) -> tinyagents_harness::host::ContentOrigin {
        tinyagents_harness::host::ContentOrigin::Agent
    }

    async fn execute(
        &self,
        state: &State,
        call_id: tinyagents_harness::ids::CallId,
        arguments: Value,
        options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<tinytools::ToolResult> {
        self.invoke_in_parent_context_for_call(state, arguments, options, parent, Some(&call_id))
            .await
            .map_err(anyhow::Error::from)
    }
}

/// Pure canonical declaration for a typed-parent sub-agent dispatcher.
///
/// Calling it through `tinytools::Tool` directly is refused because that trait
/// intentionally lacks the parent `RunContext`; register the enclosing
/// [`SubAgentTool`] with [`tinyagents_harness::tool::ToolRegistry::register_dispatch`].
struct SubAgentToolDeclaration {
    name: String,
    description: String,
    parameters: Value,
}

#[async_trait]
impl tinytools::Tool for SubAgentToolDeclaration {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }

    async fn execute(&self, _args: Value) -> anyhow::Result<tinytools::ToolResult> {
        anyhow::bail!(
            "sub-agent `{}` requires typed-parent dispatch; register SubAgentTool with ToolRegistry::register_dispatch",
            self.name
        )
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;

#[cfg(test)]
#[path = "mod_jobs_tests.rs"]
mod jobs_tests;

#[cfg(test)]
#[path = "mod_link_tests.rs"]
mod link_test;

#[cfg(test)]
#[path = "mod_inline_tests.rs"]
mod inline_test;

#[cfg(test)]
#[path = "mod_admission_tests.rs"]
mod admission_test;

#[cfg(test)]
#[path = "mod_policy_tests.rs"]
mod policy_test;

#[cfg(test)]
#[path = "mod_registry_tests.rs"]
mod registry_tests;
