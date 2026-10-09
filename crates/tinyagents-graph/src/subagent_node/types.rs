//! Type definitions for [`SubAgentNode`](super::SubAgentNode) — the graph node
//! that delegates through a host-bound [`AgentInvoker`].
//!
//! See the module [`mod`](super) docs for how these are wired into a node
//! handler. This file holds the host invocation trait ([`AgentInvoker`]), the
//! structured input/output carriers
//! ([`SubAgentInput`]/[`SubAgentOutput`]), the per-call policy
//! ([`SubAgentPolicy`]/[`SubAgentBudget`]), the parent↔child mapping aliases
//! ([`InputMapper`]/[`OutputMapper`]), and the [`SubAgentNode`] descriptor.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::Result;
use tinyagents_harness::cancel::CancellationToken;
use tinyagents_harness::context::RunConfig;
use tinyagents_harness::events::EventSink;
use tinyagents_harness::ids::{GraphId, NodeId, RunId};
use tinyagents_harness::middleware::BudgetLimits;
use tinyagents_harness::retry::RetryPolicy;
use tinyinference_llm::usage::UsageTotals;

/// An explicit request for a host-owned recursive agent invocation.
///
/// A graph never synthesizes a harness context. The host binds an invoker to a
/// graph entry point and uses the request's parent graph identity to invoke its
/// actual parent [`RunContext`](tinyagents_harness::context::RunContext), which
/// in turn creates the child through `RunContext::child`.
#[derive(Clone)]
pub struct AgentInvocation {
    /// Registered agent definition id.
    pub agent_id: String,
    /// Mapped child input.
    pub input: SubAgentInput,
    /// Graph containing the delegating node.
    pub graph_id: GraphId,
    /// Node issuing the delegation.
    pub node_id: NodeId,
    /// Immediate parent graph run.
    pub parent_run_id: RunId,
    /// Root graph run shared by all descendants.
    pub root_run_id: RunId,
    /// Parent's event sink, forwarded by the host into the child context.
    pub events: EventSink,
    /// Parent cancellation signal, forwarded by the host into the child context.
    pub cancellation: Option<CancellationToken>,
}

/// Atomic, execution-scoped host capability for recursive agent invocation.
///
/// This value is supplied to one graph run; it is never stored on a reusable
/// [`CompiledGraph`](crate::CompiledGraph). Cloning it is only for descendants
/// of that same execution tree, which preserves one parent invocation context
/// while preventing separate top-level executions from bleeding signals.
#[derive(Clone)]
pub struct AgentInvocationBinding {
    /// Host entry point bound to the parent invocation context.
    pub invoker: Arc<dyn AgentInvoker>,
    /// Parent event sink forwarded to every descendant request.
    pub events: EventSink,
    /// Parent cancellation signal forwarded to every descendant request.
    pub cancellation: CancellationToken,
}

impl AgentInvocationBinding {
    /// Creates the complete binding required for graph-to-agent recursion.
    #[must_use]
    pub fn new(
        invoker: Arc<dyn AgentInvoker>,
        events: EventSink,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            invoker,
            events,
            cancellation,
        }
    }
}

/// Object-safe host boundary for graph-to-agent recursion.
///
/// Implementations must be explicitly bound by the host to an owned or
/// `Arc`-backed parent invocation context. They must dispatch through the same
/// host entry point used for a top-level harness run and create the child with
/// `RunContext::child`. This avoids globals, task locals, downcasts, and
/// fabricated `Default` state. A graph without an invoker fails closed.
#[async_trait]
pub trait AgentInvoker: Send + Sync {
    /// Invokes a requested agent using the host-bound parent context.
    async fn invoke(&self, request: AgentInvocation) -> Result<SubAgentOutput>;
}

/// The structured input a [`SubAgentNode`] hands to a delegated agent.
///
/// `prompt` is the user-facing text the child run is seeded with; `data` is an
/// optional structured payload an agent (or an adapter) may consult.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubAgentInput {
    /// The user prompt the child run is seeded with.
    pub prompt: String,
    /// Optional structured side-channel payload.
    pub data: Option<Value>,
}

impl SubAgentInput {
    /// Builds an input carrying just a `prompt`.
    pub fn prompt(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            data: None,
        }
    }

    /// Attaches a structured `data` payload, returning `self` for chaining.
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// The structured output a delegated agent returns to its [`SubAgentNode`].
///
/// Besides the final `text` and any parsed `structured` value, the output
/// carries the child run's [`UsageTotals`] and call counts so the node can roll
/// the child's usage into the parent execution and a budget can be enforced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubAgentOutput {
    /// The child run's final assistant text.
    pub text: String,
    /// Parsed structured output, when the child run produced one.
    pub structured: Option<Value>,
    /// Cumulative token usage across the child run's model calls.
    pub usage: UsageTotals,
    /// Number of model calls the child run dispatched.
    pub model_calls: usize,
    /// Number of tool invocations the child run executed.
    pub tool_calls: usize,
}

/// An optional cap on the work a single sub-agent invocation may perform.
///
/// What is enforced, and where:
///
/// | field | enforcement |
/// | --- | --- |
/// | `max_model_calls`, `max_tool_calls` | after the child returns here ([`Self::check`]); *during* the run wherever a [`RunConfig`] is available ([`Self::apply_call_caps`]) |
/// | `max_input_tokens`, `max_output_tokens` | after the child returns ([`Self::check`]) from the reported usage; a run that overshoots is reported, not interrupted |
/// | `max_cost` | **not enforced** here: cost needs a pricing table this layer does not own. [`Self::to_budget_limits`] hands the same caps to the harness `BudgetMiddleware`, which does enforce them during the run |
///
/// A run that exceeds a checked cap fails with
/// [`TinyAgentsError::LimitExceeded`](crate::TinyAgentsError::LimitExceeded).
/// The default ([`SubAgentBudget::unlimited`]) imposes no cap.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SubAgentBudget {
    /// Maximum model calls the child run may make (`None` = unbounded).
    pub max_model_calls: Option<usize>,
    /// Maximum tool calls the child run may make (`None` = unbounded).
    pub max_tool_calls: Option<usize>,
    /// Maximum cumulative input tokens (`None` = unbounded). Checked after the
    /// child returns.
    pub max_input_tokens: Option<u64>,
    /// Maximum cumulative output tokens (`None` = unbounded). Checked after
    /// the child returns.
    pub max_output_tokens: Option<u64>,
    /// Maximum cumulative estimated cost in the pricing table's currency
    /// (`None` = unbounded). Not enforced by [`Self::check`]; see the type
    /// docs and [`Self::to_budget_limits`].
    pub max_cost: Option<f64>,
}

impl SubAgentBudget {
    /// A budget that imposes no cap.
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Caps model calls, returning `self` for chaining.
    pub fn with_max_model_calls(mut self, max: usize) -> Self {
        self.max_model_calls = Some(max);
        self
    }

    /// Caps tool calls, returning `self` for chaining.
    pub fn with_max_tool_calls(mut self, max: usize) -> Self {
        self.max_tool_calls = Some(max);
        self
    }

    /// Caps cumulative input tokens, returning `self` for chaining.
    pub fn with_max_input_tokens(mut self, max: u64) -> Self {
        self.max_input_tokens = Some(max);
        self
    }

    /// Caps cumulative output tokens, returning `self` for chaining.
    pub fn with_max_output_tokens(mut self, max: u64) -> Self {
        self.max_output_tokens = Some(max);
        self
    }

    /// Caps cumulative estimated cost, returning `self` for chaining. See the
    /// type docs: this is carried, not enforced, by this layer.
    pub fn with_max_cost(mut self, max: f64) -> Self {
        self.max_cost = Some(max);
        self
    }

    /// Tightens `config`'s call caps to this budget. A cap the config already
    /// sets lower is kept: a budget can narrow a run, never widen it.
    pub fn apply_call_caps(&self, config: &mut RunConfig) {
        if let Some(max) = self.max_model_calls {
            config.max_model_calls = Some(config.max_model_calls.map_or(max, |set| set.min(max)));
        }
        if let Some(max) = self.max_tool_calls {
            config.max_tool_calls = Some(config.max_tool_calls.map_or(max, |set| set.min(max)));
        }
    }

    /// The token and cost caps as the harness's in-run
    /// [`BudgetLimits`], for a host to attach as a `BudgetMiddleware` on the
    /// child harness — the only place a cost cap can be enforced mid-run.
    pub fn to_budget_limits(&self) -> BudgetLimits {
        BudgetLimits {
            max_input_tokens: self.max_input_tokens,
            max_output_tokens: self.max_output_tokens,
            max_cost: self.max_cost,
            ..BudgetLimits::default()
        }
    }

    /// Returns `Ok(())` when `output` is within budget, else
    /// [`TinyAgentsError::LimitExceeded`](crate::TinyAgentsError::LimitExceeded).
    pub fn check(&self, output: &SubAgentOutput, agent: &str) -> Result<()> {
        if let Some(max) = self.max_model_calls
            && output.model_calls > max
        {
            return Err(crate::TinyAgentsError::LimitExceeded(format!(
                "sub-agent `{agent}` exceeded model-call budget: {} > {max}",
                output.model_calls
            )));
        }
        if let Some(max) = self.max_tool_calls
            && output.tool_calls > max
        {
            return Err(crate::TinyAgentsError::LimitExceeded(format!(
                "sub-agent `{agent}` exceeded tool-call budget: {} > {max}",
                output.tool_calls
            )));
        }
        if let Some(max) = self.max_input_tokens
            && output.usage.usage.input_tokens > max
        {
            return Err(crate::TinyAgentsError::LimitExceeded(format!(
                "sub-agent `{agent}` exceeded input-token budget: {} > {max}",
                output.usage.usage.input_tokens
            )));
        }
        if let Some(max) = self.max_output_tokens
            && output.usage.usage.output_tokens > max
        {
            return Err(crate::TinyAgentsError::LimitExceeded(format!(
                "sub-agent `{agent}` exceeded output-token budget: {} > {max}",
                output.usage.usage.output_tokens
            )));
        }
        Ok(())
    }
}

/// Timeout, retry, and budget policy applied around a sub-agent invocation.
///
/// This is a thin graph-local struct that defers to the harness
/// [`RetryPolicy`] for retry/backoff and reuses the harness usage accounting for
/// budgeting, so a graph node and an in-harness sub-agent share the same
/// resilience semantics. The default is a *single attempt, no timeout, no
/// budget* — deliberately conservative so a node never silently re-runs a
/// non-idempotent agent.
#[derive(Clone, Debug)]
pub struct SubAgentPolicy {
    /// Optional wall-clock timeout for a single attempt. On elapse the node
    /// fails with [`TinyAgentsError::Timeout`](crate::TinyAgentsError::Timeout).
    pub timeout: Option<Duration>,
    /// Retry/backoff policy applied across attempts.
    pub retry: RetryPolicy,
    /// Optional cap on the work the child run may perform.
    pub budget: SubAgentBudget,
    /// Whether a failed attempt may be retried after the child already ran
    /// tools. Defaults to `false`: tools can have side effects, so a retry is
    /// only safe when the failed attempt provably ran none.
    ///
    /// The graph node cannot observe whether a failed [`AgentInvoker`] call
    /// ran tools (an error carries no run), so it retries exactly as the
    /// `retry` policy says; the orchestration paths (`SubagentDriver`,
    /// `SubAgentTool`) can, and honour this flag.
    pub retry_after_tool_calls: bool,
}

impl Default for SubAgentPolicy {
    fn default() -> Self {
        Self {
            timeout: None,
            retry: RetryPolicy::default().with_max_attempts(1),
            budget: SubAgentBudget::unlimited(),
            retry_after_tool_calls: false,
        }
    }
}

impl SubAgentPolicy {
    /// Sets the per-attempt timeout, returning `self` for chaining.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the retry policy, returning `self` for chaining.
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Sets the work budget, returning `self` for chaining.
    pub fn with_budget(mut self, budget: SubAgentBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Allows retrying a failed attempt even after the child ran tools,
    /// returning `self` for chaining. Only for idempotent tool sets.
    pub fn with_retry_after_tool_calls(mut self, allow: bool) -> Self {
        self.retry_after_tool_calls = allow;
        self
    }
}

/// Maps parent graph `State` into the [`SubAgentInput`] a delegated agent runs
/// over.
pub type InputMapper<State> = Arc<dyn Fn(&State) -> SubAgentInput + Send + Sync>;

/// Maps a delegated agent's [`SubAgentOutput`] into the parent graph `Update`
/// merged through the graph reducer.
pub type OutputMapper<Update> = Arc<dyn Fn(SubAgentOutput) -> Update + Send + Sync>;

/// A graph node that delegates to a harness agent resolved by name.
///
/// A `SubAgentNode` binds an agent `ComponentId` (resolved against a
/// capability registry at run time) to a pair of mappers and a
/// [`SubAgentPolicy`]. Lower it into a graph node handler with
/// [`subagent_node`](super::subagent_node).
///
/// Deprecated: `tinyagents-graph` cannot depend on `tinyagents-orchestration`
/// (the dependency runs the other way), so this node cannot be an adapter over
/// `SubagentDriver`. Run agent steps through `SubagentDriver` instead; it owns
/// spawn admission, result policy, role and the typed outcome. The type keeps
/// its behaviour unchanged.
#[deprecated(note = "use tinyagents_orchestration SubagentDriver")]
pub struct SubAgentNode<State, Update> {
    /// The registered agent name to resolve and delegate to.
    pub agent: String,
    /// Projects parent state into the child input.
    pub input_mapper: InputMapper<State>,
    /// Folds the child output back into a parent update.
    pub output_mapper: OutputMapper<Update>,
    /// Timeout/retry/budget policy applied around the invocation.
    pub policy: SubAgentPolicy,
}

impl<State, Update> Clone for SubAgentNode<State, Update> {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            input_mapper: self.input_mapper.clone(),
            output_mapper: self.output_mapper.clone(),
            policy: self.policy.clone(),
        }
    }
}
