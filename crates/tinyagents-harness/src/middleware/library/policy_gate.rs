//! Host-driven tool-call policy gate with an approval outcome.
//!
//! [`ToolPolicyMiddleware`] enforces *declarative* per-tool metadata
//! ([`tinytools::ToolPolicy`]). This module is its runtime counterpart: a host
//! supplies a [`ToolCallPolicy`] that inspects each concrete call (name,
//! arguments, the run's host context) and answers with a [`PolicyDecision`] —
//! allow, deny, or **require approval**. An [`ApprovalResolver`] is the seam a
//! host implements to actually ask a human (or a policy store) and to record
//! the outcome afterwards.
//!
//! Two middleware are provided:
//!
//! - [`ApprovalGateMiddleware`] — routes calls the resolver says need approval
//!   through [`ApprovalResolver::resolve`], short-circuits a denial as a
//!   model-consumable error result, and records the terminal outcome of an
//!   approved call.
//! - [`ToolPolicyGateMiddleware`] — consults a [`ToolCallPolicy`]. `Deny` always
//!   short-circuits. `RequireApproval` fails closed unless an
//!   [`ApprovalResolver`] is attached with
//!   [`ToolPolicyGateMiddleware::with_approval_resolver`].
//!
//! Hosts that interleave the gate with their own checks can call
//! [`ToolPolicyGate::check`] directly instead of installing the middleware.

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::RunContext;
use crate::error::Result;
use crate::middleware::{MiddlewareToolOutcome, ToolHandler, ToolMiddleware};
use tinyinference_llm::tool::ToolCall;
use tinytools::ToolResult;

/// What a [`ToolCallPolicy`] decided about one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// The call may run.
    Allow,
    /// The call may run only after an approval handoff.
    RequireApproval {
        /// Why approval is needed (shown to the model when it fails closed).
        reason: String,
    },
    /// The call must not run.
    Deny {
        /// Why the call is refused.
        reason: String,
    },
}

impl PolicyDecision {
    /// Builds a [`PolicyDecision::RequireApproval`].
    pub fn require_approval(reason: impl Into<String>) -> Self {
        Self::RequireApproval {
            reason: reason.into(),
        }
    }

    /// Builds a [`PolicyDecision::Deny`].
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// The reason a fail-closed executor must refuse the call, or `None` for
    /// [`PolicyDecision::Allow`].
    pub fn blocking_reason(&self) -> Option<&str> {
        match self {
            Self::Allow => None,
            Self::RequireApproval { reason } | Self::Deny { reason } => Some(reason.as_str()),
        }
    }
}

/// Host policy consulted before a tool call executes.
///
/// `Ctx` is the host's run-context payload ([`RunContext::data`]); the policy
/// receives the whole [`RunContext`] so it can read session, channel or
/// workspace identity without the harness knowing their shape.
#[async_trait]
pub trait ToolCallPolicy<Ctx: Send + Sync = ()>: Send + Sync {
    /// Stable policy name for logs and denial messages.
    fn name(&self) -> &str;

    /// Inspects one tool call and decides whether it can execute.
    async fn check(&self, ctx: &RunContext<Ctx>, call: &ToolCall) -> PolicyDecision;
}

/// How an [`ApprovalResolver`] settled one approval request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalResolution {
    /// Run the call. `ticket` is handed back to [`ApprovalResolver::record`]
    /// with the call's terminal result (an audit id, say); `None` records
    /// nothing.
    Allow {
        /// Opaque host identifier for the approved request.
        ticket: Option<String>,
    },
    /// Do not run the call; `reason` becomes the tool-error the model sees.
    Deny {
        /// Why the request was refused.
        reason: String,
    },
}

/// Host seam that asks for approval and audits the outcome.
#[async_trait]
pub trait ApprovalResolver<Ctx: Send + Sync = ()>: Send + Sync {
    /// Whether this call must be routed through [`Self::resolve`] at all.
    async fn requires_approval(&self, ctx: &RunContext<Ctx>, call: &ToolCall) -> bool;

    /// Settles the approval request (prompting a human, consulting a store,
    /// waiting for a decision or an expiry).
    async fn resolve(&self, ctx: &RunContext<Ctx>, call: &ToolCall) -> ApprovalResolution;

    /// Records the terminal result of an approved call. Called once, only for
    /// an [`ApprovalResolution::Allow`] carrying a ticket whose tool call
    /// returned a result.
    fn record(&self, ticket: &str, result: &ToolResult);
}

/// What a host must do with a call after [`ToolPolicyGate::evaluate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateVerdict {
    /// Run the call. `ticket` is what the resolver handed back for the
    /// approved request; pass it to [`record_approved`] with the call's result.
    Proceed {
        /// Opaque host identifier for the approved request, if any.
        ticket: Option<String>,
    },
    /// The policy refused the call, or required approval and no resolver was
    /// supplied (fail closed). Carries the policy's own decision so the host
    /// can render its wording.
    Blocked(PolicyDecision),
    /// The policy required approval and the resolver refused it; `reason` is
    /// the resolver's text, shown to the model as-is.
    Refused {
        /// Why the approval request was refused.
        reason: String,
    },
}

/// A [`ToolCallPolicy`] plus the context-free glue shared by its consumers.
pub struct ToolPolicyGate<Ctx: Send + Sync = ()> {
    policy: Arc<dyn ToolCallPolicy<Ctx>>,
}

impl<Ctx: Send + Sync> Clone for ToolPolicyGate<Ctx> {
    fn clone(&self) -> Self {
        Self {
            policy: self.policy.clone(),
        }
    }
}

impl<Ctx: Send + Sync> ToolPolicyGate<Ctx> {
    /// Wraps a host policy.
    pub fn new(policy: Arc<dyn ToolCallPolicy<Ctx>>) -> Self {
        Self { policy }
    }

    /// The wrapped policy's name.
    pub fn policy_name(&self) -> &str {
        self.policy.name()
    }

    /// The effective decision for one call. When `waive_approval` is true a
    /// [`PolicyDecision::RequireApproval`] is downgraded to
    /// [`PolicyDecision::Allow`] (the host has already decided approvals do not
    /// apply to this call); `Deny` is never waived.
    pub async fn check(
        &self,
        ctx: &RunContext<Ctx>,
        call: &ToolCall,
        waive_approval: bool,
    ) -> PolicyDecision {
        let decision = self.policy.check(ctx, call).await;
        if waive_approval && matches!(decision, PolicyDecision::RequireApproval { .. }) {
            return PolicyDecision::Allow;
        }
        decision
    }
}

impl<Ctx: Send + Sync> ToolPolicyGate<Ctx> {
    /// Decides one call end to end for hosts that interleave the gate with
    /// their own checks: the policy decision (with the `waive_approval`
    /// downgrade of [`Self::check`]), then, for a
    /// [`PolicyDecision::RequireApproval`], settlement through `resolver`.
    /// Without a resolver an approval requirement fails closed as
    /// [`GateVerdict::Blocked`].
    pub async fn evaluate(
        &self,
        ctx: &RunContext<Ctx>,
        call: &ToolCall,
        waive_approval: bool,
        resolver: Option<&dyn ApprovalResolver<Ctx>>,
    ) -> GateVerdict {
        let decision = self.check(ctx, call, waive_approval).await;
        match (decision, resolver) {
            (PolicyDecision::Allow, _) => GateVerdict::Proceed { ticket: None },
            (PolicyDecision::RequireApproval { .. }, Some(resolver)) => {
                match resolver.resolve(ctx, call).await {
                    ApprovalResolution::Allow { ticket } => GateVerdict::Proceed { ticket },
                    ApprovalResolution::Deny { reason } => GateVerdict::Refused { reason },
                }
            }
            (decision, _) => GateVerdict::Blocked(decision),
        }
    }
}

/// Records the terminal result of an approved call with its resolver. A no-op
/// when there is no ticket or the outcome is not a tool result.
pub fn record_approved<Ctx: Send + Sync>(
    resolver: &dyn ApprovalResolver<Ctx>,
    ticket: Option<&str>,
    outcome: &MiddlewareToolOutcome,
) {
    if let (Some(ticket), MiddlewareToolOutcome::Result(result)) = (ticket, outcome) {
        resolver.record(ticket, result);
    }
}

async fn run_approved<State: Send + Sync, Ctx: Send + Sync>(
    resolver: &dyn ApprovalResolver<Ctx>,
    prompt_lock: &tokio::sync::Mutex<()>,
    ctx: &RunContext<Ctx>,
    state: &State,
    call: ToolCall,
    next: ToolHandler<'_, State, Ctx>,
) -> Result<MiddlewareToolOutcome> {
    let resolution = {
        // Calls of one batch overlap, but a host routes one prompt at a time
        // (e.g. per chat thread), so only the interactive part is serialised.
        let _prompting = prompt_lock.lock().await;
        resolver.resolve(ctx, &call).await
    };
    match resolution {
        ApprovalResolution::Deny { reason } => {
            Ok(MiddlewareToolOutcome::Result(ToolResult::error(reason)))
        }
        ApprovalResolution::Allow { ticket } => {
            let outcome = next.run(ctx, state, call).await?;
            record_approved(resolver, ticket.as_deref(), &outcome);
            Ok(outcome)
        }
    }
}

/// Tool-wrap middleware that routes calls through an [`ApprovalResolver`].
pub struct ApprovalGateMiddleware<Ctx: Send + Sync = ()> {
    label: &'static str,
    resolver: Arc<dyn ApprovalResolver<Ctx>>,
    /// Serialises [`ApprovalResolver::resolve`] across the overlapping calls
    /// of a concurrent batch. Calls that need no approval never take it.
    prompt_lock: tokio::sync::Mutex<()>,
}

impl<Ctx: Send + Sync> ApprovalGateMiddleware<Ctx> {
    /// Creates the middleware under a stable event `label`.
    pub fn new(label: &'static str, resolver: Arc<dyn ApprovalResolver<Ctx>>) -> Self {
        Self {
            label,
            resolver,
            prompt_lock: tokio::sync::Mutex::new(()),
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolMiddleware<State, Ctx>
    for ApprovalGateMiddleware<Ctx>
{
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareToolOutcome> {
        if !self.resolver.requires_approval(ctx, &call).await {
            return next.run(ctx, state, call).await;
        }
        run_approved(
            self.resolver.as_ref(),
            &self.prompt_lock,
            ctx,
            state,
            call,
            next,
        )
        .await
    }
}

/// Renders the model-facing text for a blocked call: tool name, policy name,
/// the blocking decision.
pub type DenialRenderer = Arc<dyn Fn(&ToolCall, &str, &PolicyDecision) -> String + Send + Sync>;

fn default_denial(call: &ToolCall, policy: &str, decision: &PolicyDecision) -> String {
    let (action, reason) = match decision {
        PolicyDecision::RequireApproval { reason } => ("requires approval", reason.as_str()),
        PolicyDecision::Deny { reason } => ("denied", reason.as_str()),
        PolicyDecision::Allow => ("allowed", ""),
    };
    format!(
        "Tool '{}' {action} by policy '{policy}': {reason}",
        call.name
    )
}

/// Tool-wrap middleware that enforces a [`ToolCallPolicy`].
pub struct ToolPolicyGateMiddleware<Ctx: Send + Sync = ()> {
    gate: ToolPolicyGate<Ctx>,
    resolver: Option<Arc<dyn ApprovalResolver<Ctx>>>,
    render: DenialRenderer,
    /// Serialises approval prompts across the overlapping calls of a
    /// concurrent batch; calls the policy allows or denies never take it.
    prompt_lock: tokio::sync::Mutex<()>,
}

impl<Ctx: Send + Sync> ToolPolicyGateMiddleware<Ctx> {
    /// Creates a fail-closed gate over `policy`.
    pub fn new(policy: Arc<dyn ToolCallPolicy<Ctx>>) -> Self {
        Self {
            gate: ToolPolicyGate::new(policy),
            resolver: None,
            render: Arc::new(default_denial),
            prompt_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Settles [`PolicyDecision::RequireApproval`] through `resolver` instead
    /// of failing closed.
    pub fn with_approval_resolver(mut self, resolver: Arc<dyn ApprovalResolver<Ctx>>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Overrides the text returned to the model for a blocked call.
    pub fn with_denial_renderer(mut self, render: DenialRenderer) -> Self {
        self.render = render;
        self
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolMiddleware<State, Ctx>
    for ToolPolicyGateMiddleware<Ctx>
{
    fn name(&self) -> &str {
        "tool_policy_gate"
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareToolOutcome> {
        // Evaluate without the resolver first so only a call that actually
        // needs approval waits on the prompt lock; the rest stay concurrent.
        let mut verdict = self.gate.evaluate(ctx, &call, false, None).await;
        if let (GateVerdict::Blocked(PolicyDecision::RequireApproval { .. }), Some(resolver)) =
            (&verdict, self.resolver.as_deref())
        {
            let _prompting = self.prompt_lock.lock().await;
            verdict = match resolver.resolve(ctx, &call).await {
                ApprovalResolution::Allow { ticket } => GateVerdict::Proceed { ticket },
                ApprovalResolution::Deny { reason } => GateVerdict::Refused { reason },
            };
        }
        match verdict {
            GateVerdict::Proceed { ticket } => {
                let outcome = next.run(ctx, state, call).await?;
                if let Some(resolver) = &self.resolver {
                    record_approved(resolver.as_ref(), ticket.as_deref(), &outcome);
                }
                Ok(outcome)
            }
            GateVerdict::Refused { reason } => {
                Ok(MiddlewareToolOutcome::Result(ToolResult::error(reason)))
            }
            GateVerdict::Blocked(decision) => {
                let text = (self.render)(&call, self.gate.policy_name(), &decision);
                Ok(MiddlewareToolOutcome::Result(ToolResult::error(text)))
            }
        }
    }
}
