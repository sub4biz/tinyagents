//! Tool policy and selection middleware: allowlisting, classification-based
//! policy enforcement, dynamic/contextual tool selection, and human
//! approval.
//!
//! Split out of `library/mod.rs`; see that module's doc comment for the
//! full built-in middleware library overview.

use super::*;
use tinytools::{SandboxMode, ToolContent, ToolPolicy, ToolResult, ToolSideEffects};

// ── ToolAllowlistMiddleware ───────────────────────────────────────────────────

impl ToolAllowlistMiddleware {
    /// Creates an allowlist middleware permitting only the named tools.
    pub fn new(allowed: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            label: "tool_allowlist",
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns `true` if `name` is on the allowlist.
    ///
    /// Delegates to [`crate::tool::toolset::tool_name_allowed`] — the exact
    /// membership test [`crate::tool::toolset::FilteredToolSet::allowing`]
    /// uses — so this middleware and its `ToolSet` counterpart cannot drift
    /// (`docs/runtime-comparison/pydantic-ai.md` §4).
    pub fn allows(&self, name: &str) -> bool {
        crate::tool::toolset::tool_name_allowed(&self.allowed, name)
    }

    /// The enforcement shared by `before_tool` and `check_nested_tool`.
    fn check_allowed(&self, name: &str) -> Result<()> {
        if !self.allows(name) {
            return Err(TinyAgentsError::Validation(format!(
                "tool `{name}` is not on the allowlist"
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ToolAllowlistMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        self.check_allowed(&call.name)
    }

    async fn check_nested_tool(
        &self,
        _ctx: &RunContext<Ctx>,
        _state: &State,
        call: &ToolCall,
    ) -> Result<()> {
        self.check_allowed(&call.name)
    }
}

// ── ToolPolicyMiddleware ──────────────────────────────────────────────────────

impl ToolPolicyMiddleware {
    /// Creates a policy middleware from a name→policy snapshot (typically
    /// [`ToolRegistry::policies`][crate::tool::ToolRegistry::policies]).
    ///
    /// Defaults are permissive: nothing is required or denied until configured.
    /// Use [`strict`](Self::strict) for a fail-closed baseline.
    pub fn new(policies: std::collections::HashMap<String, ToolPolicy>) -> Self {
        Self {
            label: "tool_policy",
            policies,
            require_classification: false,
            require_background_safe: false,
            deny: ToolSideEffects::default(),
            require_sandbox: false,
            require_approval: false,
            approved: std::collections::HashSet::new(),
            enforce_result_bytes: false,
            exempt_discovery_bridge: false,
        }
    }

    /// Creates a fail-closed policy middleware: unclassified tools are rejected,
    /// and tools declaring `destructive` or `payment` side effects are denied.
    pub fn strict(policies: std::collections::HashMap<String, ToolPolicy>) -> Self {
        Self {
            label: "tool_policy",
            policies,
            require_classification: true,
            require_background_safe: false,
            deny: ToolSideEffects {
                destructive: true,
                payment: true,
                ..ToolSideEffects::default()
            },
            require_sandbox: false,
            require_approval: false,
            approved: std::collections::HashSet::new(),
            enforce_result_bytes: false,
            exempt_discovery_bridge: false,
        }
    }

    /// Exempts the intrinsic `tool_search` discovery-bridge name
    /// from classification/side-effect checks whenever `policies` has no
    /// entry for them. See the field doc on
    /// [`ToolPolicyMiddleware::exempt_discovery_bridge`] for why this is
    /// opt-in rather than automatic under [`strict`](Self::strict): it is
    /// only safe when `policies` is a complete registry snapshot (typically
    /// [`ToolRegistry::policies`][crate::tool::ToolRegistry::policies]), so
    /// that "no entry for this name" reliably means "not a registered tool."
    /// A host-registered tool under either reserved name still wins — the
    /// intrinsic bridge only fills a name nobody registered — and is
    /// evaluated by its own policy entry as normal, whether or not this is
    /// enabled.
    pub fn exempt_discovery_bridge(mut self, exempt: bool) -> Self {
        self.exempt_discovery_bridge = exempt;
        self
    }

    /// Requires every tool to carry a classified policy (fail closed on
    /// unclassified or unknown tools).
    pub fn require_classification(mut self, require: bool) -> Self {
        self.require_classification = require;
        self
    }

    /// Requires every exposed/executed tool to be `background_safe`.
    pub fn require_background_safe(mut self, require: bool) -> Self {
        self.require_background_safe = require;
        self
    }

    /// Denies tools declaring any side effect present in `mask`.
    pub fn deny_side_effects(mut self, mask: ToolSideEffects) -> Self {
        self.deny = mask;
        self
    }

    /// Enforces that a tool declaring
    /// [`SandboxMode::Required`](tinytools::SandboxMode::Required)
    /// only runs when the run carries a sandboxed workspace (fail closed
    /// otherwise). See [`RunContext::with_workspace`][crate::context::RunContext::with_workspace].
    pub fn require_sandbox(mut self, require: bool) -> Self {
        self.require_sandbox = require;
        self
    }

    /// Blocks any tool declaring `approval_required` unless its name is in
    /// `approved`, turning the declarative approval flag into a fail-closed gate.
    pub fn require_approval(
        mut self,
        approved: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.require_approval = true;
        self.approved = approved.into_iter().map(Into::into).collect();
        self
    }

    /// Enforces each tool's declared `max_result_bytes` cap by truncating and
    /// flagging oversized results in `after_tool`.
    pub fn enforce_result_bytes(mut self, enforce: bool) -> Self {
        self.enforce_result_bytes = enforce;
        self
    }

    /// Returns `Ok(())` if the named tool is permitted, otherwise an explanation
    /// of why it is blocked. Used by both the exposure and execution hooks so a
    /// hidden tool cannot be executed by a divergent decision.
    fn evaluate(&self, name: &str) -> std::result::Result<(), String> {
        // The intrinsic `tool_search` discovery bridge is never a registered
        // tool (see `crate::tool::discover`), so it never has a policy entry.
        // Exempting it here is opt-in (`exempt_discovery_bridge`), not
        // automatic even under `strict()`: unconditionally treating every
        // policy-less tool sharing this magic name as the safe intrinsic
        // bridge would let a real, side-effecting host tool bypass `strict()`'s
        // fail-closed checks whenever the caller's `policies` snapshot is
        // incomplete or stale. When enabled, this is still safe: the bridge
        // itself has no side effects (search only reads the run's catalogue),
        // and a deferred tool is called by its own name, so the real call is
        // evaluated against its own policy like any other. A host that
        // registers its own tool under this name still wins (the bridge only
        // fills a name nobody registered), and that registration is evaluated
        // normally since it hits the `self.policies.get` lookup below.
        if self.exempt_discovery_bridge
            && !self.policies.contains_key(name)
            && name == crate::tool::discover::TOOL_SEARCH_NAME
        {
            return Ok(());
        }
        let Some(policy) = self.policies.get(name) else {
            if self.require_classification {
                return Err(format!("tool `{name}` has no declared policy"));
            }
            return Ok(());
        };
        if self.require_classification && !policy.classified {
            return Err(format!("tool `{name}` is unclassified"));
        }
        let s = &policy.side_effects;
        let d = &self.deny;
        let denied = (d.writes_files && s.writes_files)
            || (d.network && s.network)
            || (d.installs_dependencies && s.installs_dependencies)
            || (d.destructive && s.destructive)
            || (d.external_service && s.external_service)
            || (d.payment && s.payment);
        if denied {
            return Err(format!("tool `{name}` declares a denied side effect"));
        }
        if self.require_background_safe && !policy.access.background_safe {
            return Err(format!("tool `{name}` is not background-safe"));
        }
        if self.require_approval && policy.access.approval_required && !self.approved.contains(name)
        {
            return Err(format!(
                "tool `{name}` requires approval that was not granted"
            ));
        }
        Ok(())
    }

    /// The context-aware slice of policy enforcement: the sandbox requirement
    /// depends on the run's workspace, which `evaluate` (name-only) cannot see.
    fn evaluate_sandbox<Ctx>(
        &self,
        name: &str,
        ctx: &RunContext<Ctx>,
    ) -> std::result::Result<(), String> {
        if !self.require_sandbox {
            return Ok(());
        }
        let Some(policy) = self.policies.get(name) else {
            return Ok(());
        };
        if policy.runtime.sandbox != SandboxMode::Required {
            return Ok(());
        }
        let sandboxed = ctx
            .workspace
            .as_ref()
            .is_some_and(|ws| ws.sandbox == SandboxMode::Required);
        if sandboxed {
            Ok(())
        } else {
            Err(format!(
                "tool `{name}` requires a sandbox but the run has none"
            ))
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ToolPolicyMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.tools.retain(|schema| {
            self.evaluate(&schema.name).is_ok() && self.evaluate_sandbox(&schema.name, ctx).is_ok()
        });
        Ok(())
    }

    async fn before_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        self.evaluate(&call.name)
            .and_then(|_| self.evaluate_sandbox(&call.name, ctx))
            .map_err(TinyAgentsError::Validation)
    }

    async fn check_nested_tool(
        &self,
        ctx: &RunContext<Ctx>,
        _state: &State,
        call: &ToolCall,
    ) -> Result<()> {
        self.evaluate(&call.name)
            .and_then(|_| self.evaluate_sandbox(&call.name, ctx))
            .map_err(TinyAgentsError::Validation)
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> Result<()> {
        if !self.enforce_result_bytes {
            return Ok(());
        }
        let Some(max_result_bytes) = self
            .policies
            .get(invocation.tool_name())
            .and_then(|policy| policy.runtime.max_result_bytes)
        else {
            return Ok(());
        };
        truncate_result(result, max_result_bytes);
        Ok(())
    }
}

/// Truncates each model-facing canonical representation without changing the
/// tool's reported-error or trusted-verbatim declarations. JSON content is
/// rendered to text when it exceeds the cap because a partial JSON block would
/// no longer be a valid structured value.
fn truncate_result(result: &mut ToolResult, max_result_bytes: usize) {
    let output = result.output();
    if output.len() > max_result_bytes {
        result.content = vec![ToolContent::Text {
            text: truncate_with_marker(&output, max_result_bytes),
        }];
    }
    if let Some(markdown) = &mut result.markdown_formatted
        && markdown.len() > max_result_bytes
    {
        *markdown = truncate_with_marker(markdown, max_result_bytes);
    }
}

fn truncate_with_marker(value: &str, max_bytes: usize) -> String {
    const MARKER: &str = "[truncated]";
    if max_bytes <= MARKER.len() {
        return truncate_utf8(MARKER, max_bytes);
    }
    format!(
        "{}{}",
        truncate_utf8(value, max_bytes - MARKER.len()),
        MARKER
    )
}

/// Returns the longest valid UTF-8 prefix fitting within `max_bytes`.
fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

// ── DynamicToolSelectionMiddleware ────────────────────────────────────────────

impl DynamicToolSelectionMiddleware {
    /// Creates a selection middleware exposing only tools for which `predicate`
    /// returns `true`.
    pub fn new(predicate: ToolPredicate) -> Self {
        Self {
            label: "dynamic_tool_selection",
            predicate,
        }
    }

    /// Creates a selection middleware exposing only the named tools.
    pub fn allowing(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let allowed: HashSet<String> = names.into_iter().map(Into::into).collect();
        Self::new(Arc::new(move |schema: &ToolSchema| {
            allowed.contains(&schema.name)
        }))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for DynamicToolSelectionMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        // Delegates to `PreparedToolSet`'s retain helper — see
        // `crate::tool::toolset::retain_matching_schemas`'s doc comment for
        // why this is shared rather than a second `retain` implementation.
        crate::tool::toolset::retain_matching_schemas(&mut request.tools, self.predicate.as_ref());
        Ok(())
    }
}

// ── ContextualToolSelectionMiddleware ─────────────────────────────────────────

impl ContextualToolSelectionMiddleware {
    /// Creates a selection middleware from a context-aware predicate.
    pub fn new(predicate: ContextualToolPredicate) -> Self {
        Self {
            label: "contextual_tool_selection",
            predicate,
        }
    }

    /// Builds a selection middleware from explicit allow/deny lists.
    ///
    /// Composition rules (fail-closed):
    /// - a tool named in `deny` is always hidden;
    /// - when `allow` is `Some`, a tool must be named in it to be exposed
    ///   (unknown tools are hidden);
    /// - when `allow` is `None`, everything not denied is exposed.
    pub fn from_lists(
        allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        deny: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let allow: Option<HashSet<String>> =
            allow.map(|names| names.into_iter().map(Into::into).collect());
        let deny: HashSet<String> = deny.into_iter().map(Into::into).collect();
        Self::from_resolved_lists(allow, deny)
    }

    /// Builds a selection middleware whose effective policy is a child allow/deny
    /// pair *composed with* an inherited parent policy, so a delegated sub-agent
    /// can only ever narrow — never widen — the tools its parent allowed.
    ///
    /// Inheritance rules:
    /// - **deny is additive**: the effective denylist is `parent_deny ∪ child_deny`
    ///   (a child cannot un-deny what the parent denied);
    /// - **allow is intersective**: if both parent and child restrict to an
    ///   allowlist, the effective allowlist is their intersection; if only one
    ///   restricts, that allowlist applies; if neither does, all-not-denied is
    ///   exposed.
    ///
    /// The result is fail-closed for the same reasons as [`Self::from_lists`].
    pub fn inheriting(
        parent_allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        parent_deny: impl IntoIterator<Item = impl Into<String>>,
        child_allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        child_deny: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let parent_allow: Option<HashSet<String>> =
            parent_allow.map(|n| n.into_iter().map(Into::into).collect());
        let child_allow: Option<HashSet<String>> =
            child_allow.map(|n| n.into_iter().map(Into::into).collect());
        let allow = match (parent_allow, child_allow) {
            (Some(p), Some(c)) => Some(p.intersection(&c).cloned().collect()),
            (Some(p), None) => Some(p),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };
        let mut deny: HashSet<String> = parent_deny.into_iter().map(Into::into).collect();
        deny.extend(child_deny.into_iter().map(Into::into));
        Self::from_resolved_lists(allow, deny)
    }

    fn from_resolved_lists(allow: Option<HashSet<String>>, deny: HashSet<String>) -> Self {
        Self::new(Arc::new(move |schema: &ToolSchema, _ctx| {
            if deny.contains(&schema.name) {
                return false;
            }
            match &allow {
                Some(set) => set.contains(&schema.name),
                None => true,
            }
        }))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for ContextualToolSelectionMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        let selection = ToolSelectionContext {
            run_id: ctx.config.run_id.as_str().to_string(),
            depth: ctx.config.depth(),
            tags: ctx.config.tags.clone(),
            requested_model: request.model.clone(),
        };
        let mut excluded = Vec::new();
        request.tools.retain(|schema| {
            let keep = (self.predicate)(schema, &selection);
            if !keep {
                excluded.push(schema.name.clone());
            }
            keep
        });
        // Make the exposure decision auditable when it actually withheld tools.
        if !excluded.is_empty() {
            ctx.emit(AgentEvent::ToolsFiltered {
                by: self.label.to_string(),
                explanations: excluded
                    .iter()
                    .map(|name| {
                        (
                            name.clone(),
                            crate::tool::ToolExposureExplanation::FilteredOut,
                        )
                    })
                    .collect(),
                excluded,
                remaining: request.tools.len(),
            });
        }
        Ok(())
    }
}

// ── HumanApprovalMiddleware ───────────────────────────────────────────────────

impl HumanApprovalMiddleware {
    /// Creates an approval middleware that interrupts when any flagged tool is
    /// called and no approval callback is configured.
    pub fn new(flagged: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            label: "human_approval",
            flagged: flagged.into_iter().map(Into::into).collect(),
            approve: None,
            outcome: None,
        }
    }

    /// Attaches an approval callback consulted for flagged tools. Returning
    /// `true` admits the call; `false` (or no callback) raises an interrupt.
    pub fn with_approval(mut self, approve: ApprovalFn) -> Self {
        self.approve = Some(approve);
        self
    }

    /// Attaches a callback that decides [`ApprovalOutcome::Allow`],
    /// [`ApprovalOutcome::Deny`], or [`ApprovalOutcome::Defer`] for each
    /// flagged call (A2). Takes precedence over [`Self::with_approval`].
    pub fn with_approval_outcome(mut self, outcome: ApprovalOutcomeFn) -> Self {
        self.outcome = Some(outcome);
        self
    }

    /// Resolves one flagged call to a control outcome, or a signal error the
    /// tool-admission path turns into a deferral / a denial answer.
    ///
    /// A call the resume path already approved is always allowed, so the
    /// same gate cannot defer it a second time. A `nested` call (see
    /// [`Middleware::check_nested_tool`]) never inherits an approval granted by
    /// id: nested ids (`<parent>/<n>`) are minted by the harness, so an
    /// approval recorded for a model call with the same id was not for them.
    fn decide<Ctx>(
        &self,
        ctx: &RunContext<Ctx>,
        call: &ToolCall,
        nested: bool,
    ) -> Result<MiddlewareControl> {
        if !self.flagged.contains(&call.name) || (!nested && ctx.is_call_approved(&call.id)) {
            return Ok(MiddlewareControl::Continue);
        }
        if let Some(outcome) = &self.outcome {
            return match outcome(call) {
                ApprovalOutcome::Allow => Ok(MiddlewareControl::Continue),
                ApprovalOutcome::Deny(message) => Err(TinyAgentsError::ToolFailed(message)),
                ApprovalOutcome::Defer => Err(TinyAgentsError::ApprovalRequired {
                    metadata: serde_json::json!({
                        "gate": self.label,
                        "tool": call.name,
                    }),
                }),
            };
        }
        let approved = self
            .approve
            .as_ref()
            .map(|approve| approve(call))
            .unwrap_or(false);
        if approved {
            Ok(MiddlewareControl::Continue)
        } else {
            Ok(MiddlewareControl::Interrupt {
                node: "tool".to_string(),
                message: format!("tool `{}` requires human approval", call.name),
            })
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for HumanApprovalMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        match self.decide(ctx, call, false)? {
            MiddlewareControl::Interrupt { node, message } => {
                Err(TinyAgentsError::Interrupted { node, message })
            }
            _ => Ok(()),
        }
    }

    /// Control-outcome override (A1): a flagged, unapproved call requests
    /// [`MiddlewareControl::Interrupt`] instead of erroring the run out
    /// directly. The agent loop drains the request at its next safe
    /// checkpoint — the same place any other interrupt is honored — and
    /// surfaces the identical [`TinyAgentsError::Interrupted`], so callers
    /// driving the harness through the ordinary loop see no behavior change;
    /// what changes is that the interrupt is now expressed in the shared
    /// control vocabulary a durable HITL host can also inspect via
    /// [`RunContext::take_control`][crate::context::RunContext::take_control]
    /// before it is drained, rather than only as a thrown error.
    ///
    /// With an [`ApprovalOutcomeFn`] installed (A2), `Deny` and `Defer` are
    /// returned as the `ToolFailed` / `ApprovalRequired` signals that tool
    /// admission answers or defers without failing the run.
    async fn before_tool_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<MiddlewareControl> {
        self.decide(ctx, call, false)
    }

    /// A flagged nested call is decided by the same callbacks as a model-issued
    /// one, but an outcome that would defer or interrupt fails instead: the
    /// parent is mid-execution and cannot pause.
    async fn check_nested_tool(
        &self,
        ctx: &RunContext<Ctx>,
        _state: &State,
        call: &ToolCall,
    ) -> Result<()> {
        match self.decide(ctx, call, true) {
            Ok(MiddlewareControl::Continue) => Ok(()),
            Ok(_) | Err(TinyAgentsError::ApprovalRequired { .. }) => {
                Err(TinyAgentsError::ToolFailed(format!(
                    "nested call '{}' requires approval; nested calls cannot be deferred",
                    call.name
                )))
            }
            Err(other) => Err(other),
        }
    }
}

// ── PlanModeMiddleware ─────────────────────────────────────────────────────────

impl PlanModeMiddleware {
    /// Creates a plan-mode middleware driven by `mode`, classifying tools
    /// from `policies`. The allowlist starts empty; widen it with
    /// [`Self::allow`].
    pub fn new(
        mode: RunModeHandle,
        policies: std::collections::HashMap<String, ToolPolicy>,
    ) -> Self {
        Self {
            label: "plan_mode",
            mode,
            policies,
            allow: std::collections::HashSet::new(),
        }
    }

    /// Adds tools that stay exposed and executable in [`RunMode::Plan`]
    /// regardless of their declared side effects — read-only tools and
    /// plan-mode-specific tools such as a plan-exit or review-request tool.
    pub fn allow(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allow.extend(names.into_iter().map(Into::into));
        self
    }

    /// Returns `true` when `name` declares a side effect (or has no declared
    /// policy at all — see the fail-closed note on [`PlanModeMiddleware`]).
    fn is_side_effecting(&self, name: &str) -> bool {
        let Some(policy) = self.policies.get(name) else {
            return true;
        };
        let s = &policy.side_effects;
        s.writes_files
            || s.network
            || s.installs_dependencies
            || s.destructive
            || s.external_service
            || s.payment
    }

    /// Returns `true` when `name` may be exposed/executed under
    /// [`RunMode::Plan`]: allowlisted, or classified with no side effects.
    fn allowed_in_plan_mode(&self, name: &str) -> bool {
        self.allow.contains(name) || !self.is_side_effecting(name)
    }

    /// The enforcement shared by `before_tool` and `check_nested_tool`.
    fn check_plan_mode(&self, name: &str) -> Result<()> {
        if self.mode.get() != RunMode::Plan || self.allowed_in_plan_mode(name) {
            return Ok(());
        }
        Err(TinyAgentsError::Validation(format!(
            "tool `{name}` is side-effecting and unavailable in plan mode"
        )))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for PlanModeMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        if self.mode.get() != RunMode::Plan {
            return Ok(());
        }
        let mut excluded = Vec::new();
        request.tools.retain(|schema| {
            let keep = self.allowed_in_plan_mode(&schema.name);
            if !keep {
                excluded.push(schema.name.clone());
            }
            keep
        });
        // Auditable, mirroring `ContextualToolSelectionMiddleware`: a UI or
        // log can see exactly which tools plan mode withheld and why.
        if !excluded.is_empty() {
            ctx.emit(AgentEvent::ToolsFiltered {
                by: self.label.to_string(),
                explanations: excluded
                    .iter()
                    .map(|name| {
                        (
                            name.clone(),
                            crate::tool::ToolExposureExplanation::FilteredOut,
                        )
                    })
                    .collect(),
                excluded,
                remaining: request.tools.len(),
            });
        }
        Ok(())
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        self.check_plan_mode(&call.name)
    }

    async fn check_nested_tool(
        &self,
        _ctx: &RunContext<Ctx>,
        _state: &State,
        call: &ToolCall,
    ) -> Result<()> {
        self.check_plan_mode(&call.name)
    }
}
