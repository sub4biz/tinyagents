//! The run-level rule policy and the gate the loop consults.

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::Value;
use tinytools::{
    ApprovalDirective, RuleContext, Surface, Tool, ToolExposure, ToolRuleSet, ToolRules,
    ToolSubject,
};

/// Harness-wide tool rules and the context they are evaluated in.
///
/// Set on [`RunPolicy::tool_rules`](crate::runtime::RunPolicy::tool_rules).
/// The default holds no rules and admits every tool, so a harness that never
/// sets it behaves exactly as before.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolRulePolicy {
    /// The rule layers every tool must pass.
    pub rules: Arc<ToolRuleSet>,
    /// Attributes `when` conditions match against: `channel`, `agent`, ….
    pub context: RuleContext,
}

impl ToolRulePolicy {
    /// A policy evaluating `rules` in an empty context.
    #[must_use]
    pub fn new(rules: impl Into<ToolRuleSet>) -> Self {
        Self {
            rules: Arc::new(rules.into()),
            context: RuleContext::new(),
        }
    }

    /// Replaces the evaluation context.
    #[must_use]
    pub fn with_context(mut self, context: RuleContext) -> Self {
        self.context = context;
        self
    }

    /// Whether the policy admits every tool, so the loop can skip it.
    #[must_use]
    pub fn is_permissive(&self) -> bool {
        self.rules.is_permissive()
    }
}

/// What the rules say about one model call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CallGate {
    /// The call may proceed, under this approval directive.
    Admit(ApprovalDirective),
    /// The call is refused; the message names the rule.
    Refuse(String),
}

/// The run's resolved tool gate: the definition's exact allowlist plus every
/// rule layer that applies, evaluated in one context.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolGate {
    allowed: Option<HashSet<String>>,
    rules: Option<Arc<ToolRuleSet>>,
    context: RuleContext,
}

impl ToolGate {
    /// Builds the gate from the resolved allowlist, the harness policy and a
    /// hosted definition's rules. A permissive rule set is dropped so an
    /// unconfigured run pays nothing per tool.
    pub(crate) fn new(
        allowed: Option<HashSet<String>>,
        policy: &ToolRulePolicy,
        definition: Option<&ToolRules>,
    ) -> Self {
        let rules = match definition.filter(|layer| !layer.is_permissive()) {
            Some(layer) => {
                let mut merged = (*policy.rules).clone();
                merged.push(layer.clone());
                Some(Arc::new(merged))
            }
            None => (!policy.is_permissive()).then(|| policy.rules.clone()),
        };
        Self {
            allowed,
            rules,
            context: policy.context.clone(),
        }
    }

    /// Whether the exact allowlist admits `name`. Rules are not consulted.
    pub(crate) fn allows_name(&self, name: &str) -> bool {
        self.allowed
            .as_ref()
            .is_none_or(|allowed| allowed.contains(name))
    }

    /// Whether `name` may be listed on `surface` (catalogue or search).
    /// Evaluated against `tool` when the caller has it, otherwise against the
    /// bare name.
    pub(crate) fn lists(&self, name: &str, tool: Option<&dyn Tool>, surface: Surface) -> bool {
        if !self.allows_name(name) {
            return false;
        }
        let Some(rules) = &self.rules else {
            return true;
        };
        let subject = tool.map_or_else(|| ToolSubject::named(name), ToolSubject::of);
        let decision = rules.evaluate(&subject, &self.context, surface, None);
        if !decision.visible {
            tracing::debug!(
                target: "tinyagents::tool_rules",
                tool = %name,
                surface = ?surface,
                rule = ?decision.blocked_by,
                "[tool_rules] tool withheld from listing"
            );
        }
        decision.visible
    }

    /// [`Self::lists`] on the surface a tool's exposure puts it on: search
    /// for a deferred tool, the catalogue otherwise.
    pub(crate) fn lists_tool(&self, tool: &dyn Tool) -> bool {
        let surface = if tool.exposure() == ToolExposure::Deferred {
            Surface::Search
        } else {
            Surface::Catalog
        };
        self.lists(tool.name(), Some(tool), surface)
    }

    /// The rules' answer for a harness-intrinsic tool such as the
    /// `tool_search` bridge, on `surface`.
    ///
    /// Intrinsics are not registrations, so a definition's exact allowlist
    /// (which names registered tools) does not apply; the rules do, so a
    /// host can withhold discovery itself.
    pub(crate) fn admits_intrinsic(&self, name: &str, surface: Surface) -> CallGate {
        let Some(rules) = &self.rules else {
            return CallGate::Admit(ApprovalDirective::Default);
        };
        let decision = rules.evaluate(&ToolSubject::named(name), &self.context, surface, None);
        if decision.admits(surface) {
            CallGate::Admit(decision.approval)
        } else {
            tracing::debug!(
                target: "tinyagents::tool_rules",
                tool = %name,
                surface = ?surface,
                rule = ?decision.blocked_by,
                "[tool_rules] intrinsic tool withheld"
            );
            CallGate::Refuse(decision.refusal(name))
        }
    }

    /// Decides a concrete call. The allowlist is checked by the caller before
    /// lookup; this applies the rules, including any indirect target.
    pub(crate) fn admit_call(&self, tool: &dyn Tool, args: &Value) -> CallGate {
        let Some(rules) = &self.rules else {
            return CallGate::Admit(ApprovalDirective::Default);
        };
        let decision = rules.evaluate_call(tool, &self.context, args);
        if decision.callable {
            CallGate::Admit(decision.approval)
        } else {
            tracing::debug!(
                target: "tinyagents::tool_rules",
                tool = %tool.name(),
                rule = ?decision.blocked_by,
                "[tool_rules] call refused"
            );
            CallGate::Refuse(decision.refusal(tool.name()))
        }
    }
}
