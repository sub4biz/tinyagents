//! Unit tests for the run's tool gate: how the allowlist, the harness policy
//! and a definition's rules combine.

use super::*;

use std::collections::HashSet;

use serde_json::json;
use tinytools::{ApprovalDirective, RuleContext, Surface, ToolRules};

fn deny(pattern: &str) -> ToolRules {
    ToolRules::from_allow_deny(Vec::<String>::new(), [pattern])
}

#[test]
fn a_default_gate_admits_everything() {
    let gate = ToolGate::default();
    assert!(gate.allows_name("x"));
    assert!(gate.lists("x", None, Surface::Catalog));
    assert!(ToolRulePolicy::default().is_permissive());
}

#[test]
fn the_allowlist_still_applies_first() {
    let allowed: HashSet<String> = ["a".to_string()].into();
    let gate = ToolGate::new(Some(allowed), &ToolRulePolicy::default(), None);
    assert!(gate.lists("a", None, Surface::Search));
    assert!(!gate.lists("b", None, Surface::Search));
    assert!(!gate.allows_name("b"));
}

#[test]
fn policy_and_definition_rules_both_apply() {
    let policy = ToolRulePolicy::new(deny("x_*"));
    let definition = deny("y_*");
    let gate = ToolGate::new(None, &policy, Some(&definition));
    assert!(!gate.lists("x_1", None, Surface::Catalog));
    assert!(!gate.lists("y_1", None, Surface::Catalog));
    assert!(gate.lists("z_1", None, Surface::Catalog));
}

#[test]
fn a_permissive_definition_layer_is_ignored() {
    let policy = ToolRulePolicy::new(deny("x_*"));
    let gate = ToolGate::new(None, &policy, Some(&ToolRules::allow_all()));
    assert!(!gate.lists("x_1", None, Surface::Catalog));
    let empty = ToolGate::new(None, &ToolRulePolicy::default(), Some(&ToolRules::allow_all()));
    assert!(empty.rules.is_none());
}

#[test]
fn the_policy_context_reaches_when_conditions() {
    let rules: ToolRules = serde_json::from_value(json!({ "rules": [
        { "effect": "deny", "match": { "name": "shell" }, "when": { "channel": "web" } },
    ] }))
    .expect("rules");
    let web = ToolRulePolicy::new(rules.clone()).with_context(RuleContext::new().with("channel", "web"));
    let cli = ToolRulePolicy::new(rules).with_context(RuleContext::new().with("channel", "cli"));
    assert!(!ToolGate::new(None, &web, None).lists("shell", None, Surface::Catalog));
    assert!(ToolGate::new(None, &cli, None).lists("shell", None, Surface::Catalog));
}

struct Named(&'static str, tinytools::ToolExposure);

#[async_trait::async_trait]
impl Tool for Named {
    fn name(&self) -> &str {
        self.0
    }
    fn description(&self) -> &str {
        "named"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn exposure(&self) -> tinytools::ToolExposure {
        self.1
    }
    async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success("ok"))
    }
}

#[test]
fn lists_tool_picks_the_surface_from_exposure() {
    let rules: ToolRules = serde_json::from_value(json!({ "rules": [
        { "effect": "deny", "on": ["search"], "match": { "name": "*" } },
    ] }))
    .expect("rules");
    let gate = ToolGate::new(None, &ToolRulePolicy::new(rules), None);
    assert!(gate.lists_tool(&Named("direct", tinytools::ToolExposure::Direct)));
    assert!(!gate.lists_tool(&Named("deferred", tinytools::ToolExposure::Deferred)));
}

#[test]
fn admit_call_reports_approval_and_refusals() {
    let rules: ToolRules = serde_json::from_value(json!({ "rules": [
        { "effect": "require_approval", "match": { "name": "send" } },
        { "id": "nope", "effect": "deny", "match": { "name": "drop" } },
    ] }))
    .expect("rules");
    let gate = ToolGate::new(None, &ToolRulePolicy::new(rules), None);
    let direct = tinytools::ToolExposure::Direct;
    assert_eq!(
        gate.admit_call(&Named("send", direct), &json!({})),
        CallGate::Admit(ApprovalDirective::Required)
    );
    assert_eq!(
        gate.admit_call(&Named("read", direct), &json!({})),
        CallGate::Admit(ApprovalDirective::Default)
    );
    match gate.admit_call(&Named("drop", direct), &json!({})) {
        CallGate::Refuse(message) => assert!(message.contains("rule 'nope'"), "{message}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(
        ToolGate::default().admit_call(&Named("drop", direct), &json!({})),
        CallGate::Admit(ApprovalDirective::Default)
    );
}
