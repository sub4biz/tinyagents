//! Tests for the harness [`ToolRegistry`] lookup surface: registration, name
//! listing, schema collection, and the per-tool policy snapshot.
//!
//! The tool vocabulary itself (policy builders, display helpers, results,
//! schema formats) is owned and tested by `tinytools`; `canonical_tests.rs`
//! covers dispatch, duplicates and exposure.

use super::*;
use serde_json::json;

struct EchoTool;

#[async_trait]
impl tinytools::Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "echoes its input"
    }

    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"text": {"type": "string"}}})
    }

    async fn execute(&self, arguments: Value) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success(
            arguments["text"].as_str().unwrap_or_default(),
        ))
    }
}

struct ReadOnlyTool;

#[async_trait]
impl tinytools::Tool for ReadOnlyTool {
    fn name(&self) -> &str {
        "mcp_file-read"
    }

    fn description(&self) -> &str {
        "reads files"
    }

    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"path": {"type": "string"}}})
    }

    fn policy(&self) -> tinytools::ToolPolicy {
        tinytools::ToolPolicy::read_only()
    }

    async fn execute(&self, _arguments: Value) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success("ok"))
    }
}

#[test]
fn registry_register_get_names_schemas() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool));
    assert!(registry.get("echo").is_some());
    assert!(registry.get("missing").is_none());
    assert_eq!(registry.names(), vec!["echo".to_string()]);
    assert_eq!(registry.schemas().len(), 1);
    assert_eq!(registry.schemas()[0].name, "echo");
}

#[test]
fn registry_exposes_policy_snapshot() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool));
    registry.register(Arc::new(ReadOnlyTool));
    let policies = registry.policies();
    assert_eq!(policies.len(), 2);
    assert!(!policies["echo"].classified);
    assert!(policies["mcp_file-read"].classified);
    assert!(policies["mcp_file-read"].side_effects.read_only);
}
