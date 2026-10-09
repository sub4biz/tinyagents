//! Shared test fixtures plus tests for the [`ToolSet`] trait itself and the
//! [`crate::tool::ToolRegistry`] blanket implementation.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use super::ToolSet;
use crate::context::{RunConfig, RunContext};
use crate::tool::ToolRegistry;

/// A minimal, deterministic [`tinytools::Tool`] for toolset adaptor tests:
/// echoes back its `text` argument, and reports its own declared name so
/// tests can assert on exactly what an adaptor exposed or renamed.
pub(crate) struct EchoTool {
    name: String,
}

impl EchoTool {
    pub(crate) fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl tinytools::Tool for EchoTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Echoes the `text` argument back."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success(
            args["text"].as_str().unwrap_or_default().to_string(),
        ))
    }
}

pub(crate) fn ctx() -> RunContext<()> {
    RunContext::new(RunConfig::new("run-toolset"), ())
}

#[tokio::test]
async fn tool_registry_implements_tool_set() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new("echo")));

    let ctx = ctx();
    let names: Vec<_> = ToolSet::tools(&registry, &ctx)
        .await
        .expect("tools")
        .into_iter()
        .map(|tool| tool.name().to_string())
        .collect();
    assert_eq!(names, vec!["echo".to_string()]);

    let result = ToolSet::call(&registry, "echo", json!({"text": "hi"}), &ctx)
        .await
        .expect("echo call");
    assert!(!result.is_error);
}

#[tokio::test]
async fn tool_registry_as_tool_set_hides_hidden_tools() {
    struct Hidden;

    #[async_trait]
    impl tinytools::Tool for Hidden {
        fn name(&self) -> &str {
            "hidden_tool"
        }
        fn description(&self) -> &str {
            "never advertised"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({"type": "object"})
        }
        fn exposure(&self) -> tinytools::ToolExposure {
            tinytools::ToolExposure::Hidden
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<tinytools::ToolResult> {
            Ok(tinytools::ToolResult::success("ran"))
        }
    }

    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(Hidden));

    let ctx = ctx();
    let names = ToolSet::tools(&registry, &ctx).await.expect("tools");
    assert!(names.is_empty());

    let err = ToolSet::call(&registry, "hidden_tool", json!({}), &ctx)
        .await
        .expect_err("hidden tool is not model-callable through ToolSet either");
    assert!(matches!(
        err,
        crate::error::TinyAgentsError::ToolNotFound(_)
    ));
}

#[tokio::test]
async fn unknown_tool_call_reports_tool_not_found() {
    let registry: ToolRegistry<(), ()> = ToolRegistry::new();
    let ctx = ctx();
    let err = ToolSet::call(&registry, "nope", json!({}), &ctx)
        .await
        .expect_err("nope is not registered");
    assert!(matches!(err, crate::error::TinyAgentsError::ToolNotFound(name) if name == "nope"));
}

struct TaggedDispatcher;

use super::OverrideTool;
use serde_json::Value;
use tinytools::{Tool, ToolResult};

#[async_trait::async_trait]
impl Tool for TaggedDispatcher {
    fn name(&self) -> &str {
        "execute"
    }
    fn description(&self) -> &str {
        "dispatches to a connector action"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({ "type": "object" })
    }
    fn family(&self) -> Option<&str> {
        Some("connector")
    }
    fn tags(&self) -> Vec<String> {
        vec!["composio.scope:write".into()]
    }
    fn indirect_target(&self, args: &Value) -> Option<tinytools::IndirectCall> {
        args["action"]
            .as_str()
            .map(|name| tinytools::ToolSubject::named(name).into())
    }
    async fn execute(&self, _args: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

/// A renamed or prefixed tool is the same tool to the host's tool rules.
#[test]
fn an_override_keeps_what_tool_rules_read() {
    let tool = OverrideTool::new(Arc::new(TaggedDispatcher)).with_name("connector_execute");
    assert_eq!(tool.name(), "connector_execute");
    assert_eq!(tool.family(), Some("connector"));
    assert_eq!(tool.tags(), ["composio.scope:write"]);
    assert_eq!(
        tool.indirect_target(&serde_json::json!({ "action": "GMAIL_DELETE_EMAIL" })),
        Some(tinytools::ToolSubject::named("GMAIL_DELETE_EMAIL").into())
    );
}
