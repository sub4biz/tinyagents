use std::sync::Arc;

use async_trait::async_trait;

use super::{
    batch_is_canonical_parallel_safe, map_tool_dispatch_error, should_execute_tools_concurrently,
    tool_message_from_result,
};
use crate::error::TinyAgentsError;
use tinyinference_llm::message::ContentBlock;
use tinytools::{ToolCallOptions, ToolContent, ToolResult};

struct DeclaredParallelTool {
    parallel: bool,
    injected_risk: bool,
}

#[async_trait]
impl tinytools::Tool for DeclaredParallelTool {
    fn name(&self) -> &str {
        "parallel"
    }

    fn description(&self) -> &str {
        "test tool"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type":"object"})
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }

    fn is_concurrency_safe(&self, arguments: &serde_json::Value) -> bool {
        self.parallel && arguments["risk"].as_str().is_none_or(|risk| risk == "safe")
    }

    fn injected_arguments(&self) -> Vec<tinytools::ToolInjectedArgument> {
        if self.injected_risk {
            vec![tinytools::ToolInjectedArgument::host("risk")]
        } else {
            Vec::new()
        }
    }
}

fn markdown_result() -> ToolResult {
    ToolResult {
        content: vec![
            ToolContent::Text {
                text: "plain summary".to_string(),
            },
            ToolContent::Json {
                data: serde_json::json!({"ordered": 2}),
            },
        ],
        is_error: true,
        markdown_formatted: Some("## compact failure".to_string()),
        ..ToolResult::default()
    }
}

#[test]
fn serial_and_concurrent_folds_select_the_same_markdown_and_preserve_blocks() {
    let result = markdown_result();
    let options = ToolCallOptions::prefer_markdown();

    // Both execution modes converge through this fold helper.
    let serial = tool_message_from_result("serial-call".to_string(), &result, options);
    let concurrent = tool_message_from_result("concurrent-call".to_string(), &result, options);

    for message in [&serial, &concurrent] {
        assert_eq!(
            message.content,
            vec![ContentBlock::Text("## compact failure".to_string())]
        );
        assert_eq!(message.artifact.as_ref().unwrap()["is_error"], true);
        assert!(!message.trusted_verbatim);
        assert_eq!(
            message.artifact.as_ref().unwrap()["trusted_verbatim"],
            false,
            "a tool-controlled canonical result cannot opt out of host framing"
        );
        assert_eq!(
            message.artifact.as_ref().unwrap()["tinytools_content"],
            serde_json::json!([
                {"type":"text", "text":"plain summary"},
                {"type":"json", "data":{"ordered":2}}
            ])
        );
    }
}

#[test]
fn ordinary_result_keeps_ordered_blocks_when_markdown_is_not_preferred() {
    let message = tool_message_from_result(
        "call".to_string(),
        &markdown_result(),
        ToolCallOptions::default(),
    );
    assert_eq!(
        message.content,
        vec![
            ContentBlock::Text("plain summary".to_string()),
            ContentBlock::Json(serde_json::json!({"ordered": 2})),
        ]
    );
}

#[test]
fn dispatch_error_mapping_preserves_harness_classification_without_leaking_foreign_detail() {
    let cancelled = map_tool_dispatch_error(anyhow::Error::new(TinyAgentsError::Cancelled));
    assert!(matches!(cancelled, TinyAgentsError::Cancelled));

    let foreign = map_tool_dispatch_error(anyhow::anyhow!("outer: {}", "root cause"));
    assert!(matches!(foreign, TinyAgentsError::Tool(message) if message == "tool dispatch failed"));
}

#[test]
fn canonical_concurrency_declaration_gates_the_parallel_path() {
    let call = tinyinference_llm::tool::ToolCall::new("call", "parallel", serde_json::json!({}));

    let mut serial: crate::tool::ToolRegistry<(), ()> = crate::tool::ToolRegistry::new();
    serial.register(Arc::new(DeclaredParallelTool {
        parallel: false,
        injected_risk: false,
    }));
    assert!(!batch_is_canonical_parallel_safe(
        &serial,
        std::slice::from_ref(&call)
    ));

    let mut concurrent: crate::tool::ToolRegistry<(), ()> = crate::tool::ToolRegistry::new();
    concurrent.register(Arc::new(DeclaredParallelTool {
        parallel: true,
        injected_risk: false,
    }));
    assert!(batch_is_canonical_parallel_safe(&concurrent, &[call]));
}

#[test]
fn forged_safe_injected_value_cannot_select_parallel_execution() {
    let mut registry: crate::tool::ToolRegistry<(), ()> = crate::tool::ToolRegistry::new();
    registry.register(Arc::new(DeclaredParallelTool {
        parallel: true,
        injected_risk: true,
    }));

    // A model can claim `safe`; admission will strip this and inject the
    // host's real (potentially unsafe) value. The raw value must therefore
    // never be considered a parallelization proof.
    let forged_safe = tinyinference_llm::tool::ToolCall::new(
        "call",
        "parallel",
        serde_json::json!({"risk": "safe"}),
    );
    let canonical = tinytools::ToolCall::new(
        tinytools::ToolCallId::new("call"),
        "parallel",
        forged_safe.arguments.clone(),
    );
    let mut host_values = tinytools::InjectedToolArguments::new();
    host_values.insert("risk", serde_json::json!("unsafe"));
    let authoritative = tinytools::prepare_tool_arguments(
        &canonical,
        &registry.get("parallel").unwrap().injected_arguments(),
        &host_values,
    )
    .unwrap();
    assert!(
        !registry
            .get("parallel")
            .unwrap()
            .is_concurrency_safe(&authoritative),
        "the host value is the one that makes this call unsafe"
    );
    assert!(!batch_is_canonical_parallel_safe(&registry, &[forged_safe]));
}

#[test]
fn lifecycle_middleware_no_longer_forces_the_serial_route() {
    // Regression test (I-8): lifecycle middleware used to force the
    // serial path unconditionally, on the theory that `before_tool` can
    // rewrite a call (`&mut ToolCall`) while execution is concurrently in
    // flight. That never actually applied: admission (including every
    // `before_tool` hook) is serial and completes in full, for every call
    // in the batch, before any concurrent future is built — so a
    // lifecycle middleware has nothing left to mutate once execution
    // starts. Only tool-*wrap* middleware (bypassed entirely by the
    // concurrent path) still forces serial execution.
    assert!(should_execute_tools_concurrently(2, true, true));
}

#[test]
fn concurrent_safe_tool_wrap_middleware_keeps_the_concurrent_route() {
    // The wrap onion runs inside each concurrent future (`wrap_tool` takes a
    // shared `&RunContext`), so a registered `ToolMiddleware` no longer forces
    // serial execution on its own.
    assert!(should_execute_tools_concurrently(2, true, true));
}

#[test]
fn a_wrap_that_is_not_concurrent_safe_forces_the_serial_route() {
    // `ToolMiddleware::concurrent_safe() == false` is the escape hatch.
    assert!(!should_execute_tools_concurrently(2, true, false));
}

#[test]
fn map_tool_dispatch_error_preserves_sub_agent_depth_and_limit_exceeded() {
    // M-3 regression: every non-cancel/timeout error used to collapse to
    // a generic `Tool("tool dispatch failed")`, which `is_retryable`
    // treats as unconditionally retryable. A `SubAgentDepth`/
    // `LimitExceeded` escaping a nested sub-agent tool call is
    // deterministic and will never succeed on retry, so it must keep its
    // own classification instead of masquerading as a retryable tool
    // error.
    let depth_err = anyhow::Error::from(TinyAgentsError::SubAgentDepth(4));
    assert!(matches!(
        map_tool_dispatch_error(depth_err),
        TinyAgentsError::SubAgentDepth(4)
    ));

    let limit_err = anyhow::Error::from(TinyAgentsError::LimitExceeded(
        "some sensitive detail".to_string(),
    ));
    match map_tool_dispatch_error(limit_err) {
        TinyAgentsError::LimitExceeded(message) => {
            assert!(
                !message.contains("sensitive"),
                "the original message must still be redacted: {message}"
            );
        }
        other => panic!("expected LimitExceeded, got {other:?}"),
    }
}

#[test]
fn map_tool_dispatch_error_still_redacts_a_genuine_tool_error() {
    // An ordinary tool-authored error (arbitrary text, possibly carrying
    // secrets or user data) must still be collapsed to a generic message,
    // unlike the structural errors above.
    let tool_err = anyhow::Error::from(TinyAgentsError::Model(
        "leaked api key sk-secret".to_string(),
    ));
    match map_tool_dispatch_error(tool_err) {
        TinyAgentsError::Tool(message) => {
            assert!(!message.contains("sk-secret"));
        }
        other => panic!("expected Tool, got {other:?}"),
    }
}
