//! The staged repeat escalation inside the real agent loop: a blocked call
//! must never reach its tool, and the run must pause on the second block.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::error::Result as TaResult;
use crate::middleware::{Middleware, ToolInvocationIdentity};
use crate::runtime::AgentHarness;
use crate::steering::SteeringHandle;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolPolicy, ToolResult};

struct CountingTool {
    runs: Mutex<u32>,
}

#[async_trait]
impl Tool for CountingTool {
    fn name(&self) -> &str {
        "lookup"
    }
    fn description(&self) -> &str {
        "counting tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.runs.lock().unwrap() += 1;
        Ok(ToolResult::success("same answer"))
    }
}

fn repeat_call(turn: usize) -> ModelResponse {
    let mut response = ModelResponse::assistant(format!("attempt {turn}"));
    response.message = AssistantMessage {
        id: None,
        content: vec![ContentBlock::Text(format!("attempt {turn}"))],
        tool_calls: vec![ToolCall::new(
            format!("call-{turn}"),
            "lookup",
            json!({"id": 1}),
        )],
        usage: None,
        origin: None,
    };
    response
}

#[tokio::test]
async fn blocked_calls_never_execute_and_the_second_block_pauses_the_run() {
    let steering = SteeringHandle::allow_all();
    let summary: HaltSummarySlot = Arc::new(Mutex::new(None));
    let tool = Arc::new(CountingTool {
        runs: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(
            (0..12).map(repeat_call).collect::<Vec<_>>(),
        )),
    );
    harness.register_tool(tool.clone());
    harness.push_middleware(Arc::new(RepeatProgressMiddleware::new(
        steering.clone(),
        summary.clone(),
        Arc::new(|_| false),
    )));

    let ctx = RunContext::new(RunConfig::new("repeat-loop"), ()).with_steering(steering.clone());
    let run = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .expect("a halt pauses the run rather than failing it")
        .run;

    assert_eq!(
        *tool.runs.lock().unwrap(),
        4,
        "attempts 1-4 execute; 5 and 6 are answered without running"
    );
    let texts: Vec<String> = run
        .messages
        .iter()
        .filter(|m| matches!(m, Message::Tool(_)))
        .map(|m| m.text())
        .collect();
    assert!(texts[2].contains("[repeat notice]"), "{texts:?}");
    assert!(texts[4].contains("not executed"), "{texts:?}");
    assert!(
        summary.lock().unwrap().is_some(),
        "the halt names its cause"
    );
}

/// Registered after the guard, so its `after_tool` runs before the guard's.
struct MarkerObserver(Arc<Mutex<Vec<Option<String>>>>);

#[async_trait]
impl Middleware<(), ()> for MarkerObserver {
    fn name(&self) -> &str {
        "marker_observer"
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> TaResult<()> {
        self.0
            .lock()
            .unwrap()
            .push(repeat_guard_marker(result).map(str::to_string));
        Ok(())
    }
}

#[tokio::test]
async fn a_later_registered_after_tool_sees_the_marker_on_refused_calls() {
    let steering = SteeringHandle::allow_all();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(CountingTool {
        runs: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(
            (0..12).map(repeat_call).collect::<Vec<_>>(),
        )),
    );
    harness.register_tool(tool);
    harness.push_middleware(Arc::new(RepeatProgressMiddleware::new(
        steering.clone(),
        Arc::new(Mutex::new(None)),
        Arc::new(|_| false),
    )));
    harness.push_middleware(Arc::new(MarkerObserver(seen.clone())));

    let ctx = RunContext::new(RunConfig::new("marker"), ()).with_steering(steering);
    harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .expect("a halt pauses the run");

    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.iter().map(Option::as_deref).collect::<Vec<_>>(),
        [None, None, None, None, Some("blocked"), Some("halted")],
        "executed results are unmarked; refused ones are marked for every hook"
    );
}

#[tokio::test]
async fn a_guard_halt_is_reported_as_a_halted_terminal_outcome() {
    use crate::terminal::{TerminalClass, TerminalReason};
    let steering = SteeringHandle::allow_all();
    let summary: HaltSummarySlot = Arc::new(Mutex::new(None));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(
            (0..12).map(repeat_call).collect::<Vec<_>>(),
        )),
    );
    harness.register_tool(Arc::new(CountingTool {
        runs: Mutex::new(0),
    }));
    harness.push_middleware(Arc::new(RepeatProgressMiddleware::new(
        steering.clone(),
        summary.clone(),
        Arc::new(|_| false),
    )));
    let ctx = RunContext::new(RunConfig::new("repeat-halted"), ()).with_steering(steering);
    let run = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .unwrap()
        .run;

    assert!(run.paused.is_some(), "a halt still pauses the run");
    let outcome = run.terminal.expect("outcome");
    assert_eq!(outcome.reason, TerminalReason::Halted);
    assert_eq!(outcome.class, TerminalClass::Failure);
    assert_eq!(Some(outcome.message), summary.lock().unwrap().clone());
}
