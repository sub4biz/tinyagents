//! The graph loop driver announces the same turn/message lifecycle events as
//! the harness's direct loop (`TurnStarted`, `TurnCompleted`,
//! `MessageAppended`), once each, and never for nested tool calls.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use tinyagents_graph::agent_loop::{AgentLoopGraphExt, GraphLoopDriver};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::events::AgentEvent;
use tinyagents_harness::limits::RunLimits;
use tinyagents_harness::runtime::{AgentHarness, LoopExecution, RunPolicy};
use tinyagents_harness::steering::{SteeringCommand, SteeringHandle};
use tinyagents_harness::testkit::{EventRecorder, FakeTool};
use tinyagents_harness::tool::ToolExecutionContext;
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolResult};

fn tool_call_response(id: &str, name: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, json!({}))],
            usage: None,
            origin: None,
        },
        finish_reason: Some("tool_calls".to_string()),
        ..ModelResponse::assistant("")
    }
}

fn harness_for(
    execution: LoopExecution,
    responses: Vec<ModelResponse>,
    limits: RunLimits,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(MockModel::with_responses(responses)))
        .set_default_model("mock");
    if matches!(execution, LoopExecution::Graph) {
        harness.with_loop_driver(Arc::new(GraphLoopDriver::new()));
    }
    harness.with_policy(RunPolicy {
        execution,
        limits,
        ..RunPolicy::default()
    });
    harness
}

/// The lifecycle events of a run, flattened to comparable strings.
fn lifecycle(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnStarted { turn } => Some(format!("turn.started:{turn}")),
            AgentEvent::TurnCompleted {
                turn,
                tool_call_ids,
                ..
            } => Some(format!(
                "turn.completed:{turn}:{}",
                tool_call_ids
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )),
            AgentEvent::MessageAppended {
                role,
                index,
                call_id,
                ..
            } => Some(format!(
                "append:{index}:{role}:{}",
                call_id.as_ref().map(ToString::to_string).unwrap_or_default()
            )),
            _ => None,
        })
        .collect()
}

async fn run_with(
    execution: LoopExecution,
    responses: Vec<ModelResponse>,
    input: Vec<Message>,
    steering: Option<SteeringHandle>,
) -> Vec<String> {
    let mut harness = harness_for(execution, responses, RunLimits::default());
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")));
    let recorder = EventRecorder::new();
    let mut ctx = RunContext::new(RunConfig::new("lifecycle"), ()).with_events(recorder.sink());
    if let Some(steering) = steering {
        ctx = ctx.with_steering(steering);
    }
    harness
        .invoke_in_context(&(), ctx, input)
        .await
        .expect("run completes");
    lifecycle(&recorder.events())
}

#[tokio::test]
async fn a_plain_run_announces_the_same_lifecycle_as_the_direct_loop() {
    let script = || vec![ModelResponse::assistant("done")];
    let input = || vec![Message::user("hi")];
    let direct = run_with(LoopExecution::Direct, script(), input(), None).await;
    let graph = run_with(LoopExecution::Graph, script(), input(), None).await;
    assert!(!direct.is_empty(), "the direct loop emits lifecycle events");
    assert_eq!(graph, direct);
}

#[tokio::test]
async fn a_tool_run_announces_each_message_once_and_closes_each_turn() {
    let script = || {
        vec![
            tool_call_response("call-1", "lookup"),
            ModelResponse::assistant("done"),
        ]
    };
    let input = || vec![Message::user("look it up")];
    let direct = run_with(LoopExecution::Direct, script(), input(), None).await;
    let graph = run_with(LoopExecution::Graph, script(), input(), None).await;
    assert_eq!(graph, direct);

    // Input is never announced; every later index appears exactly once.
    let mut indices: Vec<&String> = graph.iter().filter(|e| e.starts_with("append:")).collect();
    let total = indices.len();
    indices.dedup();
    assert_eq!(indices.len(), total, "no duplicate MessageAppended: {graph:?}");
    assert!(!graph.iter().any(|e| e.starts_with("append:0:")), "{graph:?}");
    assert!(graph.iter().any(|e| e == "turn.completed:1:call-1"), "{graph:?}");
}

#[tokio::test]
async fn a_steering_injected_message_is_announced_like_the_direct_loop() {
    let run = |execution| async move {
        let steering = SteeringHandle::allow_all();
        steering.send(SteeringCommand::InjectMessage(Message::user("extra")));
        run_with(
            execution,
            vec![ModelResponse::assistant("done")],
            vec![Message::user("hello")],
            Some(steering),
        )
        .await
    };
    assert_eq!(run(LoopExecution::Graph).await, run(LoopExecution::Direct).await);
}

/// A tool that calls another tool through the harness.
struct Caller;

#[async_trait]
impl Tool for Caller {
    fn name(&self) -> &str {
        "caller"
    }
    fn description(&self) -> &str {
        "calls another tool"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: Value) -> anyhow::Result<ToolResult> {
        unreachable!("dispatched through execute_with_context")
    }
    async fn execute_with_context(
        &self,
        _arguments: Value,
        _options: tinytools::ToolCallOptions,
        context: Option<&dyn tinytools::ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let harness = context
            .and_then(tinytools::ToolRunContext::host_extension)
            .and_then(|any| any.downcast_ref::<ToolExecutionContext>())
            .expect("the harness installs its context")
            .clone();
        harness.call_tool("lookup", json!({})).await?;
        Ok(ToolResult::success("caller-out"))
    }
}

#[tokio::test]
async fn nested_tool_calls_do_not_emit_message_appended_on_the_graph_driver() {
    let mut rows = Vec::new();
    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let mut harness = harness_for(
            execution,
            vec![
                tool_call_response("p1", "caller"),
                ModelResponse::assistant("done"),
            ],
            RunLimits::default().with_max_nested_depth(3),
        );
        harness.register_tool(Arc::new(FakeTool::returning("lookup", "leaf-out")));
        harness.register_tool(Arc::new(Caller));
        let recorder = EventRecorder::new();
        let ctx = RunContext::new(RunConfig::new("nested"), ()).with_events(recorder.sink());
        harness
            .invoke_in_context(&(), ctx, vec![Message::user("go")])
            .await
            .expect("run completes");
        let tool_rows: Vec<String> = lifecycle(&recorder.events())
            .into_iter()
            .filter(|e| e.contains(":tool:"))
            .collect();
        assert_eq!(tool_rows.len(), 1, "{execution:?}: {tool_rows:?}");
        assert!(tool_rows[0].ends_with(":tool:p1"), "{tool_rows:?}");
        rows.push(tool_rows);
    }
    assert_eq!(rows[0], rows[1]);
}

#[tokio::test]
async fn iter_stepping_announces_appends_once_and_never_the_input() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("c1", "lookup"),
                ModelResponse::assistant("done"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "out")));
    let harness = Arc::new(harness);
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("iter-lifecycle"), ()).with_events(recorder.sink());
    let mut iter = harness
        .iter(Arc::new(()), ctx, vec![Message::user("go")])
        .expect("iter starts");
    iter.run_to_end().await.expect("finishes");

    let events = lifecycle(&recorder.events());
    assert!(!events.iter().any(|e| e.starts_with("append:0:")), "{events:?}");
    let appended: Vec<&String> = events.iter().filter(|e| e.starts_with("append:")).collect();
    assert_eq!(appended.len(), 3, "assistant, tool, assistant: {events:?}");
    assert_eq!(
        events.iter().filter(|e| e.starts_with("turn.started")).count(),
        events.iter().filter(|e| e.starts_with("turn.completed")).count(),
        "{events:?}"
    );
}
