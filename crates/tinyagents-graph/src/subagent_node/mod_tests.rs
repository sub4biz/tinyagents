//! Tests for host-driven graph-to-agent delegation.
// The deprecated `SubAgentNode` keeps its behaviour and its coverage.
#![allow(deprecated)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::*;
use crate::builder::GraphBuilder;
use tinyagents_harness::cancel::CancellationToken;
use tinyagents_harness::events::{AgentEvent, EventSink, RecordingListener};

#[derive(Clone, Default)]
struct RecordingInvoker {
    requests: Arc<Mutex<Vec<AgentInvocation>>>,
}

impl RecordingInvoker {
    fn requests(&self) -> Vec<AgentInvocation> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl AgentInvoker for RecordingInvoker {
    async fn invoke(&self, request: AgentInvocation) -> crate::Result<SubAgentOutput> {
        self.requests.lock().unwrap().push(request.clone());
        request.events.emit(AgentEvent::StateUpdate);
        Ok(SubAgentOutput {
            text: format!("done:{}", request.input.prompt),
            model_calls: 1,
            ..SubAgentOutput::default()
        })
    }
}

struct FailingInvoker;

#[async_trait]
impl AgentInvoker for FailingInvoker {
    async fn invoke(&self, _request: AgentInvocation) -> crate::Result<SubAgentOutput> {
        Err(crate::TinyAgentsError::Model(
            "temporary failure".to_string(),
        ))
    }
}

fn graph() -> crate::CompiledGraph<String, String> {
    GraphBuilder::<String, String>::overwrite()
        .add_node(
            "delegate",
            subagent_node(SubAgentNode::from_fns(
                "researcher",
                |state: &String| SubAgentInput::prompt(state.clone()),
                |output: SubAgentOutput| output.text,
            )),
        )
        .set_entry("delegate")
        .set_finish("delegate")
        .compile()
        .unwrap()
}

fn binding(invoker: Arc<dyn AgentInvoker>) -> AgentInvocationBinding {
    AgentInvocationBinding::new(invoker, EventSink::new(), CancellationToken::new())
}

#[tokio::test]
async fn delegation_uses_carried_invoker_and_preserves_graph_lineage() {
    let invoker = Arc::new(RecordingInvoker::default());
    let graph = graph();

    let run = graph
        .run_with_agent_binding("question".to_string(), binding(invoker.clone()))
        .await
        .unwrap();

    assert_eq!(run.state, "done:question");
    let requests = invoker.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.agent_id, "researcher");
    assert_eq!(request.input.prompt, "question");
    assert_eq!(request.parent_run_id, run.run_id);
    assert_eq!(request.root_run_id, run.root_run_id);
    assert_eq!(request.node_id.as_str(), "delegate");
    assert_eq!(request.graph_id, run.graph_id);
    assert!(request.cancellation.is_some());
}

#[tokio::test]
async fn concurrent_sibling_graph_runs_have_isolated_parent_identity() {
    let left_invoker = Arc::new(RecordingInvoker::default());
    let right_invoker = Arc::new(RecordingInvoker::default());
    let graph = graph();
    let left_cancellation = CancellationToken::new();
    left_cancellation.cancel();
    let right_cancellation = CancellationToken::new();

    let (left, right) = tokio::join!(
        graph.run_with_agent_binding(
            "left".to_string(),
            AgentInvocationBinding::new(left_invoker.clone(), EventSink::new(), left_cancellation,)
        ),
        graph.run_with_agent_binding(
            "right".to_string(),
            AgentInvocationBinding::new(
                right_invoker.clone(),
                EventSink::new(),
                right_cancellation,
            )
        )
    );
    let left = left.unwrap();
    let right = right.unwrap();
    let left_request = left_invoker.requests().pop().unwrap();
    let right_request = right_invoker.requests().pop().unwrap();
    assert_ne!(left.run_id, right.run_id);
    assert_eq!(left_request.input.prompt, "left");
    assert_eq!(left_request.parent_run_id, left.run_id);
    assert_eq!(left_request.root_run_id, left.root_run_id);
    assert!(left_request.cancellation.unwrap().is_cancelled());
    assert_eq!(right_request.input.prompt, "right");
    assert_eq!(right_request.parent_run_id, right.run_id);
    assert_eq!(right_request.root_run_id, right.root_run_id);
    assert!(!right_request.cancellation.unwrap().is_cancelled());
}

#[tokio::test]
async fn missing_host_invoker_is_an_explicit_capability_error() {
    let graph = GraphBuilder::<String, String>::overwrite()
        .add_node(
            "delegate",
            subagent_node(SubAgentNode::from_fns(
                "researcher",
                |state: &String| SubAgentInput::prompt(state.clone()),
                |output: SubAgentOutput| output.text,
            )),
        )
        .set_entry("delegate")
        .set_finish("delegate")
        .compile()
        .unwrap();

    let error = graph.run("question".to_string()).await.unwrap_err();
    assert!(matches!(error, crate::TinyAgentsError::Capability(_)));
}

#[tokio::test]
async fn resume_binding_reaches_a_later_subagent_with_host_signals() {
    // The initial invocation pauses before reaching the sub-agent, so the
    // continuation must supply a fresh execution-scoped binding rather than
    // relying on mutable state retained by the compiled graph.
    let checkpointer = Arc::new(crate::checkpoint::InMemoryCheckpointer::<String>::new());
    let graph = GraphBuilder::<String, String>::overwrite()
        .add_node(
            "gate",
            |state: String, ctx: crate::builder::NodeContext| async move {
                if ctx.resume.is_some() {
                    Ok(crate::command::NodeResult::Update(state))
                } else {
                    Ok(crate::command::NodeResult::Interrupt(
                        crate::command::Interrupt::new("gate", serde_json::json!({ "ask": "go?" })),
                    ))
                }
            },
        )
        .add_node(
            "delegate",
            subagent_node(SubAgentNode::from_fns(
                "researcher",
                |state: &String| SubAgentInput::prompt(state.clone()),
                |output: SubAgentOutput| output.text,
            )),
        )
        .set_entry("gate")
        .add_edge("gate", "delegate")
        .set_finish("delegate")
        .compile()
        .unwrap()
        .with_checkpointer(checkpointer);

    let paused = graph
        .run_with_thread("resume", "question".to_string())
        .await
        .unwrap();
    assert!(paused.is_interrupted());

    let invoker = Arc::new(RecordingInvoker::default());
    let events = EventSink::new();
    let listener = Arc::new(RecordingListener::new());
    events.subscribe(listener.clone());
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let resumed = graph
        .resume_with_agent_binding(
            "resume",
            crate::command::Command::resume(serde_json::json!("approved")),
            AgentInvocationBinding::new(invoker.clone(), events, cancellation),
        )
        .await
        .unwrap();

    assert_eq!(resumed.state, "done:question");
    let request = invoker.requests().pop().expect("resumed sub-agent ran");
    assert_eq!(request.parent_run_id, resumed.run_id);
    assert_eq!(request.root_run_id, resumed.root_run_id);
    assert!(
        request
            .cancellation
            .expect("binding has cancellation")
            .is_cancelled()
    );
    assert_eq!(
        listener.len(),
        1,
        "the resumed sub-agent used the supplied event sink"
    );
}

#[tokio::test]
async fn retry_binding_is_fresh_and_reaches_the_failed_subagent() {
    // A durable failure from one execution must not capture that execution's
    // host capability. Retrying with a different binding reaches the node and
    // succeeds through the new invoker.
    let checkpointer = Arc::new(crate::checkpoint::InMemoryCheckpointer::<String>::new());
    let graph = graph().with_checkpointer(checkpointer);

    let failed = graph
        .run_with_thread_agent_binding(
            "retry",
            "question".to_string(),
            binding(Arc::new(FailingInvoker)),
        )
        .await
        .unwrap_err();
    assert!(matches!(failed, crate::TinyAgentsError::Model(_)));

    let replacement = Arc::new(RecordingInvoker::default());
    let retried = graph
        .retry_with_agent_binding("retry", binding(replacement.clone()))
        .await
        .unwrap();

    assert_eq!(retried.state, "done:question");
    let request = replacement
        .requests()
        .pop()
        .expect("retry used replacement binding");
    assert_eq!(request.parent_run_id, retried.run_id);
    assert_eq!(request.root_run_id, retried.root_run_id);
}

fn usage_output(input: u64, output: u64) -> SubAgentOutput {
    let mut out = SubAgentOutput::default();
    out.usage.usage.input_tokens = input;
    out.usage.usage.output_tokens = output;
    out
}

#[test]
fn token_budget_is_enforced_after_the_child_returns() {
    let budget = SubAgentBudget::unlimited()
        .with_max_input_tokens(100)
        .with_max_output_tokens(50);
    assert!(budget.check(&usage_output(100, 50), "a").is_ok());
    let err = budget.check(&usage_output(101, 0), "a").unwrap_err();
    assert!(matches!(err, TinyAgentsError::LimitExceeded(m) if m.contains("input-token")));
    let err = budget.check(&usage_output(0, 51), "a").unwrap_err();
    assert!(matches!(err, TinyAgentsError::LimitExceeded(m) if m.contains("output-token")));
}

#[test]
fn budget_maps_onto_the_harness_budget_limits_including_cost() {
    let limits = SubAgentBudget::unlimited()
        .with_max_input_tokens(7)
        .with_max_output_tokens(8)
        .with_max_cost(1.5)
        .to_budget_limits();
    assert_eq!(limits.max_input_tokens, Some(7));
    assert_eq!(limits.max_output_tokens, Some(8));
    assert_eq!(limits.max_cost, Some(1.5));
}

#[test]
fn call_caps_only_ever_tighten_a_run_config() {
    let budget = SubAgentBudget {
        max_model_calls: Some(3),
        max_tool_calls: Some(9),
        ..SubAgentBudget::unlimited()
    };
    let mut loose = tinyagents_harness::context::RunConfig::new("r");
    budget.apply_call_caps(&mut loose);
    assert_eq!(loose.max_model_calls, Some(3));
    assert_eq!(loose.max_tool_calls, Some(9));
    let mut tight = tinyagents_harness::context::RunConfig::new("r")
        .with_max_model_calls(2)
        .with_max_tool_calls(20);
    budget.apply_call_caps(&mut tight);
    assert_eq!(tight.max_model_calls, Some(2), "a tighter cap is kept");
    assert_eq!(tight.max_tool_calls, Some(9));
}

#[test]
fn policy_defaults_to_no_retry_after_tool_calls() {
    assert!(!SubAgentPolicy::default().retry_after_tool_calls);
    assert!(
        SubAgentPolicy::default()
            .with_retry_after_tool_calls(true)
            .retry_after_tool_calls
    );
}
