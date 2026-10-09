//! End-to-end coverage for graph sub-agent nodes through the public registry,
//! harness, and graph execution surfaces.
// The deprecated `SubAgentNode` keeps its behaviour and its coverage.
#![allow(deprecated)]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents_graph::*;
use tinyinference_llm::usage::Usage;

/// Host-owned test adapter for the graph's explicit agent-invocation boundary.
/// The graph receives it per run, so it cannot retain a fabricated harness
/// context or accidentally share run lineage across executions.
#[derive(Clone, Default)]
struct TestAgentInvoker {
    outputs: Arc<HashMap<String, SubAgentOutput>>,
}

impl TestAgentInvoker {
    fn with_output(name: &str, output: SubAgentOutput) -> Arc<Self> {
        Arc::new(Self {
            outputs: Arc::new(HashMap::from([(name.to_owned(), output)])),
        })
    }
}

#[async_trait]
impl AgentInvoker for TestAgentInvoker {
    async fn invoke(&self, request: AgentInvocation) -> tinyagents_graph::Result<SubAgentOutput> {
        self.outputs.get(&request.agent_id).cloned().ok_or_else(|| {
            TinyAgentsError::Capability(format!("unknown test agent `{}`", request.agent_id))
        })
    }
}

fn binding(invoker: Arc<dyn AgentInvoker>) -> AgentInvocationBinding {
    AgentInvocationBinding::new(
        invoker,
        tinyagents_harness::events::EventSink::new(),
        tinyagents_harness::cancel::CancellationToken::new(),
    )
}

fn graph_delegating_to(
    node: SubAgentNode<String, String>,
) -> tinyagents_graph::CompiledGraph<String, String> {
    GraphBuilder::<String, String>::overwrite()
        .add_node("delegate", subagent_node(node))
        .set_entry("delegate")
        .set_finish("delegate")
        .compile()
        .expect("graph compiles")
}

#[tokio::test]
async fn subagent_node_delegates_records_child_run_and_forwards_events() {
    let mut usage = tinyinference_llm::usage::UsageTotals::new();
    usage.record(Usage::new(7, 3));
    let invoker = TestAgentInvoker::with_output(
        "researcher",
        SubAgentOutput {
            text: "answer: 42".to_owned(),
            usage,
            model_calls: 1,
            ..SubAgentOutput::default()
        },
    );
    let node = SubAgentNode::<String, String>::from_fns(
        "researcher",
        |state: &String| SubAgentInput::prompt(format!("question: {state}")),
        |out: SubAgentOutput| {
            assert_eq!(out.text, "answer: 42");
            assert!(out.model_calls >= 1);
            assert_eq!(out.tool_calls, 0);
            out.text
        },
    );
    let graph = graph_delegating_to(node);

    let run = graph
        .run_with_agent_binding("life?".to_string(), binding(invoker))
        .await
        .expect("graph run");
    assert_eq!(run.state, "answer: 42");
    assert_eq!(run.child_runs.len(), 1);
    let child = &run.child_runs[0];
    assert_eq!(child.node.as_str(), "delegate");
    assert_eq!(child.graph_id.as_str(), "agent:researcher");
    assert_eq!(child.root_run_id, run.root_run_id);
    assert_ne!(child.run_id, run.run_id);
    assert!(child.usage.usage.effective_total() > 0);
    assert_eq!(run.run_tree().children.len(), 1);
}

#[tokio::test]
async fn subagent_node_errors_for_missing_agent_and_budget_excess() {
    let missing_node = SubAgentNode::<String, String>::from_fns(
        "missing",
        |state: &String| SubAgentInput::prompt(state.clone()).with_data(json!({ "source": "e2e" })),
        |out: SubAgentOutput| out.text,
    );
    let missing_graph = graph_delegating_to(missing_node);
    let err = missing_graph.run("go".to_string()).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Capability(_)), "got {err:?}");

    let invoker = TestAgentInvoker::with_output(
        "twostep",
        SubAgentOutput {
            text: "final".to_owned(),
            model_calls: 2,
            tool_calls: 1,
            ..SubAgentOutput::default()
        },
    );

    let policy = SubAgentPolicy::default().with_budget(SubAgentBudget {
        max_model_calls: Some(1),
        ..SubAgentBudget::unlimited()
    });
    let node = SubAgentNode::<String, String>::from_fns(
        "twostep",
        |state: &String| SubAgentInput {
            prompt: state.clone(),
            data: Some(json!({ "kind": "budget-check" })),
        },
        |out: SubAgentOutput| out.text,
    )
    .with_policy(policy);
    let graph = graph_delegating_to(node);

    let err = graph
        .run_with_agent_binding("go".to_string(), binding(invoker))
        .await
        .unwrap_err();
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );

    assert!(
        SubAgentBudget::unlimited()
            .check(
                &SubAgentOutput {
                    text: "ok".to_string(),
                    structured: None,
                    usage: Default::default(),
                    model_calls: 0,
                    tool_calls: 0,
                },
                "agent"
            )
            .is_ok()
    );
}

#[tokio::test]
async fn graph_uses_the_execution_scoped_agent_invoker() {
    let invoker = TestAgentInvoker::with_output(
        "adapter",
        SubAgentOutput {
            text: "adapter answer".to_owned(),
            model_calls: 1,
            ..SubAgentOutput::default()
        },
    );
    let node = SubAgentNode::<String, String>::from_fns(
        "adapter",
        |state: &String| SubAgentInput::prompt(state.clone()),
        |out: SubAgentOutput| out.text,
    );
    let run = graph_delegating_to(node)
        .run_with_agent_binding("delegated prompt".to_owned(), binding(invoker))
        .await
        .expect("adapter run");
    assert_eq!(run.state, "adapter answer");
}
