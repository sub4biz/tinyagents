//! Equivalence coverage for A5: the compiled-graph rendition of the agent
//! loop (`tinyagents_graph::agent_loop`) against the harness's built-in
//! direct loop.
//!
//! Each scenario below runs the identical scripted model/harness setup
//! twice — once with `RunPolicy::execution = LoopExecution::Direct` (the
//! default), once with `LoopExecution::Graph` plus
//! `AgentHarness::with_loop_driver(Arc::new(GraphLoopDriver::new()))` — and
//! asserts the two runs agree on transcript, structured output, and usage,
//! and that the direct run's `AgentEvent` kind sequence appears (in order,
//! extra graph events allowed) within the graph run's sequence. See
//! `docs/modules/harness/state-graph.md` for the documented scope of the
//! graph rendition (it is a subset of the direct loop's behavior, not a
//! byte-for-byte reimplementation).

use std::sync::Arc;

use serde_json::json;

use tinyagents_graph::agent_loop::{
    AgentLoopGraphExt, GraphLoopDriver, LoopRuntime, LoopState, compile_loop, node,
};
use tinyagents_graph::{FileCheckpointer, InMemoryCheckpointer};
use tinyagents_harness::TinyAgentsError;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::runtime::{AgentHarness, LoopExecution, RunPolicy};
use tinyagents_harness::steering::{SteeringCommand, SteeringHandle};
use tinyagents_harness::testkit::{EventRecorder, FakeTool};
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::model::{ModelResponse, ResponseFormat};
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(7, 3)),
            origin: None,
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        ..ModelResponse::assistant("")
    }
}

/// Builds a harness for `execution`, registering `model` as the default.
fn harness_for(execution: LoopExecution, model: Arc<MockModel>) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model)
        .set_default_model("mock");
    if matches!(execution, LoopExecution::Graph) {
        harness.with_loop_driver(Arc::new(GraphLoopDriver::new()));
    }
    harness.with_policy(RunPolicy {
        execution,
        ..RunPolicy::default()
    });
    harness
}

/// Asserts `expected` appears, in order, as a (not necessarily contiguous)
/// subsequence of `actual` — the "same kind sequence, extra graph events
/// allowed" contract.
///
/// The turn/message lifecycle events (`turn.*`, `message.appended`) are emitted
/// by the direct loop only for now; the graph driver does not announce them, so
/// they are excluded from the expected sequence.
fn assert_kinds_subsequence(expected: &[String], actual: &[String]) {
    let mut cursor = 0;
    let lifecycle =
        |kind: &&String| !(kind.starts_with("turn.") || kind.as_str() == "message.appended");
    for kind in expected.iter().filter(lifecycle) {
        let Some(offset) = actual[cursor..].iter().position(|k| k == kind) else {
            panic!(
                "expected event kind `{kind}` not found (in order) in graph run's kinds: \
                 {actual:?}; direct run's kinds were: {expected:?}"
            );
        };
        cursor += offset + 1;
    }
}

// ── Scenario 1: tool call ───────────────────────────────────────────────────

#[tokio::test]
async fn tool_call_scenario_matches_direct_and_graph() {
    let mut direct_run = None;
    let mut direct_kinds = None;
    let mut graph_run = None;
    let mut graph_kinds = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "lookup", json!({ "q": "x" })),
            ModelResponse::assistant("done"),
        ]));
        let mut harness = harness_for(execution, model);
        harness.register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")));

        let recorder = EventRecorder::new();
        let ctx = RunContext::new(RunConfig::new("tool-call"), ()).with_events(recorder.sink());
        let run = harness
            .invoke_in_context(&(), ctx, vec![Message::user("look something up")])
            .await
            .expect("run completes");

        match execution {
            LoopExecution::Direct => {
                direct_run = Some(run);
                direct_kinds = Some(recorder.kinds());
            }
            LoopExecution::Graph => {
                graph_run = Some(run);
                graph_kinds = Some(recorder.kinds());
            }
        }
    }

    let (direct_run, graph_run) = (direct_run.unwrap(), graph_run.unwrap());
    assert_eq!(direct_run.model_calls, graph_run.model_calls);
    assert_eq!(direct_run.tool_calls, graph_run.tool_calls);
    assert_eq!(direct_run.executed_tools, graph_run.executed_tools);
    assert_eq!(direct_run.text(), graph_run.text());
    assert_eq!(direct_run.usage, graph_run.usage);
    assert_kinds_subsequence(&direct_kinds.unwrap(), &graph_kinds.unwrap());
}

// ── Scenario 2: structured output ───────────────────────────────────────────

#[tokio::test]
async fn structured_output_scenario_matches_direct_and_graph() {
    let schema = json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
    });

    let mut direct_run = None;
    let mut graph_run = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
            r#"{"answer":"42"}"#,
        )]));
        let mut harness = harness_for(execution, model);
        let mut policy = harness.policy().clone();
        policy.default_response_format = Some(ResponseFormat::auto("answer", schema.clone()));
        harness.with_policy(policy);

        let run = harness
            .invoke_default(&(), vec![Message::user("what is the answer")])
            .await
            .expect("run completes");

        match execution {
            LoopExecution::Direct => direct_run = Some(run),
            LoopExecution::Graph => graph_run = Some(run),
        }
    }

    let (direct_run, graph_run) = (direct_run.unwrap(), graph_run.unwrap());
    assert_eq!(direct_run.structured, graph_run.structured);
    assert_eq!(direct_run.structured, Some(json!({ "answer": "42" })));
}

// ── Scenario 3: limit stop ──────────────────────────────────────────────────

#[tokio::test]
async fn limit_stop_scenario_matches_direct_and_graph() {
    let mut direct_err = None;
    let mut graph_err = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        // Always requests a tool call, so the loop never finishes on its own
        // and must hit the model-call cap.
        let model = Arc::new(MockModel::with_tool_call("lookup", json!({})));
        let mut harness = harness_for(execution, model);
        harness.register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")));

        let ctx = RunContext::new(RunConfig::new("limit-stop").with_max_model_calls(2), ());
        let err = harness
            .invoke_in_context(&(), ctx, vec![Message::user("loop forever")])
            .await
            .expect_err("run hits the model-call cap");

        match execution {
            LoopExecution::Direct => direct_err = Some(err),
            LoopExecution::Graph => graph_err = Some(err),
        }
    }

    let direct_err = direct_err.unwrap();
    let graph_err = graph_err.unwrap();
    assert!(
        matches!(direct_err, TinyAgentsError::LimitExceeded(_)),
        "direct: {direct_err:?}"
    );
    assert!(
        matches!(graph_err, TinyAgentsError::LimitExceeded(_)),
        "graph: {graph_err:?}"
    );
}

// ── Scenario 4: approval interrupt ──────────────────────────────────────────

mod interrupt_middleware {
    use async_trait::async_trait;
    use tinyagents_harness::context::{MiddlewareControl, RunContext};
    use tinyagents_harness::middleware::Middleware;
    use tinyinference_llm::model::ModelResponse;

    pub struct RequireApproval;

    #[async_trait]
    impl Middleware<(), ()> for RequireApproval {
        fn name(&self) -> &str {
            "require_approval"
        }

        async fn after_model(
            &self,
            ctx: &mut RunContext<()>,
            _state: &(),
            _response: &mut ModelResponse,
        ) -> tinyagents_harness::Result<()> {
            ctx.request_control(MiddlewareControl::Interrupt {
                node: "review".into(),
                message: "needs approval".into(),
            });
            Ok(())
        }
    }
}

#[tokio::test]
async fn approval_interrupt_scenario_matches_direct_and_graph() {
    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::constant("hi"));
        let mut harness = harness_for(execution, model);
        harness.push_middleware(Arc::new(interrupt_middleware::RequireApproval));

        let err = harness
            .invoke_default(&(), vec![Message::user("do the risky thing")])
            .await
            .expect_err("both engines surface MiddlewareControl::Interrupt as an error");

        match err {
            TinyAgentsError::Interrupted { node, message } => {
                assert_eq!(node, "review");
                assert_eq!(message, "needs approval");
            }
            other => panic!("expected Interrupted, got {other:?}"),
        }
    }
}

// ── Scenario 5: steering inject ─────────────────────────────────────────────

#[tokio::test]
async fn steering_inject_scenario_matches_direct_and_graph() {
    let mut direct_messages = None;
    let mut graph_messages = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
            "done",
        )]));
        let harness = harness_for(execution, model.clone());

        let steering = SteeringHandle::allow_all();
        steering.send(SteeringCommand::InjectMessage(Message::user(
            "ORCHESTRATOR: answer in French.",
        )));
        let ctx = RunContext::new(RunConfig::new("steer-inject"), ()).with_steering(steering);

        let run = harness
            .invoke_in_context(&(), ctx, vec![Message::user("hello")])
            .await
            .expect("run completes");
        assert_eq!(run.model_calls, 1);

        let injected = run
            .messages
            .iter()
            .any(|message| message.text().contains("ORCHESTRATOR"));
        assert!(
            injected,
            "the injected steering message must reach the transcript"
        );

        match execution {
            LoopExecution::Direct => direct_messages = Some(run.messages),
            LoopExecution::Graph => graph_messages = Some(run.messages),
        }
    }

    assert_eq!(direct_messages, graph_messages);
}

// ── Scenario 6: output retry ────────────────────────────────────────────────

#[tokio::test]
async fn output_retry_scenario_matches_direct_and_graph() {
    let schema = json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
    });

    let mut direct_run = None;
    let mut graph_run = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        // First reply is not valid JSON for the schema; second reply repairs it.
        let model = Arc::new(MockModel::with_responses(vec![
            ModelResponse::assistant("not json"),
            ModelResponse::assistant(r#"{"answer":"fixed"}"#),
        ]));
        let mut harness = harness_for(execution, model);
        let mut policy = harness.policy().clone();
        policy.default_response_format = Some(ResponseFormat::auto("answer", schema.clone()));
        harness.with_policy(policy);

        let run = harness
            .invoke_default(&(), vec![Message::user("what is the answer")])
            .await
            .expect("run completes after one repair turn");

        match execution {
            LoopExecution::Direct => direct_run = Some(run),
            LoopExecution::Graph => graph_run = Some(run),
        }
    }

    let (direct_run, graph_run) = (direct_run.unwrap(), graph_run.unwrap());
    assert_eq!(direct_run.structured, graph_run.structured);
    assert_eq!(direct_run.structured, Some(json!({ "answer": "fixed" })));
    assert_eq!(direct_run.model_calls, graph_run.model_calls);
    assert_eq!(direct_run.model_calls, 2);
}

// ── Checkpoint + resume across an interrupt ─────────────────────────────────

mod pause_middleware {
    use async_trait::async_trait;
    use tinyagents_harness::context::{MiddlewareControl, RunContext};
    use tinyagents_harness::middleware::Middleware;
    use tinyinference_llm::model::ModelResponse;

    /// Requests a graph-level interrupt (an approval gate) on the first
    /// `after_model` hook only.
    pub struct PauseOnce(pub std::sync::atomic::AtomicBool);

    impl PauseOnce {
        pub fn new() -> Self {
            Self(std::sync::atomic::AtomicBool::new(true))
        }
    }

    #[async_trait]
    impl Middleware<(), ()> for PauseOnce {
        fn name(&self) -> &str {
            "pause_once"
        }

        async fn after_model(
            &self,
            ctx: &mut RunContext<()>,
            _state: &(),
            _response: &mut ModelResponse,
        ) -> tinyagents_harness::Result<()> {
            if self.0.swap(false, std::sync::atomic::Ordering::SeqCst) {
                ctx.request_control(MiddlewareControl::Interrupt {
                    node: "review".into(),
                    message: "needs approval".into(),
                });
            }
            Ok(())
        }
    }
}

/// Drives `compile_loop`'s real `CompiledGraph` directly (not through
/// `AgentHarness::invoke`/`GraphLoopDriver`), which is what actually gets
/// checkpoint/resume: A5 item 2's "approvals surfacing as graph interrupts".
async fn checkpoint_resume_with<C>(checkpointer: Arc<C>)
where
    C: tinyagents_graph::checkpoint::Checkpointer<tinyagents_graph::agent_loop::LoopState>
        + 'static,
{
    let model = Arc::new(MockModel::with_responses(vec![
        tool_call_response("call-1", "lookup", json!({})),
        ModelResponse::assistant("done"),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model)
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")))
        .push_middleware(Arc::new(pause_middleware::PauseOnce::new()));
    let harness = Arc::new(harness);

    let rt = Arc::new(LoopRuntime::for_run(
        harness.clone(),
        Arc::new(()),
        RunContext::new(RunConfig::new("checkpoint-resume"), ()),
    ));
    let graph = compile_loop(rt)
        .expect("graph compiles")
        .with_checkpointer(checkpointer);

    let execution = graph
        .run_with_thread(
            "checkpoint-resume-thread",
            LoopState::seed(vec![Message::user("look something up")]),
        )
        .await
        .expect("first leg completes to the interrupt");
    assert_eq!(
        execution.interrupts.len(),
        1,
        "run paused at the approval gate"
    );
    assert!(!execution.state.finished);

    let resumed = graph
        .resume(
            "checkpoint-resume-thread",
            tinyagents_graph::Command {
                update: None,
                goto: Vec::new(),
                resume: Some(json!({ "approved": true })),
                resume_by_task: Default::default(),
            },
        )
        .await
        .expect("resume drives the run to completion");
    assert!(resumed.state.finished);
    assert_eq!(resumed.state.final_text.as_deref(), Some("done"));
}

#[tokio::test]
async fn checkpoint_resume_across_interrupt_in_memory() {
    checkpoint_resume_with(Arc::new(InMemoryCheckpointer::new())).await;
}

#[tokio::test]
async fn checkpoint_resume_across_interrupt_file() {
    let dir = tempdir();
    checkpoint_resume_with(Arc::new(FileCheckpointer::new(dir.path()))).await;
}

/// Minimal temp-dir helper (avoids pulling in the `tempfile` crate just for
/// this one test).
fn tempdir() -> TempDir {
    let path = std::env::temp_dir().join(format!(
        "loop_as_graph-{}-{}",
        std::process::id(),
        tinyagents_harness::ids::now_ms()
    ));
    std::fs::create_dir_all(&path).expect("create temp dir");
    TempDir(path)
}

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ── `iter()` stepping with `override_next` ──────────────────────────────────

#[tokio::test]
async fn iter_steps_node_by_node_and_honors_override_next() {
    let model = Arc::new(MockModel::with_responses(vec![
        tool_call_response("call-1", "lookup", json!({})),
        ModelResponse::assistant("done"),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model)
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")));
    let harness = Arc::new(harness);

    let ctx = RunContext::new(RunConfig::new("iter-stepping"), ());
    let mut iter = harness
        .iter(Arc::new(()), ctx, vec![Message::user("look something up")])
        .expect("iter starts");

    let step = iter.next().await.expect("plan step").expect("not finished");
    assert_eq!(step.node, node::PLAN);
    assert_eq!(step.next.as_deref(), Some(node::MODEL));

    let step = iter
        .next()
        .await
        .expect("model step")
        .expect("not finished");
    assert_eq!(step.node, node::MODEL);
    assert_eq!(step.next.as_deref(), Some(node::TOOLS));

    // Redirect the very next activation back to `plan` instead of the
    // naturally-routed `tools` — exercising `override_next` — then let the
    // (now unoverridden) routing carry the run to completion.
    iter.override_next(node::PLAN);
    let step = iter
        .next()
        .await
        .expect("overridden step")
        .expect("not finished");
    assert_eq!(step.node, node::PLAN);
    assert_eq!(step.next.as_deref(), Some(node::MODEL));

    let state = iter.run_to_end().await.expect("run finishes");
    assert!(state.finished);
}

// ── Model profile preview (#6962) ───────────────────────────────────────────

/// Both loops expose the target model's profile to `before_model` middleware
/// on `RunContext::model_profile`, so middleware can avoid shapes the model
/// handles badly (a new system message on a model that hoists them).
#[tokio::test]
async fn before_model_sees_the_model_profile_in_direct_and_graph() {
    use async_trait::async_trait;
    use std::sync::Mutex;
    use tinyagents_harness::middleware::Middleware;
    use tinyinference_llm::model::{ChatModel, ModelProfile, ModelRequest};

    struct SeenProfiles(Arc<Mutex<Vec<Option<ModelProfile>>>>);

    #[async_trait]
    impl Middleware<(), ()> for SeenProfiles {
        fn name(&self) -> &str {
            "seen-profiles"
        }

        async fn before_model(
            &self,
            ctx: &mut RunContext<()>,
            _state: &(),
            _request: &mut ModelRequest,
        ) -> tinyagents_harness::Result<()> {
            self.0.lock().unwrap().push(ctx.model_profile.clone());
            Ok(())
        }
    }

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
            "done",
        )]));
        let expected = ChatModel::<()>::profile(model.as_ref()).cloned();
        assert!(expected.is_some(), "the mock advertises a profile");
        let mut harness = harness_for(execution, model);
        let seen = Arc::new(Mutex::new(Vec::new()));
        harness.push_middleware(Arc::new(SeenProfiles(seen.clone())));

        let ctx = RunContext::new(RunConfig::new("profile-preview"), ());
        harness
            .invoke_in_context(&(), ctx, vec![Message::user("hi")])
            .await
            .expect("run completes");

        assert_eq!(
            seen.lock().unwrap().as_slice(),
            [expected],
            "{execution:?} loop must expose the target profile to before_model"
        );
    }
}

/// A hint added by `before_model` must invalidate the graph loop's initial
/// resolution, because hints participate in registry selection just like an
/// explicit model name.
#[tokio::test]
async fn graph_reresolves_when_before_model_adds_a_model_hint() {
    use async_trait::async_trait;
    use tinyagents_harness::middleware::Middleware;
    use tinyinference_llm::model::{ModelHint, ModelRequest};

    struct SelectHint;

    #[async_trait]
    impl Middleware<(), ()> for SelectHint {
        fn name(&self) -> &str {
            "select-hint"
        }

        async fn before_model(
            &self,
            _ctx: &mut RunContext<()>,
            _state: &(),
            request: &mut ModelRequest,
        ) -> tinyagents_harness::Result<()> {
            request.model_hints.push(ModelHint {
                model: "hinted".to_string(),
                priority: 100,
                reason: Some("test route".to_string()),
            });
            Ok(())
        }
    }

    let default = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
        "default",
    )]));
    let hinted = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
        "hinted",
    )]));
    let mut harness = harness_for(LoopExecution::Graph, default);
    harness
        .register_model("hinted", hinted)
        .set_default_model("mock")
        .push_middleware(Arc::new(SelectHint));

    let run = harness
        .invoke_default(&(), vec![Message::user("route me")])
        .await
        .expect("run completes");

    assert_eq!(run.text(), Some("hinted".to_string()));
}

// ── Terminal outcome parity ─────────────────────────────────────────────────

/// A `StopWithPartial` cap must be reported identically by both engines: a
/// limit outcome, not a plain completion.
#[tokio::test]
async fn model_cap_stop_reports_the_same_terminal_outcome_in_both_engines() {
    use tinyagents_harness::events::LimitKind;
    use tinyagents_harness::limits::{LimitBehavior, RunLimits};
    use tinyagents_harness::terminal::TerminalReason;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_tool_call("spin", serde_json::json!({})));
        let mut harness = harness_for(execution, model);
        harness.register_tool(Arc::new(tinyagents_harness::testkit::FakeTool::returning(
            "spin", "again",
        )));
        harness.with_policy(RunPolicy {
            execution,
            limits: RunLimits::default()
                .with_max_model_calls(2)
                .with_behavior(LimitBehavior::StopWithPartial),
            ..RunPolicy::default()
        });
        let run = harness
            .invoke_default(&(), vec![Message::user("go")])
            .await
            .expect("StopWithPartial completes the run");
        let outcome = run.terminal.expect("terminal outcome");
        assert_eq!(
            outcome.reason,
            TerminalReason::LimitReached(Some(LimitKind::ModelCalls)),
            "{execution:?}"
        );
    }
}

/// The graph engine honors `StopWithPartial` for the tool-call cap (the direct
/// loop surfaces it as a `LimitExceeded` error carrying `ToolCalls`); the stop
/// must carry `ToolCalls`, not an untyped limit.
#[tokio::test]
async fn graph_tool_cap_stop_reports_a_typed_tool_calls_limit() {
    use tinyagents_harness::events::LimitKind;
    use tinyagents_harness::limits::{LimitBehavior, RunLimits};
    use tinyagents_harness::terminal::TerminalReason;

    for execution in [LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_tool_call("spin", serde_json::json!({})));
        let mut harness = harness_for(execution, model);
        harness.register_tool(Arc::new(tinyagents_harness::testkit::FakeTool::returning(
            "spin", "again",
        )));
        harness.with_policy(RunPolicy {
            execution,
            limits: RunLimits::default()
                .with_max_tool_calls(1)
                .with_behavior(LimitBehavior::StopWithPartial),
            ..RunPolicy::default()
        });
        let run = harness
            .invoke_default(&(), vec![Message::user("go")])
            .await
            .expect("StopWithPartial completes the run");
        let outcome = run.terminal.expect("terminal outcome");
        assert_eq!(
            outcome.reason,
            TerminalReason::LimitReached(Some(LimitKind::ToolCalls)),
            "{execution:?}"
        );
        assert!(
            outcome.message.contains("tool_calls"),
            "{execution:?}: {outcome:?}"
        );
    }
}

struct BoomModel;

#[async_trait::async_trait]
impl tinyinference_llm::model::ChatModel<()> for BoomModel {
    async fn invoke(
        &self,
        _: &(),
        _: tinyinference_llm::model::ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        Err(tinyinference_llm::Error::Model("model boom".into()))
    }
}

struct FailingAfterAgent;

#[async_trait::async_trait]
impl tinyagents_harness::middleware::Middleware<(), ()> for FailingAfterAgent {
    fn name(&self) -> &str {
        "failing_after_agent"
    }

    async fn after_agent(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _run: &mut tinyagents_harness::middleware::AgentRun,
    ) -> tinyagents_harness::Result<()> {
        Err(TinyAgentsError::Middleware("cleanup boom".into()))
    }
}

/// When the run already failed, a failing `after_agent` hook must not replace
/// the originating error, in either engine.
#[tokio::test]
async fn a_failing_after_agent_keeps_the_original_error_in_both_engines() {
    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness
            .register_model("mock", Arc::new(BoomModel))
            .set_default_model("mock");
        if matches!(execution, LoopExecution::Graph) {
            harness.with_loop_driver(Arc::new(GraphLoopDriver::new()));
        }
        harness.with_policy(RunPolicy {
            execution,
            ..RunPolicy::default()
        });
        harness.push_middleware(Arc::new(FailingAfterAgent));
        let error = harness
            .invoke_default(&(), vec![Message::user("go")])
            .await
            .expect_err("the run fails");
        assert!(
            error.to_string().contains("model boom"),
            "{execution:?}: {error}"
        );
    }
}

/// A failing `after_agent` hook on an otherwise successful run is the surfaced
/// error, so the partial run's terminal outcome must be that failure rather
/// than the stale completion recorded before the hook ran.
#[tokio::test]
async fn a_failing_after_agent_on_a_successful_run_records_a_failure_outcome() {
    use tinyagents_harness::terminal::{TerminalClass, TerminalReason};

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
            "fine",
        )]));
        let mut harness = harness_for(execution, model);
        harness.push_middleware(Arc::new(FailingAfterAgent));
        let ctx = RunContext::new(tinyagents_harness::context::RunConfig::new("aa"), ());
        let partial = harness
            .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("go")])
            .await;
        let error = partial.error.expect("the hook error surfaces");
        assert!(error.to_string().contains("cleanup boom"), "{execution:?}");
        let outcome = partial.run.terminal.expect("terminal outcome");
        assert_ne!(outcome.reason, TerminalReason::Completed, "{execution:?}");
        assert_eq!(outcome.class, TerminalClass::Failure, "{execution:?}");
    }
}

/// Under `LimitBehavior::Error` the graph engine's tool-cap failure still
/// carries the concrete `ToolCalls` kind on the partial run's outcome.
#[tokio::test]
async fn graph_tool_cap_error_carries_the_tool_calls_kind() {
    use tinyagents_harness::events::LimitKind;
    use tinyagents_harness::limits::RunLimits;
    use tinyagents_harness::terminal::TerminalReason;

    let model = Arc::new(MockModel::with_tool_call("spin", serde_json::json!({})));
    let mut harness = harness_for(LoopExecution::Graph, model);
    harness.register_tool(Arc::new(tinyagents_harness::testkit::FakeTool::returning(
        "spin", "again",
    )));
    harness.with_policy(RunPolicy {
        execution: LoopExecution::Graph,
        limits: RunLimits::default().with_max_tool_calls(1),
        ..RunPolicy::default()
    });
    let ctx = RunContext::new(tinyagents_harness::context::RunConfig::new("cap"), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("go")])
        .await;
    assert!(partial.error.is_some());
    assert_eq!(
        partial.run.terminal.expect("outcome").reason,
        TerminalReason::LimitReached(Some(LimitKind::ToolCalls))
    );
}

struct LimitFromHook;

#[async_trait::async_trait]
impl tinyagents_harness::middleware::Middleware<(), ()> for LimitFromHook {
    fn name(&self) -> &str {
        "limit_from_hook"
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _call: &mut tinyinference_llm::tool::ToolCall,
    ) -> tinyagents_harness::Result<()> {
        Err(TinyAgentsError::LimitExceeded("policy budget".into()))
    }
}

/// A `LimitExceeded` raised by middleware is not the tool-call cap, so under
/// `StopWithPartial` it must fail the run in both engines instead of becoming
/// a partial stop labeled `ToolCalls`.
#[tokio::test]
async fn a_middleware_limit_error_is_not_a_tool_cap_partial_stop() {
    use tinyagents_harness::limits::{LimitBehavior, RunLimits};

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_tool_call("spin", serde_json::json!({})));
        let mut harness = harness_for(execution, model);
        harness.register_tool(Arc::new(tinyagents_harness::testkit::FakeTool::returning(
            "spin", "again",
        )));
        harness.push_middleware(Arc::new(LimitFromHook));
        harness.with_policy(RunPolicy {
            execution,
            limits: RunLimits::default().with_behavior(LimitBehavior::StopWithPartial),
            ..RunPolicy::default()
        });
        let result = harness.invoke_default(&(), vec![Message::user("go")]).await;
        assert!(
            matches!(result, Err(TinyAgentsError::LimitExceeded(_))),
            "{execution:?}: {result:?}"
        );
    }
}

/// The turn/message lifecycle events are direct-loop only for now (a documented
/// follow-up). Pin that so the gap is explicit and this parity file's filter
/// cannot silently hide a change in either direction.
#[tokio::test]
async fn lifecycle_events_are_direct_loop_only_until_the_graph_driver_emits_them() {
    for (execution, expect_lifecycle) in
        [(LoopExecution::Direct, true), (LoopExecution::Graph, false)]
    {
        let model = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
            "done",
        )]));
        let harness = harness_for(execution, model);
        let recorder = EventRecorder::new();
        let ctx = RunContext::new(RunConfig::new("lc"), ()).with_events(recorder.sink());
        harness
            .invoke_in_context(&(), ctx, vec![Message::user("hi")])
            .await
            .unwrap();
        let has_lifecycle = recorder
            .events()
            .iter()
            .any(|event| event.kind() == "turn.started" || event.kind() == "message.appended");
        assert_eq!(has_lifecycle, expect_lifecycle, "{execution:?}");
    }
}
