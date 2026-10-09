//! Tests for the `RunQueue` wiring (A4): the loop drains the `Steer` lane at
//! the turn boundary after a tool batch, the `Followup` lane when it would
//! otherwise finish, honors `RunPolicy::queue_mode`, delivers the `Collect`
//! lane on `AgentRun::collected`, and emits `QueuedMessageApplied`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::run_queue::{QueueLane, QueueMode, RunQueue, RunQueueHandle};
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::{EventRecorder, ScriptedModel};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolResult};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// A tool that returns a fixed reply and, when given a queue, pushes a steer
/// message onto it *while executing* — i.e. mid-batch.
struct QueueingTool {
    name: &'static str,
    reply: &'static str,
    push_on_execute: Option<(RunQueueHandle, &'static str)>,
}

impl QueueingTool {
    fn plain(name: &'static str, reply: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            reply,
            push_on_execute: None,
        })
    }

    fn steering(
        name: &'static str,
        reply: &'static str,
        queue: RunQueueHandle,
        steer: &'static str,
    ) -> Arc<Self> {
        Arc::new(Self {
            name,
            reply,
            push_on_execute: Some((queue, steer)),
        })
    }
}

#[async_trait]
impl Tool for QueueingTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "queueing tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Some((queue, steer)) = &self.push_on_execute {
            queue.push(QueueLane::Steer, Message::user(*steer)).await;
        }
        Ok(ToolResult::success(self.reply))
    }
}

fn response(tool_calls: Vec<ToolCall>, text: &str) -> ModelResponse {
    let content = if text.is_empty() {
        Vec::new()
    } else {
        vec![ContentBlock::Text(text.to_string())]
    };
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content,
            tool_calls,
            usage: Some(Usage::new(1, 1)),
            origin: None,
        },
        usage: Some(Usage::new(1, 1)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn tool_turn(calls: &[(&str, &str)]) -> ModelResponse {
    response(
        calls
            .iter()
            .map(|(id, name)| ToolCall::new(*id, *name, json!({})))
            .collect(),
        "",
    )
}

fn final_turn(text: &str) -> ModelResponse {
    response(Vec::new(), text)
}

/// Texts of the user messages in `request`, in order.
fn user_texts(request: &ModelRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter(|message| matches!(message, Message::User(_)))
        .map(Message::text)
        .collect()
}

/// A compact role/text rendering of a transcript for ordering assertions.
fn shape(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|message| match message {
            Message::System(_) => format!("system:{}", message.text()),
            Message::User(_) => format!("user:{}", message.text()),
            Message::Assistant(a) if !a.tool_calls.is_empty() => {
                format!("assistant:tools[{}]", a.tool_calls.len())
            }
            Message::Assistant(_) => format!("assistant:{}", message.text()),
            Message::Tool(t) => format!("tool:{}", t.tool_call_id),
            Message::Custom(_) => "custom".to_string(),
        })
        .collect()
}

fn queued_applied(events: &[AgentEvent]) -> Vec<(QueueLane, usize)> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::QueuedMessageApplied { lane, count, .. } => Some((*lane, *count)),
            _ => None,
        })
        .collect()
}

struct Fixture {
    harness: AgentHarness<()>,
    model: Arc<ScriptedModel>,
    queue: RunQueueHandle,
    recorder: EventRecorder,
}

fn fixture(responses: Vec<ModelResponse>, mode: QueueMode) -> Fixture {
    let model = Arc::new(ScriptedModel::new(responses));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_policy(RunPolicy {
        queue_mode: mode,
        ..RunPolicy::default()
    });
    Fixture {
        harness,
        model,
        queue: Arc::new(RunQueue::new()),
        recorder: EventRecorder::new(),
    }
}

impl Fixture {
    fn ctx(&self, run_id: &str) -> RunContext<()> {
        RunContext::new(RunConfig::new(run_id), ())
            .with_events(self.recorder.sink())
            .with_run_queue(Arc::clone(&self.queue))
    }
}

// ── Steer ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn steer_queued_mid_tool_batch_waits_for_the_batch_to_finish() {
    let mut fx = fixture(
        vec![
            tool_turn(&[("call-a", "a"), ("call-b", "b")]),
            final_turn("done"),
        ],
        QueueMode::All,
    );
    // Tool `a` pushes the steer while the two-call batch is executing.
    fx.harness.register_tool(QueueingTool::steering(
        "a",
        "a-result",
        Arc::clone(&fx.queue),
        "steer: be brief",
    ));
    fx.harness
        .register_tool(QueueingTool::plain("b", "b-result"));

    let ctx = fx.ctx("steer-mid-batch");
    let run = fx
        .harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let requests = fx.model.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        user_texts(&requests[0]),
        vec!["go"],
        "the tool-call turn never sees the steer"
    );
    assert_eq!(
        user_texts(&requests[1]),
        vec!["go", "steer: be brief"],
        "the next model call sees the steer as a user message"
    );
    // Both tool results land before the steer: it never interrupted the batch.
    assert_eq!(
        shape(&run.messages),
        vec![
            "user:go",
            "assistant:tools[2]",
            "tool:call-a",
            "tool:call-b",
            "user:steer: be brief",
            "assistant:done",
        ]
    );
    assert_eq!(run.text().as_deref(), Some("done"));
    assert_eq!(
        queued_applied(&fx.recorder.events()),
        vec![(QueueLane::Steer, 1)]
    );
    assert_eq!(fx.queue.status().await.total, 0);
}

#[tokio::test]
async fn one_at_a_time_applies_one_steer_per_boundary_while_all_applies_every_steer_at_once() {
    for (mode, expected_second, expected_third, expected_events) in [
        (
            QueueMode::OneAtATime,
            vec!["go", "s1"],
            vec!["go", "s1", "s2"],
            vec![(QueueLane::Steer, 1), (QueueLane::Steer, 1)],
        ),
        (
            QueueMode::All,
            vec!["go", "s1", "s2"],
            vec!["go", "s1", "s2"],
            vec![(QueueLane::Steer, 2)],
        ),
    ] {
        let mut fx = fixture(
            vec![
                tool_turn(&[("call-1", "a")]),
                tool_turn(&[("call-2", "a")]),
                final_turn("done"),
            ],
            mode,
        );
        fx.harness
            .register_tool(QueueingTool::plain("a", "a-result"));
        fx.queue.push(QueueLane::Steer, Message::user("s1")).await;
        fx.queue.push(QueueLane::Steer, Message::user("s2")).await;

        let ctx = fx.ctx("queue-mode");
        let run = fx
            .harness
            .invoke_in_context(&(), ctx, vec![Message::user("go")])
            .await
            .expect("run succeeds");

        let requests = fx.model.requests();
        assert_eq!(requests.len(), 3, "{mode:?}");
        assert_eq!(user_texts(&requests[0]), vec!["go"], "{mode:?}");
        assert_eq!(user_texts(&requests[1]), expected_second, "{mode:?}");
        assert_eq!(user_texts(&requests[2]), expected_third, "{mode:?}");
        assert_eq!(run.model_calls, 3, "{mode:?}");
        assert_eq!(
            queued_applied(&fx.recorder.events()),
            expected_events,
            "{mode:?}"
        );
        assert_eq!(fx.queue.status().await.total, 0, "{mode:?}");
    }
}

// ── Followup ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn followup_runs_one_more_turn_after_the_model_would_have_finished() {
    // Without a follow-up this script finishes after the first response. With
    // one queued, the loop appends it as a user turn and keeps going until
    // the follow-up turn (which itself uses a tool) reaches its own final.
    let mut fx = fixture(
        vec![
            final_turn("first answer"),
            tool_turn(&[("call-1", "a")]),
            final_turn("second answer"),
        ],
        QueueMode::All,
    );
    fx.harness
        .register_tool(QueueingTool::plain("a", "a-result"));
    fx.queue
        .push(QueueLane::Followup, Message::user("and then?"))
        .await;

    let ctx = fx.ctx("followup");
    let run = fx
        .harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(
        run.model_calls, 3,
        "one extra turn (plus its tool round trip)"
    );
    assert_eq!(run.text().as_deref(), Some("second answer"));
    assert_eq!(
        shape(&run.messages),
        vec![
            "user:go",
            "assistant:first answer",
            "user:and then?",
            "assistant:tools[1]",
            "tool:call-1",
            "assistant:second answer",
        ]
    );
    let requests = fx.model.requests();
    assert_eq!(user_texts(&requests[0]), vec!["go"]);
    assert_eq!(user_texts(&requests[1]), vec!["go", "and then?"]);
    assert_eq!(
        queued_applied(&fx.recorder.events()),
        vec![(QueueLane::Followup, 1)]
    );
    assert!(
        fx.recorder
            .events()
            .iter()
            .filter(|event| matches!(event, AgentEvent::RunCompleted { .. }))
            .count()
            == 1,
        "the run completes exactly once, after the follow-up turn"
    );
}

#[tokio::test]
async fn no_queue_attached_finishes_exactly_as_before() {
    let fx = fixture(vec![final_turn("done")], QueueMode::All);
    let ctx = RunContext::new(RunConfig::new("no-queue"), ()).with_events(fx.recorder.sink());
    let run = fx
        .harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.model_calls, 1);
    assert!(run.collected.is_empty());
    assert!(queued_applied(&fx.recorder.events()).is_empty());
}

// ── Collect ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn collect_lane_lands_on_the_run_and_never_reaches_the_model() {
    let fx = fixture(vec![final_turn("done")], QueueMode::All);
    fx.queue
        .push(QueueLane::Collect, Message::user("observation 1"))
        .await;
    fx.queue
        .push(QueueLane::Collect, Message::user("observation 2"))
        .await;

    let ctx = fx.ctx("collect");
    let run = fx
        .harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(
        run.collected.iter().map(Message::text).collect::<Vec<_>>(),
        vec!["observation 1", "observation 2"]
    );
    assert_eq!(shape(&run.messages), vec!["user:go", "assistant:done"]);
    for request in fx.model.requests() {
        assert_eq!(user_texts(&request), vec!["go"]);
    }
    assert!(
        queued_applied(&fx.recorder.events()).is_empty(),
        "collected items are not applied to the transcript"
    );
    assert_eq!(fx.queue.status().await.collects, 0);
}

// ── Boundary semantics ──────────────────────────────────────────────────────

#[tokio::test]
async fn steer_arriving_after_the_final_answer_is_applied_before_any_followup() {
    // A steer that lands once the model has already answered is not lost:
    // the natural-finish boundary applies it first (pi polls steering after
    // every completed turn), and only a boundary with no pending steer
    // takes a follow-up. Each gets its own turn, in that order.
    let fx = fixture(
        vec![
            final_turn("first answer"),
            final_turn("steered answer"),
            final_turn("followed-up answer"),
        ],
        QueueMode::All,
    );
    fx.queue
        .push(QueueLane::Followup, Message::user("follow-up"))
        .await;
    fx.queue
        .push(QueueLane::Steer, Message::user("actually, shorter"))
        .await;

    let ctx = fx.ctx("steer-at-finish");
    let run = fx
        .harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 3);
    assert_eq!(run.text().as_deref(), Some("followed-up answer"));
    assert_eq!(
        shape(&run.messages),
        vec![
            "user:go",
            "assistant:first answer",
            "user:actually, shorter",
            "assistant:steered answer",
            "user:follow-up",
            "assistant:followed-up answer",
        ]
    );
    assert_eq!(
        queued_applied(&fx.recorder.events()),
        vec![(QueueLane::Steer, 1), (QueueLane::Followup, 1)]
    );
    assert_eq!(fx.queue.status().await.total, 0);
}

/// Requests `StopWithFinal` after any tool result.
struct StopAfterTool;

#[async_trait]
impl crate::middleware::Middleware<(), ()> for StopAfterTool {
    fn name(&self) -> &str {
        "stop-after-tool"
    }
    async fn after_tool(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _invocation: &crate::middleware::ToolInvocationIdentity,
        _result: &mut ToolResult,
    ) -> crate::error::Result<()> {
        ctx.request_control(crate::context::MiddlewareControl::StopWithFinal(
            "stopped by middleware".to_string(),
        ));
        Ok(())
    }
}

#[tokio::test]
async fn middleware_stop_is_terminal_and_leaves_followups_queued() {
    let mut fx = fixture(
        vec![tool_turn(&[("call-1", "a")]), final_turn("never reached")],
        QueueMode::All,
    );
    fx.harness
        .register_tool(QueueingTool::plain("a", "a-result"));
    fx.harness.push_middleware(Arc::new(StopAfterTool));
    fx.queue
        .push(QueueLane::Followup, Message::user("later"))
        .await;

    let ctx = fx.ctx("middleware-stop");
    let run = fx
        .harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 1);
    assert_eq!(run.text().as_deref(), Some("stopped by middleware"));
    assert_eq!(
        fx.queue.status().await.followups,
        1,
        "a forced stop does not consume the follow-up; the host decides"
    );
    assert!(queued_applied(&fx.recorder.events()).is_empty());
}
