//! Turn and message lifecycle events (`TurnStarted`, `TurnCompleted`,
//! `MessageAppended`) and the message-carrying `QueuedMessageApplied`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::ids::CallId;
use crate::run_queue::{QueueLane, RunQueue};
use crate::runtime::{AgentHarness, PayloadCapture, RunPolicy};
use crate::testkit::{EventRecorder, ScriptedModel};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolResult};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "echo"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("echoed"))
    }
}

struct FailingModel;

#[async_trait]
impl ChatModel<()> for FailingModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        Err(tinyinference_llm::Error::Model("boom".into()))
    }
}

fn response(tool_calls: Vec<ToolCall>, text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: if text.is_empty() {
                Vec::new()
            } else {
                vec![ContentBlock::Text(text.to_string())]
            },
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

fn tool_turn(ids: &[&str]) -> ModelResponse {
    response(
        ids.iter()
            .map(|id| ToolCall::new(*id, "echo", json!({})))
            .collect(),
        "",
    )
}

fn harness(responses: Vec<ModelResponse>, capture: PayloadCapture) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(ScriptedModel::new(responses)));
    harness.register_tool(Arc::new(EchoTool));
    harness.with_policy(RunPolicy {
        capture,
        ..RunPolicy::default()
    });
    harness
}

/// A compact rendering of the lifecycle events, in emission order.
fn lifecycle(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnStarted { turn } => Some(format!("turn.started:{turn}")),
            AgentEvent::TurnCompleted {
                turn,
                tool_result_count,
                tool_call_ids,
            } => Some(format!(
                "turn.completed:{turn}:{tool_result_count}:{}",
                tool_call_ids
                    .iter()
                    .map(CallId::as_str)
                    .collect::<Vec<_>>()
                    .join(",")
            )),
            AgentEvent::MessageAppended {
                role,
                index,
                call_id,
                ..
            } => Some(format!(
                "message:{index}:{role}{}",
                call_id
                    .as_ref()
                    .map(|id| format!(":{}", id.as_str()))
                    .unwrap_or_default()
            )),
            AgentEvent::ModelStarted { .. } => Some("model.started".to_string()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn turns_and_messages_are_announced_in_transcript_order() {
    let harness = harness(
        vec![tool_turn(&["a", "b"]), response(vec![], "done")],
        PayloadCapture::default(),
    );
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("life"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .unwrap();

    assert_eq!(run.messages.len(), 5);
    assert_eq!(
        lifecycle(&recorder.events()),
        vec![
            "turn.started:1",
            "model.started",
            "message:1:assistant",
            "message:2:tool:a",
            "message:3:tool:b",
            "turn.completed:1:2:a,b",
            "turn.started:2",
            "model.started",
            "message:4:assistant",
            "turn.completed:2:0:",
        ]
    );
}

#[tokio::test]
async fn message_payloads_follow_the_capture_policy() {
    for (capture, expect_message, expect_tool) in [
        (PayloadCapture::default(), false, false),
        (PayloadCapture::all(), true, true),
    ] {
        let harness = harness(vec![tool_turn(&["a"]), response(vec![], "done")], capture);
        let recorder = EventRecorder::new();
        let ctx = RunContext::new(RunConfig::new("cap"), ()).with_events(recorder.sink());
        harness
            .invoke_in_context(&(), ctx, vec![Message::user("go")])
            .await
            .unwrap();
        let mut seen = 0;
        for event in recorder.events() {
            if let AgentEvent::MessageAppended { role, message, .. } = event {
                seen += 1;
                let expected = if role == "tool" {
                    expect_tool
                } else {
                    expect_message
                };
                assert_eq!(message.is_some(), expected, "role {role}");
            }
        }
        assert_eq!(seen, 3, "assistant, tool, assistant");
    }
}

#[tokio::test]
async fn a_failed_turn_is_still_closed() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(FailingModel));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("fail"), ()).with_events(recorder.sink());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("go")])
        .await;
    assert!(partial.error.is_some());
    assert_eq!(
        lifecycle(&recorder.events()),
        vec!["turn.started:1", "model.started", "turn.completed:1:0:"]
    );
}

#[tokio::test]
async fn queued_message_applied_carries_the_applied_messages() {
    let harness = harness(
        vec![tool_turn(&["a"]), response(vec![], "done")],
        PayloadCapture::all(),
    );
    let queue = Arc::new(RunQueue::new());
    queue
        .push(QueueLane::Steer, Message::user("be brief"))
        .await;
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("queue"), ())
        .with_events(recorder.sink())
        .with_run_queue(Arc::clone(&queue));
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .unwrap();

    let events = recorder.events();
    let applied = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::QueuedMessageApplied {
                count,
                first_index,
                messages,
                ..
            } => Some((*count, *first_index, messages.clone())),
            _ => None,
        })
        .expect("queue applied");
    assert_eq!(applied.0, 1);
    assert_eq!(applied.1, 3, "after user, assistant, tool");
    assert_eq!(applied.2.len(), 1);
    assert_eq!(
        applied.2[0],
        serde_json::to_value(Message::user("be brief")).unwrap()
    );
    // The same message is also announced as an ordinary transcript append.
    assert!(lifecycle(&events).contains(&"message:3:user".to_string()));
}

#[tokio::test]
async fn mixed_capture_keeps_one_payload_slot_per_applied_message() {
    // `tool_io` only: the queued user message is not captured, the tool one is.
    let harness = harness(
        vec![tool_turn(&["a"]), response(vec![], "done")],
        PayloadCapture {
            model_io: false,
            tool_io: true,
        },
    );
    let queue = Arc::new(RunQueue::new());
    queue
        .push(QueueLane::Steer, Message::user("uncaptured"))
        .await;
    queue
        .push(QueueLane::Steer, Message::tool("t1", "captured"))
        .await;
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("mixed"), ())
        .with_events(recorder.sink())
        .with_run_queue(Arc::clone(&queue));
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .unwrap();
    let messages = recorder
        .events()
        .iter()
        .find_map(|event| match event {
            AgentEvent::QueuedMessageApplied { messages, .. } => Some(messages.clone()),
            _ => None,
        })
        .expect("queue applied");
    assert_eq!(messages.len(), 2, "one slot per applied message");
    assert!(messages[0].is_null());
    assert_eq!(
        messages[1],
        serde_json::to_value(Message::tool("t1", "captured")).unwrap()
    );
}

// ── Transcript mirroring ────────────────────────────────────────────────────

/// Folds the lifecycle events into a role list the way a consumer mirroring
/// the transcript would. `?` marks messages known only by position (after a
/// rewrite).
pub(super) fn mirror_roles(events: &[AgentEvent], seed: &[Message]) -> Vec<String> {
    let mut roles: Vec<String> = seed
        .iter()
        .map(|m| super::lifecycle::role_of(m).to_string())
        .collect();
    for event in events {
        match event {
            AgentEvent::MessageAppended { role, index, .. } => {
                assert_eq!(*index, roles.len(), "append index follows the mirror");
                roles.push(role.clone());
            }
            AgentEvent::MessageRetracted { index } => {
                assert_eq!(*index + 1, roles.len(), "retraction pops the tail");
                roles.pop();
            }
            AgentEvent::TranscriptRewritten { len, .. } => roles = vec!["?".to_string(); *len],
            _ => {}
        }
    }
    roles
}

pub(super) fn assert_mirrors(events: &[AgentEvent], seed: &[Message], transcript: &[Message]) {
    let mirror = mirror_roles(events, seed);
    let actual: Vec<String> = transcript
        .iter()
        .map(|m| super::lifecycle::role_of(m).to_string())
        .collect();
    assert_eq!(mirror.len(), actual.len(), "{mirror:?} vs {actual:?}");
    for (m, a) in mirror.iter().zip(&actual) {
        assert!(m == "?" || m == a, "{mirror:?} vs {actual:?}");
    }
}

fn truncated_empty(cap: u32) -> ModelResponse {
    let mut r = response(vec![], "");
    r.finish_reason = Some("length".to_string());
    let _ = cap;
    r
}

#[tokio::test]
async fn a_recovery_pop_is_retracted_and_the_mirror_stays_exact() {
    let model_script = vec![
        truncated_empty(2048),
        truncated_empty(4096),
        truncated_empty(8192),
        tool_turn(&["c1"]),
        response(vec![], "done"),
    ];
    let harness = harness(model_script, PayloadCapture::default());
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(
        RunConfig::new("recover").with_max_turn_output_tokens(2048),
        (),
    )
    .with_events(recorder.sink());
    let seed = vec![Message::user("go")];
    let run = harness
        .invoke_in_context(&(), ctx, seed.clone())
        .await
        .unwrap();

    let events = recorder.events();
    let retractions = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::MessageRetracted { index: 1 }))
        .count();
    assert_eq!(retractions, 3, "all blank replies were dropped");
    // The recovery nudge is announced as an ordinary append.
    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::MessageAppended { role, .. } if role == "user"
        )),
        "the nudge is announced"
    );
    assert_mirrors(&events, &seed, &run.messages);
}

#[tokio::test]
async fn steer_between_turns_reports_the_right_first_index() {
    let harness = harness(
        vec![
            tool_turn(&["a"]),
            tool_turn(&["b"]),
            response(vec![], "done"),
        ],
        PayloadCapture::default(),
    );
    let queue = Arc::new(RunQueue::new());
    queue.push(QueueLane::Steer, Message::user("s1")).await;
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("steer"), ())
        .with_events(recorder.sink())
        .with_run_queue(Arc::clone(&queue));
    let seed = vec![Message::user("go")];
    let run = harness
        .invoke_in_context(&(), ctx, seed.clone())
        .await
        .unwrap();
    let events = recorder.events();
    let first = events.iter().find_map(|e| match e {
        AgentEvent::QueuedMessageApplied { first_index, .. } => Some(*first_index),
        _ => None,
    });
    assert_eq!(first, Some(3));
    assert_eq!(run.messages[3].text(), "s1");
    assert_mirrors(&events, &seed, &run.messages);
}

#[test]
fn tracker_retract_and_rebase_are_explicit() {
    use super::lifecycle::TurnTracker;
    let recorder = EventRecorder::new();
    let sink = recorder.sink();
    let capture = PayloadCapture::default();
    let mut tracker = TurnTracker::new(0);
    let mut messages = vec![Message::user("a"), Message::user("b"), Message::user("c")];
    tracker.flush(&sink, capture, &messages);
    messages.pop();
    tracker.retract_to(&sink, messages.len());
    // An unannounced message that is popped again emits nothing.
    messages.push(Message::user("tmp"));
    messages.pop();
    tracker.retract_to(&sink, messages.len());
    messages.insert(0, Message::system("sys"));
    tracker.rebase(&sink, messages.len(), "tool_change");
    messages.push(Message::user("d"));
    tracker.flush(&sink, capture, &messages);
    let kinds: Vec<String> = recorder
        .events()
        .iter()
        .map(|e| match e {
            AgentEvent::MessageAppended { index, .. } => format!("append:{index}"),
            AgentEvent::MessageRetracted { index } => format!("retract:{index}"),
            AgentEvent::TranscriptRewritten { len, reason } => format!("rewrite:{len}:{reason}"),
            other => other.kind().to_string(),
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            "append:0",
            "append:1",
            "append:2",
            "retract:2",
            "rewrite:3:tool_change",
            "append:3"
        ]
    );
}

/// Replays scripted responses for a model without native structured output, so
/// the `answer` tool call is the structured-output channel.
struct ToolStructuredScript {
    profile: tinyinference_llm::model::ModelProfile,
    responses: std::sync::Mutex<std::collections::VecDeque<ModelResponse>>,
}

#[async_trait]
impl ChatModel<()> for ToolStructuredScript {
    fn profile(&self) -> Option<&tinyinference_llm::model::ModelProfile> {
        Some(&self.profile)
    }
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("the model was called more often than scripted"))
    }
}

/// A mixed structured turn is closed before its queued steering is drained, so
/// the queued message is announced after `TurnCompleted` and is not counted as
/// one of that turn's tool results.
#[tokio::test]
async fn a_mixed_structured_turn_closes_before_queued_messages_are_drained() {
    let mixed = response(
        vec![
            ToolCall::new("s1", "answer", json!({"value": "first"})),
            ToolCall::new("c1", "echo", json!({})),
        ],
        "",
    );
    let last = response(
        vec![ToolCall::new("s2", "answer", json!({"value": "final"}))],
        "",
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(ToolStructuredScript {
            profile: tinyinference_llm::model::ModelProfile {
                tool_calling: true,
                native_structured_output: false,
                json_schema: false,
                ..Default::default()
            },
            responses: std::sync::Mutex::new(vec![mixed, last].into()),
        }),
    );
    harness.register_tool(Arc::new(EchoTool));
    harness.with_policy(RunPolicy {
        end_strategy: crate::runtime::EndStrategy::Exhaustive,
        default_response_format: Some(tinyinference_llm::model::ResponseFormat::auto(
            "answer",
            json!({"type": "object"}),
        )),
        ..RunPolicy::default()
    });
    let queue = Arc::new(RunQueue::new());
    queue
        .push(
            QueueLane::Steer,
            Message::tool("queued-call", "late result"),
        )
        .await;
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("mixed"), ())
        .with_events(recorder.sink())
        .with_run_queue(Arc::clone(&queue));
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .unwrap();

    let events = recorder.events();
    let lines = lifecycle(&events);
    let completed = lines
        .iter()
        .position(|line| line.starts_with("turn.completed:1:"))
        .expect("turn 1 completed");
    assert_eq!(
        lines[completed], "turn.completed:1:2:s1,c1",
        "only the turn's own tool results are counted: {lines:?}"
    );
    let queued = lines
        .iter()
        .position(|line| line.ends_with(":queued-call"))
        .expect("queued message announced");
    assert!(
        completed < queued,
        "queued message after TurnCompleted: {lines:?}"
    );
}

#[tokio::test]
async fn messages_appended_by_after_agent_middleware_are_announced() {
    use crate::middleware::{AgentRun, Middleware};
    struct Appender;
    #[async_trait::async_trait]
    impl Middleware<(), ()> for Appender {
        fn name(&self) -> &str {
            "appender"
        }
        async fn after_agent(
            &self,
            _: &mut RunContext<()>,
            _: &(),
            run: &mut AgentRun,
        ) -> crate::error::Result<()> {
            run.messages.push(Message::user("added by middleware"));
            Ok(())
        }
    }
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(crate::testkit::ScriptedModel::new(vec![
            crate::testkit::text_response("done"),
        ])),
    );
    harness.push_middleware(Arc::new(Appender));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("aa-append"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .unwrap();
    let last = run.messages.len() - 1;
    let announced: Vec<usize> = recorder
        .events()
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageAppended { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    assert_eq!(announced.last(), Some(&last), "{announced:?}");
}
