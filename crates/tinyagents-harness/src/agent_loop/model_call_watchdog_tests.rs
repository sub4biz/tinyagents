use super::*;

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem, ReasoningConfig,
    ReasoningEffort,
};

use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::runtime::{AgentHarness, ReasoningWatchdog, RunPolicy};
use crate::testkit::EventRecorder;

/// A model that replays one scripted stream per call and records the
/// requests it was given, so a test can see what the loop sent after the
/// watchdog ended a call.
struct ScriptedStreams {
    streams: Mutex<VecDeque<Vec<ModelStreamItem>>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl ScriptedStreams {
    fn new(streams: Vec<Vec<ModelStreamItem>>) -> Self {
        Self {
            streams: Mutex::new(streams.into()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn next_items(&self, request: ModelRequest) -> Vec<ModelStreamItem> {
        self.requests.lock().unwrap().push(request);
        self.streams
            .lock()
            .unwrap()
            .pop_front()
            .expect("more model calls than scripted streams")
    }
}

#[async_trait]
impl ChatModel<()> for ScriptedStreams {
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let mut accumulator = tinyinference_llm::model::StreamAccumulator::new();
        for item in self.next_items(request) {
            accumulator.push(&item);
        }
        accumulator.finish()
    }

    async fn stream(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelStream> {
        let items = self.next_items(request);
        Ok(ModelStream::new(Box::pin(futures::stream::iter(items))))
    }
}

fn reasoning_delta(chars: usize) -> ModelStreamItem {
    ModelStreamItem::MessageDelta(MessageDelta {
        text: String::new(),
        reasoning: "x".repeat(chars),
        tool_call: None,
    })
}

fn text_delta(text: &str) -> ModelStreamItem {
    ModelStreamItem::MessageDelta(MessageDelta {
        text: text.to_string(),
        reasoning: String::new(),
        tool_call: None,
    })
}

/// A stream that reasons for `chars` characters, then answers `text`.
fn reasoning_then_text(chars: usize, text: &str) -> Vec<ModelStreamItem> {
    let mut items = vec![ModelStreamItem::Started];
    for _ in 0..(chars / 100) {
        items.push(reasoning_delta(100));
    }
    items.push(text_delta(text));
    items.push(ModelStreamItem::Completed(ModelResponse::assistant(text)));
    items
}

fn harness_with(model: Arc<ScriptedStreams>, watchdog: ReasoningWatchdog) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model as _);
    harness.with_policy(RunPolicy {
        default_reasoning: Some(ReasoningConfig::effort(ReasoningEffort::High)),
        reasoning_watchdog: watchdog,
        ..RunPolicy::default()
    });
    harness
}

#[tokio::test]
async fn a_call_that_reasons_past_its_bound_with_nothing_visible_is_ended_as_a_dead_call() {
    // 4,000 characters of reasoning is about 1,000 tokens; the bound is 200.
    // The call is ended before its (late) answer arrives, and the loop's
    // truncated-empty recovery retries with reasoning off and gets the
    // second scripted stream.
    let model = Arc::new(ScriptedStreams::new(vec![
        reasoning_then_text(4_000, "late answer"),
        reasoning_then_text(0, "recovered"),
    ]));
    let harness = harness_with(Arc::clone(&model), ReasoningWatchdog::Tokens(200));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("watchdog"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the run finishes on the recovered call");
    assert_eq!(run.text(), Some("recovered".to_string()));
    let efforts: Vec<Option<ReasoningEffort>> = model
        .requests()
        .iter()
        .map(|r| r.reasoning.as_ref().and_then(|c| c.effort))
        .collect();
    assert_eq!(
        efforts,
        vec![Some(ReasoningEffort::High), Some(ReasoningEffort::None)],
        "the ended call is a dead call: the retry goes out with reasoning off"
    );
    assert!(
        recorder.events().iter().any(|e| matches!(
            e,
            AgentEvent::ControlApplied { control, .. } if control == "reasoning_watchdog"
        )),
        "the watchdog is observable; got kinds {:?}",
        recorder.kinds()
    );
}

#[tokio::test]
async fn the_bound_counts_characters_not_utf8_bytes() {
    // 240 CJK characters are 720 bytes. At three characters per token that
    // is 80 tokens, under a 100-token bound; counted in bytes it would be
    // 240 tokens and the watchdog would end a live call early.
    let mut items = vec![ModelStreamItem::Started];
    for _ in 0..4 {
        items.push(ModelStreamItem::MessageDelta(MessageDelta {
            text: String::new(),
            reasoning: "思".repeat(60),
            tool_call: None,
        }));
    }
    items.push(text_delta("answer"));
    items.push(ModelStreamItem::Completed(ModelResponse::assistant(
        "answer",
    )));
    let model = Arc::new(ScriptedStreams::new(vec![items]));
    let harness = harness_with(Arc::clone(&model), ReasoningWatchdog::Tokens(100));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("watchdog-chars"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the run finishes");
    assert_eq!(run.text(), Some("answer".to_string()));
    assert_eq!(model.requests().len(), 1);
    assert!(
        !recorder.events().iter().any(|e| matches!(
            e,
            AgentEvent::ControlApplied { control, .. } if control == "reasoning_watchdog"
        )),
        "no watchdog event; got kinds {:?}",
        recorder.kinds()
    );
}

#[tokio::test]
async fn visible_output_before_the_bound_disarms_the_watchdog() {
    // Text arrives first, then a long think, then the answer: the model is
    // answering, so the call runs to completion.
    let mut items = vec![ModelStreamItem::Started, text_delta("Working: ")];
    for _ in 0..40 {
        items.push(reasoning_delta(100));
    }
    items.push(text_delta("answer"));
    items.push(ModelStreamItem::Completed(ModelResponse::assistant(
        "Working: answer",
    )));
    let model = Arc::new(ScriptedStreams::new(vec![items]));
    let harness = harness_with(Arc::clone(&model), ReasoningWatchdog::Tokens(200));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("watchdog-disarmed"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the run finishes");
    assert_eq!(run.text(), Some("Working: answer".to_string()));
    assert_eq!(model.requests().len(), 1);
    assert!(
        !recorder.events().iter().any(|e| matches!(
            e,
            AgentEvent::ControlApplied { control, .. } if control == "reasoning_watchdog"
        )),
        "no watchdog event; got kinds {:?}",
        recorder.kinds()
    );
}

#[tokio::test]
async fn the_request_budget_is_the_default_bound_and_no_budget_means_no_bound() {
    // With the default policy the bound is the request's own
    // `reasoning.budget_tokens`; a request without one is never ended.
    let model = Arc::new(ScriptedStreams::new(vec![reasoning_then_text(
        4_000,
        "slow but fine",
    )]));
    let harness = harness_with(Arc::clone(&model), ReasoningWatchdog::RequestBudget);
    let ctx = RunContext::new(RunConfig::new("watchdog-unbounded"), ());
    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the run finishes");
    assert_eq!(run.text(), Some("slow but fine".to_string()));
    assert_eq!(model.requests().len(), 1);

    // The same stream under a request that carries a 200-token budget is
    // ended, and the retry (reasoning off, so no budget) completes.
    let model = Arc::new(ScriptedStreams::new(vec![
        reasoning_then_text(4_000, "late"),
        reasoning_then_text(0, "recovered"),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_policy(RunPolicy {
        default_reasoning: Some(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            budget_tokens: Some(200),
            summary: None,
        }),
        ..RunPolicy::default()
    });
    let ctx = RunContext::new(RunConfig::new("watchdog-budget"), ());
    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the run finishes");
    assert_eq!(run.text(), Some("recovered".to_string()));
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn the_watchdog_can_be_switched_off() {
    let model = Arc::new(ScriptedStreams::new(vec![reasoning_then_text(
        4_000,
        "late answer",
    )]));
    let harness = harness_with(Arc::clone(&model), ReasoningWatchdog::Off);
    let ctx = RunContext::new(RunConfig::new("watchdog-off"), ());
    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the run finishes");
    assert_eq!(run.text(), Some("late answer".to_string()));
    assert_eq!(model.requests().len(), 1);
}

#[tokio::test]
async fn the_watchdog_hands_the_interrupted_reasoning_to_the_retry() {
    let model = Arc::new(ScriptedStreams::new(vec![
        reasoning_then_text(4_000, "late answer"),
        reasoning_then_text(0, "recovered"),
    ]));
    let harness = harness_with(Arc::clone(&model), ReasoningWatchdog::Tokens(200));
    let ctx = RunContext::new(RunConfig::new("watchdog-carry"), ());
    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the run finishes");
    assert_eq!(run.text(), Some("recovered".to_string()));
    let last = model.requests()[1]
        .messages
        .last()
        .map(|m| m.text())
        .unwrap_or_default();
    assert!(
        last.contains("Continue from this point") && last.contains("xxxxxxxx"),
        "the streamed reasoning the watchdog cut off rides into the retry: {last:.100}"
    );
}
