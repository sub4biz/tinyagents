//! Tests for [`ContextCompressionMiddleware`]'s overflow recovery v2: several
//! compaction attempts per call, overflow detected from a successful response,
//! and the cheaper truncate-tool-results route.

#[allow(unused_imports)]
use super::*;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::{AgentEvent, RecordingListener};
use crate::middleware::{
    BoxModelFuture, ContextCompressionMiddleware, Middleware, MiddlewareStack, ModelBaseCall,
};
use crate::summarization::{
    CompressionProvenance, ResponseOverflowDetection, SummarizationPolicy, Summarizer,
    SummaryRecord,
};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, UserMessage};
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

fn ctx() -> RunContext {
    RunContext::new(RunConfig::new("test-run"), ())
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: vec![ContentBlock::Text(text.to_string())],
    })
}

fn response(usage: Option<Usage>, finish_reason: Option<&str>) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("ok".to_string())],
            tool_calls: Vec::new(),
            usage: None,
            origin: None,
        },
        usage,
        finish_reason: finish_reason.map(str::to_string),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn input_usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage {
        input_tokens,
        output_tokens,
        ..Usage::default()
    }
}

/// A provider overflow error that reports no numbers, so the kept tail is
/// sized from the transcript rather than from a (tiny) stated limit.
fn overflow_error() -> TinyAgentsError {
    TinyAgentsError::Model("context_length_exceeded".to_string())
}

type Responder = Box<dyn Fn(usize, &ModelRequest) -> Result<ModelResponse> + Send + Sync>;

/// Records every request and answers with `respond(call_number, request)`.
struct ScriptedBase {
    requests: Arc<Mutex<Vec<ModelRequest>>>,
    respond: Responder,
}

impl ScriptedBase {
    fn new(
        respond: impl Fn(usize, &ModelRequest) -> Result<ModelResponse> + Send + Sync + 'static,
    ) -> Self {
        Self {
            requests: Arc::default(),
            respond: Box::new(respond),
        }
    }

    fn calls(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

impl ModelBaseCall<(), ()> for ScriptedBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext,
        _state: &'a (),
        request: ModelRequest,
    ) -> BoxModelFuture<'a> {
        Box::pin(async move {
            let n = {
                let mut requests = self.requests.lock().unwrap();
                requests.push(request.clone());
                requests.len()
            };
            (self.respond)(n, &request)
        })
    }
}

/// Answers every request with a one-line summary and counts the requests.
#[derive(Clone, Default)]
struct ShortSummarizer {
    calls: Arc<Mutex<usize>>,
}

#[async_trait]
impl Summarizer for ShortSummarizer {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        *self.calls.lock().unwrap() += 1;
        Ok(record("short summary"))
    }
}

/// A summary larger than anything it replaces.
struct HugeSummarizer;

#[async_trait]
impl Summarizer for HugeSummarizer {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        Ok(record(&"y".repeat(200_000)))
    }
}

/// The first summary is tiny; each later one is far bigger than the history.
#[derive(Default)]
struct GrowingSummarizer {
    calls: Mutex<usize>,
}

#[async_trait]
impl Summarizer for GrowingSummarizer {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        let n = {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            *calls
        };
        Ok(record(&"y".repeat(if n == 1 { 10 } else { 50_000 * n })))
    }
}

fn record(text: &str) -> SummaryRecord {
    SummaryRecord {
        summary: Message::system(text),
        provenance: CompressionProvenance {
            source_ids: Vec::new(),
            original_token_estimate: 0,
            summary_token_estimate: 0,
            reason: "test".into(),
        },
        usage: None,
    }
}

/// 16 turns of ~75 tokens each.
fn long_transcript() -> Vec<Message> {
    (0..16)
        .map(|i| {
            if i % 2 == 0 {
                user(&format!("turn {i} {}", "word ".repeat(60)))
            } else {
                Message::assistant(format!("turn {i} {}", "word ".repeat(60)))
            }
        })
        .collect()
}

fn roomy_policy() -> SummarizationPolicy {
    SummarizationPolicy::default().with_context_window(10_000)
}

fn short_mw(policy: SummarizationPolicy) -> ContextCompressionMiddleware {
    ContextCompressionMiddleware::with_summarizer(policy, Box::new(ShortSummarizer::default()))
}

fn stack_of(mw: ContextCompressionMiddleware) -> MiddlewareStack<()> {
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(mw));
    stack
}

async fn run(
    stack: &MiddlewareStack<()>,
    base: &ScriptedBase,
    messages: Vec<Message>,
) -> Result<ModelResponse> {
    let mut c = ctx();
    stack
        .run_wrapped_model(
            &mut c,
            &(),
            ModelRequest {
                messages,
                ..Default::default()
            },
            base,
        )
        .await
        .map(|outcome| outcome.into_response())
}

fn compacted_count(recorder: &RecordingListener) -> usize {
    recorder
        .events()
        .into_iter()
        .filter(|r| matches!(r.event, AgentEvent::Compacted { .. }))
        .count()
}

// ── several attempts, each of which must shrink ───────────────────────────────

#[tokio::test]
async fn a_persistent_overflow_gets_three_compaction_attempts_by_default() {
    let base = ScriptedBase::new(|_, _| Err(overflow_error()));
    let summarizer = ShortSummarizer::default();
    let stack = stack_of(ContextCompressionMiddleware::with_summarizer(
        roomy_policy(),
        Box::new(summarizer.clone()),
    ));

    let result = run(&stack, &base, long_transcript()).await;

    assert!(result.is_err());
    assert_eq!(base.calls(), 4, "the original call plus three retries");
    assert_eq!(*summarizer.calls.lock().unwrap(), 3);
    // Each retry carries a smaller request than the one before it.
    let sizes: Vec<u64> = base
        .requests
        .lock()
        .unwrap()
        .iter()
        .map(|r| crate::token_estimation::count_tokens_approximately(&r.messages))
        .collect();
    assert!(sizes.windows(2).all(|w| w[1] < w[0]), "sizes {sizes:?}");
}

#[tokio::test]
async fn the_attempt_budget_is_configurable() {
    let base = ScriptedBase::new(|_, _| Err(overflow_error()));
    let stack = stack_of(short_mw(roomy_policy()).with_max_overflow_attempts(1));
    assert!(run(&stack, &base, long_transcript()).await.is_err());
    assert_eq!(base.calls(), 2);

    let base = ScriptedBase::new(|_, _| Err(overflow_error()));
    let stack = stack_of(short_mw(roomy_policy()).with_max_overflow_attempts(0));
    assert!(run(&stack, &base, long_transcript()).await.is_err());
    assert_eq!(base.calls(), 1, "zero attempts disables recovery");
}

#[tokio::test]
async fn recovery_stops_at_the_first_attempt_that_succeeds() {
    let base = ScriptedBase::new(|n, _| {
        if n <= 2 {
            Err(overflow_error())
        } else {
            Ok(response(None, Some("stop")))
        }
    });
    let stack = stack_of(short_mw(roomy_policy()));
    assert!(run(&stack, &base, long_transcript()).await.is_ok());
    assert_eq!(base.calls(), 3);
}

#[tokio::test]
async fn a_later_attempt_that_does_not_shrink_the_request_ends_recovery() {
    let base = ScriptedBase::new(|_, _| Err(overflow_error()));
    let stack = stack_of(ContextCompressionMiddleware::with_summarizer(
        roomy_policy(),
        Box::new(GrowingSummarizer::default()),
    ));
    let result = run(&stack, &base, long_transcript()).await;
    assert!(matches!(result, Err(TinyAgentsError::Model(_))));
    // The first attempt shrinks and is sent; the second's summary is bigger
    // than its input, so it is never sent.
    assert_eq!(base.calls(), 2);
}

// ── overflow reported by a successful response ────────────────────────────────

fn silent_overflow_then_ok() -> ScriptedBase {
    ScriptedBase::new(|n, _| {
        let tokens = if n == 1 { 9_000 } else { 50 };
        Ok(response(Some(input_usage(tokens, 20)), Some("stop")))
    })
}

fn small_window() -> SummarizationPolicy {
    SummarizationPolicy::default()
        .with_context_window(100)
        .with_threshold_fraction(0.5)
}

#[tokio::test]
async fn usage_above_the_window_on_a_successful_response_is_recovered() {
    let base = silent_overflow_then_ok();
    let stack = stack_of(
        short_mw(small_window()).with_response_overflow_detection(ResponseOverflowDetection::Usage),
    );
    let response = run(&stack, &base, long_transcript()).await.unwrap();
    assert_eq!(response.usage.unwrap().input_tokens, 50);
    assert_eq!(base.calls(), 2);
}

#[tokio::test]
async fn response_overflow_detection_can_be_switched_off() {
    let base = silent_overflow_then_ok();
    let stack = stack_of(
        short_mw(small_window()).with_response_overflow_detection(ResponseOverflowDetection::Off),
    );
    let response = run(&stack, &base, long_transcript()).await.unwrap();
    assert_eq!(response.usage.unwrap().input_tokens, 9_000);
    assert_eq!(base.calls(), 1);
}

#[tokio::test]
async fn a_length_stop_far_below_the_cap_is_recovered_only_when_opted_in() {
    let respond = |n: usize, _: &ModelRequest| {
        if n == 1 {
            Ok(response(Some(input_usage(60, 5)), Some("length")))
        } else {
            Ok(response(Some(input_usage(30, 400)), Some("stop")))
        }
    };
    let capped = |messages| ModelRequest {
        messages,
        max_tokens: Some(1_000),
        ..Default::default()
    };

    let base = ScriptedBase::new(respond);
    let stack = stack_of(short_mw(small_window()));
    let mut c = ctx();
    stack
        .run_wrapped_model(&mut c, &(), capped(long_transcript()), &base)
        .await
        .unwrap();
    assert_eq!(
        base.calls(),
        1,
        "default detection leaves a short length stop alone"
    );

    let base = ScriptedBase::new(respond);
    let stack = stack_of(
        short_mw(small_window())
            .with_response_overflow_detection(ResponseOverflowDetection::UsageAndShortLength),
    );
    let mut c = ctx();
    let out = stack
        .run_wrapped_model(&mut c, &(), capped(long_transcript()), &base)
        .await
        .unwrap()
        .into_response();
    assert_eq!(base.calls(), 2);
    assert_eq!(out.finish_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn an_unrecoverable_response_overflow_is_returned_not_turned_into_an_error() {
    // One message: nothing can be compacted, so the response stands.
    let base = silent_overflow_then_ok();
    let stack = stack_of(
        short_mw(small_window()).with_response_overflow_detection(ResponseOverflowDetection::Usage),
    );
    let response = run(&stack, &base, vec![user("hello")]).await.unwrap();
    assert_eq!(response.usage.unwrap().input_tokens, 9_000);
    assert_eq!(base.calls(), 1);
}

// ── the cheaper first route: truncate oversized tool results ──────────────────

fn call_and_result(id: &str, result_bytes: usize) -> Vec<Message> {
    vec![
        Message::Assistant(AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, "read", json!({"path": "a.txt"}))],
            usage: None,
            origin: None,
        }),
        Message::tool(id, "z".repeat(result_bytes)),
    ]
}

fn tool_text_len(request: &ModelRequest) -> usize {
    request
        .messages
        .iter()
        .filter(|m| matches!(m, Message::Tool(_)))
        .map(|m| m.text().len())
        .max()
        .unwrap_or(0)
}

/// Rejects any request still carrying a tool result over 3 000 bytes.
fn picky_base(error: &'static str) -> ScriptedBase {
    ScriptedBase::new(move |_, request| {
        if tool_text_len(request) > 3_000 {
            Err(TinyAgentsError::Model(error.to_string()))
        } else {
            Ok(response(None, Some("stop")))
        }
    })
}

const OVERFLOW_BY_2900: &str = "This model's maximum context length is 100 tokens. \
     However, your messages resulted in 3000 tokens.";

#[tokio::test]
async fn truncating_tool_results_alone_recovers_without_a_summary() {
    let summarizer = ShortSummarizer::default();
    let mw =
        ContextCompressionMiddleware::with_summarizer(roomy_policy(), Box::new(summarizer.clone()))
            .with_tool_result_truncation(2_000);
    let stack = stack_of(mw);
    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());
    let base = picky_base(OVERFLOW_BY_2900);

    let mut messages = vec![user("read it")];
    messages.extend(call_and_result("c1", 40_000));
    stack
        .run_wrapped_model(
            &mut c,
            &(),
            ModelRequest {
                messages,
                ..Default::default()
            },
            &base,
        )
        .await
        .unwrap();

    assert_eq!(base.calls(), 2);
    assert_eq!(
        *summarizer.calls.lock().unwrap(),
        0,
        "no summary was bought"
    );
    assert_eq!(compacted_count(&recorder), 0);
    assert!(tool_text_len(&base.requests.lock().unwrap()[1]) < 3_000);
}

#[tokio::test]
async fn without_a_truncation_cap_the_same_overflow_compacts() {
    let summarizer = ShortSummarizer::default();
    let stack = stack_of(ContextCompressionMiddleware::with_summarizer(
        roomy_policy(),
        Box::new(summarizer.clone()),
    ));
    let base = picky_base(OVERFLOW_BY_2900);
    let mut messages = long_transcript();
    messages.extend(call_and_result("c1", 40_000));
    // The compaction keeps the huge result in the tail, so the retry still
    // overflows: only the route differs, which is what this checks.
    let _ = run(&stack, &base, messages).await;
    assert!(*summarizer.calls.lock().unwrap() >= 1);
}

#[tokio::test]
async fn when_truncation_cannot_cover_the_overflow_it_follows_the_compaction() {
    // Overflow of ~19 900 tokens against ~9 500 reducible tokens.
    let error = "This model's maximum context length is 100 tokens. \
         However, your messages resulted in 20000 tokens.";
    let summarizer = ShortSummarizer::default();
    let stack = stack_of(
        ContextCompressionMiddleware::with_summarizer(roomy_policy(), Box::new(summarizer.clone()))
            .with_tool_result_truncation(2_000),
    );
    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());
    let base = picky_base(error);

    let mut messages = long_transcript();
    messages.extend(call_and_result("c1", 40_000));
    stack
        .run_wrapped_model(
            &mut c,
            &(),
            ModelRequest {
                messages,
                ..Default::default()
            },
            &base,
        )
        .await
        .unwrap();

    assert_eq!(compacted_count(&recorder), 1);
    let requests = base.requests.lock().unwrap();
    assert!(tool_text_len(requests.last().unwrap()) < 3_000);
}

/// A summary of a fixed size.
struct SizedSummarizer(usize);

#[async_trait]
impl Summarizer for SizedSummarizer {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        Ok(record(&"s".repeat(self.0)))
    }
}

#[tokio::test]
async fn a_compaction_is_judged_against_the_truncated_request_actually_sent() {
    // Truncation engages first (the retry carries a 200-byte result). A later
    // compaction whose summary is smaller than the raw 40 KB result but larger
    // than what is now sent would grow the outgoing request: it is refused.
    let base = ScriptedBase::new(|_, _| Err(TinyAgentsError::Model(OVERFLOW_BY_2900.to_string())));
    let stack = stack_of(
        ContextCompressionMiddleware::with_summarizer(
            roomy_policy(),
            Box::new(SizedSummarizer(12_000)),
        )
        .with_tool_result_truncation(200),
    );
    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());
    let mut messages = long_transcript();
    messages.extend(call_and_result("c1", 40_000));
    let result = stack
        .run_wrapped_model(
            &mut c,
            &(),
            ModelRequest {
                messages,
                ..Default::default()
            },
            &base,
        )
        .await;
    assert!(result.is_err());
    assert_eq!(compacted_count(&recorder), 0, "no growing compaction");
    // The first request is the raw one; every retry after truncation engaged
    // carries the cut result.
    for request in base.requests.lock().unwrap().iter().skip(1) {
        assert!(tool_text_len(request) < 3_000);
    }
}

#[tokio::test]
async fn a_response_discarded_for_the_truncation_fallback_is_still_accounted() {
    // Nothing to compact (one user message and one tool exchange), so the
    // mixed route falls back to cutting the tool results: the discarded
    // response was billed and its usage must be recorded.
    let base = ScriptedBase::new(|n, _| {
        let tokens = if n == 1 { 9_000 } else { 50 };
        Ok(response(Some(input_usage(tokens, 20)), Some("stop")))
    });
    let stack = stack_of(
        ContextCompressionMiddleware::with_summarizer(
            truncating_policy(),
            Box::new(ShortSummarizer::default()),
        )
        .with_response_overflow_detection(ResponseOverflowDetection::Usage)
        .with_before_compaction(|_| crate::summarization::CompactionDecision::Decline)
        .with_tool_result_truncation(400),
    );
    let mut messages = vec![user("read it")];
    messages.extend(call_and_result("c1", 10_000));
    let mut c = ctx();
    let response = stack
        .run_wrapped_model(
            &mut c,
            &(),
            ModelRequest {
                messages,
                ..Default::default()
            },
            &base,
        )
        .await
        .unwrap()
        .into_response();
    assert_eq!(response.usage.unwrap().input_tokens, 50);
    assert_eq!(base.calls(), 2);
    let discarded = c.take_discarded_usage();
    assert_eq!(discarded.len(), 1, "the first response's usage is kept");
    assert_eq!(discarded[0].input_tokens, 9_000);
}

#[tokio::test]
async fn the_mixed_route_cuts_the_newest_result_again_after_compacting() {
    // Truncation alone cannot reach the trigger, so the request is compacted;
    // the compaction splice rebuilds it from the untruncated transcript, and
    // the newest (otherwise spared) oversized result must be cut again.
    let summarizer = ShortSummarizer::default();
    let mw = ContextCompressionMiddleware::with_summarizer(
        truncating_policy(),
        Box::new(summarizer.clone()),
    )
    .with_keep_recent_tokens(3_000)
    .with_tool_result_truncation(400);
    let mut messages = long_transcript();
    messages.extend(long_transcript());
    messages.extend(long_transcript());
    messages.extend(call_and_result("c1", 10_000));
    let mut request = ModelRequest {
        messages,
        ..Default::default()
    };
    let mut c = ctx();
    let epoch = c.prompt_prefix_epoch();
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut request)
        .await
        .unwrap();
    assert!(*summarizer.calls.lock().unwrap() >= 1, "it compacted");
    assert!(
        request
            .messages
            .iter()
            .any(|m| matches!(m, Message::Tool(_))),
        "the result stayed in the kept tail"
    );
    assert!(tool_text_len(&request) < 1_000, "and the result is cut");
    assert_ne!(
        c.prompt_prefix_epoch(),
        epoch,
        "the compaction and its cut declare the rewritten prefix"
    );
}

#[tokio::test]
async fn a_truncation_route_that_falls_short_still_cuts_the_result_after_compacting() {
    // The route estimate (bare chars / 4) says cutting the result is enough,
    // but the measured request (per-message framing over many tiny messages)
    // is still over the trigger, so it compacts. The splice rebuilds the
    // request from the untruncated transcript and must cut the result again.
    let summarizer = ShortSummarizer::default();
    let mw = ContextCompressionMiddleware::with_summarizer(
        truncating_policy(),
        Box::new(summarizer.clone()),
    )
    .with_keep_recent_tokens(3_000)
    .with_tool_result_truncation(400);
    let mut messages: Vec<Message> = (0..500)
        .map(|i| {
            if i % 2 == 0 {
                user("a")
            } else {
                Message::assistant("b")
            }
        })
        .collect();
    messages.extend(call_and_result("c1", 10_000));
    let mut request = ModelRequest {
        messages,
        ..Default::default()
    };
    let mut c = ctx();
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut request)
        .await
        .unwrap();
    assert!(*summarizer.calls.lock().unwrap() >= 1, "it compacted");
    assert!(
        request
            .messages
            .iter()
            .any(|m| matches!(m, Message::Tool(_))),
        "the result stayed in the kept tail"
    );
    assert!(tool_text_len(&request) < 1_000, "and the result is cut");
}

#[tokio::test]
async fn the_mixed_route_reports_the_size_of_the_request_after_the_final_cut() {
    let mw = ContextCompressionMiddleware::with_summarizer(
        truncating_policy(),
        Box::new(ShortSummarizer::default()),
    )
    .with_keep_recent_tokens(3_000)
    .with_tool_result_truncation(400);
    let mut messages = long_transcript();
    messages.extend(long_transcript());
    messages.extend(long_transcript());
    messages.extend(call_and_result("c1", 10_000));
    let mut request = ModelRequest {
        messages,
        ..Default::default()
    };
    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut request)
        .await
        .unwrap();
    let sent = crate::token_estimation::count_tokens_approximately(&request.messages);
    let reported: Vec<u64> = recorder
        .events()
        .into_iter()
        .filter_map(|r| match r.event {
            AgentEvent::Compressed { to_tokens, .. } => Some(to_tokens),
            _ => None,
        })
        .collect();
    assert!(reported.contains(&sent), "{reported:?} vs sent {sent}");
}

#[tokio::test]
async fn aging_out_an_oversized_result_moves_the_prefix_epoch_only_when_it_cuts() {
    let mw =
        ContextCompressionMiddleware::new(truncating_policy()).with_tool_result_truncation(400);
    let mut messages = vec![user("read it")];
    messages.extend(call_and_result("c1", 10_000));
    let mut c = ctx();
    let mut first = ModelRequest {
        messages: messages.clone(),
        ..Default::default()
    };
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut first)
        .await
        .unwrap();
    messages.push(Message::assistant("noted"));
    messages.push(user("and then?"));
    let epoch = c.prompt_prefix_epoch();
    let mut second = ModelRequest {
        messages: messages.clone(),
        ..Default::default()
    };
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut second)
        .await
        .unwrap();
    assert_ne!(c.prompt_prefix_epoch(), epoch, "the rewrite was declared");
    // A request with nothing left to cut leaves the epoch alone.
    let epoch = c.prompt_prefix_epoch();
    let mut small = ModelRequest {
        messages: vec![user("hi")],
        ..Default::default()
    };
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut small)
        .await
        .unwrap();
    assert_eq!(c.prompt_prefix_epoch(), epoch);
}

// ── preemptive route (before_model) ───────────────────────────────────────────

fn truncating_policy() -> SummarizationPolicy {
    // 4 000-token window, 50% trigger: a 2 000-token budget, above the 512
    // tokens of headroom the truncation route insists on.
    SummarizationPolicy::default()
        .with_context_window(4_000)
        .with_threshold_fraction(0.5)
}

#[tokio::test]
async fn an_over_trigger_prompt_with_a_huge_tool_result_is_truncated_not_summarized() {
    let summarizer = ShortSummarizer::default();
    let mw = ContextCompressionMiddleware::with_summarizer(
        truncating_policy(),
        Box::new(summarizer.clone()),
    )
    .with_tool_result_truncation(400);
    let mut messages = vec![user("read it")];
    messages.extend(call_and_result("c1", 10_000));
    let mut request = ModelRequest {
        messages,
        ..Default::default()
    };
    let mut c = ctx();
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut request)
        .await
        .unwrap();

    assert_eq!(*summarizer.calls.lock().unwrap(), 0);
    assert!(tool_text_len(&request) < 1_000);
    assert!(request.messages.len() == 3, "no history was folded");
}

#[tokio::test]
async fn once_truncation_engages_it_stays_applied_for_the_run() {
    let mw =
        ContextCompressionMiddleware::new(truncating_policy()).with_tool_result_truncation(400);
    let mut messages = vec![user("read it")];
    messages.extend(call_and_result("c1", 10_000));
    let mut c = ctx();

    let mut first = ModelRequest {
        messages: messages.clone(),
        ..Default::default()
    };
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut first)
        .await
        .unwrap();

    // The loop rebuilds the next request from the untruncated transcript; the
    // model has answered since, so the result is no longer the newest.
    messages.push(Message::assistant("noted"));
    messages.push(user("and then?"));
    let mut second = ModelRequest {
        messages,
        ..Default::default()
    };
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut second)
        .await
        .unwrap();
    assert!(tool_text_len(&second) < 1_000, "still truncated");
}

#[tokio::test]
async fn truncating_mode_spares_the_results_the_model_just_asked_for() {
    let mw = ContextCompressionMiddleware::new(
        SummarizationPolicy::default()
            .with_context_window(8_000)
            .with_threshold_fraction(0.5),
    )
    .with_tool_result_truncation(400);
    let mut messages = vec![user("read it")];
    messages.extend(call_and_result("c1", 20_000));
    let mut c = ctx();
    let mut first = ModelRequest {
        messages: messages.clone(),
        ..Default::default()
    };
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut first)
        .await
        .unwrap();
    assert!(tool_text_len(&first) < 1_000, "the route itself cut it");

    // Next call: a second result arrived after the last assistant message.
    messages.push(Message::assistant("noted"));
    messages.extend(call_and_result("c2", 10_000));
    let mut second = ModelRequest {
        messages,
        ..Default::default()
    };
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut second)
        .await
        .unwrap();
    let lens: Vec<usize> = second
        .messages
        .iter()
        .filter(|m| matches!(m, Message::Tool(_)))
        .map(|m| m.text().len())
        .collect();
    assert!(lens[0] < 1_000, "older result stays cut: {lens:?}");
    assert_eq!(lens[1], 10_000, "newest result is whole: {lens:?}");
}

#[tokio::test]
async fn a_prompt_under_the_trigger_is_never_truncated() {
    let mw = short_mw(roomy_policy()).with_tool_result_truncation(400);
    let mut messages = vec![user("read it")];
    messages.extend(call_and_result("c1", 8_000));
    let original = messages.clone();
    let mut request = ModelRequest {
        messages,
        ..Default::default()
    };
    let mut c = ctx();
    Middleware::<(), ()>::before_model(&mw, &mut c, &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages, original);
}

// ── streamed calls ────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_streamed_response_is_never_discarded_for_its_usage() {
    // Its deltas already reached the consumer; a retry would stream the answer
    // twice. Error-level recovery is unaffected.
    let base = silent_overflow_then_ok();
    let stack = stack_of(
        short_mw(small_window()).with_response_overflow_detection(ResponseOverflowDetection::Usage),
    );
    let mut c = ctx();
    c.call_streamed = true;
    let out = stack
        .run_wrapped_model(
            &mut c,
            &(),
            ModelRequest {
                messages: long_transcript(),
                ..Default::default()
            },
            &base,
        )
        .await
        .unwrap()
        .into_response();
    assert_eq!(out.usage.unwrap().input_tokens, 9_000);
    assert_eq!(base.calls(), 1);
}

#[tokio::test]
async fn a_first_attempt_that_grows_the_request_is_not_used_or_persisted() {
    struct Sink(Mutex<usize>);
    impl crate::summarization::CompactionSink for Sink {
        fn persist(&self, _record: &crate::summarization::CompactionRecord) -> Result<()> {
            *self.0.lock().unwrap() += 1;
            Ok(())
        }
    }
    let base = ScriptedBase::new(|_, _| Err(overflow_error()));
    let stack = stack_of(ContextCompressionMiddleware::with_summarizer(
        roomy_policy(),
        Box::new(HugeSummarizer),
    ));
    let sink = Arc::new(Sink(Mutex::new(0)));
    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx().with_compaction_sink(sink.clone());
    c.events.subscribe(recorder.clone());
    let result = stack
        .run_wrapped_model(
            &mut c,
            &(),
            ModelRequest {
                messages: long_transcript(),
                ..Default::default()
            },
            &base,
        )
        .await;
    assert!(result.is_err());
    assert_eq!(base.calls(), 1, "a request that grew is never sent");
    assert_eq!(*sink.0.lock().unwrap(), 0, "no boundary persisted");
    assert_eq!(compacted_count(&recorder), 0);
}

#[tokio::test]
async fn an_overflow_compaction_marks_the_prompt_prefix_as_changed() {
    // The prompt cache guard keys its miss accounting on this, so the smaller
    // prompt after a compaction is not reported as a miss.
    let base = ScriptedBase::new(|n, _| {
        if n == 1 {
            Err(overflow_error())
        } else {
            Ok(response(None, Some("stop")))
        }
    });
    let stack = stack_of(short_mw(roomy_policy()));
    let mut c = ctx();
    let before = c.prompt_prefix_epoch();
    stack
        .run_wrapped_model(
            &mut c,
            &(),
            ModelRequest {
                messages: long_transcript(),
                ..Default::default()
            },
            &base,
        )
        .await
        .unwrap();
    assert_ne!(c.prompt_prefix_epoch(), before);
}
