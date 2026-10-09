//! Tests for the middleware stack and built-in middleware.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::{AgentEvent, RecordingListener};
use crate::summarization::{
    CompactionContext, CompactionDecision, CompactionRecord, CompactionSink, SummarizationPolicy,
    Summarizer, SummaryRecord, TrimStrategy,
};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, UserMessage};
use tinyinference_llm::model::{ModelRequest, ModelResponse, PromptSegment, SegmentRole};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::ToolResult;

// ── helpers ───────────────────────────────────────────────────────────────────

fn ctx() -> RunContext {
    RunContext::new(RunConfig::new("test-run"), ())
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: vec![ContentBlock::Text(text.to_string())],
    })
}

fn response_with_usage(usage: Usage) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("ok".to_string())],
            tool_calls: Vec::new(),
            usage: None,
            origin: None,
        },
        usage: Some(usage),
        finish_reason: None,
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn segment(id: &str, role: SegmentRole, cacheable: bool) -> PromptSegment {
    PromptSegment {
        id: id.to_string(),
        role,
        cacheable,
    }
}

/// Records hook firing order into a shared log for ordering assertions.
struct OrderRecorder {
    label: &'static str,
    log: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<()> for OrderRecorder {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:before", self.label));
        Ok(())
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:after", self.label));
        Ok(())
    }
}

/// Always fails its `before_model` hook to exercise short-circuiting.
struct FailingMiddleware;

#[async_trait]
impl Middleware<()> for FailingMiddleware {
    fn name(&self) -> &str {
        "failing"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> Result<()> {
        Err(TinyAgentsError::Middleware("boom".to_string()))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn before_runs_forward_after_runs_reverse() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(OrderRecorder {
        label: "a",
        log: log.clone(),
    }));
    stack.push(Arc::new(OrderRecorder {
        label: "b",
        log: log.clone(),
    }));

    let mut c = ctx();
    let mut request = ModelRequest::default();
    let mut response = response_with_usage(Usage::new(1, 1));

    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();
    stack
        .run_after_model(&mut c, &(), &mut response)
        .await
        .unwrap();

    let order = log.lock().unwrap().clone();
    assert_eq!(
        order,
        vec!["a:before", "b:before", "b:after", "a:after"],
        "before runs in registration order, after runs reversed"
    );
}

#[tokio::test]
async fn error_short_circuits_and_invokes_on_error() {
    let logging = Arc::new(LoggingMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(logging.clone());
    stack.push(Arc::new(FailingMiddleware));
    // This third middleware must never run because the second one fails first.
    let never = Arc::new(LoggingMiddleware::with_label("never"));
    stack.push(never.clone());

    let mut c = ctx();
    let mut request = ModelRequest::default();
    let result = stack.run_before_model(&mut c, &(), &mut request).await;

    assert!(matches!(result, Err(TinyAgentsError::Middleware(_))));
    // on_error fanned out to the whole stack, so the first logging mw saw it.
    assert_eq!(logging.counts().on_error, 1);
    // The first logging mw's before_model ran; the one after the failure did not.
    assert_eq!(logging.counts().before_model, 1);
    assert_eq!(never.counts().before_model, 0);
}

#[tokio::test]
async fn emits_started_and_completed_events() {
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(LoggingMiddleware::new()));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest::default();
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    let kinds: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert_eq!(
        kinds,
        vec![
            AgentEvent::MiddlewareStarted {
                name: "logging".to_string(),
                call_id: None,
            },
            AgentEvent::MiddlewareCompleted {
                name: "logging".to_string(),
                call_id: None,
            },
        ]
    );
}

#[tokio::test]
async fn failing_hook_still_emits_balanced_completed_event() {
    // A hook that returns `Err` must still close its `MiddlewareStarted` with a
    // matching `MiddlewareCompleted`, so downstream observers never see a
    // dangling, unbalanced `Started`.
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(FailingMiddleware));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest::default();
    let result = stack.run_before_model(&mut c, &(), &mut request).await;
    assert!(matches!(result, Err(TinyAgentsError::Middleware(_))));

    let brackets: Vec<AgentEvent> = recorder
        .events()
        .into_iter()
        .map(|r| r.event)
        .filter(|e| {
            matches!(
                e,
                AgentEvent::MiddlewareStarted { .. } | AgentEvent::MiddlewareCompleted { .. }
            )
        })
        .collect();
    assert_eq!(
        brackets,
        vec![
            AgentEvent::MiddlewareStarted {
                name: "failing".to_string(),
                call_id: None,
            },
            AgentEvent::MiddlewareCompleted {
                name: "failing".to_string(),
                call_id: None,
            },
        ],
        "a failing hook must emit a balanced Started/Completed pair"
    );
}

/// I-3 regression: `run_stack_hook!` must emit `AgentEvent::MiddlewareFailed`
/// for a hook that returns `Err`, not just fan `on_error` out privately. The
/// variant existed but nothing in the stack emitted it before this fix.
#[tokio::test]
async fn failing_hook_emits_middleware_failed() {
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(FailingMiddleware));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest::default();
    let _ = stack.run_before_model(&mut c, &(), &mut request).await;

    let failed: Vec<AgentEvent> = recorder
        .events()
        .into_iter()
        .map(|r| r.event)
        .filter(|e| matches!(e, AgentEvent::MiddlewareFailed { .. }))
        .collect();
    assert_eq!(
        failed,
        vec![AgentEvent::MiddlewareFailed {
            name: "failing".to_string(),
            error: TinyAgentsError::Middleware("boom".to_string()).to_string(),
        }],
    );
}

#[tokio::test]
async fn on_model_delta_hook_emits_no_bracketing_events() {
    // The per-delta hook runs on the streaming hot path, so it must NOT emit
    // `MiddlewareStarted`/`MiddlewareCompleted` events the way the other stack
    // runners do — those two events per middleware per token dominated the
    // stream loop for no observability value.
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(LoggingMiddleware::new()));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut delta = ModelDelta {
        call_id: "call-1".to_string(),
        content: "tok".to_string(),
        reasoning: String::new(),
        tool_call: None,
    };
    stack
        .run_on_model_delta(&mut c, &(), &mut delta)
        .await
        .unwrap();

    let bracketing = recorder
        .events()
        .into_iter()
        .filter(|r| {
            matches!(
                r.event,
                AgentEvent::MiddlewareStarted { .. } | AgentEvent::MiddlewareCompleted { .. }
            )
        })
        .count();
    assert_eq!(
        bracketing, 0,
        "the delta hook must not bracket middleware with events"
    );
}

#[tokio::test]
async fn on_tool_delta_hook_emits_no_bracketing_events() {
    // M-12 regression: `run_on_tool_delta` was the one delta hook still
    // routed through `run_stack_hook!`, so it emitted
    // `MiddlewareStarted`/`MiddlewareCompleted` on every streamed
    // tool-progress delta while `run_on_model_delta` (the sibling hook, same
    // hot-path shape) did not. The two delta hooks must agree.
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(LoggingMiddleware::new()));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut delta = tinyinference_llm::tool::ToolDelta {
        call_id: "call-1".to_string(),
        content: "partial args".to_string(),
        tool_name: Some("search".to_string()),
        ..Default::default()
    };
    stack
        .run_on_tool_delta(&mut c, &(), &mut delta)
        .await
        .unwrap();

    let bracketing = recorder
        .events()
        .into_iter()
        .filter(|r| {
            matches!(
                r.event,
                AgentEvent::MiddlewareStarted { .. } | AgentEvent::MiddlewareCompleted { .. }
            )
        })
        .count();
    assert_eq!(
        bracketing, 0,
        "the tool-delta hook must not bracket middleware with events"
    );
}

#[tokio::test]
async fn message_trim_middleware_shrinks_request() {
    let mw = MessageTrimMiddleware::new(TrimStrategy::KeepLast(1));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut request = ModelRequest {
        messages: vec![user("one"), user("two"), user("three")],
        ..Default::default()
    };
    let mut c = ctx();
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0], user("three"));
}

#[tokio::test]
async fn context_compression_is_noop_below_window_threshold() {
    // 1000-token window, 0.9 threshold → 900-token budget. A tiny transcript
    // stays far below it, so the middleware must leave messages untouched and
    // emit no Compressed event.
    let policy = SummarizationPolicy::default()
        .with_context_window(1000)
        .with_threshold_fraction(0.9);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let before = vec![user("one"), user("two"), user("three")];
    let mut request = ModelRequest {
        messages: before.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Messages unchanged.
    assert_eq!(request.messages, before);
    // No record produced.
    assert!(mw.records().is_empty());
    // No Compressed event emitted (only the stack's started/completed events).
    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::Compressed { .. })),
    );
}

#[tokio::test]
async fn context_compression_compresses_at_or_above_threshold() {
    // 100-token window, 0.5 threshold → 50-token budget. keep_last=1 keeps the
    // newest message verbatim; everything older is summarized.
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    // Three ~50-token (200-char) messages → ~150 tokens, well above the 50 budget.
    let big = "a".repeat(200);
    let mut request = ModelRequest {
        messages: vec![
            user(&format!("{big}-1")),
            user(&format!("{big}-2")),
            user(&format!("{big}-3")),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Compressed to: one summary message + the single kept recent message.
    // The summary is a user-role, reference-only checkpoint by default.
    assert_eq!(request.messages.len(), 2);
    assert!(matches!(request.messages[0], Message::User(_)));
    assert!(crate::summarization::is_checkpoint(&request.messages[0]));
    assert_eq!(request.messages[1].text(), format!("{big}-3"));

    // Provenance recorded: the two oldest messages were the source.
    let records = mw.records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].provenance.source_ids, vec!["msg-0", "msg-1"]);
    assert!(records[0].provenance.original_token_estimate > 0);

    // A single Compressed event was emitted. (ConcatSummarizer keeps text
    // verbatim, so the guarantee is fewer messages, not fewer tokens; the event
    // still reports the before/after token estimates.)
    let compressed: Vec<(u64, u64)> = recorder
        .events()
        .into_iter()
        .filter_map(|r| match r.event {
            AgentEvent::Compressed {
                from_tokens,
                to_tokens,
            } => Some((from_tokens, to_tokens)),
            _ => None,
        })
        .collect();
    assert_eq!(compressed.len(), 1);
    assert!(compressed[0].0 > 0);
    assert!(compressed[0].1 > 0);
}

/// A [`Summarizer`] that always fails — models the real gap the
/// `ConcatSummarizer` can never exhibit (a model-backed summarizer whose
/// provider call is rejected).
struct FailingSummarizer;

#[async_trait]
impl Summarizer for FailingSummarizer {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        Err(TinyAgentsError::Model("summarizer boom".to_string()))
    }
}

/// Build an over-threshold transcript `[system, big-1, big-2, big-3]` under a
/// 100-token window / 0.5 threshold (→ 50-token trigger budget), so the
/// compression middleware is guaranteed to fire.
fn over_threshold_request() -> (SummarizationPolicy, Vec<Message>) {
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let big = "a".repeat(200);
    let messages = vec![
        Message::system("You are a helpful assistant."),
        user(&format!("{big}-1")),
        user(&format!("{big}-2")),
        user(&format!("{big}-3")),
    ];
    (policy, messages)
}

#[tokio::test]
async fn context_compression_falls_back_to_trim_when_summarizer_errors() {
    // Regression for the "summarizer failure aborts the run" gap: under the
    // default FallbackTrim policy a failing summarizer must NOT abort. The
    // middleware front-drops the transcript to the trigger budget (keeping the
    // leading system prompt) and continues, emitting a MiddlewareFailed
    // diagnostic plus a Compressed event for the trim.
    let (policy, before) = over_threshold_request();
    let mw = Arc::new(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(FailingSummarizer),
    ));
    // Default failure policy is the trim fallback.
    assert_eq!(mw.failure_policy(), CompressionFailurePolicy::FallbackTrim);

    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: before.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .expect("summarizer failure must not abort the run under FallbackTrim");

    // Trimmed from the front — strictly fewer messages — with the system prompt
    // preserved (MaxTokens drops system messages last).
    assert!(request.messages.len() < before.len());
    assert!(matches!(request.messages[0], Message::System(_)));
    // No summary record could be produced (the summarizer failed).
    assert!(mw.records().is_empty());

    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::MiddlewareFailed { name, error }
                if name == "context_compression" && error.contains("summarizer boom")
        )),
        "a MiddlewareFailed diagnostic naming the failure must be emitted: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Compressed { .. })),
        "the fallback trim must emit a Compressed event: {events:?}"
    );
}

/// Regression: `CompressionFailurePolicy::FallbackTrim` used to trim messages
/// down to the *full* `trigger_budget` regardless of how much of that budget
/// the request's tool schemas already consumed — even though the trigger
/// itself (`should_summarize_with_tools`) charges those same schemas. A
/// request whose schemas alone consumed a meaningful share of the budget
/// therefore stayed over threshold after "recovery". The message budget must
/// reserve the schema cost first, so a schema-heavy request trims further
/// than a schema-free one under the same trigger budget.
#[tokio::test]
async fn context_compression_fallback_trim_reserves_the_tool_schema_budget() {
    // A finer-grained transcript than `over_threshold_request` (which trims
    // straight to the system-only floor in both cases here): ten ~20-token
    // messages under a 100-token trigger budget leaves room to see the
    // schema reservation actually change how many messages survive, rather
    // than both cases bottoming out at the same floor.
    let policy = SummarizationPolicy {
        keep_last: 0,
        trigger_tokens: 100,
        ..SummarizationPolicy::default()
    };
    let trigger_budget = policy.trigger_budget();
    let mut before = vec![Message::system("You are a helpful assistant.")];
    for i in 0..10 {
        before.push(user(&format!("message {i}: {}", "x".repeat(60))));
    }
    let mw = Arc::new(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(FailingSummarizer),
    ));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw);
    let mut c = ctx();

    // Baseline: no tools, trimmed to the full trigger budget.
    let mut request_no_tools = ModelRequest {
        messages: before.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request_no_tools)
        .await
        .expect("fallback trim runs");

    // Same transcript, but the request also carries a moderate tool schema
    // that eats into the same trigger budget without consuming all of it, so
    // the system prompt still survives trimming.
    let moderate_schema_text = "p".repeat(60);
    let mut request_with_tools = ModelRequest {
        messages: before.clone(),
        tools: vec![tinyinference_llm::tool::ToolSchema::new(
            "moderate_tool",
            moderate_schema_text,
            serde_json::json!({"type": "object"}),
        )],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request_with_tools)
        .await
        .expect("fallback trim runs");

    // Strict `<`, not `<=`: the pre-fix implementation trimmed both requests
    // to the full (schema-blind) `trigger_budget`, which — for this
    // transcript, where every non-system message is the same size — would
    // often keep the exact same number of messages in both cases and satisfy
    // a merely-`<=` assertion despite not actually reserving anything for the
    // schema. A `<` here is only possible because the schema budget was
    // subtracted from the message budget before trimming.
    assert!(
        request_with_tools.messages.len() < request_no_tools.messages.len(),
        "a request whose schemas already consume budget must trim strictly \
         further than one with no schemas: with_tools={}, no_tools={}",
        request_with_tools.messages.len(),
        request_no_tools.messages.len()
    );
    assert!(matches!(request_with_tools.messages[0], Message::System(_)));

    // Quantitative check: the trimmed messages plus the schema cost must fit
    // within the policy's trigger budget — the property the reservation
    // exists to guarantee, not just "fewer messages than before."
    let schema_tokens = crate::token_estimation::count_tool_schema_tokens(
        &request_with_tools.tools,
        &crate::token_estimation::TokenCountOptions::default(),
    );
    let message_tokens =
        crate::token_estimation::estimate_slice_tokens(&request_with_tools.messages);
    assert!(
        message_tokens + schema_tokens <= trigger_budget,
        "message_tokens ({message_tokens}) + schema_tokens ({schema_tokens}) must fit within \
         trigger_budget ({trigger_budget})"
    );

    // Extreme case: the schema cost alone exceeds the whole trigger budget.
    // The message budget must saturate to 0 (not underflow/panic), so the
    // fallback still returns instead of erroring or crashing.
    let huge_schema_text = "p".repeat(2_000);
    let mut request_huge_tools = ModelRequest {
        messages: before,
        tools: vec![tinyinference_llm::tool::ToolSchema::new(
            "huge_tool",
            huge_schema_text,
            serde_json::json!({"type": "object"}),
        )],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request_huge_tools)
        .await
        .expect("fallback trim runs even when schemas alone exceed the budget");
    assert!(request_huge_tools.messages.len() <= request_with_tools.messages.len());
}

#[tokio::test]
async fn context_compression_abort_policy_propagates_summarizer_error() {
    // Opt back into the legacy behaviour: Abort propagates the error so the run
    // fails, but still emits the MiddlewareFailed diagnostic first.
    let (policy, before) = over_threshold_request();
    let mw = Arc::new(
        ContextCompressionMiddleware::with_summarizer(policy, Box::new(FailingSummarizer))
            .with_failure_policy(CompressionFailurePolicy::Abort),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: before,
        ..Default::default()
    };
    let result = stack.run_before_model(&mut c, &(), &mut request).await;
    assert!(
        result.is_err(),
        "Abort must propagate the summarizer error and fail the run"
    );

    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::MiddlewareFailed { name, .. } if name == "context_compression"
        )),
        "Abort must still emit the MiddlewareFailed diagnostic: {events:?}"
    );
}

#[tokio::test]
async fn context_compression_pass_through_policy_keeps_transcript_and_continues() {
    // PassThrough leaves the (over-threshold) transcript untouched and lets the
    // run continue, emitting only the MiddlewareFailed diagnostic.
    let (policy, before) = over_threshold_request();
    let mw = Arc::new(
        ContextCompressionMiddleware::with_summarizer(policy, Box::new(FailingSummarizer))
            .with_failure_policy(CompressionFailurePolicy::PassThrough),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: before.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .expect("PassThrough must not abort the run");

    // Transcript is untouched.
    assert_eq!(request.messages, before);
    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::MiddlewareFailed { name, .. } if name == "context_compression"
        )),
        "PassThrough must emit the MiddlewareFailed diagnostic: {events:?}"
    );
    // No compression happened, so no Compressed event.
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::Compressed { .. })),
        "PassThrough must not emit a Compressed event: {events:?}"
    );
}

#[tokio::test]
async fn context_compression_records_are_bounded_by_max_records() {
    // A long-running loop that compresses repeatedly must not grow the
    // recorder without bound; cap it and confirm eviction happens.
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy).with_max_records(2));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    // The transcript grows by two messages a call, as an agent loop's does, so
    // every call has new history past the fold to compact. (Re-sending an
    // unchanged transcript re-applies the existing fold instead.)
    let big = "a".repeat(200);
    let mut c = ctx();
    let mut transcript = vec![user(&format!("{big}-0"))];
    for i in 0..10 {
        transcript.push(user(&format!("{big}-{i}a")));
        transcript.push(user(&format!("{big}-{i}b")));
        let mut request = ModelRequest {
            messages: transcript.clone(),
            ..Default::default()
        };
        stack
            .run_before_model(&mut c, &(), &mut request)
            .await
            .unwrap();
    }

    assert_eq!(mw.records().len(), 2);
}

#[tokio::test]
async fn context_compression_keeps_system_prompt_before_summary() {
    // A leading system prompt carries persistent instructions and anchors the
    // cacheable prefix; the summary of elided older turns must be inserted
    // *after* it, never at position 0.
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    let big = "a".repeat(200);
    let system_prompt = "You are a helpful assistant. Always follow these rules.";
    let mut request = ModelRequest {
        messages: vec![
            Message::system(system_prompt),
            user(&format!("{big}-1")),
            user(&format!("{big}-2")),
            user(&format!("{big}-3")),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Result is [real system prompt, summary, kept recent turn].
    assert_eq!(request.messages.len(), 3);
    assert!(matches!(request.messages[0], Message::System(_)));
    assert_eq!(
        request.messages[0].text(),
        system_prompt,
        "the real system prompt must stay at position 0"
    );
    assert!(
        matches!(request.messages[1], Message::User(_))
            && crate::summarization::is_checkpoint(&request.messages[1]),
        "the summary follows the system prompt as a user-role checkpoint"
    );
    assert_ne!(
        request.messages[1].text(),
        system_prompt,
        "position 1 is the summary, not a duplicated system prompt"
    );
    assert_eq!(request.messages[2].text(), format!("{big}-3"));
}

#[tokio::test]
async fn context_compression_none_window_falls_back_to_trigger_tokens() {
    // No context window → raw trigger_tokens gate (strict `>`). Trigger at 2
    // tokens, keep_last=1.
    let policy = SummarizationPolicy {
        trigger_tokens: 2,
        keep_last: 1,
        ..SummarizationPolicy::default()
    };
    assert_eq!(policy.context_window, None);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    // Two ~4-token (16-char) messages → ~8 tokens > 2 trigger.
    let mut request = ModelRequest {
        messages: vec![user("aaaaaaaaaaaaaaaa"), user("bbbbbbbbbbbbbbbb")],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Summary + the one kept recent message.
    assert_eq!(request.messages.len(), 2);
    assert!(crate::summarization::is_checkpoint(&request.messages[0]));
    assert_eq!(request.messages[1].text(), "bbbbbbbbbbbbbbbb");
    assert_eq!(mw.records().len(), 1);
}

// ── MicrocompactMiddleware ────────────────────────────────────────────────────
//
// These cases are the byte-for-byte parity contract ported from the OpenHuman
// in-house `MicrocompactMiddleware` this type replaces: older tool-result bodies
// are blanked to the caller's placeholder, the newest `keep_recent` are kept
// verbatim, non-tool messages are never touched, and the pass is idempotent.

const CLEARED: &str = "[Old tool result content cleared]";

#[tokio::test]
async fn microcompact_clears_older_tool_bodies_and_keeps_recent() {
    let mw = MicrocompactMiddleware::new(1, CLEARED);
    let mut request = ModelRequest {
        messages: vec![
            Message::system("sys"),
            Message::user("hello"),
            Message::tool("t1", "FIRST_BODY"),
            Message::assistant("thinking"),
            Message::tool("t2", "SECOND_BODY"),
            Message::tool("t3", "THIRD_BODY"),
        ],
        ..Default::default()
    };

    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();

    // 3 tool messages, keep_recent=1 → the two oldest cleared, newest kept.
    assert_eq!(request.messages[2].text(), CLEARED);
    assert_eq!(request.messages[4].text(), CLEARED);
    assert_eq!(request.messages[5].text(), "THIRD_BODY");
    // Non-tool messages are never touched.
    assert_eq!(request.messages[0].text(), "sys");
    assert_eq!(request.messages[1].text(), "hello");
    assert_eq!(request.messages[3].text(), "thinking");
    // Cleared tool messages keep their tool_call_id.
    match &request.messages[2] {
        Message::Tool(t) => assert_eq!(t.tool_call_id, "t1"),
        other => panic!("expected tool message, got {other:?}"),
    }
}

#[tokio::test]
async fn microcompact_is_a_noop_when_within_keep_recent() {
    let mw = MicrocompactMiddleware::new(5, CLEARED);
    let mut request = ModelRequest {
        messages: vec![Message::tool("t1", "A"), Message::tool("t2", "B")],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), "A");
    assert_eq!(request.messages[1].text(), "B");
}

#[tokio::test]
async fn microcompact_is_idempotent() {
    let mw = MicrocompactMiddleware::new(1, CLEARED);
    let mut request = ModelRequest {
        messages: vec![Message::tool("t1", "FIRST"), Message::tool("t2", "SECOND")],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), CLEARED);
    // Second pass leaves the already-cleared body as the placeholder.
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), CLEARED);
    assert_eq!(request.messages[1].text(), "SECOND");
}

// ── token-budget gate (issue tinyhumansai/openhuman#4755) ──────────────────────

#[tokio::test]
async fn microcompact_with_token_budget_is_a_noop_below_budget() {
    // More than `keep_recent` tool results, but the whole transcript fits the
    // configured budget: NOTHING is blanked, so the request stays byte-stable
    // across iterations and the provider KV-cache prefix is preserved. This is
    // the fix — the ungated middleware would have blanked the two oldest here
    // (see `microcompact_clears_older_tool_bodies_and_keeps_recent`), churning
    // the cache to reclaim tokens the model had ample room for.
    let mw = MicrocompactMiddleware::new(1, CLEARED).with_token_budget(100_000);
    assert_eq!(mw.token_budget(), Some(100_000));
    let mut request = ModelRequest {
        messages: vec![
            Message::system("sys"),
            Message::tool("t1", "FIRST_BODY"),
            Message::tool("t2", "SECOND_BODY"),
            Message::tool("t3", "THIRD_BODY"),
        ],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[1].text(), "FIRST_BODY");
    assert_eq!(request.messages[2].text(), "SECOND_BODY");
    assert_eq!(request.messages[3].text(), "THIRD_BODY");
}

#[tokio::test]
async fn microcompact_with_token_budget_blanks_once_over_budget() {
    // Same shape, but the (un-blanked) transcript exceeds the tiny budget, so
    // compaction is genuinely needed to stay under the window: the gate lets the
    // usual keep-recent blanking run and reclaims tokens.
    let body = "x".repeat(400); // ~100 estimated tokens each (chars / 4)
    let mw = MicrocompactMiddleware::new(1, CLEARED).with_token_budget(10);
    let mut request = ModelRequest {
        messages: vec![
            Message::tool("t1", body.clone()),
            Message::tool("t2", body.clone()),
            Message::tool("t3", body.clone()),
        ],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), CLEARED);
    assert_eq!(request.messages[1].text(), CLEARED);
    assert_eq!(request.messages[2].text(), body);
}

#[tokio::test]
async fn microcompact_with_token_budget_zero_disables_the_gate() {
    // `budget == 0` is an explicit opt-out: the gate is `None`, so blanking
    // reverts to the legacy count-only behaviour even on a tiny transcript.
    let mw = MicrocompactMiddleware::new(1, CLEARED).with_token_budget(0);
    assert_eq!(mw.token_budget(), None);
    let mut request = ModelRequest {
        messages: vec![
            Message::tool("t1", "A"),
            Message::tool("t2", "B"),
            Message::tool("t3", "C"),
        ],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), CLEARED);
    assert_eq!(request.messages[1].text(), CLEARED);
    assert_eq!(request.messages[2].text(), "C");
}

#[tokio::test]
async fn microcompact_emits_no_event_by_default() {
    // Default construction is silent: bodies are still cleared, but no
    // Compressed event is emitted (parity with the OpenHuman in-house version).
    let mw = Arc::new(MicrocompactMiddleware::new(1, CLEARED));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: vec![Message::tool("t1", "FIRST"), Message::tool("t2", "SECOND")],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    assert_eq!(request.messages[0].text(), CLEARED);
    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::Compressed { .. })),
        "no Compressed event should be emitted when events are off"
    );
}

#[tokio::test]
async fn microcompact_emits_compressed_event_when_enabled() {
    let mw = Arc::new(MicrocompactMiddleware::new(1, CLEARED).with_events(true));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: vec![
            Message::tool("t1", "x".repeat(400)),
            Message::tool("t2", "y".repeat(400)),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    let compressed: Vec<(u64, u64)> = recorder
        .events()
        .into_iter()
        .filter_map(|r| match r.event {
            AgentEvent::Compressed {
                from_tokens,
                to_tokens,
            } => Some((from_tokens, to_tokens)),
            _ => None,
        })
        .collect();
    assert_eq!(
        compressed.len(),
        1,
        "one Compressed event when a body cleared"
    );
    assert!(
        compressed[0].0 > compressed[0].1,
        "tokens dropped after clear"
    );
}

#[tokio::test]
async fn usage_accounting_accumulates_across_calls() {
    let mw = Arc::new(UsageAccountingMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    let mut r1 = response_with_usage(Usage::new(10, 5));
    let mut r2 = response_with_usage(Usage::new(3, 2));
    stack.run_after_model(&mut c, &(), &mut r1).await.unwrap();
    stack.run_after_model(&mut c, &(), &mut r2).await.unwrap();

    let totals = mw.totals();
    assert_eq!(totals.calls, 2);
    assert_eq!(totals.usage.input_tokens, 13);
    assert_eq!(totals.usage.output_tokens, 7);
    assert_eq!(totals.usage.total_tokens, 20);
}

#[tokio::test]
async fn prompt_cache_guard_detects_prefix_change() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();

    // First call establishes a cacheable prefix [sys].
    let mut req1 = ModelRequest {
        cache_segments: vec![segment("sys", SegmentRole::System, true)],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut req1)
        .await
        .unwrap();
    assert!(mw.layout_events().is_empty(), "no prior layout to compare");

    // Second call changes the stable prefix -> a layout event is recorded.
    let mut req2 = ModelRequest {
        cache_segments: vec![segment("sys2", SegmentRole::System, true)],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut req2)
        .await
        .unwrap();

    let events = mw.layout_events();
    assert_eq!(events.len(), 1);
    assert!(events[0].changed_prefix);
    assert_eq!(events[0].segment_ids_before, vec!["sys".to_string()]);
    assert_eq!(events[0].segment_ids_after, vec!["sys2".to_string()]);
}

#[tokio::test]
async fn prompt_cache_guard_ignores_trimmed_history_when_stable_prefix_is_unchanged() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("system", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![
        Message::system("same prompt"),
        user("old request"),
        Message::assistant("old answer"),
    ])
    .with_cache_segments(segments.clone());
    before.prompt_fingerprint = Some("same-stable-prefix".into());
    let mut after = ModelRequest::new(vec![
        Message::system("same prompt"),
        user("summary of old request"),
        user("new request"),
    ])
    .with_cache_segments(segments);
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    let before_layout = crate::cache::PromptCacheLayout::from_request(&before);
    let after_layout = crate::cache::PromptCacheLayout::from_request(&after);
    assert_eq!(before.messages[0].text(), "same prompt");
    assert_eq!(after.messages[0], before.messages[0]);
    assert_eq!(before_layout.prefix_ids(), &["system"]);
    assert_eq!(after_layout.prefix_ids(), before_layout.prefix_ids());
    assert_eq!(after_layout.fingerprint(), before_layout.fingerprint());
    assert!(
        mw.layout_events().is_empty(),
        "history compaction retains the stable prefix"
    );
}

#[tokio::test]
async fn prompt_cache_guard_reports_same_id_stable_content_change() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("system", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![Message::system("prompt A"), user("question")])
        .with_cache_segments(segments.clone());
    before.prompt_fingerprint = Some("prompt-a".into());
    let mut after = ModelRequest::new(vec![Message::system("prompt B"), user("question")])
        .with_cache_segments(segments);
    after.prompt_fingerprint = Some("prompt-b".into());

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    let events = mw.layout_events();
    assert_eq!(events.len(), 1);
    assert!(events[0].content_only_change);
}

#[tokio::test]
async fn prompt_cache_guard_detects_rewritten_system_with_stale_annotation() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("system", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![Message::system("prompt A"), user("question")])
        .with_cache_segments(segments.clone());
    before.prompt_fingerprint = Some("builder-value".into());
    let mut after = ModelRequest::new(vec![Message::system("prompt B"), user("question")])
        .with_cache_segments(segments);
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    assert_eq!(mw.layout_events().len(), 1);
}

#[tokio::test]
async fn prompt_cache_guard_detects_a_new_leading_system_message() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("system", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![Message::system("stable"), user("question")])
        .with_cache_segments(segments.clone());
    before.prompt_fingerprint = Some("builder-value".into());
    let mut after = ModelRequest::new(vec![
        Message::system("new instruction"),
        Message::system("stable"),
        user("question"),
    ])
    .with_cache_segments(segments);
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    assert_eq!(mw.layout_events().len(), 1);
}

#[tokio::test]
async fn dynamic_prompt_preserves_the_original_declared_system_tier() {
    let guard = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(
        crate::middleware::library::DynamicPromptMiddleware::<(), ()>::from_fn(|_, _| {
            Some("dynamic".into())
        }),
    ));
    stack.push(guard.clone());
    let mut c = ctx();
    let segments = vec![segment("system", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![Message::system("original A"), user("question")])
        .with_cache_segments(segments.clone());
    before.prompt_fingerprint = Some("stale-builder-value".into());
    let mut after = ModelRequest::new(vec![Message::system("original B"), user("question")])
        .with_cache_segments(segments);
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    let expected_segments = vec![
        segment("system", SegmentRole::System, true),
        segment("system.1", SegmentRole::System, true),
    ];
    assert_eq!(before.cache_segments, expected_segments);
    assert_eq!(after.cache_segments, expected_segments);
    assert_eq!(before.messages[0].text(), "dynamic");
    assert_eq!(before.messages[1].text(), "original A");
    assert_eq!(after.messages[1].text(), "original B");
    assert_eq!(guard.layout_events().len(), 1);
}

#[tokio::test]
async fn prompt_cache_guard_detects_custom_dynamic_prompt_rewrite() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("tenant-prompt", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![Message::system("tenant A"), user("question")])
        .with_cache_segments(segments.clone());
    before.prompt_fingerprint = Some("builder-value".into());
    let mut after = ModelRequest::new(vec![Message::system("tenant B"), user("question")])
        .with_cache_segments(segments);
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    assert_eq!(mw.layout_events().len(), 1);
}

#[tokio::test]
async fn prompt_cache_guard_accepts_a_custom_layout_tail_append() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("tenant-prompt", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![Message::system("stable"), user("question")])
        .with_cache_segments(segments.clone());
    before.prompt_fingerprint = Some("builder-value".into());
    let mut after = ModelRequest::new(vec![
        Message::system("stable"),
        user("question"),
        Message::assistant("answer"),
    ])
    .with_cache_segments(segments);
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    assert!(mw.layout_events().is_empty());
}

#[tokio::test]
async fn prompt_cache_guard_uses_full_request_when_boundary_is_unknown() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("system", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![Message::system("stable"), user("question")])
        .with_cache_segments(segments.clone());
    let mut after = ModelRequest::new(vec![
        Message::system("stable"),
        Message::system("changing history summary"),
        user("question"),
    ])
    .with_cache_segments(segments);

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    assert_eq!(mw.layout_events().len(), 1);
}

#[tokio::test]
async fn prompt_cache_guard_checks_history_without_declared_segments() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let mut before = ModelRequest::new(vec![Message::system("prompt A"), user("question")]);
    before.prompt_fingerprint = Some("stale-builder-value".into());
    let mut after = ModelRequest::new(vec![Message::system("prompt B"), user("question")]);
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    assert_eq!(mw.layout_events().len(), 1);
}

#[tokio::test]
async fn prompt_cache_guard_detects_same_id_system_edit_without_a_fingerprint() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("system", SegmentRole::System, true)];
    let mut before = ModelRequest::new(vec![Message::system("prompt A"), user("question")])
        .with_cache_segments(segments.clone());
    let mut after = ModelRequest::new(vec![Message::system("prompt B"), user("question")])
        .with_cache_segments(segments);

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    assert_eq!(mw.layout_events().len(), 1);
}

#[tokio::test]
async fn prompt_cache_guard_detects_tool_schema_change_without_a_fingerprint() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let segments = vec![segment("tools", SegmentRole::Tools, true)];
    let tool = |description| {
        tinyinference_llm::tool::ToolSchema::new(
            "search",
            description,
            serde_json::json!({"type": "object"}),
        )
    };
    let mut before = ModelRequest::new(vec![user("question")])
        .with_cache_segments(segments.clone())
        .with_tools(vec![tool("search files")]);
    let mut after = ModelRequest::new(vec![user("question")])
        .with_cache_segments(segments)
        .with_tools(vec![tool("search files recursively")]);

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    assert_eq!(mw.layout_events().len(), 1);
}

#[tokio::test]
async fn prompt_cache_guard_reports_custom_volatile_segment_changes() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    let mut c = ctx();
    let mut before = ModelRequest::new(vec![Message::system("stable"), user("first")])
        .with_cache_segments(vec![
            segment("system", SegmentRole::System, true),
            segment("turn-1", SegmentRole::Volatile, false),
        ]);
    let mut after = ModelRequest::new(vec![Message::system("stable"), user("first")])
        .with_cache_segments(vec![
            segment("system", SegmentRole::System, true),
            segment("turn-2", SegmentRole::Volatile, false),
        ]);
    before.prompt_fingerprint = Some("stable-system".into());
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    stack
        .run_before_model(&mut c, &(), &mut before)
        .await
        .unwrap();
    stack
        .run_before_model(&mut c, &(), &mut after)
        .await
        .unwrap();

    // A noncanonical annotation takes the same full-request fallback as
    // dispatch, where even volatile segment metadata can change the key.
    assert_eq!(mw.layout_events().len(), 1);
}

#[tokio::test]
async fn prompt_cache_guard_events_are_bounded_by_max_events() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new().with_max_events(2));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    // Each call after the first changes the stable prefix, producing an
    // event; run far more iterations than the cap and confirm eviction.
    for i in 0..10 {
        let mut req = ModelRequest {
            cache_segments: vec![segment(&format!("sys{i}"), SegmentRole::System, true)],
            ..Default::default()
        };
        stack.run_before_model(&mut c, &(), &mut req).await.unwrap();
    }

    assert_eq!(mw.layout_events().len(), 2);
}

// ── wrap middleware: model ────────────────────────────────────────────────────

/// Builds a `ModelResponse` whose single text block is `text`.
fn response_text(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: None,
            origin: None,
        },
        usage: None,
        finish_reason: None,
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// A model base that records how many times it was invoked and fails its first
/// `fail_times` calls (with a retryable-looking error) before succeeding.
struct CountingModelBase {
    calls: Arc<Mutex<usize>>,
    fail_times: usize,
    text: &'static str,
}

impl ModelBaseCall<(), ()> for CountingModelBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext,
        _state: &'a (),
        _request: ModelRequest,
    ) -> BoxModelFuture<'a> {
        Box::pin(async move {
            let attempt = {
                let mut n = self.calls.lock().unwrap();
                *n += 1;
                *n
            };
            if attempt <= self.fail_times {
                Err(TinyAgentsError::Middleware("transient".to_string()))
            } else {
                Ok(response_text(self.text))
            }
        })
    }
}

/// Wrap middleware that returns a canned response without calling `next`.
struct ShortCircuitModel {
    text: &'static str,
}

#[async_trait]
impl ModelMiddleware<()> for ShortCircuitModel {
    fn name(&self) -> &str {
        "short_circuit_model"
    }

    async fn wrap_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _request: ModelRequest,
        _next: ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        Ok(MiddlewareModelOutcome::Response(response_text(self.text)))
    }
}

/// Wrap middleware that calls `next` then mutates the resulting response.
struct MutateAfterModel;

#[async_trait]
impl ModelMiddleware<()> for MutateAfterModel {
    fn name(&self) -> &str {
        "mutate_after_model"
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext,
        state: &(),
        request: ModelRequest,
        next: ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        let mut response = next.run(ctx, state, request).await?.into_response();
        response.finish_reason = Some("mutated".to_string());
        Ok(response.into())
    }
}

/// Wrap middleware that retries `next` up to `max` times until it succeeds.
struct RetryModel {
    max: usize,
}

#[async_trait]
impl ModelMiddleware<()> for RetryModel {
    fn name(&self) -> &str {
        "retry_model"
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext,
        state: &(),
        request: ModelRequest,
        next: ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match next.run(ctx, state, request.clone()).await {
                Ok(outcome) => return Ok(outcome),
                Err(_) if attempt < self.max => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

#[tokio::test]
async fn wrap_model_short_circuits_without_calling_base() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingModelBase {
        calls: calls.clone(),
        fail_times: 0,
        text: "from-base",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(ShortCircuitModel { text: "canned" }));

    let mut c = ctx();
    let response = stack
        .run_wrapped_model(&mut c, &(), ModelRequest::default(), &base)
        .await
        .unwrap()
        .into_response();

    assert_eq!(response.text(), "canned");
    // Base was never invoked because the wrap middleware short-circuited.
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn wrap_model_calls_next_then_mutates_response() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingModelBase {
        calls: calls.clone(),
        fail_times: 0,
        text: "from-base",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(MutateAfterModel));

    let mut c = ctx();
    let response = stack
        .run_wrapped_model(&mut c, &(), ModelRequest::default(), &base)
        .await
        .unwrap()
        .into_response();

    // Forwarded the base response, then mutated it.
    assert_eq!(response.text(), "from-base");
    assert_eq!(response.finish_reason.as_deref(), Some("mutated"));
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn wrap_model_retries_next_until_success() {
    let calls = Arc::new(Mutex::new(0));
    // Fails twice, succeeds on the third attempt.
    let base = CountingModelBase {
        calls: calls.clone(),
        fail_times: 2,
        text: "eventually",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(RetryModel { max: 5 }));

    let mut c = ctx();
    let response = stack
        .run_wrapped_model(&mut c, &(), ModelRequest::default(), &base)
        .await
        .unwrap()
        .into_response();

    assert_eq!(response.text(), "eventually");
    // Two failures + one success = three base invocations.
    assert_eq!(*calls.lock().unwrap(), 3);
}

#[tokio::test]
async fn wrap_model_onion_orders_outer_to_inner() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingModelBase {
        calls: calls.clone(),
        fail_times: 0,
        text: "base",
    };
    // Outer = mutate-after (sees the canned response from the inner layer and
    // stamps finish_reason); inner = short-circuit (never reaches base).
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(MutateAfterModel));
    stack.push_model_middleware(Arc::new(ShortCircuitModel { text: "canned" }));

    let mut c = ctx();
    let response = stack
        .run_wrapped_model(&mut c, &(), ModelRequest::default(), &base)
        .await
        .unwrap()
        .into_response();

    assert_eq!(response.text(), "canned");
    assert_eq!(response.finish_reason.as_deref(), Some("mutated"));
    // Inner layer short-circuited, so the base call never ran.
    assert_eq!(*calls.lock().unwrap(), 0);
    assert_eq!(stack.model_middleware_len(), 2);
}

// ── wrap middleware: tool ───────────────────────────────────────────────────--

/// A tool base that records invocations and fails its first `fail_times` calls.
struct CountingToolBase {
    calls: Arc<Mutex<usize>>,
    fail_times: usize,
    content: &'static str,
}

impl ToolBaseCall<(), ()> for CountingToolBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a RunContext,
        _state: &'a (),
        _call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            let attempt = {
                let mut n = self.calls.lock().unwrap();
                *n += 1;
                *n
            };
            if attempt <= self.fail_times {
                Err(TinyAgentsError::Middleware("transient".to_string()))
            } else {
                Ok(ToolResult::success(self.content))
            }
        })
    }
}

fn tool_call() -> ToolCall {
    ToolCall {
        id: "call-1".to_string(),
        name: "fake".to_string(),
        arguments: serde_json::Value::Null,
        invalid: None,
    }
}

/// Wrap middleware that returns a canned result without calling `next`.
struct ShortCircuitTool {
    content: &'static str,
}

#[async_trait]
impl ToolMiddleware<()> for ShortCircuitTool {
    fn name(&self) -> &str {
        "short_circuit_tool"
    }

    async fn wrap_tool(
        &self,
        _ctx: &RunContext,
        _state: &(),
        _call: ToolCall,
        _next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        Ok(MiddlewareToolOutcome::Result(ToolResult::success(
            self.content,
        )))
    }
}

/// Wrap middleware that calls `next` then mutates the resulting result.
struct MutateAfterTool;

#[async_trait]
impl ToolMiddleware<()> for MutateAfterTool {
    fn name(&self) -> &str {
        "mutate_after_tool"
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext,
        state: &(),
        call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        let mut result = next.run(ctx, state, call).await?.into_result();
        result.content = vec![tinytools::ToolContent::Text {
            text: format!("{}!", result.output()),
        }];
        Ok(result.into())
    }
}

/// Wrap middleware that retries `next` up to `max` times until it succeeds.
struct RetryTool {
    max: usize,
}

#[async_trait]
impl ToolMiddleware<()> for RetryTool {
    fn name(&self) -> &str {
        "retry_tool"
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext,
        state: &(),
        call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match next.run(ctx, state, call.clone()).await {
                Ok(outcome) => return Ok(outcome),
                Err(_) if attempt < self.max => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

#[tokio::test]
async fn wrap_tool_short_circuits_without_calling_base() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingToolBase {
        calls: calls.clone(),
        fail_times: 0,
        content: "from-base",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_tool_middleware(Arc::new(ShortCircuitTool { content: "canned" }));

    let c = ctx();
    let result = stack
        .run_wrapped_tool(&c, &(), tool_call(), &base)
        .await
        .unwrap()
        .into_result();

    assert_eq!(result.output(), "canned");
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn wrap_tool_calls_next_then_mutates_result() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingToolBase {
        calls: calls.clone(),
        fail_times: 0,
        content: "ok",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_tool_middleware(Arc::new(MutateAfterTool));

    let c = ctx();
    let result = stack
        .run_wrapped_tool(&c, &(), tool_call(), &base)
        .await
        .unwrap()
        .into_result();

    assert_eq!(result.output(), "ok!");
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn wrap_tool_retries_next_until_success() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingToolBase {
        calls: calls.clone(),
        fail_times: 2,
        content: "eventually",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_tool_middleware(Arc::new(RetryTool { max: 5 }));

    let c = ctx();
    let result = stack
        .run_wrapped_tool(&c, &(), tool_call(), &base)
        .await
        .unwrap()
        .into_result();

    assert_eq!(result.output(), "eventually");
    assert_eq!(*calls.lock().unwrap(), 3);
    assert_eq!(stack.tool_middleware_len(), 1);
}

/// Wrap middleware that opts out of overlapping invocations.
struct SerialOnlyTool;

#[async_trait]
impl ToolMiddleware<()> for SerialOnlyTool {
    fn name(&self) -> &str {
        "serial_only_tool"
    }

    fn concurrent_safe(&self) -> bool {
        false
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext,
        state: &(),
        call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        next.run(ctx, state, call).await
    }
}

#[test]
fn tool_middleware_concurrent_safe_is_the_conjunction_of_every_wrap() {
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    assert!(stack.tool_middleware_concurrent_safe(), "vacuously true");
    stack.push_tool_middleware(Arc::new(MutateAfterTool));
    assert!(stack.tool_middleware_concurrent_safe(), "default is true");
    stack.push_tool_middleware(Arc::new(SerialOnlyTool));
    assert!(!stack.tool_middleware_concurrent_safe());
}

#[tokio::test]
async fn agent_run_text_reflects_final_response() {
    let mut run = AgentRun::new();
    assert_eq!(run.text(), None);
    run.final_response = Some(response_with_usage(Usage::new(1, 1)));
    assert_eq!(run.text(), Some("ok".to_string()));
}

// ── ContextCompressionMiddleware: overflow → compact → retry ──────────────────

/// A model base that fails its first `fail_times` calls with a classified
/// context-overflow error, then succeeds.
struct OverflowThenSucceedBase {
    calls: Arc<Mutex<usize>>,
    fail_times: usize,
}

impl ModelBaseCall<(), ()> for OverflowThenSucceedBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext,
        _state: &'a (),
        _request: ModelRequest,
    ) -> BoxModelFuture<'a> {
        Box::pin(async move {
            let attempt = {
                let mut n = self.calls.lock().unwrap();
                *n += 1;
                *n
            };
            if attempt <= self.fail_times {
                Err(TinyAgentsError::Model(
                    "This model's maximum context length is 100 tokens. However, your \
                     messages resulted in 900 tokens."
                        .to_string(),
                ))
            } else {
                Ok(response_text("recovered"))
            }
        })
    }
}

/// A large-enough transcript that `find_cut_point` finds a real cut under a
/// small `keep_recent_tokens` budget: several long user/assistant turns, no
/// tool calls (pairing is exercised separately by `summarization::compaction`
/// tests).
fn overflow_prone_messages() -> Vec<Message> {
    let big = "word ".repeat(60);
    vec![
        user(&format!("first {big}")),
        Message::assistant(format!("second {big}")),
        user(&format!("third {big}")),
        Message::assistant(format!("fourth {big}")),
        user(&format!("fifth {big}")),
    ]
}

/// A summarizer whose summary is far smaller than its input, so an overflow
/// compaction actually shrinks the request (`ConcatSummarizer` never does).
struct TinySummarizer;

#[async_trait]
impl Summarizer for TinySummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        Ok(SummaryRecord {
            summary: Message::system("tiny summary"),
            provenance: crate::summarization::CompressionProvenance {
                source_ids: Vec::new(),
                original_token_estimate: messages.len() as u64,
                summary_token_estimate: 2,
                reason: "test".into(),
            },
            usage: None,
        })
    }
}

fn tiny_middleware(policy: SummarizationPolicy) -> ContextCompressionMiddleware {
    ContextCompressionMiddleware::with_summarizer(policy, Box::new(TinySummarizer))
}

fn small_window_policy() -> SummarizationPolicy {
    SummarizationPolicy::default()
        .with_context_window(100)
        .with_threshold_fraction(0.5)
}

#[tokio::test]
async fn context_compression_overflow_retries_once_and_compacts() {
    let calls = Arc::new(Mutex::new(0));
    let base = OverflowThenSucceedBase {
        calls: calls.clone(),
        fail_times: 1,
    };
    let mw = Arc::new(tiny_middleware(small_window_policy()).with_max_overflow_attempts(1));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let request = ModelRequest {
        messages: overflow_prone_messages(),
        ..Default::default()
    };
    let response = stack
        .run_wrapped_model(&mut c, &(), request, &base)
        .await
        .unwrap()
        .into_response();

    assert_eq!(response.text(), "recovered");
    // One failing call + one successful retry = exactly two base invocations.
    assert_eq!(*calls.lock().unwrap(), 2);

    let compacted: Vec<AgentEvent> = recorder
        .events()
        .into_iter()
        .map(|r| r.event)
        .filter(|e| matches!(e, AgentEvent::Compacted { .. }))
        .collect();
    assert_eq!(compacted.len(), 1);
    assert!(matches!(
        compacted[0],
        AgentEvent::Compacted {
            reason: crate::summarization::CompactionReason::Overflow,
            ..
        }
    ));
}

#[tokio::test]
async fn context_compression_overflow_propagates_after_second_failure() {
    // Fails every call: the retry itself also overflows, so the middleware
    // must give up after exactly one retry rather than looping.
    let calls = Arc::new(Mutex::new(0));
    let base = OverflowThenSucceedBase {
        calls: calls.clone(),
        fail_times: usize::MAX,
    };
    let mw = Arc::new(tiny_middleware(small_window_policy()).with_max_overflow_attempts(1));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(mw.clone());

    let mut c = ctx();
    let request = ModelRequest {
        messages: overflow_prone_messages(),
        ..Default::default()
    };
    let result = stack.run_wrapped_model(&mut c, &(), request, &base).await;

    assert!(result.is_err());
    // Original call + exactly one retry = two base invocations, not more.
    assert_eq!(*calls.lock().unwrap(), 2);
}

#[tokio::test]
async fn context_compression_wrap_model_ignores_unrelated_errors() {
    // A non-overflow failure must propagate untouched, with no compaction
    // attempted and no retry.
    let calls = Arc::new(Mutex::new(0));
    struct AlwaysFailsBase {
        calls: Arc<Mutex<usize>>,
    }
    impl ModelBaseCall<(), ()> for AlwaysFailsBase {
        fn call<'a>(
            &'a self,
            _ctx: &'a mut RunContext,
            _state: &'a (),
            _request: ModelRequest,
        ) -> BoxModelFuture<'a> {
            Box::pin(async move {
                *self.calls.lock().unwrap() += 1;
                Err(TinyAgentsError::Tool("boom".to_string()))
            })
        }
    }
    let base = AlwaysFailsBase {
        calls: calls.clone(),
    };
    let mw = Arc::new(ContextCompressionMiddleware::new(small_window_policy()));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(mw);

    let mut c = ctx();
    let request = ModelRequest {
        messages: overflow_prone_messages(),
        ..Default::default()
    };
    let result = stack.run_wrapped_model(&mut c, &(), request, &base).await;

    assert!(matches!(result, Err(TinyAgentsError::Tool(_))));
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn context_compression_before_compaction_decline_leaves_transcript_untouched() {
    let calls = Arc::new(Mutex::new(0));
    let base = OverflowThenSucceedBase {
        calls: calls.clone(),
        fail_times: 1,
    };
    let mw = Arc::new(
        tiny_middleware(small_window_policy())
            .with_before_compaction(|_ctx: &CompactionContext| CompactionDecision::Decline),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(mw);

    let mut c = ctx();
    let request = ModelRequest {
        messages: overflow_prone_messages(),
        ..Default::default()
    };
    let result = stack.run_wrapped_model(&mut c, &(), request, &base).await;

    // Declined: no retry happens, the original overflow error propagates.
    assert!(result.is_err());
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn context_compression_threshold_decline_leaves_transcript_untouched() {
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(
        ContextCompressionMiddleware::new(policy)
            .with_before_compaction(|_ctx: &CompactionContext| CompactionDecision::Decline),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    let big = "a".repeat(200);
    let before = vec![
        user(&format!("{big}-1")),
        user(&format!("{big}-2")),
        user(&format!("{big}-3")),
    ];
    let mut request = ModelRequest {
        messages: before.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    assert_eq!(request.messages, before);
    assert!(mw.records().is_empty());
}

/// An in-memory [`CompactionSink`] that records every persisted
/// [`CompactionRecord`], for asserting the durable-persistence contract
/// without depending on `tinyagents-session`.
#[derive(Default)]
struct RecordingCompactionSink {
    records: Mutex<Vec<CompactionRecord>>,
}

impl CompactionSink for RecordingCompactionSink {
    fn persist(&self, record: &CompactionRecord) -> Result<()> {
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }
}

#[tokio::test]
async fn context_compression_persists_compaction_when_sink_is_attached() {
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let sink = Arc::new(RecordingCompactionSink::default());
    let mut c = ctx().with_compaction_sink(sink.clone());

    let big = "a".repeat(200);
    let mut request = ModelRequest {
        messages: vec![
            user(&format!("{big}-1")),
            user(&format!("{big}-2")),
            user(&format!("{big}-3")),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    let persisted = sink.records.lock().unwrap();
    assert_eq!(persisted.len(), 1);
    assert_eq!(
        persisted[0].reason,
        crate::summarization::CompactionReason::Threshold
    );
    assert_eq!(persisted[0].first_kept_index, 2);
    assert!(persisted[0].tokens_before > 0);
}

#[tokio::test]
async fn context_compression_no_sink_means_no_persistence_attempt() {
    // No sink attached: compaction still runs and emits `Compacted`, it just
    // has nowhere to persist to. This is mostly a "doesn't panic" check.
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw);

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());
    assert!(c.compaction_sink.is_none());

    let big = "a".repeat(200);
    let mut request = ModelRequest {
        messages: vec![
            user(&format!("{big}-1")),
            user(&format!("{big}-2")),
            user(&format!("{big}-3")),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    let compacted = recorder
        .events()
        .into_iter()
        .filter(|r| matches!(r.event, AgentEvent::Compacted { .. }))
        .count();
    assert_eq!(compacted, 1);
}

#[tokio::test]
async fn context_compression_iterative_summary_threads_previous_summary() {
    // Two successive threshold-triggered compactions on the same middleware
    // instance: the second must see the first's summary as
    // `SummaryRequest::previous_summary`.
    let requests: Arc<Mutex<Vec<crate::summarization::SummaryRequest>>> =
        Arc::new(Mutex::new(Vec::new()));

    struct RecordingSummarizer {
        requests: Arc<Mutex<Vec<crate::summarization::SummaryRequest>>>,
    }

    #[async_trait]
    impl Summarizer for RecordingSummarizer {
        async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
            self.summarize_request(&crate::summarization::SummaryRequest::new(
                messages.to_vec(),
            ))
            .await
        }

        async fn summarize_request(
            &self,
            request: &crate::summarization::SummaryRequest,
        ) -> Result<SummaryRecord> {
            self.requests.lock().unwrap().push(request.clone());
            crate::summarization::ConcatSummarizer
                .summarize(&request.messages)
                .await
        }
    }

    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(RecordingSummarizer {
            requests: requests.clone(),
        }),
    ));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    let big = "a".repeat(200);

    let mut request = ModelRequest {
        messages: vec![
            user(&format!("{big}-1")),
            user(&format!("{big}-2")),
            user(&format!("{big}-3")),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Grow the (already-compacted) transcript back over threshold and compact
    // again.
    request.messages.push(user(&format!("{big}-4")));
    request.messages.push(user(&format!("{big}-5")));
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    let seen = requests.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].previous_summary, None);
    assert!(seen[1].previous_summary.is_some());
}
