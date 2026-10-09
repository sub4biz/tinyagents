//! Tool-wrap middleware no longer forces serial execution (C8): a
//! multi-call batch of concurrency-safe tools runs its wrap onion inside each
//! concurrent future, unless a registered wrap opts out through
//! `ToolMiddleware::concurrent_safe`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use crate::context::{MiddlewareControl, RunConfig, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::AgentEvent;
use crate::middleware::{MiddlewareToolOutcome, ToolHandler, ToolMiddleware};
use crate::runtime::AgentHarness;
use crate::testkit::EventRecorder;
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolContent, ToolResult};

const DELAY: Duration = Duration::from_millis(100);

// ── Helpers ─────────────────────────────────────────────────────────────────

/// A concurrency-safe tool that sleeps, then answers `<name>-out`. Every
/// execution (and the high-water mark of overlapping ones) is recorded.
struct SleepTool {
    name: &'static str,
    delay: Duration,
    ran: Arc<Mutex<Vec<&'static str>>>,
    active: Arc<std::sync::atomic::AtomicUsize>,
    max_seen: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait]
impl Tool for SleepTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "sleeps"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn is_concurrency_safe(&self, _arguments: &serde_json::Value) -> bool {
        true
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        use std::sync::atomic::Ordering;
        self.ran.lock().unwrap().push(self.name);
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_seen.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::success(format!("{}-out", self.name)))
    }
}

/// What a probe wrap does with a call.
#[derive(Clone, Copy)]
enum Mode {
    /// Call `next` and stamp the result.
    Stamp,
    /// Answer `call-b` with a canned result without calling `next`.
    ShortCircuitB,
    /// Fail `call-b` with a fatal error without calling `next`.
    FailB,
    /// Defer `call-b` for approval without calling `next`.
    DeferB,
    /// Answer `call-b` with a `Command` outcome (a control request, no result).
    CommandB,
}

struct ProbeWrap {
    mode: Mode,
    concurrent_safe: bool,
}

#[async_trait]
impl ToolMiddleware<()> for ProbeWrap {
    fn name(&self) -> &str {
        "probe_wrap"
    }

    fn concurrent_safe(&self) -> bool {
        self.concurrent_safe
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext<()>,
        state: &(),
        call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        if call.id == "call-b" {
            match self.mode {
                Mode::Stamp => {}
                Mode::ShortCircuitB => {
                    return Ok(ToolResult::success("canned").into());
                }
                Mode::FailB => return Err(TinyAgentsError::Tool("wrap boom".to_string())),
                Mode::DeferB => {
                    return Err(TinyAgentsError::ApprovalRequired {
                        metadata: json!({"why": "wrap"}),
                    });
                }
                Mode::CommandB => {
                    return Ok(MiddlewareToolOutcome::Command {
                        control: MiddlewareControl::StopWithFinal("stopped by wrap".to_string()),
                    });
                }
            }
        }
        let mut result = next.run(ctx, state, call).await?.into_result();
        result.content = vec![ToolContent::Text {
            text: format!("[w] {}", result.output()),
        }];
        Ok(result.into())
    }
}

fn batch_response() -> ModelResponse {
    let tool_calls = ["a", "b", "c"]
        .iter()
        .map(|id| {
            let name = match *id {
                "a" => "alpha",
                "b" => "beta",
                _ => "gamma",
            };
            ToolCall::new(format!("call-{id}"), name, json!({}))
        })
        .collect();
    ModelResponse {
        message: AssistantMessage {
            id: Some("msg-batch".to_string()),
            content: Vec::new(),
            tool_calls,
            usage: Some(Usage::new(7, 3)),
            origin: None,
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

struct Rig {
    harness: AgentHarness<()>,
    ran: Arc<Mutex<Vec<&'static str>>>,
    max_seen: Arc<std::sync::atomic::AtomicUsize>,
}

/// Three sleeping tools (alpha slowest, gamma fastest) behind one probe wrap.
fn rig(mode: Mode, concurrent_safe: bool) -> Rig {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            batch_response(),
            crate::testkit::text_response("done").with_usage(Usage::new(4, 2)),
        ])),
    );
    let ran = Arc::new(Mutex::new(Vec::new()));
    let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for (name, delay) in [
        ("alpha", DELAY),
        ("beta", DELAY * 3 / 4),
        ("gamma", DELAY / 2),
    ] {
        harness.register_tool(Arc::new(SleepTool {
            name,
            delay,
            ran: ran.clone(),
            active: active.clone(),
            max_seen: max_seen.clone(),
        }));
    }
    harness.push_tool_middleware(Arc::new(ProbeWrap {
        mode,
        concurrent_safe,
    }));
    Rig {
        harness,
        ran,
        max_seen,
    }
}

fn middleware_balance(recorder: &EventRecorder) -> (usize, usize) {
    let events = recorder.events();
    let started = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::MiddlewareStarted { name, .. } if name == "probe_wrap"))
        .count();
    let completed = events
        .iter()
        .filter(
            |e| matches!(e, AgentEvent::MiddlewareCompleted { name, .. } if name == "probe_wrap"),
        )
        .count();
    (started, completed)
}

fn tool_text(messages: &[Message], call_id: &str) -> Option<String> {
    messages.iter().find_map(|m| match m {
        Message::Tool(t) if t.tool_call_id == call_id => Some(m.text()),
        _ => None,
    })
}

fn ctx(recorder: &EventRecorder) -> RunContext<()> {
    RunContext::new(RunConfig::new("wrap-concurrency"), ()).with_events(recorder.sink())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn wrapped_batch_runs_in_max_not_sum_of_tool_latency() {
    let rig = rig(Mode::Stamp, true);
    let recorder = EventRecorder::new();

    let started = tokio::time::Instant::now();
    let run = rig
        .harness
        .invoke_in_context(&(), ctx(&recorder), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    let elapsed = started.elapsed();

    assert_eq!(run.tool_calls, 3);
    assert_eq!(
        rig.max_seen.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "wrapped tools must overlap"
    );
    assert!(
        elapsed >= DELAY && elapsed < DELAY * 2,
        "wall time must be ~max ({DELAY:?}), not ~sum; got {elapsed:?}"
    );
    // The wrap onion still ran around every call.
    assert_eq!(
        tool_text(&run.messages, "call-a").as_deref(),
        Some("[w] alpha-out")
    );
}

#[tokio::test(start_paused = true)]
async fn a_wrap_that_is_not_concurrent_safe_forces_the_serial_route() {
    let rig = rig(Mode::Stamp, false);
    let recorder = EventRecorder::new();

    let started = tokio::time::Instant::now();
    let run = rig
        .harness
        .invoke_in_context(&(), ctx(&recorder), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    let elapsed = started.elapsed();

    assert_eq!(run.tool_calls, 3);
    assert_eq!(rig.max_seen.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(
        elapsed >= DELAY * 2,
        "serial route sums latency; got {elapsed:?}"
    );
    assert_eq!(
        tool_text(&run.messages, "call-c").as_deref(),
        Some("[w] gamma-out")
    );
}

#[tokio::test(start_paused = true)]
async fn middleware_events_stay_balanced_and_results_keep_call_order() {
    let rig = rig(Mode::Stamp, true);
    let recorder = EventRecorder::new();

    let run = rig
        .harness
        .invoke_in_context(&(), ctx(&recorder), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(middleware_balance(&recorder), (3, 3));
    // alpha finishes last, gamma first; the transcript still lists a, b, c.
    let order: Vec<String> = run
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(t) => Some(t.tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(order, ["call-a", "call-b", "call-c"]);
    let completions: Vec<String> = recorder
        .events()
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolCompleted { call_id, .. } => Some(call_id.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(completions, ["call-a", "call-b", "call-c"]);
}

#[tokio::test(start_paused = true)]
async fn a_short_circuiting_wrap_on_one_call_leaves_siblings_untouched() {
    let rig = rig(Mode::ShortCircuitB, true);
    let recorder = EventRecorder::new();

    let run = rig
        .harness
        .invoke_in_context(&(), ctx(&recorder), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(
        tool_text(&run.messages, "call-a").as_deref(),
        Some("[w] alpha-out")
    );
    assert_eq!(
        tool_text(&run.messages, "call-b").as_deref(),
        Some("canned")
    );
    assert_eq!(
        tool_text(&run.messages, "call-c").as_deref(),
        Some("[w] gamma-out")
    );
    let mut ran = rig.ran.lock().unwrap().clone();
    ran.sort_unstable();
    assert_eq!(ran, ["alpha", "gamma"], "beta's tool never ran");
    assert_eq!(middleware_balance(&recorder), (3, 3));
}

#[tokio::test(start_paused = true)]
async fn a_wrap_error_on_one_call_fails_the_turn_without_starving_siblings() {
    let rig = rig(Mode::FailB, true);
    let recorder = EventRecorder::new();

    let error = rig
        .harness
        .invoke_in_context(&(), ctx(&recorder), vec![Message::user("go")])
        .await
        .expect_err("a fatal wrap error fails the turn");

    assert!(error.to_string().contains("wrap boom"), "{error}");
    let mut ran = rig.ran.lock().unwrap().clone();
    ran.sort_unstable();
    assert_eq!(ran, ["alpha", "gamma"], "siblings ran to completion");
    assert_eq!(middleware_balance(&recorder), (3, 3));
    // Every announced call got exactly one terminal event.
    let events = recorder.events();
    let started = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolStarted { .. }))
        .count();
    let terminal = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                AgentEvent::ToolCompleted { .. } | AgentEvent::ToolFailed { .. }
            )
        })
        .count();
    assert_eq!(started, 3);
    assert_eq!(terminal, 3);
}

#[tokio::test(start_paused = true)]
async fn a_deferral_raised_inside_a_wrap_folds_like_any_other_deferral() {
    let rig = rig(Mode::DeferB, true);
    let recorder = EventRecorder::new();

    let run = rig
        .harness
        .invoke_in_context(&(), ctx(&recorder), vec![Message::user("go")])
        .await
        .expect("a deferral is not an error");

    let deferred = run.deferred.expect("the run reports the pending approval");
    assert_eq!(deferred.approvals.len(), 1);
    assert_eq!(deferred.approvals[0].id, "call-b");
    assert_eq!(
        tool_text(&run.messages, "call-a").as_deref(),
        Some("[w] alpha-out")
    );
    assert_eq!(
        tool_text(&run.messages, "call-c").as_deref(),
        Some("[w] gamma-out")
    );
    assert!(tool_text(&run.messages, "call-b").is_none());
    assert_eq!(middleware_balance(&recorder), (3, 3));
    assert!(recorder.events().iter().any(|e| matches!(
        e,
        AgentEvent::ToolDeferred { call_id, .. } if call_id.as_str() == "call-b"
    )));
}

#[tokio::test(start_paused = true)]
async fn a_wrap_command_outcome_is_applied_by_the_concurrent_fold() {
    let rig = rig(Mode::CommandB, true);
    let recorder = EventRecorder::new();

    let run = rig
        .harness
        .invoke_in_context(&(), ctx(&recorder), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    // The queued control ends the run with the wrap's final answer, and the
    // siblings of the commanding call still ran and were answered.
    assert_eq!(run.text().as_deref(), Some("stopped by wrap"));
    assert_eq!(run.model_calls, 1, "the control ended the loop");
    assert_eq!(
        tool_text(&run.messages, "call-a").as_deref(),
        Some("[w] alpha-out")
    );
    assert_eq!(
        tool_text(&run.messages, "call-c").as_deref(),
        Some("[w] gamma-out")
    );
    assert_eq!(middleware_balance(&recorder), (3, 3));
}

#[tokio::test(start_paused = true)]
async fn tool_wrap_middleware_events_carry_the_call_id() {
    let rig = rig(Mode::Stamp, true);
    let recorder = EventRecorder::new();

    rig.harness
        .invoke_in_context(&(), ctx(&recorder), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let ids = |started: bool| {
        let mut ids: Vec<String> = recorder
            .events()
            .iter()
            .filter_map(|e| match e {
                AgentEvent::MiddlewareStarted { call_id, .. } if started => {
                    call_id.as_ref().map(|id| id.to_string())
                }
                AgentEvent::MiddlewareCompleted { call_id, .. } if !started => {
                    call_id.as_ref().map(|id| id.to_string())
                }
                _ => None,
            })
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(ids(true), ["call-a", "call-b", "call-c"]);
    assert_eq!(ids(false), ["call-a", "call-b", "call-c"]);
}
