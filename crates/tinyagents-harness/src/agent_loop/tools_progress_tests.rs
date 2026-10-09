//! Tests for live mid-tool progress: a tool reports through
//! `ToolRunContext::report_progress` and the loop turns each update into an
//! `AgentEvent::ToolProgress` that sits between that call's `ToolStarted` and
//! its terminal event, and replays it to `Middleware::on_tool_delta`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::error::Result;
use crate::events::AgentEvent;
use crate::middleware::{Middleware, ToolInvocationIdentity};
use crate::runtime::AgentHarness;
use crate::testkit::{EventRecorder, ScriptedModel};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::{ToolCall, ToolDelta};
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolCallOptions, ToolProgress, ToolResult, ToolRunContext};

/// A tool that reports `updates` (yielding between each so concurrent siblings
/// interleave) and then returns. When `stash` is set it keeps a clone of its
/// context so the test can report *after* the call has settled.
struct ProgressTool {
    name: &'static str,
    updates: Vec<&'static str>,
    safe: bool,
    stash: Option<Arc<Mutex<Option<crate::tool::ToolExecutionContext>>>>,
}

#[async_trait]
impl Tool for ProgressTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "reports progress"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn is_concurrency_safe(&self, _arguments: &serde_json::Value) -> bool {
        self.safe
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        unreachable!("the harness dispatches through execute_with_context")
    }
    async fn execute_with_context(
        &self,
        _arguments: serde_json::Value,
        _options: ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let context = context.expect("the loop supplies a context");
        if let Some(stash) = &self.stash {
            let harness = context
                .host_extension()
                .and_then(|any| any.downcast_ref::<crate::tool::ToolExecutionContext>())
                .expect("harness context");
            *stash.lock().unwrap() = Some(harness.clone());
        }
        for update in &self.updates {
            context.report_progress(ToolProgress::message(*update));
            tokio::task::yield_now().await;
        }
        Ok(ToolResult::success("done"))
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

fn calls(names: &[(&str, &str)]) -> ModelResponse {
    response(
        names
            .iter()
            .map(|(id, name)| ToolCall::new(*id, *name, json!({})))
            .collect(),
        "",
    )
}

struct Fixture {
    harness: AgentHarness<()>,
    recorder: EventRecorder,
}

fn fixture(first_turn: ModelResponse) -> Fixture {
    let model = Arc::new(ScriptedModel::new(vec![first_turn, response(vec![], "ok")]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model as _);
    Fixture {
        harness,
        recorder: EventRecorder::new(),
    }
}

impl Fixture {
    async fn run(&self) {
        let ctx =
            RunContext::new(RunConfig::new("run-progress"), ()).with_events(self.recorder.sink());
        self.harness
            .invoke_in_context(&(), ctx, vec![Message::user("go")])
            .await
            .expect("run succeeds");
    }
}

/// One event of interest, reduced to something comparable.
#[derive(Debug, PartialEq)]
enum Seen {
    Started(String),
    Progress(String, String),
    Completed(String),
    Failed(String),
}

fn seen(recorder: &EventRecorder) -> Vec<Seen> {
    recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::ToolStarted { call_id, .. } => Some(Seen::Started(call_id.to_string())),
            AgentEvent::ToolProgressDetail {
                call_id, message, ..
            } => Some(Seen::Progress(call_id.to_string(), message)),
            AgentEvent::ToolCompleted { call_id, .. } => Some(Seen::Completed(call_id.to_string())),
            AgentEvent::ToolFailed { call_id, .. } => Some(Seen::Failed(call_id.to_string())),
            _ => None,
        })
        .collect()
}

fn progress(call: &str, message: &str) -> Seen {
    Seen::Progress(call.to_string(), message.to_string())
}

#[tokio::test]
async fn three_updates_arrive_in_order_before_the_call_completes() {
    let mut fx = fixture(calls(&[("c1", "build")]));
    fx.harness.register_tool(Arc::new(ProgressTool {
        name: "build",
        updates: vec!["p1", "p2", "p3"],
        safe: false,
        stash: None,
    }));
    fx.run().await;

    assert_eq!(
        seen(&fx.recorder),
        vec![
            Seen::Started("c1".into()),
            progress("c1", "p1"),
            progress("c1", "p2"),
            progress("c1", "p3"),
            Seen::Completed("c1".into()),
        ]
    );
}

#[tokio::test]
async fn an_update_reported_after_completion_is_dropped() {
    let stash = Arc::new(Mutex::new(None));
    let mut fx = fixture(calls(&[("c1", "build")]));
    fx.harness.register_tool(Arc::new(ProgressTool {
        name: "build",
        updates: vec!["p1"],
        safe: false,
        stash: Some(stash.clone()),
    }));
    fx.run().await;
    let before = seen(&fx.recorder);

    let late = stash
        .lock()
        .unwrap()
        .take()
        .expect("tool stashed its context");
    late.report_progress(ToolProgress::message("too late"));

    assert_eq!(
        seen(&fx.recorder),
        before,
        "no event after the terminal one"
    );
    assert_eq!(before.last(), Some(&Seen::Completed("c1".into())));
}

#[tokio::test]
async fn concurrent_calls_interleave_but_each_precedes_its_own_terminal_event() {
    let mut fx = fixture(calls(&[("a", "alpha"), ("b", "beta")]));
    for (name, updates) in [("alpha", vec!["a1", "a2"]), ("beta", vec!["b1", "b2"])] {
        fx.harness.register_tool(Arc::new(ProgressTool {
            name,
            updates,
            safe: true,
            stash: None,
        }));
    }
    fx.run().await;
    let events = seen(&fx.recorder);

    let position = |wanted: &Seen| {
        events
            .iter()
            .position(|e| e == wanted)
            .unwrap_or_else(|| panic!("{wanted:?} missing from {events:?}"))
    };
    for (call, updates) in [("a", ["a1", "a2"]), ("b", ["b1", "b2"])] {
        let started = position(&Seen::Started(call.into()));
        let first = position(&progress(call, updates[0]));
        let second = position(&progress(call, updates[1]));
        let completed = position(&Seen::Completed(call.into()));
        assert!(
            started < first && first < second && second < completed,
            "{events:?}"
        );
    }
}

/// Records every tool delta and the point where `after_tool` ran.
struct DeltaLog(Arc<Mutex<Vec<String>>>);

#[async_trait]
impl Middleware<(), ()> for DeltaLog {
    fn name(&self) -> &str {
        "delta-log"
    }

    async fn on_tool_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut ToolDelta,
    ) -> Result<()> {
        self.0.lock().unwrap().push(format!(
            "{}:{}:{}",
            delta.call_id,
            delta.tool_name.clone().unwrap_or_default(),
            delta.content
        ));
        Ok(())
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        invocation: &ToolInvocationIdentity,
        _result: &mut ToolResult,
    ) -> Result<()> {
        self.0
            .lock()
            .unwrap()
            .push(format!("after:{}", invocation.call_id()));
        Ok(())
    }
}

struct RejectDelta(Arc<Mutex<Vec<String>>>);

#[async_trait]
impl Middleware<(), ()> for RejectDelta {
    fn name(&self) -> &str {
        "reject-delta"
    }

    async fn on_tool_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut ToolDelta,
    ) -> Result<()> {
        self.0.lock().unwrap().push(delta.content.clone());
        Err(crate::error::TinyAgentsError::Middleware(
            "delta rejected".into(),
        ))
    }
}

#[tokio::test]
async fn a_failing_delta_hook_does_not_fail_the_call_or_stop_replay() {
    let mut fx = fixture(calls(&[("c1", "build")]));
    let rejected = Arc::new(Mutex::new(Vec::new()));
    fx.harness
        .push_middleware(Arc::new(RejectDelta(rejected.clone())));
    fx.harness.register_tool(Arc::new(ProgressTool {
        name: "build",
        updates: vec!["p1", "p2"],
        safe: false,
        stash: None,
    }));

    fx.run().await;
    assert_eq!(
        seen(&fx.recorder),
        vec![
            Seen::Started("c1".into()),
            progress("c1", "p1"),
            progress("c1", "p2"),
            Seen::Completed("c1".into()),
        ]
    );
    assert_eq!(*rejected.lock().unwrap(), vec!["p1", "p2"]);
}

#[tokio::test]
async fn middleware_observes_every_update_before_the_call_settles() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut fx = fixture(calls(&[("c1", "build")]));
    fx.harness.push_middleware(Arc::new(DeltaLog(log.clone())));
    fx.harness.register_tool(Arc::new(ProgressTool {
        name: "build",
        updates: vec!["p1", "p2", "p3"],
        safe: false,
        stash: None,
    }));
    fx.run().await;

    assert_eq!(
        *log.lock().unwrap(),
        vec!["c1:build:p1", "c1:build:p2", "c1:build:p3", "after:c1"]
    );
}

#[tokio::test]
async fn middleware_observes_concurrent_progress_per_call_in_order() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut fx = fixture(calls(&[("a", "alpha"), ("b", "beta")]));
    fx.harness.push_middleware(Arc::new(DeltaLog(log.clone())));
    for (name, updates) in [("alpha", vec!["a1", "a2"]), ("beta", vec!["b1", "b2"])] {
        fx.harness.register_tool(Arc::new(ProgressTool {
            name,
            updates,
            safe: true,
            stash: None,
        }));
    }
    fx.run().await;

    let log = log.lock().unwrap();
    let index = |entry: &str| {
        log.iter()
            .position(|l| l == entry)
            .unwrap_or_else(|| panic!("{entry} missing from {log:?}"))
    };
    assert!(index("a:alpha:a1") < index("a:alpha:a2") && index("a:alpha:a2") < index("after:a"));
    assert!(index("b:beta:b1") < index("b:beta:b2") && index("b:beta:b2") < index("after:b"));
}

/// Reports once, then fails the way a crashed tool does.
struct ReportThenFail;

#[async_trait]
impl Tool for ReportThenFail {
    fn name(&self) -> &str {
        "crash"
    }
    fn description(&self) -> &str {
        "reports then fails"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        unreachable!("the harness dispatches through execute_with_context")
    }
    async fn execute_with_context(
        &self,
        _arguments: serde_json::Value,
        _options: ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        if let Some(context) = context {
            context.report_progress(ToolProgress::message("about to fail"));
        }
        Err(anyhow::anyhow!("boom"))
    }
}

#[tokio::test]
async fn progress_from_a_call_that_then_fails_still_precedes_its_failure() {
    let mut fx = fixture(calls(&[("c1", "crash")]));
    fx.harness.register_tool(Arc::new(ReportThenFail));
    let ctx = RunContext::new(RunConfig::new("run-progress"), ()).with_events(fx.recorder.sink());
    let outcome = fx
        .harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await;

    let events = seen(&fx.recorder);
    let progress_at = events
        .iter()
        .position(|e| e == &progress("c1", "about to fail"))
        .unwrap_or_else(|| panic!("progress missing from {events:?} (run: {outcome:?})"));
    let terminal_at = events
        .iter()
        .position(|e| e == &Seen::Failed("c1".into()))
        .expect("the call has a terminal event");
    assert!(progress_at < terminal_at, "{events:?}");
}

/// Reports once, hands its context to the test, then never returns.
struct HangAfterReport(Arc<Mutex<Option<crate::tool::ToolExecutionContext>>>);

#[async_trait]
impl Tool for HangAfterReport {
    fn name(&self) -> &str {
        "hang"
    }
    fn description(&self) -> &str {
        "reports then hangs"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        unreachable!("the harness dispatches through execute_with_context")
    }
    async fn execute_with_context(
        &self,
        _arguments: serde_json::Value,
        _options: ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let context = context.expect("context");
        let harness = context
            .host_extension()
            .and_then(|any| any.downcast_ref::<crate::tool::ToolExecutionContext>())
            .expect("harness context");
        *self.0.lock().unwrap() = Some(harness.clone());
        context.report_progress(ToolProgress::message("working"));
        std::future::pending().await
    }
}

#[tokio::test]
async fn cancelling_the_run_mid_call_silences_a_sink_a_spawned_task_still_holds() {
    let stash = Arc::new(Mutex::new(None));
    let mut fx = fixture(calls(&[("c1", "hang")]));
    fx.harness
        .register_tool(Arc::new(HangAfterReport(stash.clone())));
    // Dropping the run future (as a cancelled or timed-out run does) is the
    // only way this call ends.
    let _ = tokio::time::timeout(std::time::Duration::from_millis(100), fx.run()).await;
    let before = seen(&fx.recorder);
    assert!(before.contains(&progress("c1", "working")), "{before:?}");

    let held = stash.lock().unwrap().take().expect("tool ran");
    held.report_progress(ToolProgress::message("from a straggler"));

    assert_eq!(seen(&fx.recorder), before);
}
