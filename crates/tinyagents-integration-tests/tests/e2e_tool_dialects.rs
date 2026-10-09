//! End-to-end coverage for tool dialects through the harness.
//!
//! The protocol crate (`tinytools-agent`) owns how a call is rendered and
//! parsed; these tests pin the *host* half: a native model that narrates a
//! call as text — in any grammar — still dispatches it with a harness-minted
//! id; a forced text dialect strips the schemas off the wire and renders the
//! protocol instead; streamed text never shows tool-call markup to a
//! consumer; and a P-Format run parses positional calls.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::config::ToolDispatcher;
use tinyagents_harness::context::RunContext;
use tinyagents_harness::events::{AgentEvent, RecordingListener};
use tinyagents_harness::middleware::{
    Middleware, MiddlewareModelOutcome, ModelHandler, ModelMiddleware,
};
use tinyagents_harness::runtime::{AgentHarness, EndStrategy, RunPolicy};
use tinyagents_harness::testkit::{FakeTool, ScriptedModel, StreamingMock};
use tinyinference_llm::message::{Message, MessageDelta};
use tinyinference_llm::model::{
    ChatModel, ModelDelta, ModelProfile, ModelRequest, ModelResponse, ModelStreamItem,
    ResponseFormat, ToolChoice,
};
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolExposure, ToolResult};

struct CaptureMiddleware {
    listener: Arc<RecordingListener>,
}

#[async_trait]
impl Middleware<(), ()> for CaptureMiddleware {
    fn name(&self) -> &str {
        "capture"
    }

    async fn before_agent(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
    ) -> tinyagents_harness::Result<()> {
        ctx.events.subscribe(self.listener.clone());
        Ok(())
    }
}

/// A tool with a real parameter, so P-Format has a slot to render.
struct Lookup;

#[async_trait]
impl Tool for Lookup {
    fn name(&self) -> &str {
        "lookup"
    }

    fn description(&self) -> &str {
        "Looks something up."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "q": { "type": "string" } },
            "required": ["q"]
        })
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("tool-output"))
    }
}

/// Every tool-call id the run dispatched, from the tool-started events.
fn dispatched_ids(listener: &RecordingListener) -> Vec<String> {
    listener
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::ToolStarted { call_id, .. } => Some(call_id.to_string()),
            _ => None,
        })
        .collect()
}

/// A native-profile model that answers with text only, once, then finishes.
fn narrating_model(text: &str) -> MockModel {
    MockModel::with_responses(vec![
        ModelResponse::assistant(text),
        ModelResponse::assistant("done"),
    ])
}

fn harness_with(
    model: Arc<dyn ChatModel<()>>,
    listener: &Arc<RecordingListener>,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model)
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }));
    harness
}

/// Mimics the failure in a real conversation: a greeting precedes a request
/// to fetch email, and a discovered tool must still be invoked after search.
struct EmailContinuationModel {
    calls: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for EmailContinuationModel {
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let mut calls = self.calls.lock().unwrap();
        let response = match *calls {
            0 => "<tool_call>tool_search(query=\"fetch emails from Gmail\")</tool_call>",
            1 => {
                let latest_user = request
                    .messages
                    .iter()
                    .rev()
                    .find(|message| matches!(message, Message::User(_)))
                    .map(Message::text)
                    .unwrap_or_default();
                if latest_user.contains("fetch my latest email")
                    && latest_user.contains("GMAIL_FETCH_EMAILS")
                {
                    "<tool_call>GMAIL_FETCH_EMAILS()</tool_call>"
                } else {
                    "Hey! What's up?"
                }
            }
            _ => "Email fetched.",
        };
        *calls += 1;
        Ok(ModelResponse::assistant(response))
    }
}

struct DeferredEmail {
    calls: Mutex<usize>,
}

#[async_trait]
impl Tool for DeferredEmail {
    fn name(&self) -> &str {
        "GMAIL_FETCH_EMAILS"
    }

    fn description(&self) -> &str {
        "Fetch email messages from Gmail."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Deferred
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success("one email found"))
    }
}

#[tokio::test]
async fn a_greeting_does_not_replace_the_email_request_after_tool_search() {
    let model = Arc::new(EmailContinuationModel {
        calls: Mutex::new(0),
    });
    let email = Arc::new(DeferredEmail {
        calls: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(email.clone())
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Python,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(
            &(),
            vec![
                Message::user("Hi"),
                Message::assistant("Hey! What's up?"),
                Message::user("fetch my latest email"),
            ],
        )
        .await
        .expect("email continuation completes");

    assert_eq!(*email.calls.lock().unwrap(), 1, "deferred email tool ran");
    assert_eq!(*model.calls.lock().unwrap(), 3, "search, fetch, answer");
    assert_eq!(run.text().as_deref(), Some("Email fetched."));
}

#[tokio::test]
async fn a_native_model_narrating_a_call_in_any_grammar_dispatches_it() {
    for text in [
        "Let me check. <tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">{\"q\":\"x\"}</｜DSML｜invoke></｜DSML｜tool_calls>",
        "<｜DSML｜tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</｜DSML｜tool_call>",
        "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"q\":\"x\"}<｜tool▁call▁end｜>",
        "<|channel|>commentary to=functions.lookup<|message|>{\"q\":\"x\"}<|call|>",
        "<tool_call>{\"name\":\"functions.lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
    ] {
        let listener = Arc::new(RecordingListener::new());
        let mut harness = harness_with(Arc::new(narrating_model(text)), &listener);
        harness.with_policy(RunPolicy {
            text_dialect_recovery: tinyagents_harness::runtime::TextDialectRecovery::On,
            ..RunPolicy::default()
        });
        let run = harness
            .invoke_default(&(), vec![Message::user("go")])
            .await
            .expect("run succeeds");
        assert_eq!(run.tool_calls, 1, "{text}");
        let ids = dispatched_ids(&listener);
        assert_eq!(ids.len(), 1, "{text}");
        assert!(
            ids[0].ends_with("-tool-1"),
            "harness-minted id, got {}: {text}",
            ids[0]
        );
    }
}

#[tokio::test]
async fn an_unknown_narrated_tool_is_not_invented_into_a_known_one() {
    let listener = Arc::new(RecordingListener::new());
    let harness = harness_with(
        Arc::new(narrating_model(
            "<tool_call>{\"name\":\"launch_missiles\",\"arguments\":{}}</tool_call>",
        )),
        &listener,
    );
    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run completes under the default unknown-tool policy");
    // The name reaches the unknown-tool policy as written; it is never
    // fuzzed onto the one registered tool.
    let started: Vec<String> = listener
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::ToolStarted { tool_name, .. } => Some(tool_name),
            _ => None,
        })
        .collect();
    assert!(!started.iter().any(|name| name == "lookup"), "{started:?}");
    assert!(run.model_calls >= 1);
}

#[tokio::test]
async fn a_narrated_call_is_not_dispatched_when_tool_choice_is_forced_to_none() {
    // `apply_to_request` already skips its own dialect rewrite for
    // `ToolChoice::None`, but that alone did not stop recovery: the offered
    // tool names were still recorded for the scrubber/`recover_text_calls`
    // regardless of the effective choice, so a model that narrated
    // `<tool_call>` markup as plain text anyway still had it parsed and
    // dispatched as a real, side-effecting call despite the caller's
    // explicit "no tool calls this turn".
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(
        Arc::new(narrating_model(
            "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        )),
        &listener,
    );
    harness.push_middleware(Arc::new(ForceToolChoice(ToolChoice::None)));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run completes");

    assert_eq!(
        run.tool_calls, 0,
        "a narrated call must not be dispatched when the effective tool_choice is None"
    );
    let started: Vec<String> = listener
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::ToolStarted { tool_name, .. } => Some(tool_name),
            _ => None,
        })
        .collect();
    assert!(started.is_empty(), "{started:?}");
}

#[tokio::test]
async fn a_forced_xml_dialect_renders_the_protocol_and_sends_no_schemas() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model.clone(), &listener);
    harness.with_policy(RunPolicy {
        tool_dialect: ToolDispatcher::Xml,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert!(request.tools.is_empty(), "no schema goes on the wire");
        let system = request
            .messages
            .iter()
            .find(|m| matches!(m, Message::System(_)))
            .expect("a system turn carries the protocol")
            .text();
        assert!(system.contains("## Tool Use Protocol"));
        assert!(system.contains("**lookup**"));
    }
    // The second request replays the call as text and folds the result.
    let replay = &requests[1];
    let assistant = replay
        .messages
        .iter()
        .find(|m| matches!(m, Message::Assistant(_)))
        .expect("assistant turn replayed")
        .text();
    assert!(assistant.contains("<tool_call>"), "{assistant}");
    assert!(
        replay
            .messages
            .iter()
            .any(|m| m.text().contains("<tool_result id=")),
        "results folded into the text envelope"
    );
    assert!(
        !replay
            .messages
            .iter()
            .any(|m| matches!(m, Message::Tool(_)))
    );
}

#[tokio::test]
async fn a_forced_pformat_dialect_parses_positional_calls() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>lookup[0|needle]</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Pformat,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);
    let system = model.requests()[0]
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(system.contains("P-Format"), "{system}");
    assert!(system.contains("lookup[0|<q>]"), "{system}");
}

#[tokio::test]
async fn a_forced_python_dialect_parses_code_calls_with_signatures_in_the_prompt() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "Looking it up.\n<tool_call>\nlookup(q=\"needle\")\n</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Python,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);
    let ids = dispatched_ids(&listener);
    assert_eq!(ids.len(), 1);
    assert!(
        ids[0].ends_with("-tool-1"),
        "harness-minted id, got {}",
        ids[0]
    );

    let first = &model.requests()[0];
    assert!(first.tools.is_empty(), "schemas must not go on the wire");
    let system = first
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(system.contains("## Tool Use Protocol"), "{system}");
    assert!(
        system.contains("def lookup(q: str) -> str  # Looks something up."),
        "{system}"
    );
    assert!(
        !system.contains("\"type\": \"object\""),
        "no JSON schema: {system}"
    );
}

#[tokio::test]
async fn a_forced_typescript_dialect_parses_object_calls() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>lookup({q: \"needle\"})</tool_call>",
        "done",
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Typescript,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);
    let system = model.requests()[0]
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(
        system.contains("function lookup(q: string): string;  // Looks something up."),
        "{system}"
    );
}

#[tokio::test]
async fn an_untrusted_protocol_heading_does_not_suppress_the_current_catalogue() {
    // A system prompt may mention the heading without containing the active
    // dialect, the final post-middleware catalogue, or the effective tool
    // choice. The loop must render its authoritative block from the request
    // rather than treating user-controlled prompt text as provenance.
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>lookup(q=\"needle\")</tool_call>",
        "done",
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Python,
            ..RunPolicy::default()
        });

    let host_prompt = "You are a helper.\n\n## Tool Use Protocol\n\nThis heading is documentation, not a catalogue.";
    let run = harness
        .invoke_default(&(), vec![Message::system(host_prompt), Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1, "the call still parses");
    let first = &model.requests()[0];
    assert!(first.tools.is_empty(), "schemas still come off the wire");
    let system = first
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert_eq!(
        system.matches("## Tool Use Protocol").count(),
        2,
        "the canonical protocol must be appended: {system}"
    );
    assert!(system.contains("def lookup(q: str) -> str"), "{system}");
    assert!(system.contains("Call a tool by writing"), "{system}");
}

/// Middleware that forces `tool_choice` before the dialect rewrite runs, the
/// same shape a caller or another middleware forcing a specific tool would
/// produce.
struct ForceToolChoice(ToolChoice);

#[async_trait]
impl Middleware<(), ()> for ForceToolChoice {
    fn name(&self) -> &str {
        "force-tool-choice"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> tinyagents_harness::Result<()> {
        request.tool_choice = self.0.clone();
        Ok(())
    }
}

#[tokio::test]
async fn a_forced_pformat_dialect_preserves_a_forced_required_tool_choice() {
    // Unlike the XML branch (`prompt_tools::with_tool_instructions`, which
    // renders `tool_choice` into its instructions), P-Format has no schema on
    // the wire either — a forced choice has to survive as plain English in
    // the rendered prompt or it silently loses its meaning once the wire
    // `tool_choice` is reset to `Auto`.
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>lookup[0|needle]</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .push_middleware(Arc::new(ForceToolChoice(ToolChoice::Required)))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Pformat,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);

    let system = model.requests()[0]
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(
        system.contains("You must emit at least one tool call."),
        "{system}"
    );
    // The wire choice is reset to `Auto` (no schema is on the wire for a
    // text dialect), so this asserts the prompt carries the constraint
    // instead, not that the wire field kept it.
    assert_eq!(model.requests()[0].tool_choice, ToolChoice::Auto);
}

#[tokio::test]
async fn a_forced_pformat_dialect_preserves_a_forced_named_tool_choice() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>lookup[0|needle]</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .push_middleware(Arc::new(ForceToolChoice(ToolChoice::Tool(
            "lookup".to_string(),
        ))))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Pformat,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);

    let system = model.requests()[0]
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(
        system.contains("You must call the `lookup` tool."),
        "{system}"
    );
}

/// Middleware recording every visible text delta the harness emits.
struct DeltaRecorder {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<(), ()> for DeltaRecorder {
    fn name(&self) -> &str {
        "deltas"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut ModelDelta,
    ) -> tinyagents_harness::Result<()> {
        self.seen.lock().unwrap().push(delta.content.clone());
        Ok(())
    }
}

/// Middleware that redacts every occurrence of `"too"` from a visible delta,
/// standing in for any redaction/policy/transformation middleware a host
/// might install on [`Middleware::on_model_delta`].
struct RedactMiddleware;

#[async_trait]
impl Middleware<(), ()> for RedactMiddleware {
    fn name(&self) -> &str {
        "redact"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut ModelDelta,
    ) -> tinyagents_harness::Result<()> {
        delta.content = delta.content.replace("too", "REDACTED");
        Ok(())
    }
}

#[tokio::test]
async fn a_flushed_stream_tail_that_was_only_a_marker_false_alarm_still_hits_delta_middleware() {
    // A fragment such as `<too` looks like it could be opening a tool-call
    // marker, so the scrubber holds it back rather than forwarding it live.
    // The stream ends without ever completing a marker, so the held-back
    // remainder turns out to have been ordinary text all along and is
    // released from `StreamScrubber::flush` at `Completed`. That release
    // must go through the same `on_model_delta` middleware pipeline as every
    // other delta — a redaction/policy/transformation middleware installed
    // by the host has to see it too, or this one piece of visible text
    // silently bypasses every such middleware while everything around it
    // does not.
    let chunks = ["Hi there ", "<too"];
    let full: String = chunks.concat();
    let mut items = vec![ModelStreamItem::Started];
    items.extend(
        chunks
            .iter()
            .map(|chunk| ModelStreamItem::MessageDelta(MessageDelta::text(*chunk))),
    );
    items.push(ModelStreamItem::Completed(ModelResponse::assistant(full)));
    let model = Arc::new(StreamingMock::new(items));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness
        // Order matters: middleware runs in push order, and `DeltaRecorder`
        // must observe the *result* of redaction, not race it.
        .push_middleware(Arc::new(RedactMiddleware))
        .push_middleware(Arc::new(DeltaRecorder { seen: seen.clone() }))
        .with_policy(RunPolicy {
            limits: tinyagents_harness::limits::RunLimits {
                max_model_calls: 1,
                ..tinyagents_harness::limits::RunLimits::default()
            },
            ..RunPolicy::default()
        });

    let _ = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await;

    let deltas = seen.lock().unwrap().clone();
    let joined = deltas.concat();
    assert!(
        joined.contains("REDACTED"),
        "the flushed tail never reached on_model_delta: {deltas:?}"
    );
    assert!(
        !joined.contains("too"),
        "the flushed tail bypassed redaction: {deltas:?}"
    );
}

/// Middleware that suppresses a visible delta entirely, standing in for a
/// redaction/policy middleware that decides a whole fragment must not reach
/// the transcript.
struct SuppressAllMiddleware;

#[async_trait]
impl Middleware<(), ()> for SuppressAllMiddleware {
    fn name(&self) -> &str {
        "suppress-all"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut ModelDelta,
    ) -> tinyagents_harness::Result<()> {
        delta.content.clear();
        Ok(())
    }
}

#[tokio::test]
async fn a_fully_suppressed_flushed_tail_does_not_restore_raw_provider_content() {
    // The whole stream is a single fragment that looks like it could be
    // opening a tool-call marker (`<too`), so the scrubber holds it back and
    // no ordinary delta ever reaches `on_model_delta` through the live path.
    // The held-back text turns out to be ordinary after all and is released
    // at `Completed`, where a middleware suppresses it entirely (as a
    // redaction middleware legitimately might). Reconciling terminal content
    // must still happen in that case — gating it on the *post*-middleware
    // content being non-empty would skip reconciliation and leave the
    // provider's raw (un-suppressed) `Completed` text in the transcript,
    // silently restoring exactly what the middleware just removed.
    let text = "<too";
    let items = vec![
        ModelStreamItem::Started,
        ModelStreamItem::MessageDelta(MessageDelta::text(text)),
        ModelStreamItem::Completed(ModelResponse::assistant(text)),
    ];
    let model = Arc::new(StreamingMock::new(items));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness
        .push_middleware(Arc::new(SuppressAllMiddleware))
        .with_policy(RunPolicy {
            limits: tinyagents_harness::limits::RunLimits {
                max_model_calls: 1,
                behavior: tinyagents_harness::limits::LimitBehavior::StopWithPartial,
                ..tinyagents_harness::limits::RunLimits::default()
            },
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run stops cleanly");

    let assistant = run
        .messages
        .iter()
        .find(|m| matches!(m, Message::Assistant(_)))
        .expect("assistant turn");
    assert!(
        !assistant.text().contains("too"),
        "the raw provider content was restored, defeating the suppression middleware: {assistant:?}"
    );
}

#[tokio::test]
async fn streamed_tool_call_markup_never_reaches_consumers() {
    let chunks = [
        "Sure, ",
        "<tool_",
        "call>{\"name\":\"lookup\",",
        "\"arguments\":{\"q\":\"x\"}}</tool_call>",
        " checking.",
    ];
    let full: String = chunks.concat();
    let mut items = vec![ModelStreamItem::Started];
    items.extend(
        chunks
            .iter()
            .map(|chunk| ModelStreamItem::MessageDelta(MessageDelta::text(*chunk))),
    );
    items.push(ModelStreamItem::Completed(ModelResponse::assistant(full)));
    // The scripted stream replays the same call every turn; one model call is
    // enough to observe the dispatch and the scrubbed deltas.
    let model = Arc::new(StreamingMock::new(items));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness
        .push_middleware(Arc::new(DeltaRecorder { seen: seen.clone() }))
        .with_policy(RunPolicy {
            limits: tinyagents_harness::limits::RunLimits {
                max_model_calls: 1,
                ..tinyagents_harness::limits::RunLimits::default()
            },
            ..RunPolicy::default()
        });

    let _ = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await;

    let deltas = seen.lock().unwrap().clone();
    let joined = deltas.concat();
    assert!(!joined.contains("<tool_call"), "markup leaked: {deltas:?}");
    assert!(
        !joined.contains("</tool_call>"),
        "markup leaked: {deltas:?}"
    );
    assert!(joined.contains("Sure, "), "{deltas:?}");
    assert!(joined.contains(" checking."), "{deltas:?}");
    let ids = dispatched_ids(&listener);
    assert_eq!(ids.len(), 1, "the scrubbed call still dispatches once");
    assert!(ids[0].ends_with("-tool-1"));
}

#[tokio::test]
async fn a_pure_tool_call_stream_leaves_no_raw_markup_in_the_terminal_response() {
    // A response that is *only* tool-call markup, with no ordinary text
    // around it, suppresses every delta (the scrubber holds all of it back),
    // so terminal-content reconciliation must not depend on having seen any
    // *ordinary* streamed text — only on the scrubber having recovered a
    // call. Otherwise the raw `<tool_call>` text produced by the provider
    // (not the scrubbed one) survives in the terminal response's content
    // block, gets persisted into the transcript, and is replayed to the
    // model on the very next turn alongside the structured call.
    let markup = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
    let items = vec![
        ModelStreamItem::Started,
        ModelStreamItem::MessageDelta(MessageDelta::text(markup)),
        ModelStreamItem::Completed(ModelResponse::assistant(markup)),
    ];
    let model = Arc::new(StreamingMock::new(items));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness
        .push_middleware(Arc::new(DeltaRecorder { seen: seen.clone() }))
        .with_policy(RunPolicy {
            limits: tinyagents_harness::limits::RunLimits {
                max_model_calls: 1,
                behavior: tinyagents_harness::limits::LimitBehavior::StopWithPartial,
                ..tinyagents_harness::limits::RunLimits::default()
            },
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run stops cleanly with the partial transcript at the call cap");

    let deltas = seen.lock().unwrap().clone();
    let joined = deltas.concat();
    assert!(joined.is_empty(), "no ordinary text streamed: {deltas:?}");

    let ids = dispatched_ids(&listener);
    assert_eq!(ids.len(), 1, "the scrubbed call still dispatches once");

    // The persisted transcript must not carry the raw markup anywhere,
    // including on the assistant turn the terminal response became.
    for message in &run.messages {
        assert!(
            !message.text().contains("<tool_call"),
            "raw markup leaked into the transcript: {:?}",
            run.messages
        );
    }
}

#[tokio::test]
async fn a_native_call_and_a_narrated_text_call_are_both_dispatched_non_streaming() {
    // A provider can legitimately return one native structured call *and*
    // narrate a second one as text in the same response. `recover_text_calls`
    // used to return immediately whenever `tool_calls` was already
    // non-empty, silently dropping the narrated call — never authorized,
    // never executed. It must now parse the text regardless and append the
    // recovered call(s) to the native one(s).
    let markup = "<tool_call>{\"name\":\"second\",\"arguments\":{\"q\":\"y\"}}</tool_call>";
    let mut mixed = ModelResponse::assistant(markup);
    mixed
        .message
        .tool_calls
        .push(ToolCall::new("native-1", "lookup", json!({"q": "x"})));
    let model = Arc::new(ScriptedModel::new(vec![
        mixed,
        ModelResponse::assistant("done"),
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "lookup-output")))
        .register_tool(Arc::new(FakeTool::returning("second", "second-output")))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.tool_calls, 2, "{:?}", run.messages);
    let started: Vec<String> = listener
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::ToolStarted { tool_name, .. } => Some(tool_name),
            _ => None,
        })
        .collect();
    assert!(started.contains(&"lookup".to_string()), "{started:?}");
    assert!(started.contains(&"second".to_string()), "{started:?}");
}

#[tokio::test]
async fn a_native_call_and_a_narrated_text_call_are_both_dispatched_when_streamed() {
    // The streaming counterpart of the non-streaming test above: the
    // `DeltaScrubber`-recovered call attach point in `model_call.rs` had the
    // identical bug (gated on `tool_calls.is_empty()`), and fixing only
    // `recover_text_calls` would not cover it — by the time the terminal
    // response reaches `recover_text_calls`, the streaming reconciliation
    // path has already scrubbed the narrated markup out of the visible
    // text, so there is nothing left in `response.text()` for
    // `recover_text_calls` to recover a second time.
    let markup = "<tool_call>{\"name\":\"second\",\"arguments\":{\"q\":\"y\"}}</tool_call>";
    let mut completed = ModelResponse::assistant(markup);
    completed
        .message
        .tool_calls
        .push(ToolCall::new("native-1", "lookup", json!({"q": "x"})));
    let items = vec![
        ModelStreamItem::Started,
        ModelStreamItem::MessageDelta(MessageDelta::text(markup)),
        ModelStreamItem::Completed(completed),
    ];
    let model = Arc::new(StreamingMock::new(items));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "lookup-output")))
        .register_tool(Arc::new(FakeTool::returning("second", "second-output")))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .with_policy(RunPolicy {
            limits: tinyagents_harness::limits::RunLimits {
                max_model_calls: 1,
                behavior: tinyagents_harness::limits::LimitBehavior::StopWithPartial,
                ..tinyagents_harness::limits::RunLimits::default()
            },
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run stops cleanly");

    assert_eq!(run.tool_calls, 2, "{:?}", run.messages);
    let started: Vec<String> = listener
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::ToolStarted { tool_name, .. } => Some(tool_name),
            _ => None,
        })
        .collect();
    assert!(started.contains(&"lookup".to_string()), "{started:?}");
    assert!(started.contains(&"second".to_string()), "{started:?}");
}

#[tokio::test]
async fn a_signalled_but_missing_tool_call_is_re_prompted_then_recovered() {
    let mut promised = ModelResponse::assistant("");
    promised.finish_reason = Some("tool_calls".into());
    let model = Arc::new(ScriptedModel::new(vec![
        promised,
        ModelResponse::assistant(
            "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        ),
        ModelResponse::assistant("done"),
    ]));
    let listener = Arc::new(RecordingListener::new());
    let harness = harness_with(model.clone(), &listener);

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 3, "one nudge, one call, one final");
    assert_eq!(run.tool_calls, 1);
    let second = &model.requests()[1];
    let last = second.messages.last().expect("nudge appended").text();
    assert!(last.contains("issue the actual tool call now"), "{last}");
}

#[tokio::test]
async fn dropped_tool_call_nudges_are_bounded() {
    let mut promised = ModelResponse::assistant("");
    promised.finish_reason = Some("tool_calls".into());
    let model = Arc::new(ScriptedModel::new(vec![
        promised.clone(),
        promised.clone(),
        promised.clone(),
        promised,
    ]));
    let listener = Arc::new(RecordingListener::new());
    let harness = harness_with(model, &listener);

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run ends instead of looping");
    assert_eq!(
        run.model_calls, 4,
        "three nudges, then the answer is taken as final"
    );
    assert_eq!(run.tool_calls, 0);
}

#[tokio::test]
async fn exhausted_model_budget_does_not_schedule_a_dropped_call_nudge() {
    use tinyagents_harness::TinyAgentsError;
    use tinyagents_harness::context::RunConfig;
    use tinyagents_harness::events::LimitKind;
    use tinyagents_harness::limits::{LimitBehavior, RunLimits};

    let mut missing_call = ModelResponse::assistant("");
    missing_call.finish_reason = Some("tool_calls".into());
    for (response, dialect) in [
        (missing_call, ToolDispatcher::Auto),
        (
            ModelResponse::assistant(UNDECODABLE_CALL),
            ToolDispatcher::Xml,
        ),
    ] {
        for behavior in [LimitBehavior::Error, LimitBehavior::StopWithPartial] {
            let model = Arc::new(ScriptedModel::new(vec![response.clone()]));
            let listener = Arc::new(RecordingListener::new());
            let mut harness = harness_with(model.clone(), &listener);
            harness.with_policy(RunPolicy {
                limits: RunLimits {
                    max_model_calls: 1,
                    behavior,
                    ..RunLimits::default()
                },
                tool_dialect: dialect,
                ..RunPolicy::default()
            });

            let outcome = harness
                .invoke_collecting_partial(
                    &(),
                    (),
                    RunConfig::new("exhausted-nudge"),
                    vec![Message::user("go")],
                )
                .await;

            assert_eq!(model.requests().len(), 1);
            assert_eq!(outcome.run.tool_calls, 0);
            match behavior {
                LimitBehavior::Error => assert!(matches!(
                    outcome.error,
                    Some(TinyAgentsError::LimitExceeded(_))
                )),
                LimitBehavior::StopWithPartial => assert!(outcome.error.is_none()),
            }
            let events = listener.events();
            assert!(events.iter().any(|record| matches!(
                record.event,
                AgentEvent::LimitReached {
                    kind: LimitKind::ModelCalls
                }
            )));
            assert!(
                !events
                    .iter()
                    .any(|record| matches!(record.event, AgentEvent::RetryScheduled { .. })),
                "no model call remains for the scheduled retry: {events:?}; transcript: {:?}",
                outcome.run.messages
            );
            let user_messages: Vec<_> = outcome
                .run
                .messages
                .iter()
                .filter(|message| matches!(message, Message::User(_)))
                .map(Message::text)
                .collect();
            assert_eq!(user_messages, vec!["go"], "no unused recovery prompt");
        }
    }
}

#[tokio::test]
async fn no_dropped_call_nudge_is_issued_when_the_turn_could_not_accept_a_tool_call() {
    // A provider/router can report `finish_reason == "tool_calls"` with no
    // actual call even when this turn's effective `tool_choice` is `None`
    // (set by a `before_model` middleware) — nudging the model to "issue the
    // call" in that situation asks for something that could never have been
    // accepted, wasting `dropped_tool_call_nudges` model calls before
    // falling through to the same terminal outcome a single call would have
    // reached immediately.
    let mut promised = ModelResponse::assistant("");
    promised.finish_reason = Some("tool_calls".into());
    let model = Arc::new(ScriptedModel::new(vec![promised]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness.push_middleware(Arc::new(ForceToolChoice(ToolChoice::None)));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run ends on the first call instead of nudging");
    assert_eq!(
        run.model_calls, 1,
        "no nudge should be spent on a turn that could not accept a tool call"
    );
    let nudges = listener
        .events()
        .into_iter()
        .filter(|record| matches!(record.event, AgentEvent::RetryScheduled { .. }))
        .count();
    assert_eq!(nudges, 0, "{:?}", listener.events());
}

/// A queued model with a fixed, caller-chosen profile, so a test can force
/// `StructuredStrategy::ToolCall` (a profile with `tool_calling` but not
/// `native_structured_output && json_schema`) while still scripting a
/// specific sequence of responses. [`ScriptedModel`] cannot do this: it
/// advertises no profile at all, which `StructuredStrategy::for_profile`
/// resolves to `ProviderSchema`, not `ToolCall`.
struct ProfiledScriptedModel {
    profile: ModelProfile,
    queue: Mutex<std::collections::VecDeque<ModelResponse>>,
    received: Mutex<Vec<ModelRequest>>,
}

impl ProfiledScriptedModel {
    fn new(profile: ModelProfile, responses: Vec<ModelResponse>) -> Self {
        Self {
            profile,
            queue: Mutex::new(responses.into()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Every request received so far, in order.
    fn requests(&self) -> Vec<ModelRequest> {
        self.received.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatModel<()> for ProfiledScriptedModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.received.lock().unwrap().push(request);
        self.queue
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| tinyinference_llm::Error::Model("queue exhausted".to_string()))
    }
}

/// Builds an assistant response carrying several tool calls in one turn, with
/// no text.
fn multi_tool_call_response(calls: Vec<(&str, &str)>) -> ModelResponse {
    let mut response = ModelResponse::assistant("");
    response.finish_reason = Some("tool_calls".to_string());
    response.message.tool_calls = calls
        .into_iter()
        .map(|(id, name)| ToolCall::new(id, name, json!({})))
        .collect();
    response
}

#[tokio::test]
async fn dropped_call_nudge_budget_resets_after_a_mixed_structured_and_tool_turn() {
    // Regression: the mixed-turn branch (a structured payload alongside real
    // tool calls in the same response) runs its real tools and continues the
    // loop, but — unlike the ordinary tool-calling path and the
    // dropped-call-recovered path, both of which do — it used to leave
    // `dropped_tool_call_nudges_used` unreset. A nudge spent before a mixed
    // turn would then leak into a later, unrelated dropped-call turn and
    // receive fewer than the policy's configured number of re-prompts.
    let mut promised = ModelResponse::assistant("");
    promised.finish_reason = Some("tool_calls".into());

    let profile = ModelProfile {
        tool_calling: true,
        native_structured_output: false,
        json_schema: false,
        ..ModelProfile::default()
    };
    let model = Arc::new(ProfiledScriptedModel::new(
        profile,
        vec![
            promised.clone(), // dropped call #1: spends the only nudge.
            multi_tool_call_response(vec![("s1", "answer"), ("t1", "lookup")]), // mixed turn: must reset the nudge budget.
            promised, // dropped call #2: must be nudged again, not treated
            // as already out of budget.
            multi_tool_call_response(vec![("s2", "answer")]), // final: satisfies structured extraction.
        ],
    ));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .with_policy(RunPolicy {
            dropped_tool_call_nudges: 1,
            end_strategy: EndStrategy::Exhaustive,
            default_response_format: Some(ResponseFormat::auto(
                "answer",
                json!({"type": "object"}),
            )),
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("both dropped-call turns are recoverable under their own nudge budget");

    assert_eq!(
        run.model_calls, 4,
        "dropped -> nudged -> mixed turn -> dropped -> nudged again -> final"
    );
    let nudges = listener
        .events()
        .into_iter()
        .filter(|record| matches!(record.event, AgentEvent::RetryScheduled { .. }))
        .count();
    assert_eq!(
        nudges, 2,
        "each dropped-call turn gets its own full nudge budget, proving the \
         mixed turn reset the counter rather than leaving it spent"
    );
}

#[tokio::test]
async fn auto_dialect_falls_back_to_xml_for_a_model_that_cannot_make_native_tool_calls() {
    // `ToolDispatcher::Auto` is documented as "provider-native tool calls
    // when the provider supports them, otherwise Xml". Resolving Auto to the
    // host-side no-op `RunDialect::Native` unconditionally (regardless of
    // the resolved model's capability) left that fallback unenforced: a
    // model with `tool_calling: false` would receive a request that kept
    // depending on provider-native tools, with no host-rendered text
    // protocol to fall back to.
    let model = Arc::new(ProfiledScriptedModel::new(
        ModelProfile {
            tool_calling: false,
            ..ModelProfile::default()
        },
        vec![
            ModelResponse::assistant(
                "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
            ),
            ModelResponse::assistant("done"),
        ],
    ));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }));
    // Default policy: `tool_dialect` defaults to `Auto`.

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);

    let requests = model.requests();
    assert!(!requests.is_empty());
    let first = &requests[0];
    assert!(
        first.tools.is_empty(),
        "Auto must fall back to the text dialect (no schema on the wire) \
         for a model that cannot make native tool calls"
    );
    let system = first
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("a system turn carries the protocol")
        .text();
    assert!(system.contains("Tool Use Protocol"), "{system}");
}

#[tokio::test]
async fn a_terminal_only_stream_with_no_preceding_deltas_still_recovers_the_call() {
    // A provider may emit a single `Completed` item with no preceding
    // `MessageDelta`s at all (e.g. a short response sent in one frame). The
    // per-delta `DeltaScrubber` in `model_call.rs` never sees this text, so
    // it cannot flag it as recovered — but that scrubber is not the only
    // recovery path: `run_loop.rs` unconditionally runs
    // `dialect::recover_text_calls` on the returned response afterward,
    // regardless of whether anything streamed. This pins that second pass as
    // the safety net for exactly this case, rather than assuming (as a
    // superficial read of `model_call.rs` alone might) that terminal-only
    // content without any preceding delta is unrecoverable.
    let text = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
    let items = vec![
        ModelStreamItem::Started,
        ModelStreamItem::Completed(ModelResponse::assistant(text)),
    ];
    let model = Arc::new(StreamingMock::new(items));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness.with_policy(RunPolicy {
        limits: tinyagents_harness::limits::RunLimits {
            max_model_calls: 1,
            behavior: tinyagents_harness::limits::LimitBehavior::StopWithPartial,
            ..tinyagents_harness::limits::RunLimits::default()
        },
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run stops cleanly");

    assert_eq!(run.tool_calls, 1, "{:?}", run.messages);
    let assistant = run
        .messages
        .iter()
        .find(|m| matches!(m, Message::Assistant(_)))
        .expect("assistant turn");
    assert!(
        !assistant.text().contains("<tool_call"),
        "raw markup must not survive in the transcript: {assistant:?}"
    );
}

/// A block the tagged grammar claims but cannot decode into a call. Under a
/// text dialect it is scrubbed from the visible text and yields no call, so
/// only the lead-in prose is left (openhuman#6723).
const UNDECODABLE_CALL: &str = "Let me look.\n\n<tool_call>\nnot a call at all\n</tool_call>";
const UNDECODABLE_NUDGE_MARKER: &str = "could not be parsed";

fn xml_policy() -> RunPolicy {
    RunPolicy {
        tool_dialect: ToolDispatcher::Xml,
        ..RunPolicy::default()
    }
}

#[tokio::test]
async fn an_undecodable_text_dialect_call_is_nudged_instead_of_ending_the_turn() {
    let model = Arc::new(ScriptedModel::replies(vec![
        UNDECODABLE_CALL,
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model.clone(), &listener);
    harness.with_policy(xml_policy());

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 3, "one nudge, one call, one final");
    assert_eq!(run.tool_calls, 1, "the re-issued call ran");
    let second = &model.requests()[1];
    let nudge = second.messages.last().expect("nudge appended").text();
    assert!(nudge.contains(UNDECODABLE_NUDGE_MARKER), "{nudge}");
}

#[tokio::test]
async fn a_streamed_undecodable_text_dialect_call_is_nudged_within_the_budget() {
    // The scrubber removes the block from the streamed text, so the terminal
    // response no longer carries it: only the scrubber's own count can tell
    // the loop a call was attempted. The mock replays the same stream every
    // call, so the run is bounded by the nudge budget.
    let chunks = [
        "Let me look.\n\n<tool_",
        "call>\nnot a call",
        " at all\n</tool_call>",
    ];
    assert_eq!(chunks.concat(), UNDECODABLE_CALL);
    let mut items = vec![ModelStreamItem::Started];
    items.extend(
        chunks
            .iter()
            .map(|chunk| ModelStreamItem::MessageDelta(MessageDelta::text(*chunk))),
    );
    items.push(ModelStreamItem::Completed(ModelResponse::assistant(
        UNDECODABLE_CALL,
    )));
    let model = Arc::new(StreamingMock::new(items));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness.with_policy(xml_policy());

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run ends instead of looping");

    assert_eq!(
        run.model_calls, 4,
        "three nudges, then the answer is taken as final"
    );
    assert_eq!(run.tool_calls, 0);
}

#[tokio::test]
async fn a_decodable_or_plain_text_dialect_reply_is_not_nudged() {
    // Control for the two tests above: the nudge keys on an undecodable
    // block, not on the text dialect or on a no-call answer.
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        "Let me know if you need anything else.",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model.clone(), &listener);
    harness.with_policy(xml_policy());

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 2, "call, then a plain final with no nudge");
    assert_eq!(run.tool_calls, 1);
    for request in model.requests() {
        assert!(
            !request
                .messages
                .iter()
                .any(|m| m.text().contains(UNDECODABLE_NUDGE_MARKER)),
            "no undecodable-call nudge was sent"
        );
    }
}

/// Streams `chunks` then completes with their concatenation and `finish`.
fn text_stream(chunks: &[&str], finish: &str) -> Vec<ModelStreamItem> {
    let mut items = vec![ModelStreamItem::Started];
    items.extend(
        chunks
            .iter()
            .map(|chunk| ModelStreamItem::MessageDelta(MessageDelta::text(*chunk))),
    );
    let mut completed = ModelResponse::assistant(chunks.concat());
    completed.finish_reason = Some(finish.to_string());
    items.push(ModelStreamItem::Completed(completed));
    items
}

/// Turns a failed model call into a fixed plain answer, the way a
/// degradation middleware might, without issuing another attempt.
struct AnswerOnFailure;

#[async_trait]
impl ModelMiddleware<(), ()> for AnswerOnFailure {
    fn name(&self) -> &str {
        "answer_on_failure"
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<()>,
        state: &(),
        request: ModelRequest,
        next: ModelHandler<'_, (), ()>,
    ) -> tinyagents_harness::Result<MiddlewareModelOutcome> {
        match next.run(ctx, state, request).await {
            Ok(outcome) => Ok(outcome),
            Err(_) => Ok(ModelResponse::assistant("Here is the answer.").into()),
        }
    }
}

#[tokio::test]
async fn a_failed_attempts_undecodable_block_does_not_nudge_the_answer_that_replaced_it() {
    // The streamed attempt scrubs an undecodable block and then fails; a wrap
    // middleware answers instead. That answer made no call attempt, so the
    // failed attempt's count must not turn it into a nudge.
    let mut items = vec![ModelStreamItem::Started];
    items.push(ModelStreamItem::MessageDelta(MessageDelta::text(
        UNDECODABLE_CALL,
    )));
    items.push(ModelStreamItem::Failed("stream reset".to_string()));
    let model = Arc::new(StreamingMock::new(items));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness
        .push_model_middleware(Arc::new(AnswerOnFailure))
        .with_policy(xml_policy());

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("the middleware's answer ends the run");

    assert_eq!(
        run.model_calls, 1,
        "the replacement answer is final, not nudged"
    );
    assert_eq!(run.tool_calls, 0);
}

#[tokio::test]
async fn a_stream_that_stops_inside_a_call_block_is_nudged() {
    // The model stopped (`stop`, not `length`) before closing its block:
    // it attempted a call and forgot the closer. The block is unterminated,
    // not malformed, and must count the same way.
    let items = text_stream(
        &[
            "Let me look.\n\n<tool_",
            "call>{\"name\":\"lookup\",",
            "\"arguments\":{\"q\":\"x",
        ],
        "stop",
    );
    let model = Arc::new(StreamingMock::new(items));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness.with_policy(xml_policy());

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run ends instead of looping");

    assert_eq!(
        run.model_calls, 4,
        "three nudges, then the answer is taken as final"
    );
    assert_eq!(run.tool_calls, 0);
}

#[tokio::test]
async fn a_length_cut_inside_a_call_block_is_left_to_truncation_handling() {
    // Control for the test above: a block cut by the token cap is truncation,
    // not a forgotten closer, and is not answered with the parse nudge.
    let items = text_stream(
        &[
            "Let me look.\n\n<tool_",
            "call>{\"name\":\"lookup\",",
            "\"arguments\":{\"q\":\"x",
        ],
        "length",
    );
    let model = Arc::new(StreamingMock::new(items));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness.with_policy(xml_policy());

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 1, "no parse nudge for a length cut");
}

#[tokio::test]
async fn an_undecodable_block_under_native_text_recovery_is_nudged() {
    // A native model with text recovery on parses every grammar out of its
    // prose, so a claimed-but-undecodable block there is the same dropped
    // call as under a forced text dialect.
    let profile = ModelProfile {
        tool_calling: true,
        ..ModelProfile::default()
    };
    let model = Arc::new(ProfiledScriptedModel::new(
        profile,
        vec![
            ModelResponse::assistant(UNDECODABLE_CALL),
            ModelResponse::assistant("done"),
        ],
    ));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model.clone(), &listener);
    harness.with_policy(RunPolicy {
        text_dialect_recovery: tinyagents_harness::runtime::TextDialectRecovery::On,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let requests = model.requests();
    assert!(
        !requests[0].tools.is_empty(),
        "precondition: native dialect, schemas on the wire"
    );
    assert_eq!(run.model_calls, 2, "one nudge, then the final answer");
    let nudge = requests[1].messages.last().expect("nudge appended").text();
    assert!(nudge.contains(UNDECODABLE_NUDGE_MARKER), "{nudge}");
}

#[tokio::test]
async fn an_undecodable_block_is_nudged_before_an_empty_response_retry() {
    // A streamed reply that is only an undecodable block scrubs to empty
    // text. With empty-response retries on, a blind retry must not pre-empt
    // (or add to) the nudges that tell the model why nothing ran: the run is
    // bounded by the nudge budget alone.
    let items = text_stream(
        &["<tool_", "call>\nnot a call at all\n</tool_call>"],
        "stop",
    );
    let model = Arc::new(StreamingMock::new(items));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness.with_policy(RunPolicy {
        empty_response_retries: 1,
        ..xml_policy()
    });

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run ends instead of looping");

    assert_eq!(
        run.model_calls, 4,
        "three nudges and the final reply, no blind empty-response retry"
    );
}

/// A real call a small model wrapped in a bare fence, and the same call
/// quoted as an example under a language-tagged fence. `tinytools-agent`
/// owns the fence policy: a bare fence is not protected, a tagged one is
/// (openhuman#6732). Both model paths must apply that one policy.
const BARE_FENCED_CALL: &str =
    "```\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>\n```";
const XML_FENCED_CALL: &str = "Example:\n```xml\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>\n```";

/// Calls dispatched when an xml-dialect run gets `text` as its first reply,
/// either streamed or unary, with one model call allowed.
async fn dispatched_from_first_reply(text: &str, streamed: bool) -> usize {
    let listener = Arc::new(RecordingListener::new());
    let model: Arc<dyn ChatModel<()>> = if streamed {
        Arc::new(StreamingMock::new(vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(MessageDelta::text(text)),
            ModelStreamItem::Completed(ModelResponse::assistant(text)),
        ]))
    } else {
        Arc::new(ScriptedModel::replies(vec![text]))
    };
    let mut harness = harness_with(model, &listener);
    harness.with_policy(RunPolicy {
        tool_dialect: ToolDispatcher::Xml,
        limits: tinyagents_harness::limits::RunLimits {
            max_model_calls: 1,
            behavior: tinyagents_harness::limits::LimitBehavior::StopWithPartial,
            ..tinyagents_harness::limits::RunLimits::default()
        },
        ..RunPolicy::default()
    });
    let result = if streamed {
        harness
            .invoke_streaming_default(&(), vec![Message::user("go")])
            .await
    } else {
        harness.invoke_default(&(), vec![Message::user("go")]).await
    };
    // The one-call cap stops the run cleanly after a dispatched call; an
    // error here would mean the count below was read from a failed run.
    assert!(result.is_ok(), "run failed: {:?}", result.err());
    dispatched_ids(&listener).len()
}

#[tokio::test]
async fn a_bare_fenced_call_is_dispatched_unary_and_streamed() {
    assert_eq!(
        dispatched_from_first_reply(BARE_FENCED_CALL, false).await,
        1,
        "unary"
    );
    assert_eq!(
        dispatched_from_first_reply(BARE_FENCED_CALL, true).await,
        1,
        "streamed"
    );
}

#[tokio::test]
async fn a_language_tagged_fenced_call_is_not_dispatched_unary_or_streamed() {
    assert_eq!(
        dispatched_from_first_reply(XML_FENCED_CALL, false).await,
        0,
        "unary"
    );
    assert_eq!(
        dispatched_from_first_reply(XML_FENCED_CALL, true).await,
        0,
        "streamed"
    );
}

/// A tool with a real schema that records the arguments of every call it
/// receives, so an acceptance test can assert what was actually dispatched.
struct RecordingTool {
    name: &'static str,
    schema: serde_json::Value,
    calls: Arc<Mutex<Vec<serde_json::Value>>>,
}

#[async_trait]
impl Tool for RecordingTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "Records its calls."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.calls.lock().unwrap().push(arguments);
        Ok(ToolResult::success("ok"))
    }
}

/// A `todo` tool and a `search_repositories` tool shaped like the ones in the
/// production thread behind openhuman#6722, under the Python code dialect,
/// with `text` as the model's first reply. Returns each tool's recorded calls.
async fn python_dialect_dispatch(text: &str) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let todo_calls = Arc::new(Mutex::new(Vec::new()));
    let search_calls = Arc::new(Mutex::new(Vec::new()));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(ScriptedModel::replies(vec![text, "done"])))
        .set_default_model("mock")
        .register_tool(Arc::new(RecordingTool {
            name: "todo",
            schema: json!({
                "type": "object",
                "properties": { "todos": { "type": "array", "items": { "type": "object" } } }
            }),
            calls: todo_calls.clone(),
        }))
        .register_tool(Arc::new(RecordingTool {
            name: "search_repositories",
            schema: json!({
                "type": "object",
                "properties": {
                    "q": { "type": "string" },
                    "sort": { "type": "string" },
                    "per_page": { "type": "integer" }
                },
                "required": ["q"]
            }),
            calls: search_calls.clone(),
        }))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Python,
            ..RunPolicy::default()
        });
    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    let todo = todo_calls.lock().unwrap().clone();
    let search = search_calls.lock().unwrap().clone();
    (todo, search)
}

#[tokio::test]
async fn a_named_invoke_wrapped_in_tool_call_is_dispatched_under_the_python_dialect() {
    // The shape of the production record behind openhuman#6722 (sanitized):
    // a `todo` element closed by a stray `</tool_call>`, a code fence opened on
    // the same line as the next `<tool_call>`, and a named `<invoke>` with
    // `string=` attributes inside it.
    let text = "<todo>\n<todos>\n[{\"status\": \"in_progress\", \"description\": \"Find the repositories\"}, {\"status\": \"pending\", \"description\": \"Fetch the issues\"}]\n</todos>\n</tool_call>\n```<tool_call>\n<invoke name=\"search_repositories\">\n<parameter name=\"q\" string=\"true\">example</parameter>\n<parameter name=\"sort\" string=\"true\">stars</parameter>\n<parameter name=\"per_page\" string=\"false\">20</parameter>\n</invoke>\n</tool_call>";

    let (_todo, search) = python_dialect_dispatch(text).await;

    assert_eq!(search.len(), 1, "the wrapped invoke is dispatched once");
    assert_eq!(search[0]["q"], "example");
    assert_eq!(search[0]["sort"], "stars");
    assert_eq!(
        search[0]["per_page"], 20,
        "string=\"false\" decodes as JSON"
    );
}

#[tokio::test]
async fn a_todo_element_call_is_dispatched_under_the_python_dialect() {
    // The `<TOOL><param>…</param></TOOL>` element form the same model used
    // for its `todo` updates (openhuman#6722).
    let text = "Let me fetch the issues.\n\n<todo>\n<todos>\n[{\"status\": \"in_progress\", \"description\": \"Fetch the issues\"}, {\"status\": \"pending\", \"description\": \"Build the roadmap\"}]\n</todos>\n</todo>";

    let (todo, search) = python_dialect_dispatch(text).await;

    assert_eq!(todo.len(), 1, "the element call is dispatched once");
    assert_eq!(todo[0]["todos"][0]["status"], "in_progress");
    assert_eq!(todo[0]["todos"].as_array().map(Vec::len), Some(2));
    assert!(search.is_empty());
}
