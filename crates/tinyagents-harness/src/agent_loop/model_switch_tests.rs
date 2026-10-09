//! A live [`SteeringCommand::SwitchModel`] is applied at the next model-call
//! boundary, *before* the model binding is resolved, so everything derived
//! from the binding (events, `ctx.model_profile`, the handoff transform, the
//! fallback chain) belongs to the new model.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::middleware::{Middleware, MiddlewareModelOutcome, ModelHandler, ModelMiddleware};
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::runtime::{AgentHarness, RunPolicy};
use crate::steering::{SteeringCommand, SteeringHandle};
use crate::testkit::{EventRecorder, FakeTool, ScriptedModel, tool_call_response};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, MessageOrigin};
use tinyinference_llm::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStatus, ProviderError,
};
use tinyinference_llm::tool::ToolCall;

fn profile(provider: &str, model: &str) -> ModelProfile {
    ModelProfile {
        provider: Some(provider.into()),
        model: Some(model.into()),
        ..ModelProfile::default()
    }
}

fn switch(model: &str) -> SteeringCommand {
    SteeringCommand::SwitchModel {
        model: model.into(),
    }
}

fn call(id: &str) -> ModelResponse {
    tool_call_response(ToolCall::new(id, "lookup", json!({})))
}

fn context(handle: &SteeringHandle, recorder: &EventRecorder) -> RunContext<()> {
    RunContext::new(RunConfig::new("switch-run"), ())
        .with_events(recorder.sink())
        .with_steering(handle.clone())
}

fn model_started(recorder: &EventRecorder) -> Vec<String> {
    recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::ModelStarted { model, .. } => Some(model),
            _ => None,
        })
        .collect()
}

fn steered_rejections(recorder: &EventRecorder) -> usize {
    recorder
        .events()
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Steered { command_kind, accepted: false }
                    if command_kind == "switch_model"
            )
        })
        .count()
}

/// Every `Steered` event reported for a `switch_model` command, in order.
fn switch_outcomes(recorder: &EventRecorder) -> Vec<bool> {
    recorder
        .events()
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Steered {
                command_kind,
                accepted,
            } if command_kind == "switch_model" => Some(*accepted),
            _ => None,
        })
        .collect()
}

fn skipped(recorder: &EventRecorder) -> Vec<(String, String)> {
    recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::ModelOverrideSkipped {
                requested,
                resolved,
            } => Some((requested, resolved)),
            _ => None,
        })
        .collect()
}

struct ShortCircuitModelMiddleware;

#[async_trait]
impl ModelMiddleware<()> for ShortCircuitModelMiddleware {
    fn name(&self) -> &str {
        "short_circuit_model"
    }

    async fn wrap_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _request: ModelRequest,
        _next: ModelHandler<'_, (), ()>,
    ) -> crate::Result<MiddlewareModelOutcome> {
        Ok(ModelResponse::assistant("short-circuited").into())
    }
}

/// Two registered models; `a` is the registry default.
fn two_models(
    a: ScriptedModel,
    b: ScriptedModel,
) -> (AgentHarness<()>, Arc<ScriptedModel>, Arc<ScriptedModel>) {
    let (a, b) = (Arc::new(a), Arc::new(b));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("a", a.clone());
    harness.register_model("b", b.clone());
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "ok")));
    (harness, a, b)
}

/// Answers from a script and, on its first call, queues `command` on the
/// run's steering handle: an orchestrator reacting mid-run.
struct SteersOnFirstCall {
    handle: SteeringHandle,
    command: SteeringCommand,
    script: Mutex<VecDeque<ModelResponse>>,
    calls: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for SteersOnFirstCall {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        if *calls == 1 {
            self.handle.send(self.command.clone());
        }
        Ok(self.script.lock().unwrap().pop_front().expect("scripted"))
    }
}

#[tokio::test]
async fn model_started_names_the_switched_model() {
    let (harness, a, b) = two_models(
        ScriptedModel::replies(vec!["from a"]),
        ScriptedModel::replies(vec!["from b"]),
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(run.text(), Some("from b".to_string()));
    assert_eq!(model_started(&recorder), vec!["b"]);
    assert!(a.requests().is_empty());
    assert_eq!(b.requests().len(), 1);
}

#[tokio::test]
async fn short_circuit_wrap_does_not_announce_model_switch() {
    let (mut harness, a, _b) = two_models(
        ScriptedModel::replies(vec!["from a"]),
        ScriptedModel::replies(vec!["from b"]),
    );
    harness.push_model_middleware(Arc::new(ShortCircuitModelMiddleware));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(run.text(), Some("short-circuited".to_string()));
    assert!(switch_outcomes(&recorder).is_empty());
    assert!(a.requests().is_empty());
}

#[tokio::test]
async fn mid_run_switch_applies_to_the_next_call_and_stays_sticky() {
    let handle = SteeringHandle::allow_all_with_model_switch();
    let a = Arc::new(SteersOnFirstCall {
        handle: handle.clone(),
        command: switch("b"),
        script: Mutex::new(VecDeque::from([call("c1")])),
        calls: Mutex::new(0),
    });
    let b = Arc::new(ScriptedModel::new(vec![
        call("c2"),
        ModelResponse::assistant("done"),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("a", a);
    harness.register_model("b", b.clone());
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "ok")));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(run.text(), Some("done".to_string()));
    // The in-flight call on `a` is untouched; every later call is on `b`.
    assert_eq!(model_started(&recorder), vec!["a", "b", "b"]);
    assert_eq!(b.requests().len(), 2);
}

#[tokio::test]
async fn switch_runs_the_handoff_transform_for_the_new_providers_profile() {
    let foreign = AssistantMessage {
        id: None,
        content: vec![
            ContentBlock::RedactedThinking {
                data: "opaque".into(),
            },
            ContentBlock::Text("earlier answer".into()),
        ],
        tool_calls: vec![],
        usage: None,
        origin: Some(MessageOrigin {
            provider: "openai".into(),
            api: "chat_completions".into(),
            model: "gpt-5".into(),
        }),
    };
    let transcript = vec![
        Message::user("hi"),
        Message::Assistant(foreign),
        Message::user("continue"),
    ];
    let make = || {
        two_models(
            ScriptedModel::replies(vec!["from a"]).with_profile(profile("openai", "gpt-5")),
            ScriptedModel::replies(vec!["from b"])
                .with_profile(profile("anthropic", "claude-opus-4-6")),
        )
    };
    let handoffs = |recorder: &EventRecorder| {
        recorder
            .events()
            .iter()
            .filter(|event| matches!(event, AgentEvent::HandoffTransformApplied { .. }))
            .count()
    };

    // Control: without a switch the transcript is same-origin for `a`.
    let (harness, _, _) = make();
    let handle = SteeringHandle::allow_all_with_model_switch();
    let recorder = EventRecorder::new();
    harness
        .invoke_in_context(&(), context(&handle, &recorder), transcript.clone())
        .await
        .unwrap();
    assert_eq!(handoffs(&recorder), 0);

    // Switched to the Anthropic model, the OpenAI-origin message is foreign.
    let (harness, _, b) = make();
    let recorder = EventRecorder::new();
    handle.send(switch("b"));
    harness
        .invoke_in_context(&(), context(&handle, &recorder), transcript)
        .await
        .unwrap();
    assert_eq!(handoffs(&recorder), 1);
    let sent = b.requests();
    assert!(!sent[0].messages.iter().any(|message| matches!(
        message,
        Message::Assistant(assistant)
            if assistant.content.iter().any(|block| matches!(block, ContentBlock::RedactedThinking { .. }))
    )));
}

/// Records the profile `before_model` middleware saw on `ctx`.
struct SeenProfile(Arc<Mutex<Vec<Option<ModelProfile>>>>);

#[async_trait]
impl Middleware<()> for SeenProfile {
    fn name(&self) -> &str {
        "seen-profile"
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> crate::error::Result<()> {
        self.0.lock().unwrap().push(ctx.model_profile.clone());
        Ok(())
    }
}

#[tokio::test]
async fn before_model_middleware_sees_the_switched_models_profile() {
    let b_profile = profile("anthropic", "claude-opus-4-6");
    let (mut harness, _, _) = two_models(
        ScriptedModel::replies(vec!["from a"]).with_profile(profile("openai", "gpt-5")),
        ScriptedModel::replies(vec!["from b"]).with_profile(b_profile.clone()),
    );
    let seen = Arc::new(Mutex::new(Vec::new()));
    harness.push_middleware(Arc::new(SeenProfile(seen.clone())));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(seen.lock().unwrap().as_slice(), [Some(b_profile)]);
}

#[tokio::test]
async fn unknown_model_is_rejected_and_the_run_continues_on_the_current_model() {
    let (harness, a, _) = two_models(
        ScriptedModel::new(vec![call("c1"), ModelResponse::assistant("done")]),
        ScriptedModel::replies(vec!["never"]),
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("nope"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("a rejected switch never fails the run");

    assert_eq!(run.text(), Some("done".to_string()));
    assert_eq!(model_started(&recorder), vec!["a", "a"]);
    assert_eq!(a.requests().len(), 2);
    // Reported once, not on every later call.
    assert_eq!(steered_rejections(&recorder), 1);
    assert_eq!(
        skipped(&recorder),
        vec![("nope".to_string(), "a".to_string())]
    );
    assert_eq!(handle.model_override(), None);
}

#[tokio::test]
async fn retired_model_is_rejected_like_an_unknown_one() {
    let retired = ModelProfile {
        status: ModelStatus::Retired,
        ..profile("openai", "old")
    };
    let (harness, a, b) = two_models(
        ScriptedModel::replies(vec!["from a"]),
        ScriptedModel::replies(vec!["from b"]).with_profile(retired),
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("from a".to_string()));
    assert_eq!(a.requests().len(), 1);
    assert!(b.requests().is_empty());
    assert_eq!(steered_rejections(&recorder), 1);
    assert_eq!(skipped(&recorder), vec![("b".to_string(), "a".to_string())]);
}

#[tokio::test]
async fn switch_without_opt_in_is_ignored() {
    let (harness, a, b) = two_models(
        ScriptedModel::replies(vec!["from a"]),
        ScriptedModel::replies(vec!["from b"]),
    );
    let handle = SteeringHandle::allow_all();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("from a".to_string()));
    assert_eq!(a.requests().len(), 1);
    assert!(b.requests().is_empty());
    assert_eq!(steered_rejections(&recorder), 1);
}

struct AlwaysFails {
    attempts: Mutex<usize>,
}

impl AlwaysFails {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            attempts: Mutex::new(0),
        })
    }
    fn attempts(&self) -> usize {
        *self.attempts.lock().unwrap()
    }
}

#[async_trait]
impl ChatModel<()> for AlwaysFails {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        Err(tinyinference_llm::Error::Provider(Box::new(
            ProviderError {
                provider: "test".into(),
                status: Some(401),
                message: "invalid api key".into(),
                retryable: false,
                ..ProviderError::default()
            },
        )))
    }
}

fn failover_harness(
    chain: &[&str],
    models: Vec<(&str, Arc<dyn ChatModel<()>>)>,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    for (name, model) in models {
        harness.register_model(name, model);
    }
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy::new(chain.iter().copied())),
        ..RunPolicy::default()
    });
    harness
}

#[tokio::test]
async fn fallback_walks_on_from_the_switched_models_position_in_the_chain() {
    let (a, b, c) = (
        Arc::new(ScriptedModel::replies(vec!["from a"])),
        AlwaysFails::new(),
        Arc::new(ScriptedModel::replies(vec!["from c"])),
    );
    let harness = failover_harness(
        &["a", "b", "c"],
        vec![("a", a.clone()), ("b", b.clone()), ("c", c.clone())],
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("c answers after b fails");

    assert_eq!(run.text(), Some("from c".to_string()));
    assert_eq!(b.attempts(), 1);
    assert!(
        a.requests().is_empty(),
        "the walk continues after b, not from the head"
    );
    assert_eq!(model_started(&recorder)[0], "b");
}

#[tokio::test]
async fn fallback_for_a_switch_outside_the_chain_starts_from_the_chain_head() {
    let (a, x, b) = (
        Arc::new(ScriptedModel::replies(vec!["from a"])),
        AlwaysFails::new(),
        Arc::new(ScriptedModel::replies(vec!["from b"])),
    );
    let harness = failover_harness(
        &["a", "b"],
        vec![("a", a.clone()), ("b", b.clone()), ("x", x.clone())],
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("x"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("the primary's chain answers after x fails");

    assert_eq!(run.text(), Some("from a".to_string()));
    assert_eq!(x.attempts(), 1);
    assert_eq!(a.requests().len(), 1);
    assert!(b.requests().is_empty());
}

#[tokio::test]
async fn a_childs_rejected_switch_does_not_clear_the_parents_switch() {
    // The parent switched to a real model; the child (a separate run sharing
    // the queue) is told to switch to a model its registry lacks.
    let (harness, _, _) = two_models(
        ScriptedModel::replies(vec!["child answer"]),
        ScriptedModel::replies(vec!["unused"]),
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    let recorder = EventRecorder::new();
    let mut parent = context(&handle, &recorder);
    crate::steering::apply_pending_steering(&mut parent, &mut Vec::new()).unwrap();
    handle.send(switch("b"));
    crate::steering::apply_pending_steering(&mut parent, &mut Vec::new()).unwrap();
    assert_eq!(handle.model_override(), Some("b".to_string()));

    let child = parent.child(RunConfig::new("child"), ()).unwrap();
    let child_handle = child.steering.clone().unwrap();
    handle.send_to(
        crate::steering::SteeringTarget::Run(child.run_id().clone()),
        switch("missing"),
    );
    let run = harness
        .invoke_in_context(&(), child, vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("child answer".to_string()));
    assert_eq!(
        child_handle.model_override(),
        None,
        "child rejected its own switch"
    );
    assert_eq!(handle.model_override(), Some("b".to_string()));
}

/// A `before_model` hook that optionally picks a model and optionally raises
/// the request's required capabilities (as a tool-adding middleware does).
struct Tweak {
    model: Option<&'static str>,
    tool_calling: bool,
}

#[async_trait]
impl Middleware<()> for Tweak {
    fn name(&self) -> &str {
        "tweak"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> crate::error::Result<()> {
        if let Some(model) = self.model {
            request.model = Some(model.into());
        }
        if self.tool_calling {
            request
                .required_capabilities
                .get_or_insert_default()
                .tool_calling = true;
        }
        Ok(())
    }
}

/// Raises the required capabilities only from the second model call on.
struct RequiresToolsAfterFirstCall {
    calls: Mutex<usize>,
}

#[async_trait]
impl Middleware<()> for RequiresToolsAfterFirstCall {
    fn name(&self) -> &str {
        "requires_tools_after_first_call"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> crate::error::Result<()> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        // Only once a tool result is in the transcript (second model call).
        if request.messages.len() > 1 {
            request
                .required_capabilities
                .get_or_insert_default()
                .tool_calling = true;
        }
        Ok(())
    }
}

fn tool_capable(provider: &str, model: &str) -> ModelProfile {
    ModelProfile {
        tool_calling: true,
        ..profile(provider, model)
    }
}

#[tokio::test]
async fn rejection_after_middleware_restores_request_model_and_reports_once() {
    // `b` is eligible when the switch is first applied; middleware then
    // raises the capability requirements and `b` no longer qualifies.
    let (mut harness, a, b) = two_models(
        ScriptedModel::replies(vec!["from a"]).with_profile(tool_capable("openai", "gpt-5")),
        ScriptedModel::replies(vec!["from b"]).with_profile(profile("openai", "mini")),
    );
    harness.push_middleware(Arc::new(Tweak {
        model: None,
        tool_calling: true,
    }));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("a rejected switch never fails the run");

    assert_eq!(run.text(), Some("from a".to_string()));
    assert_eq!(model_started(&recorder), vec!["a"]);
    assert!(b.requests().is_empty());
    assert_eq!(steered_rejections(&recorder), 1);
    assert_eq!(skipped(&recorder), vec![("b".to_string(), "a".to_string())]);
    // The rejected name must not reach the provider adapter.
    assert_eq!(a.requests()[0].model, None);
    assert_eq!(handle.model_override(), None);
}

#[tokio::test]
async fn switch_wins_over_a_model_chosen_by_before_model_middleware() {
    let (mut harness, _, b) = two_models(
        ScriptedModel::replies(vec!["from a"]),
        ScriptedModel::replies(vec!["from b"]),
    );
    harness.register_model("c", Arc::new(ScriptedModel::replies(vec!["from c"])));
    harness.push_middleware(Arc::new(Tweak {
        model: Some("c"),
        tool_calling: false,
    }));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("from b".to_string()));
    assert_eq!(model_started(&recorder), vec!["b"]);
    assert_eq!(b.requests().len(), 1);
}

#[tokio::test]
async fn rejected_switch_keeps_the_model_a_middleware_picked() {
    let (mut harness, a, b) = two_models(
        ScriptedModel::replies(vec!["from a"]).with_profile(tool_capable("openai", "gpt-5")),
        ScriptedModel::replies(vec!["from b"]).with_profile(profile("openai", "mini")),
    );
    let c = Arc::new(
        ScriptedModel::replies(vec!["from c"]).with_profile(tool_capable("openai", "gpt-5-mini")),
    );
    harness.register_model("c", c.clone());
    harness.push_middleware(Arc::new(Tweak {
        model: Some("c"),
        tool_calling: true,
    }));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("from c".to_string()));
    assert_eq!(model_started(&recorder), vec!["c"]);
    assert_eq!(steered_rejections(&recorder), 1);
    assert_eq!(skipped(&recorder), vec![("b".to_string(), "c".to_string())]);
    assert!(a.requests().is_empty() && b.requests().is_empty());
    assert_eq!(c.requests()[0].model.as_deref(), Some("c"));
}

#[tokio::test]
async fn a_written_off_switched_model_is_not_resent_the_revoked_key() {
    // `b` is revoked on the first call; `c` answers. The next call must go
    // straight to `c` rather than retry the sticky, revoked switch target.
    let b = ScriptedOutcomesRevoked::new();
    let c = Arc::new(ScriptedModel::new(vec![
        call("c1"),
        ModelResponse::assistant("done"),
    ]));
    let mut harness = failover_harness(
        &["a", "b", "c"],
        vec![
            ("a", Arc::new(ScriptedModel::replies(vec!["from a"]))),
            ("b", b.clone()),
            ("c", c.clone()),
        ],
    );
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "ok")));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let run = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("done".to_string()));
    assert_eq!(b.attempts(), 1, "revoked model tried once per run");
    assert_eq!(c.requests().len(), 2);
}

/// Fails every call as a revoked credential (a permanent write-off).
struct ScriptedOutcomesRevoked(AlwaysFailsWith);

type AlwaysFailsWith = Mutex<usize>;

impl ScriptedOutcomesRevoked {
    fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(0)))
    }
    fn attempts(&self) -> usize {
        *self.0.lock().unwrap()
    }
}

#[async_trait]
impl ChatModel<()> for ScriptedOutcomesRevoked {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.0.lock().unwrap() += 1;
        Err(tinyinference_llm::Error::Provider(Box::new(
            ProviderError {
                provider: "test".into(),
                status: Some(401),
                message: "API key has been revoked".into(),
                retryable: false,
                ..ProviderError::default()
            },
        )))
    }
}

#[tokio::test]
async fn model_names_are_trimmed_before_they_are_stored() {
    let (harness, _, b) = two_models(
        ScriptedModel::replies(vec!["from a"]),
        ScriptedModel::replies(vec!["from b"]),
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("  b \n"));
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(model_started(&recorder), vec!["b"]);
    assert_eq!(b.requests().len(), 1);
}

#[tokio::test]
async fn an_applied_switch_reports_one_accepted_outcome_across_calls() {
    let (harness, _, _) = two_models(
        ScriptedModel::replies(vec!["never"]),
        ScriptedModel::new(vec![call("c1"), ModelResponse::assistant("done")]),
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("nope-first"));
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    // Two model calls on the sticky switch. The replaced `nope-first` is
    // explicitly rejected, then `b` is reported when it is applied.
    assert_eq!(model_started(&recorder), vec!["b", "b"]);
    assert_eq!(switch_outcomes(&recorder), vec![false, true]);
}

#[tokio::test]
async fn a_rejected_switch_reports_only_the_rejection() {
    let (harness, _, _) = two_models(
        ScriptedModel::new(vec![call("c1"), ModelResponse::assistant("done")]),
        ScriptedModel::replies(vec!["never"]),
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("nope"));
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(switch_outcomes(&recorder), vec![false]);
}

#[tokio::test]
async fn a_switch_rejected_after_middleware_never_looks_accepted() {
    let (mut harness, _, _) = two_models(
        ScriptedModel::replies(vec!["from a"]).with_profile(tool_capable("openai", "gpt-5")),
        ScriptedModel::replies(vec!["from b"]).with_profile(profile("openai", "mini")),
    );
    harness.push_middleware(Arc::new(Tweak {
        model: None,
        tool_calling: true,
    }));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(switch_outcomes(&recorder), vec![false]);
}

#[tokio::test]
async fn fallback_from_a_steered_model_retargets_request_model() {
    let (a, b, c) = (
        Arc::new(ScriptedModel::replies(vec!["from a"])),
        AlwaysFails::new(),
        Arc::new(ScriptedModel::replies(vec!["from c"])),
    );
    let harness = failover_harness(
        &["a", "b", "c"],
        vec![("a", a.clone()), ("b", b.clone()), ("c", c.clone())],
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("c answers after b fails");

    // A provider that honours `request.model` must be asked for the fallback,
    // not for the steered model that just failed.
    assert_eq!(c.requests()[0].model, Some("c".to_string()));
}

#[tokio::test]
async fn chain_head_fallback_retargets_request_model() {
    let (a, x, b) = (
        Arc::new(ScriptedModel::replies(vec!["from a"])),
        AlwaysFails::new(),
        Arc::new(ScriptedModel::replies(vec!["from b"])),
    );
    let harness = failover_harness(
        &["a", "b"],
        vec![("a", a.clone()), ("b", b.clone()), ("x", x.clone())],
    );
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("x"));
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("a answers after x fails");

    assert_eq!(a.requests()[0].model, Some("a".to_string()));
}

#[tokio::test]
async fn fallback_from_a_plain_request_model_override_retargets_request_model() {
    // No steering: a middleware (like an SDK caller) pins `b`, which fails.
    let (a, b, c) = (
        Arc::new(ScriptedModel::replies(vec!["from a"])),
        AlwaysFails::new(),
        Arc::new(ScriptedModel::replies(vec!["from c"])),
    );
    let mut harness = failover_harness(
        &["b", "c"],
        vec![("a", a.clone()), ("b", b.clone()), ("c", c.clone())],
    );
    harness.push_middleware(Arc::new(Tweak {
        model: Some("b"),
        tool_calling: false,
    }));

    let run = harness
        .invoke(&(), (), RunConfig::new("plain"), vec![Message::user("hi")])
        .await
        .expect("c answers after b fails");

    assert_eq!(run.text(), Some("from c".to_string()));
    assert_eq!(c.requests()[0].model, Some("c".to_string()));
}

/// A `before_model_control` hook that ends the run before dispatch.
struct EndBeforeCall;

#[async_trait]
impl Middleware<()> for EndBeforeCall {
    fn name(&self) -> &str {
        "end-before-call"
    }

    async fn before_model_control(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> crate::error::Result<crate::context::MiddlewareControl> {
        Ok(crate::context::MiddlewareControl::JumpTo(
            crate::context::LoopTarget::End,
        ))
    }
}

#[tokio::test]
async fn a_switch_whose_call_is_cancelled_by_control_reports_no_outcome() {
    let (mut harness, a, b) = two_models(
        ScriptedModel::replies(vec!["from a"]),
        ScriptedModel::replies(vec!["from b"]),
    );
    harness.push_middleware(Arc::new(EndBeforeCall));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    let _ = harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await;

    assert!(a.requests().is_empty() && b.requests().is_empty());
    assert_eq!(switch_outcomes(&recorder), Vec::<bool>::new());
}

#[tokio::test]
async fn fallback_keeps_an_absent_request_model_absent() {
    let (a, b) = (
        AlwaysFails::new(),
        Arc::new(ScriptedModel::replies(vec!["from b"])),
    );
    let harness = failover_harness(&["a", "b"], vec![("a", a.clone()), ("b", b.clone())]);
    let handle = SteeringHandle::allow_all_with_model_switch();
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("b answers after a fails");

    // A registry alias is not a provider model id: do not invent one.
    assert_eq!(b.requests()[0].model, None);
}

#[tokio::test]
async fn a_switch_that_becomes_ineligible_after_being_reported_keeps_one_outcome() {
    let (mut harness, _, _) = two_models(
        ScriptedModel::replies(vec!["from a"]).with_profile(tool_capable("openai", "gpt-5")),
        ScriptedModel::new(vec![call("c1")]).with_profile(profile("openai", "mini")),
    );
    harness.push_middleware(Arc::new(RequiresToolsAfterFirstCall {
        calls: Mutex::new(0),
    }));
    let handle = SteeringHandle::allow_all_with_model_switch();
    handle.send(switch("b"));
    let recorder = EventRecorder::new();

    harness
        .invoke_in_context(&(), context(&handle, &recorder), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(model_started(&recorder), vec!["b", "a"]);
    // Accepted on the first call; the later rejection is not a second outcome.
    assert_eq!(switch_outcomes(&recorder), vec![true]);
    assert_eq!(skipped(&recorder).len(), 1);
}
