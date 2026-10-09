//! The loop reports a typed [`TerminalOutcome`] on `RunCompleted`/`RunFailed`
//! and on `AgentRun::terminal`, on every exit path.

use std::sync::Arc;

use async_trait::async_trait;

use crate::cancel::CancellationToken;
use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::events::{AgentEvent, LimitKind};
use crate::limits::{LimitBehavior, RunLimits};
use crate::retry::FailoverReason;
use crate::runtime::{AgentHarness, RunPolicy};
use crate::steering::{SteeringCommand, SteeringHandle, SteeringPolicy};
use crate::terminal::{TerminalClass, TerminalOutcome, TerminalReason, TimeoutPhase};
use crate::testkit::{EventRecorder, ScriptedModel};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

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

struct FailingModel(&'static str);

#[async_trait]
impl ChatModel<()> for FailingModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        Err(tinyinference_llm::Error::Model(self.0.to_string()))
    }
}

struct PendingModel;

#[async_trait]
impl ChatModel<()> for PendingModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        std::future::pending().await
    }
}

fn harness_with(model: Arc<dyn ChatModel<()>>) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model);
    harness
}

fn terminal_events(events: &[AgentEvent]) -> Vec<&AgentEvent> {
    events
        .iter()
        .filter(|e| {
            matches!(
                e,
                AgentEvent::RunCompleted { .. } | AgentEvent::RunFailed { .. }
            )
        })
        .collect()
}

#[tokio::test]
async fn a_finished_run_reports_a_completed_outcome() {
    let harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("ok"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .unwrap();

    let outcome = run.terminal.clone().expect("terminal outcome on the run");
    assert_eq!(outcome.reason, TerminalReason::Completed);
    assert_eq!(outcome.class, TerminalClass::Success);
    assert!(outcome.provider_started);
    let events = recorder.events();
    let terminal = terminal_events(&events);
    assert_eq!(terminal.len(), 1);
    assert!(matches!(
        terminal[0],
        AgentEvent::RunCompleted { outcome: Some(o), .. } if *o == outcome
    ));
}

#[tokio::test]
async fn a_stop_with_partial_cap_completes_with_a_limit_outcome() {
    let harness = {
        let mut h = harness_with(Arc::new(ScriptedModel::new(vec![response(
            vec![ToolCall::new("c1", "spin", serde_json::json!({}))],
            "",
        )])));
        h.with_policy(RunPolicy {
            limits: RunLimits::default()
                .with_max_model_calls(1)
                .with_behavior(LimitBehavior::StopWithPartial),
            ..RunPolicy::default()
        });
        h
    };
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("cap"), ()).with_events(recorder.sink());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("go")])
        .await;
    // The first call returns a tool call for an unregistered tool, then the
    // second iteration trips the cap.
    assert!(partial.error.is_none(), "{:?}", partial.error);
    let outcome = partial.run.terminal.expect("outcome");
    assert_eq!(
        outcome.reason,
        TerminalReason::LimitReached(Some(LimitKind::ModelCalls))
    );
    assert_eq!(outcome.class, TerminalClass::Failure);
    let events = recorder.events();
    assert!(matches!(
        terminal_events(&events)[0],
        AgentEvent::RunCompleted { outcome: Some(o), .. }
            if o.reason == TerminalReason::LimitReached(Some(LimitKind::ModelCalls))
    ));
}

#[tokio::test]
async fn a_provider_failure_is_classified_on_run_failed_and_the_partial_run() {
    let harness = harness_with(Arc::new(FailingModel("HTTP 429 too many requests")));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("fail"), ()).with_events(recorder.sink());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("hi")])
        .await;

    let error = partial.error.expect("run fails");
    let outcome = partial.run.terminal.expect("outcome on the partial run");
    assert_eq!(
        outcome.reason,
        TerminalReason::ProviderFailed(Some(FailoverReason::RateLimit))
    );
    assert!(outcome.provider_started, "the call had started");
    assert_eq!(outcome.message, error.to_string());
    let events = recorder.events();
    match terminal_events(&events)[0] {
        AgentEvent::RunFailed {
            error: e,
            outcome: Some(o),
            ..
        } => {
            assert_eq!(*e, error.to_string(), "legacy string preserved");
            assert_eq!(*o, outcome);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn a_pre_cancelled_run_reports_cancellation_before_the_provider() {
    let harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "x")])));
    let token = CancellationToken::new();
    token.cancel();
    let ctx = RunContext::new(RunConfig::new("cancel"), ()).with_cancellation(token);
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("hi")])
        .await;
    let outcome = partial.run.terminal.expect("outcome");
    assert_eq!(outcome.reason, TerminalReason::Cancelled);
    assert_eq!(outcome.class, TerminalClass::Cancellation);
    assert!(!outcome.provider_started);
}

#[tokio::test]
async fn a_run_deadline_during_a_provider_call_reports_a_provider_phase_timeout() {
    let harness = harness_with(Arc::new(PendingModel));
    let ctx = RunContext::new(RunConfig::new("deadline").with_timeout_ms(30), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("hi")])
        .await;
    assert!(matches!(partial.error, Some(TinyAgentsError::Timeout(_))));
    let outcome = partial.run.terminal.expect("outcome");
    assert_eq!(outcome.reason, TerminalReason::Timeout);
    assert_eq!(outcome.class, TerminalClass::Timeout);
    assert_eq!(outcome.timeout_phase, Some(TimeoutPhase::Provider));
    assert!(outcome.provider_started);
}

#[tokio::test]
async fn a_paused_run_reports_a_suspended_outcome_without_run_completed() {
    let harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "x")])));
    let handle = SteeringHandle::new(SteeringPolicy::allow_all());
    handle.send(SteeringCommand::Pause);
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("pause"), ())
        .with_steering(handle)
        .with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .unwrap();
    assert!(run.paused.is_some());
    let outcome = run.terminal.expect("outcome");
    assert_eq!(outcome.reason, TerminalReason::Paused);
    assert_eq!(outcome.class, TerminalClass::Suspended);
    assert!(terminal_events(&recorder.events()).is_empty());
}

#[test]
fn outcome_helper_is_usable_from_hosts() {
    // Public constructors compose with merge for hosts that race signals.
    let merged = TerminalOutcome::completed().merge(TerminalOutcome::halted("loop"));
    assert_eq!(merged.reason, TerminalReason::Halted);
}

#[tokio::test]
async fn a_limit_exceeded_failure_carries_the_limit_kind() {
    let harness = harness_with(Arc::new(ScriptedModel::new(vec![response(
        vec![ToolCall::new("c1", "spin", serde_json::json!({}))],
        "",
    )])));
    let ctx = RunContext::new(RunConfig::new("cap-error").with_max_model_calls(1), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("go")])
        .await;
    assert!(matches!(
        partial.error,
        Some(TinyAgentsError::LimitExceeded(_))
    ));
    assert_eq!(
        partial.run.terminal.expect("outcome").reason,
        TerminalReason::LimitReached(Some(LimitKind::ModelCalls))
    );
}

#[tokio::test]
async fn after_agent_middleware_can_read_the_terminal_outcome() {
    use crate::middleware::{AgentRun, Middleware};
    use std::sync::Mutex;
    struct Reader(Arc<Mutex<Option<TerminalOutcome>>>);
    #[async_trait]
    impl Middleware<(), ()> for Reader {
        fn name(&self) -> &str {
            "reader"
        }
        async fn after_agent(
            &self,
            _: &mut RunContext<()>,
            _: &(),
            run: &mut AgentRun,
        ) -> crate::error::Result<()> {
            *self.0.lock().unwrap() = run.terminal.clone();
            Ok(())
        }
    }
    let seen = Arc::new(Mutex::new(None));
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    harness.push_middleware(Arc::new(Reader(seen.clone())));
    harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .unwrap();
    assert_eq!(
        seen.lock().unwrap().as_ref().map(|o| o.reason),
        Some(TerminalReason::Completed)
    );
}

#[tokio::test]
async fn a_timeout_outcome_set_by_middleware_on_an_ok_return_is_not_completed() {
    use crate::ids::ExecutionStatus;
    use crate::middleware::{AgentRun, Middleware};
    struct ForceTimeout;
    #[async_trait]
    impl Middleware<(), ()> for ForceTimeout {
        fn name(&self) -> &str {
            "force_timeout"
        }
        async fn after_agent(
            &self,
            _: &mut RunContext<()>,
            _: &(),
            run: &mut AgentRun,
        ) -> crate::error::Result<()> {
            run.terminal = Some(TerminalOutcome::new(TerminalReason::Timeout, "deadline"));
            Ok(())
        }
    }
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    harness.push_middleware(Arc::new(ForceTimeout));
    let ctx = RunContext::new(RunConfig::new("to"), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("hi")])
        .await;
    assert!(partial.error.is_none());
    assert_eq!(partial.status.status, ExecutionStatus::Failed);
}

#[tokio::test]
async fn a_cache_served_run_does_not_claim_the_provider_started() {
    use crate::cache::InMemoryResponseCache;
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    harness.with_response_cache(Arc::new(InMemoryResponseCache::new()));
    let input = vec![Message::user("same request")];
    let first = harness.invoke_default(&(), input.clone()).await.unwrap();
    assert!(first.terminal.unwrap().provider_started);
    let second = harness.invoke_default(&(), input).await.unwrap();
    assert_eq!(second.text(), Some("done".to_string()));
    assert!(!second.terminal.unwrap().provider_started);
}

#[tokio::test]
async fn an_after_model_timeout_is_classified_after_the_provider_call() {
    use crate::middleware::Middleware;
    struct SlowAfterModel;
    #[async_trait]
    impl Middleware<(), ()> for SlowAfterModel {
        fn name(&self) -> &str {
            "slow_after_model"
        }
        async fn after_model(
            &self,
            _: &mut RunContext<()>,
            _: &(),
            _: &mut ModelResponse,
        ) -> crate::error::Result<()> {
            Err(TinyAgentsError::Timeout("hook deadline".into()))
        }
    }
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    harness.push_middleware(Arc::new(SlowAfterModel));
    let ctx = RunContext::new(RunConfig::new("am"), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("hi")])
        .await;
    assert!(partial.error.is_some());
    let outcome = partial.run.terminal.expect("outcome");
    assert_eq!(outcome.reason, TerminalReason::Timeout);
    assert_eq!(outcome.timeout_phase, Some(TimeoutPhase::AfterTurn));
    assert!(outcome.provider_started);
}

#[tokio::test]
async fn a_wrap_model_rejection_before_dispatch_does_not_claim_the_provider_started() {
    use crate::middleware::{MiddlewareModelOutcome, ModelHandler, ModelMiddleware};
    struct Reject;
    #[async_trait]
    impl ModelMiddleware<(), ()> for Reject {
        fn name(&self) -> &str {
            "reject"
        }
        async fn wrap_model(
            &self,
            _: &mut RunContext<()>,
            _: &(),
            _: ModelRequest,
            _: ModelHandler<'_, (), ()>,
        ) -> crate::error::Result<MiddlewareModelOutcome> {
            Err(TinyAgentsError::Timeout("rejected before dispatch".into()))
        }
    }
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    harness.push_model_middleware(Arc::new(Reject));
    let ctx = RunContext::new(RunConfig::new("rej"), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("hi")])
        .await;
    assert!(partial.error.is_some());
    let outcome = partial.run.terminal.expect("outcome");
    assert!(!outcome.provider_started);
    assert_eq!(outcome.timeout_phase, Some(TimeoutPhase::BeforeProvider));
}

#[tokio::test]
async fn run_completed_reports_the_outcome_after_after_agent_middleware_replaced_it() {
    use crate::middleware::{AgentRun, Middleware};
    struct Replace;
    #[async_trait]
    impl Middleware<(), ()> for Replace {
        fn name(&self) -> &str {
            "replace"
        }
        async fn after_agent(
            &self,
            _: &mut RunContext<()>,
            _: &(),
            run: &mut AgentRun,
        ) -> crate::error::Result<()> {
            run.terminal = Some(TerminalOutcome::new(TerminalReason::Timeout, "forced"));
            Ok(())
        }
    }
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    harness.push_middleware(Arc::new(Replace));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("rep"), ()).with_events(recorder.sink());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .unwrap();
    let events = recorder.events();
    assert!(matches!(
        terminal_events(&events)[0],
        AgentEvent::RunCompleted { outcome: Some(o), .. } if o.reason == TerminalReason::Timeout
    ));
}

#[tokio::test]
async fn a_later_call_rejected_before_dispatch_is_not_a_provider_phase_failure() {
    use crate::middleware::{MiddlewareModelOutcome, ModelHandler, ModelMiddleware};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct RejectSecond(AtomicUsize);
    #[async_trait]
    impl ModelMiddleware<(), ()> for RejectSecond {
        fn name(&self) -> &str {
            "reject_second"
        }
        async fn wrap_model(
            &self,
            ctx: &mut RunContext<()>,
            state: &(),
            request: ModelRequest,
            next: ModelHandler<'_, (), ()>,
        ) -> crate::error::Result<MiddlewareModelOutcome> {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                next.run(ctx, state, request).await
            } else {
                Err(TinyAgentsError::Timeout("rejected before dispatch".into()))
            }
        }
    }
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![
        response(vec![ToolCall::new("c1", "t", serde_json::json!({}))], ""),
        response(vec![], "done"),
    ])));
    harness.push_model_middleware(Arc::new(RejectSecond(AtomicUsize::new(0))));
    let ctx = RunContext::new(RunConfig::new("second"), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("hi")])
        .await;
    assert!(partial.error.is_some());
    let outcome = partial.run.terminal.expect("outcome");
    assert_eq!(outcome.timeout_phase, Some(TimeoutPhase::BeforeProvider));
    assert!(
        outcome.provider_started,
        "an earlier call reached the provider"
    );
}

#[tokio::test]
async fn a_suspended_outcome_set_by_middleware_is_interrupted_not_completed() {
    use crate::ids::ExecutionStatus;
    use crate::middleware::{AgentRun, Middleware};
    struct Suspend;
    #[async_trait]
    impl Middleware<(), ()> for Suspend {
        fn name(&self) -> &str {
            "suspend"
        }
        async fn after_agent(
            &self,
            _: &mut RunContext<()>,
            _: &(),
            run: &mut AgentRun,
        ) -> crate::error::Result<()> {
            run.terminal = Some(TerminalOutcome::new(TerminalReason::Paused, "hold"));
            Ok(())
        }
    }
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    harness.push_middleware(Arc::new(Suspend));
    let ctx = RunContext::new(RunConfig::new("susp"), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("hi")])
        .await;
    assert_eq!(partial.status.status, ExecutionStatus::Interrupted);
}

#[tokio::test]
async fn a_cache_hit_followed_by_an_after_model_error_does_not_claim_the_provider() {
    use crate::cache::InMemoryResponseCache;
    use crate::middleware::Middleware;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct FailSecond(AtomicUsize);
    #[async_trait]
    impl Middleware<(), ()> for FailSecond {
        fn name(&self) -> &str {
            "fail_second"
        }
        async fn after_model(
            &self,
            _: &mut RunContext<()>,
            _: &(),
            _: &mut ModelResponse,
        ) -> crate::error::Result<()> {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(())
            } else {
                Err(TinyAgentsError::Timeout("late".into()))
            }
        }
    }
    let mut harness = harness_with(Arc::new(ScriptedModel::new(vec![response(vec![], "done")])));
    harness.with_response_cache(Arc::new(InMemoryResponseCache::new()));
    harness.push_middleware(Arc::new(FailSecond(AtomicUsize::new(0))));
    let input = vec![Message::user("same request")];
    harness.invoke_default(&(), input.clone()).await.unwrap();
    let ctx = RunContext::new(RunConfig::new("cachehit"), ());
    let partial = harness
        .invoke_in_context_collecting_partial(&(), ctx, input)
        .await;
    assert!(partial.error.is_some());
    assert!(!partial.run.terminal.expect("outcome").provider_started);
}
