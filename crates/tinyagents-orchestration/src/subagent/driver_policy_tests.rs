//! Policy behaviour of the lifecycle driver: timeout, retry, budget, role and
//! result policy, each observed through a real `SubagentDriver::run`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tinyagents_harness::CancellationToken;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::retry::RetryPolicy;
use tinyagents_runtime::ToolSnapshot;
use tinyinference_llm::message::Message;

use super::*;

type Planned = Arc<dyn Fn(PreparedSubagent<String>) -> PreparedSubagent<String> + Send + Sync>;

struct Planner(Planned);

#[async_trait]
impl SubagentPlanner<String> for Planner {
    async fn prepare(
        &self,
        request: SubagentRequest<String>,
    ) -> Result<PreparedSubagent<String>, SubagentError> {
        let parts = request.into_parts();
        Ok((self.0)(PreparedSubagent::new(
            parts.task_key.task_id,
            "agent",
            vec![Message::user(parts.input)],
            ToolSnapshot::new(vec![]).unwrap(),
            parts.run_context,
        )))
    }
}

type Behaviour = Arc<
    dyn Fn(u32, &SubagentExecution<String>) -> Result<SubagentOutcome, SubagentError> + Send + Sync,
>;

struct Executor {
    attempts: AtomicU32,
    seen_tools: Mutex<Vec<Vec<String>>>,
    seen_caps: Mutex<Vec<(Option<usize>, Option<usize>)>>,
    child_tokens: Mutex<Vec<CancellationToken>>,
    hang: bool,
    behaviour: Behaviour,
}

impl Executor {
    fn new(behaviour: Behaviour) -> Arc<Self> {
        Arc::new(Self {
            attempts: AtomicU32::new(0),
            seen_tools: Mutex::default(),
            seen_caps: Mutex::default(),
            child_tokens: Mutex::default(),
            hang: false,
            behaviour,
        })
    }

    fn hanging() -> Arc<Self> {
        Arc::new(Self {
            hang: true,
            ..Arc::into_inner(Self::new(Arc::new(|_, _| unreachable!()))).unwrap()
        })
    }
}

#[async_trait]
impl SubagentExecutor<String> for Executor {
    async fn execute(
        &self,
        execution: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        self.seen_tools.lock().unwrap().push(
            execution
                .prepared
                .tools
                .specs()
                .iter()
                .map(|s| s.name.clone())
                .collect(),
        );
        let config = &execution.prepared.run_context.config;
        self.seen_caps
            .lock()
            .unwrap()
            .push((config.max_model_calls, config.max_tool_calls));
        self.child_tokens
            .lock()
            .unwrap()
            .push(execution.cancellation.clone());
        if self.hang {
            std::future::pending::<()>().await;
        }
        (self.behaviour)(attempt, &execution)
    }
}

#[derive(Default)]
struct Memory(Mutex<HashMap<SubagentTaskKey, SubagentOutcome>>);

#[async_trait]
impl SubagentPersistence for Memory {
    async fn load_terminal(
        &self,
        key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(self.0.lock().unwrap().get(key).cloned())
    }
    async fn load(&self, _: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError> {
        Ok(None)
    }
    async fn load_pause(
        &self,
        _: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(None)
    }
    async fn save_pause(
        &self,
        _: PersistedSubagentPause,
    ) -> Result<SubagentPausePersistenceDisposition, SubagentError> {
        unreachable!("policy tests never pause")
    }
    async fn record_terminal(
        &self,
        key: &SubagentTaskKey,
        outcome: &SubagentOutcome,
        _: Option<&SubagentResume>,
    ) -> Result<SubagentTerminalPersistenceDisposition, SubagentError> {
        self.0.lock().unwrap().insert(key.clone(), outcome.clone());
        Ok(SubagentTerminalPersistenceDisposition::Inserted)
    }
}

fn completed(task_id: &str, output: &str) -> SubagentOutcome {
    SubagentOutcome {
        output: output.into(),
        status: SubagentOutcomeKind::Completed,
        ..SubagentOutcome::cancelled(task_id)
    }
}

fn ok_with(output: &'static str) -> Behaviour {
    Arc::new(move |_, e| Ok(completed(&e.prepared.task_id, output)))
}

fn request(task_id: &str) -> SubagentRequest<String> {
    let parent = RunContext::new(RunConfig::new(format!("parent-{task_id}")), "d".to_owned());
    let child = parent
        .child(RunConfig::new(format!("run-{task_id}")), "d".to_owned())
        .unwrap();
    SubagentRequest::fresh_from_parent(&parent, child, task_id, (), "go", None).unwrap()
}

fn driver(plan: Planned, executor: Arc<Executor>) -> SubagentDriver<String> {
    SubagentDriver::new(SubagentCapabilities {
        planner: Some(Arc::new(Planner(plan))),
        executor: Some(executor),
        persistence: Some(Arc::new(Memory::default())),
    })
    .unwrap()
}

fn plain() -> Planned {
    Arc::new(|p| p)
}

fn retry_policy(attempts: usize) -> SubAgentPolicy {
    SubAgentPolicy::default().with_retry(
        RetryPolicy::default()
            .with_max_attempts(attempts)
            .with_backoff_sleep(false),
    )
}

fn with_factory(p: PreparedSubagent<String>) -> PreparedSubagent<String> {
    p.with_retry_context(Arc::new(|n| {
        Ok(RunContext::new(
            RunConfig::new(format!("retry-{n}")),
            "d".to_owned(),
        ))
    }))
}

fn transient(tools_ran: bool) -> SubagentError {
    SubagentError::Transient {
        message: "flaky".into(),
        tools_ran,
    }
}

fn flaky(failures: u32, tools_ran: bool) -> Behaviour {
    Arc::new(move |attempt, e| {
        if attempt < failures {
            Err(transient(tools_ran))
        } else {
            Ok(completed(&e.prepared.task_id, "recovered"))
        }
    })
}

#[tokio::test]
async fn timeout_cancels_the_child_and_ends_incomplete_without_cancelling_the_lifecycle() {
    let executor = Executor::hanging();
    let plan: Planned = Arc::new(|p| {
        p.with_policy(SubAgentPolicy::default().with_timeout(Duration::from_millis(30)))
    });
    let lifecycle = CancellationToken::new();
    let result = driver(plan, executor.clone())
        .run(request("t-timeout"), lifecycle.clone())
        .await
        .unwrap();
    match result.outcome.status {
        SubagentOutcomeKind::Incomplete(ref inc) => assert_eq!(inc.kind, IncompleteKind::Timeout),
        ref other => panic!("expected typed timeout, got {other:?}"),
    }
    assert!(
        executor.child_tokens.lock().unwrap()[0].is_cancelled(),
        "the child was cancelled"
    );
    assert!(!lifecycle.is_cancelled(), "the caller's token is untouched");
    assert_eq!(
        executor.attempts.load(Ordering::SeqCst),
        1,
        "timeouts are not retried"
    );
}

#[tokio::test]
async fn retries_a_transient_failure_that_ran_no_tools() {
    let executor = Executor::new(flaky(2, false));
    let plan: Planned = Arc::new(|p| with_factory(p.with_policy(retry_policy(3))));
    let result = driver(plan, executor.clone())
        .run(request("t-retry"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.outcome.output, "recovered");
    assert_eq!(executor.attempts.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn does_not_retry_after_tools_ran_unless_allowed() {
    let executor = Executor::new(flaky(1, true));
    let plan: Planned = Arc::new(|p| with_factory(p.with_policy(retry_policy(3))));
    let err = driver(plan, executor.clone())
        .run(request("t-side-effects"), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        SubagentError::Transient {
            tools_ran: true,
            ..
        }
    ));
    assert_eq!(executor.attempts.load(Ordering::SeqCst), 1);

    let executor = Executor::new(flaky(1, true));
    let plan: Planned = Arc::new(|p| {
        with_factory(p.with_policy(retry_policy(3).with_retry_after_tool_calls(true)))
    });
    let result = driver(plan, executor.clone())
        .run(request("t-side-effects-ok"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.outcome.output, "recovered");
}

#[tokio::test]
async fn unclassified_failures_and_missing_factories_are_never_retried() {
    let executor = Executor::new(Arc::new(|_, _| {
        Err(SubagentError::Execution("boom".into()))
    }));
    let plan: Planned = Arc::new(|p| with_factory(p.with_policy(retry_policy(3))));
    driver(plan, executor.clone())
        .run(request("t-exec"), CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(executor.attempts.load(Ordering::SeqCst), 1);

    let executor = Executor::new(flaky(1, false));
    let plan: Planned = Arc::new(|p| p.with_policy(retry_policy(3)));
    driver(plan, executor.clone())
        .run(request("t-nofactory"), CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(
        executor.attempts.load(Ordering::SeqCst),
        1,
        "no factory, no retry"
    );
}

#[tokio::test]
async fn call_budget_tightens_the_child_run_config() {
    let executor = Executor::new(ok_with("x"));
    let plan: Planned = Arc::new(|p| {
        p.with_policy(
            SubAgentPolicy::default().with_budget(
                SubAgentBudget::unlimited()
                    .with_max_model_calls(4)
                    .with_max_tool_calls(9),
            ),
        )
    });
    driver(plan, executor.clone())
        .run(request("t-caps"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(executor.seen_caps.lock().unwrap()[0], (Some(4), Some(9)));
}

#[tokio::test]
async fn token_budget_overshoot_ends_incomplete_with_a_typed_kind() {
    let executor = Executor::new(Arc::new(|_, e| {
        let mut outcome = completed(&e.prepared.task_id, "long answer");
        outcome.usage.usage.output_tokens = 500;
        Ok(outcome)
    }));
    let plan: Planned = Arc::new(|p| {
        p.with_policy(
            SubAgentPolicy::default()
                .with_budget(SubAgentBudget::unlimited().with_max_output_tokens(100)),
        )
    });
    let result = driver(plan, executor)
        .run(request("t-tokens"), CancellationToken::new())
        .await
        .unwrap();
    match result.outcome.status {
        SubagentOutcomeKind::Incomplete(inc) => {
            assert_eq!(inc.kind, IncompleteKind::BudgetExceeded);
            assert!(inc.reason.contains("output-token"));
        }
        other => panic!("expected budget incomplete, got {other:?}"),
    }
    assert_eq!(result.outcome.output, "long answer", "work done is kept");
}

fn spec(name: &str) -> tinytools::ToolSpec {
    tinytools::ToolSpec {
        name: name.into(),
        description: "d".into(),
        parameters: serde_json::json!({}),
    }
}

#[tokio::test]
async fn leaf_role_strips_delegation_tools_and_ceiling_blocks_widening() {
    let executor = Executor::new(ok_with("x"));
    let plan: Planned = Arc::new(|mut p| {
        p.tools = ToolSnapshot::new(vec![
            spec("read"),
            spec("shell"),
            spec(SUBAGENT_JOBS_TOOL),
            spec("delegate_coder"),
        ])
        .unwrap();
        p.with_role(SubagentRole::Leaf)
            .with_delegation_tools(vec!["delegate_coder".into()])
            .with_tool_ceiling(
                ToolSnapshot::new(vec![spec("read"), spec(SUBAGENT_JOBS_TOOL)]).unwrap(),
            )
    });
    driver(plan, executor.clone())
        .run(request("t-role"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(executor.seen_tools.lock().unwrap()[0], ["read"]);
}

#[tokio::test]
async fn result_policy_trims_output_and_surfaces_a_schema_error() {
    let executor = Executor::new(ok_with("0123456789"));
    let plan: Planned = Arc::new(|p| {
        p.with_result_policy(
            ResultPolicy::new()
                .with_max_chars(4)
                .with_schema(serde_json::json!({"type": "object"})),
        )
    });
    let result = driver(plan, executor)
        .run(request("t-result"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.outcome.output, "0123", "hard-cut within the cap");
    assert!(result.outcome.schema_error.is_some());
    assert_eq!(result.outcome.status, SubagentOutcomeKind::Completed);
}

#[tokio::test]
async fn artifact_overflow_without_a_store_surfaces_artifact_error_on_the_outcome() {
    let executor = Executor::new(ok_with("0123456789"));
    let plan: Planned = Arc::new(|p| {
        p.with_result_policy(
            ResultPolicy::new()
                .with_max_chars(4)
                .with_overflow(ResultOverflow::Artifact),
        )
    });
    let result = driver(plan, executor)
        .run(request("t-artifact"), CancellationToken::new())
        .await
        .unwrap();
    assert!(
        result
            .outcome
            .artifact_error
            .unwrap()
            .contains("no artifact store")
    );
}

#[tokio::test]
async fn the_default_plan_changes_nothing() {
    let executor = Executor::new(ok_with("untouched output"));
    let result = driver(plain(), executor.clone())
        .run(request("t-default"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.outcome.output, "untouched output");
    assert!(result.outcome.schema_error.is_none());
    assert_eq!(executor.seen_caps.lock().unwrap()[0], (None, None));
}

#[tokio::test]
async fn call_caps_apply_to_every_retry_context() {
    let executor = Executor::new(flaky(2, false));
    let plan: Planned = Arc::new(|p| {
        with_factory(
            p.with_policy(
                retry_policy(3).with_budget(
                    SubAgentBudget::unlimited()
                        .with_max_model_calls(4)
                        .with_max_tool_calls(9),
                ),
            ),
        )
    });
    driver(plan, executor.clone())
        .run(request("t-retry-caps"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        *executor.seen_caps.lock().unwrap(),
        vec![(Some(4), Some(9)); 3],
        "the first attempt and both retries are capped"
    );
}

#[tokio::test]
async fn an_executor_that_cancels_its_token_is_not_retried() {
    let executor = Executor::new(Arc::new(|_, e| {
        e.cancellation.cancel();
        Err(transient(false))
    }));
    let plan: Planned = Arc::new(|p| with_factory(p.with_policy(retry_policy(3))));
    let _ = driver(plan, executor.clone())
        .run(request("t-exec-cancel"), CancellationToken::new())
        .await;
    assert_eq!(executor.attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn an_over_budget_run_still_gets_the_result_policy() {
    let executor = Executor::new(Arc::new(|_, e| {
        let mut outcome = completed(&e.prepared.task_id, &"y".repeat(500));
        outcome.usage.usage.output_tokens = 500;
        Ok(outcome)
    }));
    let plan: Planned = Arc::new(|p| {
        p.with_policy(
            SubAgentPolicy::default()
                .with_budget(SubAgentBudget::unlimited().with_max_output_tokens(100)),
        )
        .with_result_policy(ResultPolicy::new().with_max_chars(80))
    });
    let result = driver(plan, executor)
        .run(request("t-over-trim"), CancellationToken::new())
        .await
        .unwrap();
    assert!(matches!(
        result.outcome.status,
        SubagentOutcomeKind::Incomplete(ref inc) if inc.kind == IncompleteKind::BudgetExceeded
    ));
    assert!(result.outcome.output.chars().count() <= 80);
}
