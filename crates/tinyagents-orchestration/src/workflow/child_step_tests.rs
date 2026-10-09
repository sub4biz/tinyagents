//! Workflow agent children run through `SubagentDriver`: configured driver
//! policy takes effect on a workflow phase.

use super::*;

use std::sync::Arc;

use serde_json::json;
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::WorkflowRunStatus;

use crate::subagent::{ResultPolicy, SpawnAdmission, SpawnPolicy};
use crate::workflow::tests::{FakeExecutor, MemoryStore, definition};
use crate::workflow::WorkflowEngine;

fn run_with(
    config: AgentStepConfig,
) -> (
    Arc<MemoryStore>,
    Arc<FakeExecutor>,
    WorkflowEngine<MemoryStore, FakeExecutor>,
) {
    let store = Arc::new(MemoryStore::default());
    let executor = Arc::new(FakeExecutor::default());
    let engine = WorkflowEngine::new(store.clone(), executor.clone()).with_step_config(config);
    (store, executor, engine)
}

async fn drive(
    engine: &WorkflowEngine<MemoryStore, FakeExecutor>,
    store: &MemoryStore,
) -> serde_json::Value {
    let def = definition();
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    serde_json::to_value(store.load("run").unwrap().unwrap().phase_states).unwrap()
}

#[tokio::test]
async fn result_policy_cap_trims_a_workflow_child_output() {
    let (store, _executor, engine) = run_with(
        AgentStepConfig::default().with_result_policy(ResultPolicy::new().with_max_chars(4)),
    );
    let states = drive(&engine, &store).await;
    let out = states["plan"]["outputs"][0]["output"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_ne!(out, "plan output", "ResultPolicy trimmed the child output");
    assert!(
        out.chars().count() <= 4 + 64,
        "capped near max_chars: {out}"
    );
    // Trimmed output is carried verbatim as the raw output too.
    assert_eq!(
        states["plan"]["outputs"][0]["metadata"]["rawOutput"],
        json!(out)
    );
}

#[tokio::test]
async fn spawn_policy_denial_fails_the_phase_and_never_runs_the_child() {
    let (store, executor, engine) = run_with(AgentStepConfig::default().with_admission(
        SpawnAdmission::new(SpawnPolicy {
            allowed_targets: Some(vec!["only-this-agent".into()]),
            ..Default::default()
        }),
    ));
    drive(&engine, &store).await;
    let run = store.load("run").unwrap().unwrap();
    assert_eq!(run.status, WorkflowRunStatus::Failed);
    assert!(executor.calls.lock().is_empty(), "no child was executed");
    assert!(
        run.phase_states.to_string().contains("not admitted"),
        "{}",
        run.phase_states
    );
}

#[tokio::test]
async fn default_step_config_leaves_the_output_untouched() {
    let (store, _executor, engine) = run_with(AgentStepConfig::default());
    let states = drive(&engine, &store).await;
    assert_eq!(states["plan"]["outputs"][0]["output"], json!("plan output"));
    assert_eq!(
        store.load("run").unwrap().unwrap().status,
        WorkflowRunStatus::Completed
    );
}

#[tokio::test]
async fn timed_out_child_is_cancelled_by_its_registered_id() {
    use std::time::Duration;

    use crate::workflow::tests::BlockingExecutor;
    use crate::subagent::SubAgentPolicy;

    let store = Arc::new(MemoryStore::default());
    let executor = Arc::new(BlockingExecutor::default());
    let engine = WorkflowEngine::new(store.clone(), executor.clone()).with_step_config(
        AgentStepConfig::default()
            .with_policy(SubAgentPolicy::default().with_timeout(Duration::from_millis(30))),
    );
    let def = definition();
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    let run = store.load("run").unwrap().unwrap();
    assert_eq!(run.status, WorkflowRunStatus::Failed);
    assert!(
        run.phase_states.to_string().contains("timed out"),
        "{}",
        run.phase_states
    );
    assert!(
        executor
            .cancelled
            .lock()
            .iter()
            .any(|id| id == "live-plan-0"),
        "the timed-out child was cancelled: {:?}",
        executor.cancelled.lock()
    );
}
