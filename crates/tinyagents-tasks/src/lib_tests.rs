//! Unit tests for graph orchestration: tool argument parsing and outcomes for
//! `spawn`/`await`/`cancel`/`kill`/`status`/`list`/`timeout`/`race`/`yield`/
//! `steer`, `TaskStore` transition validation and filtering for both
//! [`InMemoryTaskStore`] and [`JsonlTaskStore`] (including log replay and
//! history), `reconcile_orphaned_tasks` sweep semantics, and
//! `DetachedTaskRegistry` ownership/steering/cancellation behavior.

use std::sync::Arc;

use serde_json::json;

use super::*;
use tinyagents_harness::cancel::CancellationToken;
use tinyagents_harness::ids::TaskId;
use tinyagents_harness::steering::SteeringHandle;
use tinyagents_harness::tool::ToolRegistry;
use tinytools::{Tool, ToolContent, ToolResult, ToolRunContext};

fn graph_spec(id: &str) -> OrchestrationTaskSpec {
    OrchestrationTaskSpec::new(
        id,
        OrchestrationTaskKind::Graph {
            graph_id: "child".into(),
        },
    )
}

struct TestToolContext;

impl ToolRunContext for TestToolContext {}

async fn run(tool: &OrchestrationTool, args: serde_json::Value) -> anyhow::Result<ToolResult> {
    tool.execute_with_context(args, Default::default(), Some(&TestToolContext))
        .await
}

fn raw(result: &ToolResult) -> &serde_json::Value {
    result
        .content
        .iter()
        .find_map(|content| match content {
            ToolContent::Json { data } => Some(data),
            ToolContent::Text { .. } | ToolContent::Image { .. } | ToolContent::File { .. } => None,
        })
        .expect("orchestration tool returns a JSON payload")
}

#[test]
fn in_memory_store_tracks_task_lifecycle() {
    let store = InMemoryTaskStore::new();
    let task_id = TaskId::new("task-a");

    let pending = store.insert(graph_spec(task_id.as_str())).unwrap();
    assert_eq!(pending.status, OrchestrationTaskStatus::Pending);

    let running = store.mark_running(&task_id).unwrap();
    assert_eq!(running.status, OrchestrationTaskStatus::Running);
    assert!(running.started_at.is_some());

    let completed = store
        .complete(&task_id, OrchestrationTaskResult::text("done"))
        .unwrap();
    assert_eq!(completed.status, OrchestrationTaskStatus::Completed);
    assert!(completed.ended_at.is_some());
    assert!(completed.is_terminal());
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RuntimeStatus {
    Running,
    Completed(String),
}

fn runtime_registry() -> DetachedTaskRegistry<String, RuntimeStatus> {
    DetachedTaskRegistry::new(SteeringRegistry::new(), 2, |status| {
        matches!(status, RuntimeStatus::Completed(_))
    })
}

fn detached_handles() -> (
    tokio::sync::watch::Sender<RuntimeStatus>,
    tokio::sync::watch::Receiver<RuntimeStatus>,
    CancellationToken,
    tokio::task::JoinHandle<()>,
) {
    let (tx, rx) = tokio::sync::watch::channel(RuntimeStatus::Running);
    let cancellation = CancellationToken::new();
    let join = tokio::spawn(std::future::pending());
    (tx, rx, cancellation, join)
}

#[tokio::test]
async fn detached_registry_enforces_owner_and_waits_for_terminal_status() {
    let registry = runtime_registry();
    let task_id = TaskId::new("detached-wait");
    let (tx, rx, cancellation, join) = detached_handles();
    registry
        .register(
            task_id.clone(),
            "parent-a",
            "researcher".to_string(),
            rx,
            cancellation,
            join.abort_handle(),
        )
        .unwrap();

    assert_eq!(
        registry.snapshot(&task_id, "parent-b").unwrap_err(),
        DetachedTaskRegistryError::NotOwned
    );
    assert_eq!(
        registry.snapshot(&task_id, "parent-a").unwrap().metadata,
        "researcher"
    );

    assert_eq!(
        registry
            .wait(&task_id, "parent-b", std::time::Duration::from_millis(1))
            .await,
        Err(DetachedTaskRegistryError::NotOwned)
    );
    tx.send(RuntimeStatus::Completed("done".into())).unwrap();
    assert_eq!(
        registry
            .wait(&task_id, "parent-a", std::time::Duration::from_secs(1))
            .await
            .unwrap(),
        DetachedTaskWaitOutcome::Terminal(RuntimeStatus::Completed("done".into()))
    );
    assert!(registry.is_empty().unwrap());
    join.abort();
}

#[tokio::test]
async fn detached_registry_remembers_steer_request_ids_per_task() {
    let registry = runtime_registry();
    let (first, rx_first, cancel_first, join_first) = detached_handles();
    let (second, rx_second, cancel_second, join_second) = detached_handles();
    let _keep = (first, second);
    let first_id = TaskId::new("steer-a");
    let second_id = TaskId::new("steer-b");
    for (id, rx, cancel, join) in [
        (&first_id, rx_first, cancel_first, &join_first),
        (&second_id, rx_second, cancel_second, &join_second),
    ] {
        registry
            .register(
                id.clone(),
                "p",
                "m".to_string(),
                rx,
                cancel,
                join.abort_handle(),
            )
            .unwrap();
    }

    assert_eq!(registry.claim_steer_request(&first_id, "req-1"), Ok(true));
    assert_eq!(
        registry.claim_steer_request(&first_id, "req-1"),
        Ok(false),
        "the same request id on the same task is a duplicate"
    );
    assert_eq!(
        registry.claim_steer_request(&second_id, "req-1"),
        Ok(true),
        "request ids are scoped to a task"
    );
    assert_eq!(
        registry.claim_steer_request(&TaskId::new("nope"), "req-1"),
        Err(DetachedTaskRegistryError::Unknown)
    );
    join_first.abort();
    join_second.abort();
}

#[tokio::test]
async fn detached_registry_timeout_keeps_task_registered() {
    let registry = runtime_registry();
    let task_id = TaskId::new("detached-timeout");
    let (_tx, rx, cancellation, join) = detached_handles();
    registry
        .register(
            task_id.clone(),
            "parent",
            "worker".to_string(),
            rx,
            cancellation,
            join.abort_handle(),
        )
        .unwrap();

    assert_eq!(
        registry
            .wait(&task_id, "parent", std::time::Duration::from_millis(1))
            .await
            .unwrap(),
        DetachedTaskWaitOutcome::TimedOut(RuntimeStatus::Running)
    );
    assert_eq!(registry.len().unwrap(), 1);
    join.abort();
}

#[tokio::test]
async fn detached_registry_cancel_is_cooperative_then_aborts_and_returns_metadata() {
    let registry = runtime_registry();
    let task_id = TaskId::new("detached-cancel");
    let (_tx, rx, cancellation, join) = detached_handles();
    registry
        .register(
            task_id.clone(),
            "parent",
            "worker-meta".to_string(),
            rx,
            cancellation.clone(),
            join.abort_handle(),
        )
        .unwrap();

    let cancelled = registry.cancel(&task_id, "parent").unwrap();
    assert_eq!(cancelled.metadata, "worker-meta");
    assert!(cancellation.is_cancelled());
    assert!(join.await.unwrap_err().is_cancelled());
    assert!(registry.is_empty().unwrap());
}

#[tokio::test]
async fn detached_registry_uses_shared_steering_and_sweeps_terminal_at_soft_cap() {
    let steering = SteeringRegistry::new();
    let registry = DetachedTaskRegistry::new(steering.clone(), 1, |status: &RuntimeStatus| {
        matches!(status, RuntimeStatus::Completed(_))
    });
    let first = TaskId::new("detached-first");
    let (first_tx, first_rx, first_cancel, first_join) = detached_handles();
    registry
        .register(
            first.clone(),
            "parent",
            "first".to_string(),
            first_rx,
            first_cancel,
            first_join.abort_handle(),
        )
        .unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(first.clone(), handle);
    assert!(registry.steering_handle(&first, "parent").is_ok());
    first_tx
        .send(RuntimeStatus::Completed("done".into()))
        .unwrap();

    let second = TaskId::new("detached-second");
    let (_second_tx, second_rx, second_cancel, second_join) = detached_handles();
    registry
        .register(
            second.clone(),
            "parent",
            "second".to_string(),
            second_rx,
            second_cancel,
            second_join.abort_handle(),
        )
        .unwrap();

    assert_eq!(registry.len().unwrap(), 1);
    assert!(steering.get(&first).is_none());
    assert_eq!(
        registry.snapshots(Some("parent")).unwrap()[0].task_id,
        second
    );
    first_join.abort();
    second_join.abort();
}

#[derive(Debug)]
struct PanickingMetadata;

impl Clone for PanickingMetadata {
    fn clone(&self) -> Self {
        panic!("poison registry lock during metadata clone")
    }
}

#[tokio::test]
async fn detached_registry_reports_a_poisoned_lock() {
    let registry =
        DetachedTaskRegistry::new(SteeringRegistry::new(), 2, |status: &RuntimeStatus| {
            matches!(status, RuntimeStatus::Completed(_))
        });
    let task_id = TaskId::new("detached-poison");
    let (_tx, rx, cancellation, join) = detached_handles();
    registry
        .register(
            task_id,
            "parent",
            PanickingMetadata,
            rx,
            cancellation,
            join.abort_handle(),
        )
        .unwrap();

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = registry.snapshots(None);
    }));
    assert!(panic.is_err());
    assert_eq!(registry.len(), Err(DetachedTaskRegistryError::LockPoisoned));
    join.abort();
}

#[tokio::test]
async fn cooperative_registration_cancels_without_removing_or_aborting() {
    let registry = runtime_registry();
    let task_id = TaskId::new("coop-cancel");
    let (tx, rx, _cancellation, join) = detached_handles();
    let cancellation = CancellationToken::new();
    registry
        .register_cooperative(
            task_id.clone(),
            "parent",
            "meta".to_string(),
            rx,
            cancellation.clone(),
        )
        .unwrap();
    assert!(registry.holds_cancellation(&task_id).unwrap());

    assert_eq!(
        registry.cancel_cooperative(&task_id, "other").unwrap_err(),
        DetachedTaskRegistryError::NotOwned
    );
    assert!(!cancellation.is_cancelled());

    let snapshot = registry.cancel_cooperative(&task_id, "parent").unwrap();
    assert_eq!(snapshot.status, RuntimeStatus::Running);
    assert!(cancellation.is_cancelled());
    // The entry stays registered and the token is released.
    assert_eq!(registry.len().unwrap(), 1);
    assert!(!registry.holds_cancellation(&task_id).unwrap());

    tx.send(RuntimeStatus::Completed("x".into())).unwrap();
    assert_eq!(
        registry.cancel_cooperative(&task_id, "parent").unwrap_err(),
        DetachedTaskRegistryError::AlreadyDone
    );
    join.abort();
}

#[tokio::test]
async fn release_cancellation_drops_the_token_without_cancelling_it() {
    let registry = runtime_registry();
    let task_id = TaskId::new("coop-release");
    let (_tx, rx, _c, join) = detached_handles();
    let cancellation = CancellationToken::new();
    registry
        .register_cooperative(task_id.clone(), "p", "m".to_string(), rx, cancellation.clone())
        .unwrap();
    registry.release_cancellation(&task_id).unwrap();
    assert!(!registry.holds_cancellation(&task_id).unwrap());
    assert!(!cancellation.is_cancelled());
    // Hard cancel still works on an entry that holds no abort handle.
    let cancelled = registry.cancel_trusted(&task_id).unwrap();
    assert_eq!(cancelled.metadata, "m");
    join.abort();
}

fn unique_log_path(tag: &str) -> std::path::PathBuf {
    // Deterministic-per-test path in the system temp dir (no clock/random ids).
    std::env::temp_dir().join(format!("tinyagents-taskstore-{tag}.jsonl"))
}

#[test]
fn jsonl_store_survives_restart_and_keeps_history() {
    let path = unique_log_path("restart");
    let _ = std::fs::remove_file(&path);
    let task_id = TaskId::new("task-a");

    {
        let store = JsonlTaskStore::open(&path).unwrap();
        store.insert(graph_spec(task_id.as_str())).unwrap();
        store.mark_running(&task_id).unwrap();
        store
            .complete(&task_id, OrchestrationTaskResult::text("done"))
            .unwrap();
        // Full lifecycle history is retained (pending → running → completed).
        let history = store.history(&task_id);
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].status, OrchestrationTaskStatus::Pending);
        assert_eq!(history[2].status, OrchestrationTaskStatus::Completed);
    }

    // Re-open: state and history are reconstructed from the append log.
    let reopened = JsonlTaskStore::open(&path).unwrap();
    let record = reopened.get(&task_id).expect("task survives restart");
    assert_eq!(record.status, OrchestrationTaskStatus::Completed);
    assert_eq!(reopened.history(&task_id).len(), 3);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn filter_by_kind_and_created_window() {
    let store = InMemoryTaskStore::new();
    store.insert(graph_spec("g1")).unwrap();
    store
        .insert(OrchestrationTaskSpec::new(
            "s1",
            OrchestrationTaskKind::SubAgent {
                agent: "worker".into(),
            },
        ))
        .unwrap();

    let sub_agents = store.list(OrchestrationTaskFilter::default().with_kind("sub_agent"));
    assert_eq!(sub_agents.len(), 1);
    assert_eq!(sub_agents[0].spec.task_id.as_str(), "s1");

    // A created-before bound in the past excludes everything.
    let none = store.list(
        OrchestrationTaskFilter::default().created_between(None, Some(std::time::UNIX_EPOCH)),
    );
    assert!(none.is_empty());
    // A created-after bound in the past includes everything.
    let all = store.list(
        OrchestrationTaskFilter::default().created_between(Some(std::time::UNIX_EPOCH), None),
    );
    assert_eq!(all.len(), 2);
}

#[test]
fn terminal_tasks_reject_further_control() {
    let store = InMemoryTaskStore::new();
    let task_id = TaskId::new("task-a");

    store.insert(graph_spec(task_id.as_str())).unwrap();
    store
        .fail(&task_id, "child failed".to_string())
        .expect("live task can fail");

    let err = store
        .request_cancel(&task_id)
        .expect_err("terminal task cannot be cancelled");
    assert!(err.to_string().contains("cannot transition"));
}

#[test]
fn register_orchestration_tools_adds_normal_tool_names() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();

    register_orchestration_tools(&mut registry, store);

    assert!(registry.get("orchestrate_spawn").is_some());
    assert!(registry.get("orchestrate_cancel").is_some());
    assert!(registry.names().contains(&"orchestrate_status".to_string()));
}

#[tokio::test]
async fn spawn_and_status_run_through_tool_trait() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let spawn = OrchestrationTool::new(OrchestrationToolKind::Spawn, store.clone());
    let status = OrchestrationTool::new(OrchestrationToolKind::Status, store);

    let spawned = run(
        &spawn,
        json!({
            "kind": "graph",
            "target": "planner",
            "timeout_ms": 1000
        }),
    )
    .await
    .unwrap();
    let task_id = raw(&spawned)["spec"]["task_id"]
        .as_str()
        .unwrap()
        .to_string();

    let inspected = run(&status, json!({ "task_id": task_id })).await.unwrap();

    assert_eq!(raw(&inspected)["status"], "pending");
}

#[tokio::test]
async fn list_tool_honors_created_window_and_kind() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let spawn = OrchestrationTool::new(OrchestrationToolKind::Spawn, store.clone());
    let list = OrchestrationTool::new(OrchestrationToolKind::List, store);

    run(&spawn, json!({ "kind": "graph", "target": "planner" }))
        .await
        .unwrap();
    run(&spawn, json!({ "kind": "sub_agent", "target": "writer" }))
        .await
        .unwrap();

    // Kind filter routes through the tool.
    let sub_agents = run(&list, json!({ "kind": "sub_agent" })).await.unwrap();
    let sub_agents = raw(&sub_agents);
    assert_eq!(sub_agents.as_array().unwrap().len(), 1);
    assert_eq!(sub_agents[0]["spec"]["kind"]["type"], "sub_agent");

    // An impossibly-early upper bound excludes everything created just now.
    let none = run(&list, json!({ "created_before_ms": 0 })).await.unwrap();
    assert!(raw(&none).as_array().unwrap().is_empty());

    // A window opening at the epoch includes both tasks.
    let all = run(&list, json!({ "created_after_ms": 0 })).await.unwrap();
    assert_eq!(raw(&all).as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn spawn_tool_preserves_every_task_kind_input_and_timeout() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let spawn = OrchestrationTool::new(OrchestrationToolKind::Spawn, store);

    let cases = [
        ("graph", "planner", "graph", "graph_id"),
        ("sub_agent", "writer", "sub_agent", "agent"),
        ("tool", "search", "tool", "tool"),
        (
            "external_process",
            "sandboxed-worker",
            "external_process",
            "label",
        ),
    ];

    for (kind, target, serialized_kind, target_field) in cases {
        let result = run(
            &spawn,
            json!({
                "kind": kind,
                "target": target,
                "input": { "topic": target },
                "timeout_ms": 250
            }),
        )
        .await
        .unwrap();
        let result = raw(&result);
        assert_eq!(result["status"], "pending");
        assert_eq!(result["spec"]["kind"]["type"], serialized_kind);
        assert_eq!(result["spec"]["kind"][target_field], target);
        assert_eq!(result["spec"]["input"]["topic"], target);
        assert_eq!(result["spec"]["timeout_ms"], 250);
    }
}

#[tokio::test]
async fn await_cancel_kill_timeout_and_yield_tools_return_control_records() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let task_a = TaskId::new("task-a");
    let task_b = TaskId::new("task-b");
    let task_c = TaskId::new("task-c");
    store.insert(graph_spec(task_a.as_str())).unwrap();
    store.insert(graph_spec(task_b.as_str())).unwrap();
    store.insert(graph_spec(task_c.as_str())).unwrap();

    let timeout = OrchestrationTool::new(OrchestrationToolKind::Timeout, store.clone());
    let awaited = OrchestrationTool::new(OrchestrationToolKind::Await, store.clone());
    let cancel = OrchestrationTool::new(OrchestrationToolKind::Cancel, store.clone());
    let kill = OrchestrationTool::new(OrchestrationToolKind::Kill, store.clone());
    let yield_interrupt = OrchestrationTool::new(OrchestrationToolKind::YieldInterrupt, store);

    let timed = run(
        &timeout,
        json!({ "task_id": task_a.as_str(), "timeout_ms": 500 }),
    )
    .await
    .unwrap();
    assert_eq!(raw(&timed)["spec"]["timeout_ms"], 500);

    let records = run(
        &awaited,
        json!({
            "task_ids": [task_a.as_str(), task_b.as_str()],
            "timeout_ms": 50,
            "mode": "all"
        }),
    )
    .await
    .unwrap();
    assert_eq!(raw(&records).as_array().unwrap().len(), 2);

    let cancelled = run(&cancel, json!({ "task_id": task_b.as_str() }))
        .await
        .unwrap();
    assert_eq!(raw(&cancelled)["status"], "cancel_requested");
    assert_eq!(raw(&cancelled)["message"], "cancellation requested");

    let killed = run(&kill, json!({ "task_id": task_c.as_str() }))
        .await
        .unwrap();
    assert_eq!(raw(&killed)["status"], "abandoned");

    let yielded = run(
        &yield_interrupt,
        json!({
            "message": "need human input",
            "resume_schema": { "type": "object" }
        }),
    )
    .await
    .unwrap();
    assert_eq!(raw(&yielded)["status"], "interrupt_requested");
    assert_eq!(raw(&yielded)["message"], "need human input");
}

#[tokio::test]
async fn race_tool_reports_completed_winner_and_cancels_live_losers() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let winner = TaskId::new("winner");
    let loser = TaskId::new("loser");
    let terminal_loser = TaskId::new("terminal-loser");
    store.insert(graph_spec(winner.as_str())).unwrap();
    store.insert(graph_spec(loser.as_str())).unwrap();
    store.insert(graph_spec(terminal_loser.as_str())).unwrap();
    store.mark_running(&winner).unwrap();
    store.mark_running(&loser).unwrap();
    store.mark_running(&terminal_loser).unwrap();
    store
        .complete(&winner, OrchestrationTaskResult::text("done"))
        .unwrap();
    store
        .fail(&terminal_loser, "already failed".to_string())
        .unwrap();

    let race = OrchestrationTool::new(OrchestrationToolKind::Race, store.clone());
    let result = run(
        &race,
        json!({
            "task_ids": [loser.as_str(), winner.as_str(), terminal_loser.as_str()],
            "cancel_losers": true
        }),
    )
    .await
    .unwrap();
    assert_eq!(raw(&result)["winner"]["spec"]["task_id"], winner.as_str());
    assert_eq!(
        store.get(&loser).unwrap().status,
        OrchestrationTaskStatus::CancelRequested
    );
    assert_eq!(
        store.get(&terminal_loser).unwrap().status,
        OrchestrationTaskStatus::Failed
    );
}

#[tokio::test]
async fn orchestration_tool_validation_rejects_bad_model_arguments() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let spawn = OrchestrationTool::new(OrchestrationToolKind::Spawn, store.clone());
    let awaited = OrchestrationTool::new(OrchestrationToolKind::Await, store.clone());
    let timeout = OrchestrationTool::new(OrchestrationToolKind::Timeout, store);

    let err = run(&spawn, json!({ "kind": "unknown", "target": "x" }))
        .await
        .expect_err("schema rejects unsupported task kind enum");
    assert!(err.to_string().contains("kind"));

    let err = run(&awaited, json!({ "task_ids": [] }))
        .await
        .expect_err("empty task list is rejected");
    assert!(err.to_string().contains("at least one task id"));

    let err = run(
        &timeout,
        json!({ "task_id": "task-a", "timeout_ms": "soon" }),
    )
    .await
    .expect_err("schema rejects wrong timeout type");
    assert!(err.to_string().contains("timeout_ms"));
}

#[tokio::test]
async fn steer_tool_delivers_command_through_steering_registry() {
    use tinyagents_harness::steering::{SteeringCommand, SteeringHandle};

    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();

    // Spawn a task and mark it running; register a steering handle for it.
    let task_id = TaskId::new("child-1");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(task_id.clone(), handle.clone());

    let steer = OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone())
        .with_steering(steering.clone());

    let result = run(
        &steer,
        json!({ "task_id": task_id.as_str(), "command": "pause" }),
    )
    .await
    .unwrap();

    // The command was accepted and actually delivered to the live handle.
    assert_eq!(raw(&result)["accepted"], true);
    let drained = handle.drain();
    assert_eq!(drained.len(), 1);
    assert!(matches!(drained[0], SteeringCommand::Pause));
}

#[tokio::test]
async fn steer_tool_reports_not_delivered_without_registered_handle() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let task_id = TaskId::new("child-2");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();

    // No steering registry attached -> recorded but not delivered.
    let steer = OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone());
    let result = run(
        &steer,
        json!({ "task_id": task_id.as_str(), "command": "pause" }),
    )
    .await
    .unwrap();
    assert_eq!(raw(&result)["accepted"], false);
}

#[tokio::test]
async fn steer_tool_delivers_inject_message_and_metadata_payloads() {
    use tinyagents_harness::steering::{SteeringCommand, SteeringHandle};

    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();
    let task_id = TaskId::new("child-payloads");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(task_id.clone(), handle.clone());
    let steer =
        OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone()).with_steering(steering);

    run(
        &steer,
        json!({
            "task_id": task_id.as_str(),
            "command": "inject_message",
            "payload": { "content": "new user hint" }
        }),
    )
    .await
    .unwrap();
    run(
        &steer,
        json!({
            "task_id": task_id.as_str(),
            "command": "set_metadata",
            "payload": { "priority": "high" }
        }),
    )
    .await
    .unwrap();

    let drained = handle.drain();
    assert!(matches!(
        &drained[0],
        SteeringCommand::InjectMessage(message) if message.text() == "new user hint"
    ));
    assert!(matches!(
        &drained[1],
        SteeringCommand::SetMetadata { metadata } if metadata["priority"] == "high"
    ));
}

#[tokio::test]
async fn steer_tool_accepts_terminal_task_but_does_not_deliver() {
    use tinyagents_harness::steering::SteeringHandle;

    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();
    let task_id = TaskId::new("child-done");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store
        .complete(&task_id, OrchestrationTaskResult::text("done"))
        .unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(task_id.clone(), handle.clone());

    let steer =
        OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone()).with_steering(steering);
    let result = run(
        &steer,
        json!({ "task_id": task_id.as_str(), "command": "cancel" }),
    )
    .await
    .unwrap();

    assert_eq!(raw(&result)["accepted"], false);
    assert!(handle.drain().is_empty());
}

#[tokio::test]
async fn steer_tool_delivers_redirect_via_payload() {
    use tinyagents_harness::steering::{SteeringCommand, SteeringHandle};

    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();
    let task_id = TaskId::new("child-r");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(task_id.clone(), handle.clone());

    let steer = OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone())
        .with_steering(steering.clone());

    // redirect carries its instruction in the schema-allowed `payload` field.
    let result = run(
        &steer,
        json!({
            "task_id": task_id.as_str(),
            "command": "redirect",
            "payload": "go north"
        }),
    )
    .await
    .unwrap();
    assert_eq!(raw(&result)["accepted"], true);
    let drained = handle.drain();
    assert!(matches!(
        &drained[0],
        SteeringCommand::Redirect { instruction } if instruction == "go north"
    ));
}

#[tokio::test]
async fn steer_tool_redirect_without_payload_is_rejected() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();
    let task_id = TaskId::new("child-r2");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    steering.register(
        task_id.clone(),
        tinyagents_harness::steering::SteeringHandle::allow_all(),
    );

    let steer =
        OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone()).with_steering(steering);
    let err = run(
        &steer,
        json!({ "task_id": task_id.as_str(), "command": "redirect" }),
    )
    .await
    .expect_err("redirect without payload is rejected");
    assert!(matches!(
        err.downcast_ref::<tinyagents_harness::error::TinyAgentsError>(),
        Some(tinyagents_harness::error::TinyAgentsError::Validation(_))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn jsonl_store_persists_inside_multi_thread_runtime() {
    // Exercises the `block_in_place` path: persist runs on a tokio worker.
    let path = unique_log_path("multi-thread-runtime");
    let _ = std::fs::remove_file(&path);
    let task_id = TaskId::new("task-mt");

    let store = JsonlTaskStore::open(&path).unwrap();
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    store
        .complete(&task_id, OrchestrationTaskResult::text("done"))
        .unwrap();

    assert_eq!(store.history(&task_id).len(), 3);
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn jsonl_store_persists_inside_current_thread_runtime() {
    // A current-thread runtime must not panic (block_in_place is skipped).
    let path = unique_log_path("current-thread-runtime");
    let _ = std::fs::remove_file(&path);
    let task_id = TaskId::new("task-ct");

    let store = JsonlTaskStore::open(&path).unwrap();
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();

    assert_eq!(store.history(&task_id).len(), 2);
    let _ = std::fs::remove_file(&path);
}

// --- TaskStoreRegistry ---------------------------------------------------

#[test]
fn registry_opens_each_key_once_and_shares_the_same_store() {
    let opens = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&opens);
    let registry: TaskStoreRegistry<String> = TaskStoreRegistry::new(move |_key| {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Arc::new(InMemoryTaskStore::new()) as Arc<dyn TaskStore>
    });

    let first = registry.get_or_open(&"a".to_string()).expect("opens");
    let second = registry.get_or_open(&"a".to_string()).expect("reuses");

    assert!(
        Arc::ptr_eq(&first, &second),
        "same key must reuse one store"
    );
    assert_eq!(opens.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(registry.len().expect("len"), 1);
}

#[test]
fn registry_keeps_distinct_keys_isolated() {
    let registry: TaskStoreRegistry<String> =
        TaskStoreRegistry::new(|_key| Arc::new(InMemoryTaskStore::new()) as Arc<dyn TaskStore>);

    let a = registry.get_or_open(&"a".to_string()).expect("opens a");
    let b = registry.get_or_open(&"b".to_string()).expect("opens b");
    assert!(!Arc::ptr_eq(&a, &b), "distinct keys must not share a store");

    a.insert(graph_spec("only-in-a")).expect("insert");
    assert_eq!(a.list(OrchestrationTaskFilter::default()).len(), 1);
    assert!(
        b.list(OrchestrationTaskFilter::default()).is_empty(),
        "records must not leak across scopes"
    );
    assert_eq!(registry.len().expect("len"), 2);
}

#[test]
fn registry_get_does_not_open_and_clear_forces_a_reopen() {
    let registry: TaskStoreRegistry<String> =
        TaskStoreRegistry::new(|_key| Arc::new(InMemoryTaskStore::new()) as Arc<dyn TaskStore>);

    assert!(registry.is_empty().expect("is_empty"));
    assert!(registry.get(&"a".to_string()).expect("get").is_none());
    assert!(registry.is_empty().expect("still empty"));

    let first = registry.get_or_open(&"a".to_string()).expect("opens");
    assert!(registry.get(&"a".to_string()).expect("get").is_some());

    registry.clear().expect("clear");
    assert!(registry.is_empty().expect("cleared"));

    let reopened = registry.get_or_open(&"a".to_string()).expect("reopens");
    assert!(!Arc::ptr_eq(&first, &reopened), "clear must force a reopen");
}

#[test]
fn registry_values_returns_every_open_store() {
    let registry: TaskStoreRegistry<String> =
        TaskStoreRegistry::new(|_key| Arc::new(InMemoryTaskStore::new()) as Arc<dyn TaskStore>);
    assert!(registry.values().expect("values").is_empty());

    registry.get_or_open(&"a".to_string()).expect("opens a");
    registry.get_or_open(&"b".to_string()).expect("opens b");

    let stores = registry.values().expect("values");
    assert_eq!(stores.len(), 2);

    // A record written through one store is visible through exactly one of the
    // returned handles, so callers can sweep across scopes.
    registry
        .get_or_open(&"a".to_string())
        .expect("reuses a")
        .insert(graph_spec("only-in-a"))
        .expect("insert");
    let total: usize = stores
        .iter()
        .map(|store| store.list(OrchestrationTaskFilter::default()).len())
        .sum();
    assert_eq!(total, 1);
}

#[test]
fn registry_debug_reports_open_store_count() {
    let registry: TaskStoreRegistry<String> =
        TaskStoreRegistry::new(|_key| Arc::new(InMemoryTaskStore::new()) as Arc<dyn TaskStore>);
    registry.get_or_open(&"a".to_string()).expect("opens");
    assert!(format!("{registry:?}").contains("TaskStoreRegistry"));
}

#[test]
fn registry_lock_error_displays_the_poison_detail() {
    let err = TaskStoreRegistryError::Lock("poisoned".to_string());
    assert!(err.to_string().contains("poisoned"));
}

#[test]
fn jsonl_fallback_opens_a_durable_store_and_replays_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("nested").join("tasks.jsonl");

    let store = open_jsonl_task_store_or_memory(&path);
    store.insert(graph_spec("t1")).expect("insert");
    assert!(path.exists(), "durable store must create its log");

    // Reopening replays the log rather than starting empty.
    let reopened = open_jsonl_task_store_or_memory(&path);
    assert_eq!(reopened.list(OrchestrationTaskFilter::default()).len(), 1);
}

#[test]
fn jsonl_fallback_degrades_to_memory_when_the_log_is_unreadable() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A directory where the log file should be: `JsonlTaskStore::open` cannot
    // use it, so the caller must still get a working store.
    let path = dir.path().join("tasks.jsonl");
    std::fs::create_dir(&path).expect("create dir in the log's place");

    let store = open_jsonl_task_store_or_memory(&path);
    store
        .insert(graph_spec("t1"))
        .expect("memory store still accepts work");
    assert_eq!(store.list(OrchestrationTaskFilter::default()).len(), 1);
}

// --- reconcile_orphaned_tasks -------------------------------------------

#[test]
fn reconcile_settles_live_tasks_and_leaves_terminal_ones_alone() {
    let store = InMemoryTaskStore::new();

    store.insert(graph_spec("pending")).expect("insert");

    store.insert(graph_spec("running")).expect("insert");
    store
        .mark_running(&TaskId::new("running"))
        .expect("running");

    store.insert(graph_spec("awaiting")).expect("insert");
    store
        .mark_running(&TaskId::new("awaiting"))
        .expect("running");
    store
        .mark_awaiting(&TaskId::new("awaiting"))
        .expect("awaiting");

    store.insert(graph_spec("done")).expect("insert");
    store.mark_running(&TaskId::new("done")).expect("running");
    store
        .complete(&TaskId::new("done"), OrchestrationTaskResult::text("ok"))
        .expect("complete");

    let report = reconcile_orphaned_tasks(&store, OrchestrationTaskFilter::default(), &|_record| {
        "driver died".to_string()
    });

    assert_eq!(
        report.reconciled_count(),
        3,
        "three live tasks were settled"
    );
    assert_eq!(report.error_count(), 0);
    assert!(
        report
            .tasks
            .iter()
            .all(|task| task.outcome == ReconcileOutcome::Failed),
        "live-but-not-cancelling tasks settle as failed"
    );

    for id in ["pending", "running", "awaiting"] {
        let record = store.get(&TaskId::new(id)).expect("record");
        assert_eq!(record.status, OrchestrationTaskStatus::Failed);
        assert_eq!(record.error.as_deref(), Some("driver died"));
    }
    // The already-terminal task was not touched.
    let done = store.get(&TaskId::new("done")).expect("record");
    assert_eq!(done.status, OrchestrationTaskStatus::Completed);
}

#[test]
fn reconcile_honours_a_pending_cancellation() {
    let store = InMemoryTaskStore::new();
    store.insert(graph_spec("cancelling")).expect("insert");
    store
        .mark_running(&TaskId::new("cancelling"))
        .expect("running");
    store
        .request_cancel(&TaskId::new("cancelling"))
        .expect("cancel");

    let report = reconcile_orphaned_tasks(&store, OrchestrationTaskFilter::default(), &|_| {
        "driver died".to_string()
    });

    assert_eq!(report.tasks.len(), 1);
    assert_eq!(report.tasks[0].outcome, ReconcileOutcome::Cancelled);
    assert_eq!(
        report.tasks[0].prior_status,
        OrchestrationTaskStatus::CancelRequested
    );
    assert_eq!(
        store
            .get(&TaskId::new("cancelling"))
            .expect("record")
            .status,
        OrchestrationTaskStatus::Cancelled
    );
}

#[test]
fn reconcile_reason_closure_sees_the_record_being_settled() {
    let store = InMemoryTaskStore::new();
    store.insert(graph_spec("t1")).expect("insert");
    store.mark_running(&TaskId::new("t1")).expect("running");

    reconcile_orphaned_tasks(&store, OrchestrationTaskFilter::default(), &|record| {
        format!("orphaned (was `{}`)", task_status_label(record.status))
    });

    assert_eq!(
        store
            .get(&TaskId::new("t1"))
            .expect("record")
            .error
            .as_deref(),
        Some("orphaned (was `running`)")
    );
}

#[test]
fn reconcile_respects_the_filter() {
    let store = InMemoryTaskStore::new();
    store
        .insert(OrchestrationTaskSpec::new(
            "agent-task",
            OrchestrationTaskKind::SubAgent {
                agent: "worker".into(),
            },
        ))
        .expect("insert");
    store.insert(graph_spec("graph-task")).expect("insert");

    let report = reconcile_orphaned_tasks(
        &store,
        OrchestrationTaskFilter::default().with_kind("sub_agent"),
        &|_| "driver died".to_string(),
    );

    assert_eq!(report.tasks.len(), 1);
    assert_eq!(report.tasks[0].task_id.as_str(), "agent-task");
    assert_eq!(
        store
            .get(&TaskId::new("graph-task"))
            .expect("record")
            .status,
        OrchestrationTaskStatus::Pending,
        "a filtered-out task must not be swept"
    );
}

#[test]
fn reconcile_on_an_empty_store_reports_nothing() {
    let store = InMemoryTaskStore::new();
    let report = reconcile_orphaned_tasks(&store, OrchestrationTaskFilter::default(), &|_| {
        "driver died".to_string()
    });
    assert!(report.is_empty());
    assert_eq!(report.reconciled_count(), 0);
    assert_eq!(report.error_count(), 0);
    assert_eq!(report.settled().count(), 0);
}

#[test]
fn reconcile_outcome_error_is_not_counted_as_settled() {
    let error = ReconcileOutcome::Error("boom".to_string());
    assert!(!error.is_settled());
    assert!(ReconcileOutcome::Failed.is_settled());
    assert!(ReconcileOutcome::Cancelled.is_settled());
}

#[test]
fn task_status_label_covers_every_status() {
    for (status, expected) in [
        (OrchestrationTaskStatus::Pending, "pending"),
        (OrchestrationTaskStatus::Running, "running"),
        (OrchestrationTaskStatus::Awaiting, "awaiting"),
        (OrchestrationTaskStatus::CancelRequested, "cancel_requested"),
        (OrchestrationTaskStatus::Completed, "completed"),
        (OrchestrationTaskStatus::Failed, "failed"),
        (OrchestrationTaskStatus::Cancelled, "cancelled"),
        (OrchestrationTaskStatus::TimedOut, "timed_out"),
        (OrchestrationTaskStatus::Abandoned, "abandoned"),
    ] {
        assert_eq!(task_status_label(status), expected);
    }
}
