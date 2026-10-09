use std::sync::Arc;
use std::time::Duration;

use tinyagents_harness::run_queue::{QueueLane, RunQueue, RunQueueHandle};
use tinyagents_tasks::{
    CompletionRouter, CompletionStatus, InMemoryCompletionStore, InMemoryTaskStore, NotifyMode,
    OrchestrationTaskStatus, TaskStore,
};
use tinyinference_llm::message::Message;
use tokio::sync::watch;

use super::*;

fn router() -> Arc<CompletionRouter> {
    Arc::new(CompletionRouter::new(Arc::new(
        InMemoryCompletionStore::new(),
    )))
}

fn target(router: &Arc<CompletionRouter>) -> DetachedCompletionTarget {
    DetachedCompletionTarget::new(router.clone(), "p", "worker").with_label("indexer")
}

async fn settle(router: &CompletionRouter, parent: &str) {
    for _ in 0..100 {
        if !router.pending_for(parent).is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn spawned_store(task: &str) -> Arc<dyn TaskStore> {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    record_spawned(
        store.as_ref(),
        &SpawnedSubagent {
            task_id: task,
            agent_id: "worker",
            parent_session: "p",
            session_parent_prefix: None,
            subagent_session_id: None,
            workspace_dir: "/w",
            parent_thread_id: None,
        },
    )
    .unwrap();
    store
}

#[test]
fn only_final_statuses_are_completions() {
    let router = router();
    let target = target(&router);
    assert!(
        target
            .record_for("t", &DetachedSubagentStatus::Running)
            .is_none()
    );
    assert!(
        target
            .record_for(
                "t",
                &DetachedSubagentStatus::AwaitingUser {
                    question: "which?".into()
                }
            )
            .is_none()
    );
    let done = target
        .record_for(
            "t",
            &DetachedSubagentStatus::Completed {
                output: "ok".into(),
                iterations: 2,
            },
        )
        .unwrap();
    assert_eq!(done.status, CompletionStatus::Success);
    assert_eq!(done.label.as_deref(), Some("indexer"));
    let failed = target
        .record_for(
            "t",
            &DetachedSubagentStatus::Failed {
                error: "bad".into(),
            },
        )
        .unwrap();
    assert_eq!(failed.status, CompletionStatus::Failed);
    assert_eq!(failed.result.text, "bad");
}

#[tokio::test]
async fn the_watcher_mirrors_the_status_and_records_the_completion() {
    let router = router();
    let store = spawned_store("t1");
    let (tx, rx) = watch::channel(DetachedSubagentStatus::Running);
    spawn_status_watcher_with_completions(store.clone(), "t1".into(), rx, target(&router));
    tx.send(DetachedSubagentStatus::Completed {
        output: "found it".into(),
        iterations: 1,
    })
    .unwrap();
    settle(&router, "p").await;
    let pending = router.pending_for("p");
    assert_eq!(pending[0].result.text, "found it");
    let record = store.get(&"t1".into()).unwrap();
    assert_eq!(record.status, OrchestrationTaskStatus::Completed);
}

#[tokio::test]
async fn a_dropped_sender_is_recorded_as_a_failure() {
    let router = router();
    let store = spawned_store("t1");
    let (tx, rx) = watch::channel(DetachedSubagentStatus::Running);
    spawn_status_watcher_with_completions(store, "t1".into(), rx, target(&router));
    drop(tx);
    settle(&router, "p").await;
    assert_eq!(router.pending_for("p")[0].status, CompletionStatus::Failed);
}

#[tokio::test]
async fn the_watcher_pushes_to_a_live_parent_by_notify_mode() {
    let router = router();
    let queue: RunQueueHandle = Arc::new(RunQueue::<Message>::new());
    router.attach_parent("p", queue.clone());
    let store = spawned_store("t1");
    let (tx, rx) = watch::channel(DetachedSubagentStatus::Running);
    spawn_status_watcher_with_completions(
        store,
        "t1".into(),
        rx,
        target(&router).with_notify_mode(NotifyMode::Collect),
    );
    tx.send(DetachedSubagentStatus::Completed {
        output: "x".into(),
        iterations: 1,
    })
    .unwrap();
    for _ in 0..100 {
        if queue.status().await.total > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(queue.drain(QueueLane::Collect).await.len(), 1);
}

#[tokio::test]
async fn a_paused_child_records_nothing() {
    let router = router();
    let store = spawned_store("t1");
    let (tx, rx) = watch::channel(DetachedSubagentStatus::Running);
    spawn_status_watcher_with_completions(store, "t1".into(), rx, target(&router));
    tx.send(DetachedSubagentStatus::AwaitingUser {
        question: "which?".into(),
    })
    .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(router.pending_for("p").is_empty());
}
