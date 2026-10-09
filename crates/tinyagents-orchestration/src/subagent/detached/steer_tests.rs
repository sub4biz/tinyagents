use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tinyagents_harness::CancellationToken;
use tinyagents_harness::ids::TaskId;
use tinyagents_harness::run_queue::QueueLane;
use tinyagents_harness::steering::SteeringHandle;
use tinyagents_tasks::{DetachedTaskRegistry, SteeringRegistry};
use tokio::sync::watch;

use super::*;

#[derive(Clone)]
struct Meta {
    thread: Option<&'static str>,
}

impl SubagentIdentity for Meta {
    fn agent_id(&self) -> &str {
        "researcher"
    }
    fn subagent_session_id(&self) -> Option<&str> {
        None
    }
    fn parent_thread_id(&self) -> Option<&str> {
        self.thread
    }
}

type Registry = DetachedTaskRegistry<Meta, DetachedSubagentStatus>;

fn registry(steering: SteeringRegistry) -> Registry {
    DetachedTaskRegistry::new(steering, 256, DetachedSubagentStatus::is_terminal)
}

fn add(
    reg: &Registry,
    task: &str,
    owner: &str,
    thread: Option<&'static str>,
) -> watch::Sender<DetachedSubagentStatus> {
    let (tx, rx) = watch::channel(DetachedSubagentStatus::Running);
    reg.register(
        TaskId::new(task),
        owner,
        Meta { thread },
        rx,
        CancellationToken::new(),
        tokio::spawn(async {}).abort_handle(),
    )
    .unwrap();
    tx
}

fn counting() -> (
    Arc<AtomicUsize>,
    impl FnOnce(Meta, QueueLane, String) -> std::future::Ready<()>,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    (calls, move |_, _, _| {
        seen.fetch_add(1, Ordering::SeqCst);
        std::future::ready(())
    })
}

#[test]
fn lane_names_and_commands_are_literal() {
    assert_eq!(queue_lane_name(QueueLane::Steer), "steer");
    assert_eq!(queue_lane_name(QueueLane::Followup), "followup");
    assert_eq!(queue_lane_name(QueueLane::Collect), "collect");
    assert!(steering_command_for_lane(QueueLane::Steer, "x").is_some());
    assert!(steering_command_for_lane(QueueLane::Collect, "x").is_some());
    assert!(steering_command_for_lane(QueueLane::Followup, "x").is_none());
}

#[tokio::test]
async fn steer_rejects_followup_unknown_unowned_and_terminal() {
    let reg = registry(SteeringRegistry::default());
    let tx = add(&reg, "t1", "p1", None);

    let (calls, fb) = counting();
    assert_eq!(
        steer_detached(
            &reg,
            "t1",
            SteerAccess::Owner("p1"),
            "x".into(),
            QueueLane::Followup,
            fb
        )
        .await,
        Err(SteerError::UnsupportedLane)
    );
    let (_, fb) = counting();
    assert_eq!(
        steer_detached(
            &reg,
            "nope",
            SteerAccess::Trusted,
            "x".into(),
            QueueLane::Steer,
            fb
        )
        .await,
        Err(SteerError::Unknown)
    );
    let (_, fb) = counting();
    assert_eq!(
        steer_detached(
            &reg,
            "t1",
            SteerAccess::Owner("other"),
            "x".into(),
            QueueLane::Steer,
            fb
        )
        .await,
        Err(SteerError::NotOwned)
    );
    tx.send(DetachedSubagentStatus::Failed { error: "e".into() })
        .unwrap();
    let (_, fb) = counting();
    assert_eq!(
        steer_detached(
            &reg,
            "t1",
            SteerAccess::Trusted,
            "x".into(),
            QueueLane::Steer,
            fb
        )
        .await,
        Err(SteerError::AlreadyDone)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn steer_prefers_the_live_handle_then_falls_back() {
    let steering = SteeringRegistry::default();
    let reg = registry(steering.clone());
    let _tx = add(&reg, "t1", "p1", None);

    let (calls, fb) = counting();
    assert_eq!(
        steer_detached(
            &reg,
            "t1",
            SteerAccess::Owner("p1"),
            "go".into(),
            QueueLane::Steer,
            fb
        )
        .await,
        Ok(SteerRoute::Fallback)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    steering.register(TaskId::new("t1"), SteeringHandle::allow_all());
    let (calls, fb) = counting();
    assert_eq!(
        steer_detached(
            &reg,
            "t1",
            SteerAccess::Trusted,
            "go".into(),
            QueueLane::Collect,
            fb
        )
        .await,
        Ok(SteerRoute::Registry)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancel_for_thread_cancels_only_that_threads_subagents() {
    let reg = registry(SteeringRegistry::default());
    let _a = add(&reg, "a", "p1", Some("thread-1"));
    let _b = add(&reg, "b", "p1", Some("thread-1"));
    let _c = add(&reg, "c", "p1", Some("thread-2"));
    let _d = add(&reg, "d", "p1", None);

    let cancelled = cancel_for_thread(&reg, "thread-1").unwrap();
    let mut ids: Vec<_> = cancelled
        .iter()
        .map(|c| c.task_id.as_str().to_string())
        .collect();
    ids.sort();
    assert_eq!(ids, ["a", "b"]);
    assert_eq!(reg.len().unwrap(), 2);

    let rest = reg.cancel_all().unwrap();
    assert_eq!(distinct_parent_threads(&rest), ["thread-2"]);
}

#[tokio::test]
async fn steer_with_a_repeated_request_id_is_delivered_once() {
    let reg = registry(SteeringRegistry::default());
    let _tx = add(&reg, "t1", "p1", None);
    let (calls, fb) = counting();
    let first = steer_detached_with_request_id(
        &reg,
        "t1",
        SteerAccess::Owner("p1"),
        "go".into(),
        QueueLane::Steer,
        Some("req-1"),
        fb,
    )
    .await
    .unwrap();
    assert_eq!(first.route, Some(SteerRoute::Fallback));
    assert!(!first.duplicate);

    let (more_calls, fb) = counting();
    let second = steer_detached_with_request_id(
        &reg,
        "t1",
        SteerAccess::Owner("p1"),
        "go".into(),
        QueueLane::Steer,
        Some("req-1"),
        fb,
    )
    .await
    .expect("a duplicate is a success");
    assert!(second.duplicate);
    assert_eq!(second.route, None);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(more_calls.load(Ordering::SeqCst), 0, "not enqueued again");

    // A different id, and a call with no id, are always delivered.
    for request_id in [Some("req-2"), None, None] {
        let (calls, fb) = counting();
        let receipt = steer_detached_with_request_id(
            &reg,
            "t1",
            SteerAccess::Trusted,
            "go".into(),
            QueueLane::Steer,
            request_id,
            fb,
        )
        .await
        .unwrap();
        assert!(!receipt.duplicate);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn rejected_steer_does_not_consume_its_request_id() {
    let reg = registry(SteeringRegistry::default());
    let _tx = add(&reg, "t1", "p1", None);
    let (_, fb) = counting();
    assert_eq!(
        steer_detached_with_request_id(
            &reg,
            "t1",
            SteerAccess::Owner("intruder"),
            "go".into(),
            QueueLane::Steer,
            Some("req-1"),
            fb,
        )
        .await,
        Err(SteerError::NotOwned)
    );
    let (calls, fb) = counting();
    let receipt = steer_detached_with_request_id(
        &reg,
        "t1",
        SteerAccess::Owner("p1"),
        "go".into(),
        QueueLane::Steer,
        Some("req-1"),
        fb,
    )
    .await
    .unwrap();
    assert!(!receipt.duplicate);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn oversized_request_ids_are_rejected_before_delivery() {
    let reg = registry(SteeringRegistry::default());
    let _tx = add(&reg, "t1", "p1", None);
    let (calls, fb) = counting();
    let too_long = "x".repeat(129);
    assert_eq!(
        steer_detached_with_request_id(
            &reg,
            "t1",
            SteerAccess::Owner("p1"),
            "go".into(),
            QueueLane::Steer,
            Some(&too_long),
            fb,
        )
        .await,
        Err(SteerError::RequestIdTooLong)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
