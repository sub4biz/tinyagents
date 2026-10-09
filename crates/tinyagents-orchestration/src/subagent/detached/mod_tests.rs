use std::sync::Arc;
use std::time::Duration;

use tinyagents_harness::CancellationToken;
use tinyagents_harness::ids::TaskId;
use tinyagents_tasks::{
    DetachedTaskRegistry, InMemoryTaskStore, OrchestrationControlOutcome, OrchestrationTaskFilter,
    OrchestrationTaskKind, OrchestrationTaskRecord, OrchestrationTaskResult, OrchestrationTaskSpec,
    OrchestrationTaskStatus, SteeringRegistry, TaskStore,
};
use tokio::sync::watch;

use super::*;

#[derive(Clone)]
struct Meta {
    agent: &'static str,
    session: Option<&'static str>,
}

impl SubagentIdentity for Meta {
    fn agent_id(&self) -> &str {
        self.agent
    }
    fn subagent_session_id(&self) -> Option<&str> {
        self.session
    }
}

type Registry = DetachedTaskRegistry<Meta, DetachedSubagentStatus>;

fn registry() -> Registry {
    DetachedTaskRegistry::new(
        SteeringRegistry::default(),
        256,
        DetachedSubagentStatus::is_terminal,
    )
}

fn add(
    reg: &Registry,
    task: &str,
    owner: &str,
    meta: Meta,
) -> watch::Sender<DetachedSubagentStatus> {
    let (tx, rx) = watch::channel(DetachedSubagentStatus::Running);
    reg.register(
        TaskId::new(task),
        owner,
        meta,
        rx,
        CancellationToken::new(),
        tokio::spawn(async {}).abort_handle(),
    )
    .unwrap();
    tx
}

fn spawned<'a>(task: &'a str) -> SpawnedSubagent<'a> {
    SpawnedSubagent {
        task_id: task,
        agent_id: "researcher",
        parent_session: "p1",
        session_parent_prefix: Some("rootrun__child"),
        subagent_session_id: Some("sub-1"),
        workspace_dir: "/ws",
        parent_thread_id: Some("thread-7"),
    }
}

#[test]
fn status_labels_and_outcomes_are_literal() {
    use DetachedSubagentStatus as S;
    let cases = [
        (S::Running, "running", false, None),
        (
            S::Completed {
                output: "o".into(),
                iterations: 1,
            },
            "completed",
            true,
            Some("completed"),
        ),
        (
            S::AwaitingUser {
                question: "q".into(),
            },
            "awaiting_user",
            true,
            None,
        ),
        (
            S::Failed { error: "e".into() },
            "failed",
            true,
            Some("failed"),
        ),
    ];
    for (status, label, terminal, finished) in cases {
        assert_eq!(status.label(), label);
        assert_eq!(status.is_terminal(), terminal);
        assert_eq!(
            status.finished_outcome().map(FinishedOutcome::as_str),
            finished
        );
    }
    assert_eq!(
        S::ended_without_result(),
        S::Failed {
            error: "sub-agent task ended without reporting a result".into()
        }
    );
}

#[test]
fn ledger_record_has_literal_metadata_and_status_mirroring() {
    let store = InMemoryTaskStore::new();
    record_spawned(&store, &spawned("t1")).unwrap();
    let rec = store.get(&TaskId::new("t1")).unwrap();
    assert_eq!(rec.status, OrchestrationTaskStatus::Running);
    assert_eq!(rec.spec.timeout_ms, Some(120_000));
    let meta: Vec<(&str, &str)> = {
        let mut m: Vec<_> = rec
            .spec
            .metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        m.sort();
        m
    };
    assert_eq!(
        meta,
        vec![
            ("defaultWaitTimeoutMs", "120000"),
            ("parentSession", "p1"),
            ("parentThreadId", "thread-7"),
            ("rootSession", "rootrun"),
            ("sessionParentPrefix", "rootrun__child"),
            ("subagentSessionId", "sub-1"),
            ("workspaceDir", "/ws"),
        ]
    );
    assert_eq!(record_parent_session(&rec), Some("p1"));
    assert_eq!(record_subagent_session_id(&rec), Some("sub-1"));
    assert_eq!(record_agent_id(&rec), "researcher");

    record_status(
        &store,
        "t1",
        &DetachedSubagentStatus::Completed {
            output: "done".into(),
            iterations: 3,
        },
    )
    .unwrap();
    // First writer wins: a later failure does not rewrite the terminal state.
    record_status(
        &store,
        "t1",
        &DetachedSubagentStatus::Failed { error: "x".into() },
    )
    .unwrap();
    let rec = store.get(&TaskId::new("t1")).unwrap();
    assert_eq!(rec.status, OrchestrationTaskStatus::Completed);

    record_spawned(&store, &spawned("t2")).unwrap();
    record_cancelled(&store, "t2").unwrap();
    assert_eq!(
        store.get(&TaskId::new("t2")).unwrap().status,
        OrchestrationTaskStatus::Cancelled
    );
    assert_eq!(list_subagent_records(&store).len(), 2);
}

#[test]
fn root_run_falls_back_to_parent_session() {
    let store = InMemoryTaskStore::new();
    let mut s = spawned("t1");
    s.session_parent_prefix = None;
    record_spawned(&store, &s).unwrap();
    let rec = store.get(&TaskId::new("t1")).unwrap();
    assert_eq!(rec.spec.metadata.get("rootSession").unwrap(), "p1");
    assert!(!rec.spec.metadata.contains_key("sessionParentPrefix"));
}

#[test]
fn durable_records_map_to_literal_wait_outcomes() {
    let store = InMemoryTaskStore::new();
    let mk = |id: &str| {
        store
            .insert(
                OrchestrationTaskSpec::new(
                    id.to_string(),
                    OrchestrationTaskKind::SubAgent { agent: "a".into() },
                )
                .with_metadata("parentSession", "p1".to_string()),
            )
            .unwrap();
        store.mark_running(&TaskId::new(id)).unwrap();
        TaskId::new(id)
    };
    let id = mk("completed");
    store
        .complete(&id, OrchestrationTaskResult::text("final".to_string()))
        .unwrap();
    let id = mk("failed");
    store.fail(&id, "boom".to_string()).unwrap();
    let id = mk("awaiting");
    store.mark_awaiting(&id).unwrap();
    let id = mk("cancelled");
    store.request_cancel(&id).unwrap();
    store.mark_cancelled(&id).unwrap();
    mk("running");

    let mut seen = Vec::new();
    for id in ["completed", "failed", "awaiting", "cancelled", "running"] {
        let rec = subagent_record_for_task(&store, id, "p1").unwrap();
        seen.push(format!("{id}: {:?}", record_to_wait_outcome(rec)));
    }
    assert_eq!(
        seen,
        vec![
            "completed: Terminal(Completed { output: \"final\", iterations: 0 })",
            "failed: Terminal(Failed { error: \"boom\" })",
            "awaiting: Terminal(AwaitingUser { question: \"sub-agent is awaiting user input; no clarification text was available from the durable task store\" })",
            "cancelled: Terminal(Failed { error: \"sub-agent was cancelled\" })",
            "running: TimedOut(Running)",
        ]
    );
    assert_eq!(
        subagent_record_for_task(&store, "failed", "other").unwrap_err(),
        WaitError::NotOwned
    );
    assert_eq!(
        subagent_record_for_task(&store, "nope", "p1").unwrap_err(),
        WaitError::Unknown
    );
}

#[test]
fn status_labels_for_durable_statuses() {
    use OrchestrationTaskStatus as S;
    let all = [
        (S::Pending, "pending"),
        (S::Running, "running"),
        (S::Awaiting, "awaiting"),
        (S::Completed, "completed"),
        (S::Failed, "failed"),
        (S::CancelRequested, "cancel_requested"),
        (S::Cancelled, "cancelled"),
        (S::TimedOut, "timed_out"),
        (S::Abandoned, "abandoned"),
    ];
    for (status, label) in all {
        assert_eq!(task_status_label(status), label);
    }
    assert_eq!(
        orphaned_subagent_reason(S::Running),
        "sub-agent orphaned by core restart (was `running`)"
    );
}

#[tokio::test]
async fn watcher_mirrors_terminal_and_dropped_sender() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    record_spawned(store.as_ref(), &spawned("w1")).unwrap();
    record_spawned(store.as_ref(), &spawned("w2")).unwrap();
    let (tx1, rx1) = watch::channel(DetachedSubagentStatus::Running);
    let (tx2, rx2) = watch::channel(DetachedSubagentStatus::Running);
    spawn_status_watcher(store.clone(), "w1".into(), rx1);
    spawn_status_watcher(store.clone(), "w2".into(), rx2);
    tx1.send(DetachedSubagentStatus::AwaitingUser {
        question: "q".into(),
    })
    .unwrap();
    drop(tx2);
    for _ in 0..100 {
        tokio::task::yield_now().await;
        let a = store.get(&TaskId::new("w1")).unwrap().status;
        let b = store.get(&TaskId::new("w2")).unwrap().status;
        if a == OrchestrationTaskStatus::Awaiting && b == OrchestrationTaskStatus::Failed {
            break;
        }
    }
    assert_eq!(
        store.get(&TaskId::new("w1")).unwrap().status,
        OrchestrationTaskStatus::Awaiting
    );
    let failed = store.get(&TaskId::new("w2")).unwrap();
    assert_eq!(failed.status, OrchestrationTaskStatus::Failed);
    assert_eq!(
        failed.error.as_deref(),
        Some("sub-agent task ended without reporting a result")
    );
}

#[tokio::test]
async fn wait_returns_terminal_times_out_and_maps_closed_channel() {
    let reg = registry();
    let tx = add(
        &reg,
        "a",
        "p1",
        Meta {
            agent: "r",
            session: None,
        },
    );
    let r = wait_detached(&reg, "a", "p1", Duration::from_millis(10))
        .await
        .unwrap();
    assert!(matches!(
        r,
        WaitOutcome::TimedOut(DetachedSubagentStatus::Running)
    ));
    tx.send(DetachedSubagentStatus::Completed {
        output: "ok".into(),
        iterations: 2,
    })
    .unwrap();
    let r = wait_detached(&reg, "a", "p1", Duration::from_secs(1))
        .await
        .unwrap();
    assert!(matches!(
        r,
        WaitOutcome::Terminal(DetachedSubagentStatus::Completed { iterations: 2, .. })
    ));

    let tx = add(
        &reg,
        "b",
        "p1",
        Meta {
            agent: "r",
            session: None,
        },
    );
    drop(tx);
    let r = wait_detached(&reg, "b", "p1", Duration::from_secs(1))
        .await
        .unwrap();
    assert!(matches!(
        r,
        WaitOutcome::Terminal(DetachedSubagentStatus::Failed { ref error })
            if error == "sub-agent task ended without reporting a result"
    ));

    let _tx = add(
        &reg,
        "c",
        "p1",
        Meta {
            agent: "r",
            session: None,
        },
    );
    assert_eq!(
        wait_detached(&reg, "c", "other", Duration::from_millis(5))
            .await
            .unwrap_err(),
        WaitError::NotOwned
    );
    assert_eq!(
        wait_detached(&reg, "zzz", "p1", Duration::from_millis(5))
            .await
            .unwrap_err(),
        WaitError::Unknown
    );
}

#[tokio::test]
async fn roster_resolution_and_resume_refs() {
    let reg = registry();
    let _a = add(
        &reg,
        "t-a",
        "p1",
        Meta {
            agent: "researcher",
            session: Some("s-a"),
        },
    );
    let tx_b = add(
        &reg,
        "t-b",
        "p1",
        Meta {
            agent: "code_executor",
            session: Some("s-b"),
        },
    );
    let _o = add(
        &reg,
        "t-o",
        "p2",
        Meta {
            agent: "researcher",
            session: Some("s-o"),
        },
    );
    tx_b.send(DetachedSubagentStatus::AwaitingUser {
        question: "which?".into(),
    })
    .unwrap();

    let snap = snapshot_for_owner(&reg, "p1").unwrap();
    assert_eq!(
        snap,
        vec![
            SubagentSnapshot {
                agent_id: "code_executor".into(),
                subagent_session_id: Some("s-b".into()),
                task_id: "t-b".into(),
                status: "awaiting_user",
            },
            SubagentSnapshot {
                agent_id: "researcher".into(),
                subagent_session_id: Some("s-a".into()),
                task_id: "t-a".into(),
                status: "running",
            },
        ]
    );

    assert_eq!(task_id_for_session(&reg, "s-a", "p1").unwrap(), "t-a");
    assert_eq!(
        task_id_for_session(&reg, "s-o", "p1").unwrap_err(),
        WaitError::NotOwned
    );
    assert_eq!(
        task_id_for_session(&reg, "none", "p1").unwrap_err(),
        WaitError::Unknown
    );
    let r = resume_ref_for_task(&reg, "t-b", "p1").unwrap();
    assert_eq!(
        r,
        SubagentResumeRef {
            task_id: "t-b".into(),
            agent_id: "code_executor".into(),
            subagent_session_id: Some("s-b".into()),
        }
    );
    assert_eq!(
        resume_ref_for_task(&reg, "t-b", "p2").unwrap_err(),
        WaitError::NotOwned
    );
}

#[tokio::test]
async fn live_task_preferred_over_terminal_for_same_session() {
    let reg = registry();
    let tx_old = add(
        &reg,
        "t-old",
        "p1",
        Meta {
            agent: "r",
            session: Some("s"),
        },
    );
    tx_old
        .send(DetachedSubagentStatus::Completed {
            output: "x".into(),
            iterations: 1,
        })
        .unwrap();
    let _new = add(
        &reg,
        "t-new",
        "p1",
        Meta {
            agent: "r",
            session: Some("s"),
        },
    );
    assert_eq!(task_id_for_session(&reg, "s", "p1").unwrap(), "t-new");
}

#[test]
fn session_resolution_from_durable_records() {
    let store = InMemoryTaskStore::new();
    record_spawned(&store, &spawned("d1")).unwrap();
    let records = list_subagent_records(&store);
    assert_eq!(
        task_id_for_session_in_records(records.clone(), "sub-1", "p1").unwrap(),
        "d1"
    );
    assert_eq!(
        task_id_for_session_in_records(records.clone(), "sub-1", "px").unwrap_err(),
        WaitError::NotOwned
    );
    assert_eq!(
        task_id_for_session_in_records(records.clone(), "zz", "p1").unwrap_err(),
        WaitError::Unknown
    );
    let r = resume_ref_from_record("d1", &records[0]);
    assert_eq!(r.agent_id, "researcher");
    assert_eq!(r.subagent_session_id.as_deref(), Some("sub-1"));
}

#[test]
fn awaiting_question_is_persisted_and_read_back() {
    let store = InMemoryTaskStore::new();
    record_spawned(&store, &spawned("q1")).unwrap();
    record_status(
        &store,
        "q1",
        &DetachedSubagentStatus::AwaitingUser {
            question: "which branch?".into(),
        },
    )
    .unwrap();
    let rec = subagent_record_for_task(&store, "q1", "p1").unwrap();
    assert_eq!(rec.status, OrchestrationTaskStatus::Awaiting);
    assert_eq!(
        format!("{:?}", record_to_wait_outcome(rec)),
        "Terminal(AwaitingUser { question: \"which branch?\" })"
    );
}

#[test]
fn reused_task_id_surfaces_the_insert_failure_and_leaves_the_record() {
    let store = InMemoryTaskStore::new();
    record_spawned(&store, &spawned("dup")).unwrap();
    record_status(
        &store,
        "dup",
        &DetachedSubagentStatus::Completed {
            output: "old".into(),
            iterations: 1,
        },
    )
    .unwrap();
    assert!(record_spawned(&store, &spawned("dup")).is_err());
    let rec = store.get(&TaskId::new("dup")).unwrap();
    assert_eq!(rec.status, OrchestrationTaskStatus::Completed);
}

#[test]
fn registry_errors_map_onto_wait_errors() {
    use tinyagents_tasks::DetachedTaskRegistryError as E;
    assert_eq!(WaitError::from(E::NotOwned), WaitError::NotOwned);
    assert_eq!(
        WaitError::from(E::LockPoisoned),
        WaitError::RegistryPoisoned
    );
    assert_eq!(WaitError::from(E::AlreadyDone), WaitError::Unknown);
}

struct FailingStore(InMemoryTaskStore);

impl TaskStore for FailingStore {
    fn insert(
        &self,
        spec: OrchestrationTaskSpec,
    ) -> tinyagents_harness::Result<OrchestrationTaskRecord> {
        self.0.insert(spec)
    }
    fn get(&self, id: &TaskId) -> Option<OrchestrationTaskRecord> {
        self.0.get(id)
    }
    fn list(&self, f: OrchestrationTaskFilter) -> Vec<OrchestrationTaskRecord> {
        self.0.list(f)
    }
    fn mark_running(&self, id: &TaskId) -> tinyagents_harness::Result<OrchestrationTaskRecord> {
        self.0.mark_running(id)
    }
    fn mark_awaiting(&self, id: &TaskId) -> tinyagents_harness::Result<OrchestrationTaskRecord> {
        self.0.mark_awaiting(id)
    }
    fn complete(
        &self,
        _id: &TaskId,
        _result: OrchestrationTaskResult,
    ) -> tinyagents_harness::Result<OrchestrationTaskRecord> {
        Err(tinyagents_harness::TinyAgentsError::Tool(
            "disk full".into(),
        ))
    }
    fn fail(&self, id: &TaskId, e: String) -> tinyagents_harness::Result<OrchestrationTaskRecord> {
        self.0.fail(id, e)
    }
    fn timeout(
        &self,
        id: &TaskId,
        e: String,
    ) -> tinyagents_harness::Result<OrchestrationTaskRecord> {
        self.0.timeout(id, e)
    }
    fn request_cancel(
        &self,
        id: &TaskId,
    ) -> tinyagents_harness::Result<OrchestrationControlOutcome> {
        self.0.request_cancel(id)
    }
    fn mark_cancelled(&self, id: &TaskId) -> tinyagents_harness::Result<OrchestrationTaskRecord> {
        self.0.mark_cancelled(id)
    }
    fn kill(&self, id: &TaskId) -> tinyagents_harness::Result<OrchestrationControlOutcome> {
        self.0.kill(id)
    }
    fn set_timeout_ms(
        &self,
        id: &TaskId,
        ms: u64,
    ) -> tinyagents_harness::Result<OrchestrationTaskRecord> {
        self.0.set_timeout_ms(id, ms)
    }
}

#[test]
fn record_status_propagates_store_failures_but_not_first_writer_races() {
    let store = FailingStore(InMemoryTaskStore::new());
    record_spawned(&store, &spawned("f1")).unwrap();
    let done = DetachedSubagentStatus::Completed {
        output: "x".into(),
        iterations: 1,
    };
    assert!(record_status(&store, "f1", &done).is_err());
    // Already terminal: a later writer is a benign no-op.
    store.fail(&TaskId::new("f1"), "boom".into()).unwrap();
    assert!(record_status(&store, "f1", &done).is_ok());
    // Unknown task: nothing to mirror.
    assert!(record_status(&store, "missing", &done).is_ok());
}

#[test]
fn record_cancelled_is_a_noop_when_terminal_or_missing() {
    let store = InMemoryTaskStore::new();
    record_spawned(&store, &spawned("c1")).unwrap();
    record_cancelled(&store, "c1").unwrap();
    // Already terminal: benign no-op.
    record_cancelled(&store, "c1").unwrap();
    record_cancelled(&store, "missing").unwrap();
}
