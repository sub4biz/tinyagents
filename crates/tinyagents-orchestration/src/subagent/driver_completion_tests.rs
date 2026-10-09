//! The driver's opt-in completion recording, observed through a real
//! `SubagentDriver::run` and a real `CompletionRouter`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tinyagents_harness::CancellationToken;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::run_queue::{QueueLane, RunQueue, RunQueueHandle};
use tinyagents_runtime::ToolSnapshot;
use tinyagents_tasks::{CompletionRouter, CompletionStatus, InMemoryCompletionStore, NotifyMode};
use tinyinference_llm::message::Message;

use super::*;

type Behaviour = Arc<dyn Fn(&str) -> Result<SubagentOutcome, SubagentError> + Send + Sync>;

struct Planner {
    mode: Option<NotifyMode>,
    parent: Option<&'static str>,
}

#[async_trait]
impl SubagentPlanner<String> for Planner {
    async fn prepare(
        &self,
        request: SubagentRequest<String>,
    ) -> Result<PreparedSubagent<String>, SubagentError> {
        let parts = request.into_parts();
        let mut prepared = PreparedSubagent::new(
            parts.task_key.task_id,
            "worker",
            vec![Message::user(parts.input)],
            ToolSnapshot::new(vec![]).unwrap(),
            parts.run_context,
        );
        if let Some(mode) = self.mode {
            prepared = prepared.with_notify_mode(mode);
        }
        if let Some(parent) = self.parent {
            prepared = prepared.with_completion_parent(parent);
        }
        Ok(prepared)
    }
}

struct Executor(Behaviour);

#[async_trait]
impl SubagentExecutor<String> for Executor {
    async fn execute(
        &self,
        execution: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        (self.0)(&execution.prepared.task_id)
    }
}

#[derive(Default)]
struct Memory {
    terminals: Mutex<HashMap<SubagentTaskKey, SubagentOutcome>>,
    pauses: Mutex<HashMap<SubagentTaskKey, SubagentOutcome>>,
}

#[async_trait]
impl SubagentPersistence for Memory {
    async fn load_terminal(
        &self,
        key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(self.terminals.lock().unwrap().get(key).cloned())
    }
    async fn load(&self, key: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError> {
        Ok(self
            .pauses
            .lock()
            .unwrap()
            .get(key)
            .and_then(|outcome| match &outcome.status {
                SubagentOutcomeKind::AwaitingInput(pause) => Some(pause.resume.clone()),
                _ => None,
            }))
    }
    async fn load_pause(
        &self,
        key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(self.pauses.lock().unwrap().get(key).cloned())
    }
    async fn save_pause(
        &self,
        pause: PersistedSubagentPause,
    ) -> Result<SubagentPausePersistenceDisposition, SubagentError> {
        self.pauses.lock().unwrap().insert(pause.key, pause.outcome);
        Ok(SubagentPausePersistenceDisposition::Inserted)
    }
    async fn record_terminal(
        &self,
        key: &SubagentTaskKey,
        outcome: &SubagentOutcome,
        _: Option<&SubagentResume>,
    ) -> Result<SubagentTerminalPersistenceDisposition, SubagentError> {
        let mut terminals = self.terminals.lock().unwrap();
        if terminals.contains_key(key) {
            return Ok(SubagentTerminalPersistenceDisposition::Existing);
        }
        self.pauses.lock().unwrap().remove(key);
        terminals.insert(key.clone(), outcome.clone());
        Ok(SubagentTerminalPersistenceDisposition::Inserted)
    }
}

fn request(task_id: &str) -> SubagentRequest<String> {
    let parent = RunContext::new(RunConfig::new("parent-run"), "d".to_owned());
    let child = parent
        .child(RunConfig::new(format!("run-{task_id}")), "d".to_owned())
        .unwrap();
    SubagentRequest::fresh_from_parent(&parent, child, task_id, (), "go", None).unwrap()
}

fn driver(
    mode: Option<NotifyMode>,
    parent: Option<&'static str>,
    behaviour: Behaviour,
) -> SubagentDriver<String> {
    SubagentDriver::new(SubagentCapabilities {
        planner: Some(Arc::new(Planner { mode, parent })),
        executor: Some(Arc::new(Executor(behaviour))),
        persistence: Some(Arc::new(Memory::default())),
    })
    .unwrap()
}

fn router() -> Arc<CompletionRouter> {
    Arc::new(CompletionRouter::new(Arc::new(
        InMemoryCompletionStore::new(),
    )))
}

fn ok() -> Behaviour {
    Arc::new(|task| Ok(SubagentOutcome::completed(task, "all done")))
}

#[tokio::test]
async fn a_finished_child_is_recorded_under_its_parent() {
    let router = router();
    let driver = driver(Some(NotifyMode::Followup), Some("thread-1"), ok())
        .with_completion_router(router.clone());
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    let pending = router.pending_for("thread-1");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].task_id, "t1");
    assert_eq!(pending[0].agent_id, "worker");
    assert_eq!(pending[0].status, CompletionStatus::Success);
    assert_eq!(pending[0].result.text, "all done");
    assert_eq!(pending[0].notify_mode, NotifyMode::Followup);
}

#[tokio::test]
async fn the_parent_key_falls_back_to_the_parent_run() {
    let router = router();
    let driver = driver(Some(NotifyMode::Off), None, ok()).with_completion_router(router.clone());
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(router.pending_for("parent-run").len(), 1);
}

#[tokio::test]
async fn each_notify_mode_reaches_its_lane() {
    for (mode, lane) in [
        (NotifyMode::Followup, Some(QueueLane::Followup)),
        (NotifyMode::Collect, Some(QueueLane::Collect)),
        (NotifyMode::HoldForNextTurn, None),
        (NotifyMode::Off, None),
    ] {
        let router = router();
        let queue: RunQueueHandle = Arc::new(RunQueue::<Message>::new());
        router.attach_parent("p", queue.clone());
        let driver = driver(Some(mode), Some("p"), ok()).with_completion_router(router.clone());
        driver
            .run(request("t1"), CancellationToken::new())
            .await
            .unwrap();
        let status = queue.status().await;
        match lane {
            Some(QueueLane::Followup) => {
                assert_eq!((status.followups, status.total), (1, 1), "{mode:?}")
            }
            Some(QueueLane::Collect) => {
                assert_eq!((status.collects, status.total), (1, 1), "{mode:?}")
            }
            _ => assert_eq!(status.total, 0, "{mode:?}"),
        }
        // A live push stays pending (leased) until the host acknowledges it.
        assert_eq!(router.pending_for("p").len(), 1, "{mode:?}");
        assert_eq!(
            router.in_flight_for("p").len(),
            usize::from(lane.is_some()),
            "{mode:?}"
        );
    }
}

#[tokio::test]
async fn a_child_the_parent_already_collected_is_never_pushed() {
    let router = router();
    let queue: RunQueueHandle = Arc::new(RunQueue::<Message>::new());
    router.attach_parent("p", queue.clone());
    router.tombstone("t1").unwrap();
    let driver =
        driver(Some(NotifyMode::Followup), Some("p"), ok()).with_completion_router(router.clone());
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(queue.status().await.total, 0);
    assert!(router.pending_for("p").is_empty());
}

#[tokio::test]
async fn an_incomplete_child_is_recorded_with_its_reason() {
    let router = router();
    let behaviour: Behaviour = Arc::new(|task| {
        Ok(SubagentOutcome::incomplete(
            task,
            SubagentIncomplete::new("ran out of budget"),
        ))
    });
    let driver =
        driver(Some(NotifyMode::Off), Some("p"), behaviour).with_completion_router(router.clone());
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    let pending = router.pending_for("p");
    assert_eq!(pending[0].status, CompletionStatus::Incomplete);
    assert_eq!(pending[0].result.text, "ran out of budget");
}

#[tokio::test]
async fn an_executor_failure_is_returned_and_recorded_as_failed() {
    let router = router();
    let behaviour: Behaviour = Arc::new(|_| Err(SubagentError::Execution("boom".into())));
    let failing =
        driver(Some(NotifyMode::Off), Some("p"), behaviour).with_completion_router(router.clone());
    // The caller of `run` still gets the error, unchanged.
    let error = failing
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(error, SubagentError::Execution(_)), "{error:?}");
    let pending = router.pending_for("p");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].task_id, "t1");
    assert_eq!(pending[0].status, CompletionStatus::Failed);
    assert!(pending[0].result.text.contains("boom"), "{:?}", pending[0]);
    // The router keeps the first record per task id: a re-run under the same
    // id does not add a second completion.
    let retry =
        driver(Some(NotifyMode::Off), Some("p"), ok()).with_completion_router(router.clone());
    retry
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(router.pending_for("p").len(), 1);
}

#[tokio::test]
async fn a_transient_failure_that_exhausts_its_retries_is_recorded_as_failed() {
    let router = router();
    let behaviour: Behaviour = Arc::new(|_| {
        Err(SubagentError::Transient {
            message: "provider unavailable".into(),
            tools_ran: false,
        })
    });
    let failing =
        driver(Some(NotifyMode::Off), Some("p"), behaviour).with_completion_router(router.clone());
    assert!(
        failing
            .run(request("t1"), CancellationToken::new())
            .await
            .is_err()
    );
    let pending = router.pending_for("p");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].status, CompletionStatus::Failed);
}

#[tokio::test]
async fn a_host_seam_error_is_not_a_child_failure() {
    let router = router();
    let behaviour: Behaviour = Arc::new(|_| Ok(SubagentOutcome::completed("other-task", "x")));
    let mismatched =
        driver(Some(NotifyMode::Off), Some("p"), behaviour).with_completion_router(router.clone());
    assert!(matches!(
        mismatched
            .run(request("t1"), CancellationToken::new())
            .await,
        Err(SubagentError::TaskIdMismatch { .. })
    ));
    assert!(router.pending_for("p").is_empty());
}

#[tokio::test]
async fn an_executor_failure_without_a_notify_mode_is_not_recorded() {
    let router = router();
    let behaviour: Behaviour = Arc::new(|_| Err(SubagentError::Execution("boom".into())));
    let failing = driver(None, Some("p"), behaviour).with_completion_router(router.clone());
    assert!(
        failing
            .run(request("t1"), CancellationToken::new())
            .await
            .is_err()
    );
    assert!(router.pending_for("p").is_empty());
}

#[tokio::test]
async fn an_executor_failure_without_a_router_is_unchanged() {
    let behaviour: Behaviour = Arc::new(|_| Err(SubagentError::Execution("boom".into())));
    let plain = driver(Some(NotifyMode::Followup), Some("p"), behaviour);
    let error = plain
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(error, SubagentError::Execution(m) if m == "boom"));
}

/// A behaviour that pauses on its first call and completes on the next.
fn pause_then_complete() -> Behaviour {
    let calls = Arc::new(Mutex::new(0usize));
    Arc::new(move |task| {
        let mut calls = calls.lock().unwrap();
        *calls += 1;
        if *calls == 1 {
            let mut outcome = SubagentOutcome::completed(task, "waiting");
            outcome.status = SubagentOutcomeKind::AwaitingInput(SubagentPause {
                reason: "need approval".into(),
                resume: SubagentResume::default(),
            });
            Ok(outcome)
        } else {
            Ok(SubagentOutcome::completed(task, "approved and done"))
        }
    })
}

#[tokio::test]
async fn a_paused_child_is_not_recorded_until_it_finishes() {
    let router = router();
    let driver = driver(Some(NotifyMode::Off), Some("p"), pause_then_complete())
        .with_completion_router(router.clone());
    let first = driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert!(matches!(
        first.outcome.status,
        SubagentOutcomeKind::AwaitingInput(_)
    ));
    assert!(
        router.pending_for("p").is_empty(),
        "a pause is not terminal"
    );
    // The resume finishes the same task id and records exactly once.
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    let pending = router.pending_for("p");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].status, CompletionStatus::Success);
    assert_eq!(pending[0].result.text, "approved and done");
}

#[tokio::test]
async fn a_resume_that_errors_leaves_the_paused_task_unrecorded() {
    let router = router();
    let calls = Arc::new(Mutex::new(0usize));
    let behaviour: Behaviour = {
        let pause = pause_then_complete();
        Arc::new(move |task| {
            let mut calls = calls.lock().unwrap();
            *calls += 1;
            match *calls {
                1 => pause(task),
                2 => Err(SubagentError::Execution("resume crashed".into())),
                _ => Ok(SubagentOutcome::completed(task, "finally done")),
            }
        })
    };
    let driver =
        driver(Some(NotifyMode::Off), Some("p"), behaviour).with_completion_router(router.clone());
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert!(
        driver
            .run(request("t1"), CancellationToken::new())
            .await
            .is_err()
    );
    // The durable pause is still there, so the task is not finished: no
    // failed record may shadow its eventual success.
    assert!(router.pending_for("p").is_empty());
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    let pending = router.pending_for("p");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].status, CompletionStatus::Success);
}

#[tokio::test]
async fn a_spawn_without_a_notify_mode_is_not_recorded() {
    let router = router();
    let queue: RunQueueHandle = Arc::new(RunQueue::<Message>::new());
    router.attach_parent("p", queue.clone());
    let foreground = driver(None, Some("p"), ok()).with_completion_router(router.clone());
    foreground
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert!(router.pending_for("p").is_empty());
    assert_eq!(queue.status().await.total, 0);
}

#[tokio::test]
async fn a_cancelled_child_is_recorded_as_cancelled() {
    let router = router();
    let token = CancellationToken::new();
    let behaviour: Behaviour = {
        let token = token.clone();
        Arc::new(move |task| {
            token.cancel();
            Ok(SubagentOutcome::completed(task, "late"))
        })
    };
    let driver =
        driver(Some(NotifyMode::Off), Some("p"), behaviour).with_completion_router(router.clone());
    let result = driver.run(request("t1"), token).await.unwrap();
    assert!(matches!(
        result.outcome.status,
        SubagentOutcomeKind::Cancelled
    ));
    let pending = router.pending_for("p");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].task_id, "t1");
    assert_eq!(pending[0].status, CompletionStatus::Cancelled);
}

#[tokio::test]
async fn a_cancel_before_the_child_launches_is_recorded_once_it_has_a_notify_mode() {
    struct CancellingPlanner(CancellationToken);

    #[async_trait]
    impl SubagentPlanner<String> for CancellingPlanner {
        async fn prepare(
            &self,
            request: SubagentRequest<String>,
        ) -> Result<PreparedSubagent<String>, SubagentError> {
            let prepared = Planner {
                mode: Some(NotifyMode::Off),
                parent: Some("p"),
            }
            .prepare(request)
            .await?;
            self.0.cancel();
            Ok(prepared)
        }
    }

    let router = router();
    let token = CancellationToken::new();
    let driver = SubagentDriver::new(SubagentCapabilities {
        planner: Some(Arc::new(CancellingPlanner(token.clone()))),
        executor: Some(Arc::new(Executor(ok()))),
        persistence: Some(Arc::new(Memory::default())),
    })
    .unwrap()
    .with_completion_router(router.clone());
    let result = driver.run(request("t1"), token).await.unwrap();
    assert!(matches!(
        result.outcome.status,
        SubagentOutcomeKind::Cancelled
    ));
    let pending = router.pending_for("p");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].status, CompletionStatus::Cancelled);
}

/// An executor that returns `SubagentError::Cancelled` is converted to a
/// cancelled outcome before persistence, so it is recorded as `Cancelled` and
/// the call returns `Ok`.
#[tokio::test]
async fn an_executor_cancelled_error_is_recorded_as_cancelled() {
    let router = router();
    let behaviour: Behaviour = Arc::new(|_| Err(SubagentError::Cancelled));
    let driver =
        driver(Some(NotifyMode::Off), Some("p"), behaviour).with_completion_router(router.clone());
    let result = driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert!(matches!(
        result.outcome.status,
        SubagentOutcomeKind::Cancelled
    ));
    let pending = router.pending_for("p");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].status, CompletionStatus::Cancelled);
}

#[tokio::test]
async fn a_cancelled_child_without_a_router_is_unchanged() {
    let token = CancellationToken::new();
    let behaviour: Behaviour = {
        let token = token.clone();
        Arc::new(move |task| {
            token.cancel();
            Ok(SubagentOutcome::completed(task, "late"))
        })
    };
    let plain = driver(Some(NotifyMode::Off), Some("p"), behaviour);
    let result = plain.run(request("t1"), token).await.unwrap();
    assert!(matches!(
        result.outcome.status,
        SubagentOutcomeKind::Cancelled
    ));
}

#[tokio::test]
async fn a_replayed_terminal_result_is_not_recorded_again() {
    let router = router();
    let driver =
        driver(Some(NotifyMode::Off), Some("p"), ok()).with_completion_router(router.clone());
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(router.pull("p", 10).unwrap().len(), 1);
    router.mark_delivered(&["t1"]).unwrap();
    let replay = driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert!(!replay.should_emit_host_effects());
    assert!(router.pending_for("p").is_empty());
    assert!(router.pull("p", 10).unwrap().is_empty());
}

#[tokio::test]
async fn without_a_router_the_run_is_unchanged() {
    let with_router_driver =
        driver(Some(NotifyMode::Followup), Some("p"), ok()).with_completion_router(router());
    let plain = driver(Some(NotifyMode::Followup), Some("p"), ok());
    let a = plain
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    let b = with_router_driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn applied_results_convert_without_loss() {
    let applied = AppliedResult {
        text: "head...tail".into(),
        omitted_chars: 40,
        artifact: Some(ArtifactReference {
            id: "art-1".into(),
            media_type: Some("text/plain".into()),
            metadata: Default::default(),
        }),
        ..Default::default()
    };
    let result = tinyagents_tasks::CompletionResult::from(&applied);
    assert_eq!(result.text, "head...tail");
    assert_eq!(result.omitted_chars, 40);
    assert_eq!(result.artifact.unwrap().id, "art-1");
}

#[tokio::test]
async fn truncation_metadata_reaches_the_completion() {
    struct Capped;
    #[async_trait]
    impl SubagentPlanner<String> for Capped {
        async fn prepare(
            &self,
            request: SubagentRequest<String>,
        ) -> Result<PreparedSubagent<String>, SubagentError> {
            let parts = request.into_parts();
            Ok(PreparedSubagent::new(
                parts.task_key.task_id,
                "worker",
                vec![Message::user(parts.input)],
                ToolSnapshot::new(vec![]).unwrap(),
                parts.run_context,
            )
            .with_notify_mode(NotifyMode::Off)
            .with_completion_parent("p")
            .with_result_policy(ResultPolicy::new().with_max_chars(20)))
        }
    }
    let router = router();
    let driver = SubagentDriver::new(SubagentCapabilities {
        planner: Some(Arc::new(Capped)),
        executor: Some(Arc::new(Executor(Arc::new(|task| {
            Ok(SubagentOutcome::completed(task, "x".repeat(500)))
        })))),
        persistence: Some(Arc::new(Memory::default())),
    })
    .unwrap()
    .with_completion_router(router.clone());
    driver
        .run(request("t1"), CancellationToken::new())
        .await
        .unwrap();
    assert!(router.pending_for("p")[0].result.omitted_chars > 0);
}

#[test]
fn the_overflow_artifact_is_the_one_a_completion_names() {
    let origin_task = SubagentTaskKey {
        root_run_id: "r".into(),
        parent_run_id: "pr".into(),
        thread_id: Some("p".into()),
        task_id: "t1".into(),
    };
    let prepared = PreparedSubagent::new(
        "t1",
        "worker",
        vec![],
        ToolSnapshot::new(vec![]).unwrap(),
        RunContext::new(RunConfig::new("c"), String::new()),
    )
    .with_notify_mode(NotifyMode::Off);
    let origin = completion::CompletionOrigin::new(&origin_task, &prepared).unwrap();
    let mut outcome = SubagentOutcome::completed("t1", "preview");
    for id in ["pre-existing", "overflow"] {
        outcome.artifacts.push(ArtifactReference {
            id: id.into(),
            ..Default::default()
        });
    }
    let record = origin.record_for_outcome(&outcome, 9).unwrap();
    assert_eq!(record.result.artifact.unwrap().id, "overflow");
    assert_eq!(record.parent_key, "p");
}
