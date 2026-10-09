//! Member steps run through `SubagentDriver`: configured driver policy takes
//! effect, and the default step is behaviour-neutral.

use super::*;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::subagent::{ResultPolicy, SpawnAdmission, SpawnPolicy, SubAgentPolicy};

#[derive(Default, Clone)]
struct Seen {
    completed: Arc<Mutex<Vec<String>>>,
    failed: Arc<Mutex<Vec<String>>>,
}

async fn run(step: MemberStep, seen: &Seen, worker: MemberOutcome) -> Result<()> {
    let (c, f) = (seen.completed.clone(), seen.failed.clone());
    let worker = Arc::new(Mutex::new(Some(worker)));
    run_member_graph_with(
        None,
        step,
        move || {
            let worker = worker.clone();
            async move { Ok(worker.lock().unwrap().take().expect("worker ran once")) }
        },
        move |o| {
            let c = c.clone();
            async move {
                c.lock().unwrap().push(o);
                Ok(())
            }
        },
        move |r| {
            let f = f.clone();
            async move {
                f.lock().unwrap().push(r);
                Ok(())
            }
        },
    )
    .await
}

fn done(text: &str) -> MemberOutcome {
    MemberOutcome::Completed {
        output: text.into(),
    }
}

#[tokio::test]
async fn result_policy_cap_applies_to_a_member_output() {
    let seen = Seen::default();
    let config =
        AgentStepConfig::default().with_result_policy(ResultPolicy::new().with_max_chars(8));
    let long = "x".repeat(500);
    run(MemberStep::new(config, "team", "m1"), &seen, done(&long))
        .await
        .unwrap();
    let completed = seen.completed.lock().unwrap();
    assert_eq!(completed.len(), 1);
    assert!(
        completed[0].chars().count() < 500,
        "output was trimmed by ResultPolicy"
    );
}

#[tokio::test]
async fn spawn_policy_denial_fails_the_member_without_running_the_worker() {
    let seen = Seen::default();
    let config = AgentStepConfig::default().with_admission(SpawnAdmission::new(SpawnPolicy {
        allowed_targets: Some(vec!["allowed".into()]),
        ..Default::default()
    }));
    // The worker closure panics if it is ever taken: run() unwraps `take()`.
    run(
        MemberStep::new(config, "team", "intruder"),
        &seen,
        done("never"),
    )
    .await
    .unwrap();
    assert!(seen.completed.lock().unwrap().is_empty());
    let failed = seen.failed.lock().unwrap();
    assert_eq!(failed.len(), 1);
    assert!(failed[0].contains("not admitted"), "got {failed:?}");
}

#[tokio::test]
async fn shared_admission_caps_members_across_a_team() {
    let seen = Seen::default();
    let config = AgentStepConfig::default().with_admission(SpawnAdmission::new(SpawnPolicy {
        max_total_per_root: Some(1),
        ..Default::default()
    }));
    run(
        MemberStep::new(config.clone(), "team", "m1"),
        &seen,
        done("a"),
    )
    .await
    .unwrap();
    run(MemberStep::new(config, "team", "m2"), &seen, done("b"))
        .await
        .unwrap();
    assert_eq!(seen.completed.lock().unwrap().as_slice(), ["a"]);
    assert_eq!(seen.failed.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn timeout_policy_fails_a_hung_member() {
    let seen = Seen::default();
    let config = AgentStepConfig::default()
        .with_policy(SubAgentPolicy::default().with_timeout(Duration::from_millis(20)));
    let f = seen.failed.clone();
    run_member_graph_with(
        None,
        MemberStep::new(config, "team", "m1"),
        || async {
            std::future::pending::<()>().await;
            Ok(done("never"))
        },
        |_| async { Ok(()) },
        move |r| {
            let f = f.clone();
            async move {
                f.lock().unwrap().push(r);
                Ok(())
            }
        },
    )
    .await
    .unwrap();
    assert!(seen.failed.lock().unwrap()[0].contains("timed out"));
}

#[tokio::test]
async fn default_step_is_neutral_and_worker_errors_still_fail_the_graph() {
    let seen = Seen::default();
    run(MemberStep::default(), &seen, done("same"))
        .await
        .unwrap();
    assert_eq!(seen.completed.lock().unwrap().as_slice(), ["same"]);
    let err = run_member_graph(
        None,
        || async { Err::<MemberOutcome, _>(anyhow::anyhow!("worker exploded")) },
        |_| async { Ok(()) },
        |_| async { Ok(()) },
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("worker exploded"), "{err}");
}

#[tokio::test]
async fn pre_cancelled_member_never_runs_its_worker() {
    let seen = Seen::default();
    let token = CancellationToken::new();
    token.cancel();
    let step = MemberStep::default().with_cancellation(token);
    let f = seen.failed.clone();
    let ran = Arc::new(Mutex::new(0u32));
    let counter = ran.clone();
    run_member_graph_with(
        None,
        step,
        move || {
            *counter.lock().unwrap() += 1;
            async { Ok(done("never")) }
        },
        |_| async { Ok(()) },
        move |r| {
            let f = f.clone();
            async move {
                f.lock().unwrap().push(r);
                Ok(())
            }
        },
    )
    .await
    .unwrap();
    assert_eq!(*ran.lock().unwrap(), 0);
    assert!(seen.failed.lock().unwrap()[0].contains("cancelled"));
}

#[tokio::test]
async fn cancelling_while_the_worker_runs_routes_to_on_failed() {
    let seen = Seen::default();
    let token = CancellationToken::new();
    let step = MemberStep::default().with_cancellation(token.clone());
    let f = seen.failed.clone();
    let cancel = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        cancel.cancel();
    });
    run_member_graph_with(
        None,
        step,
        || async {
            std::future::pending::<()>().await;
            Ok(done("never"))
        },
        |_| async { Ok(()) },
        move |r| {
            let f = f.clone();
            async move {
                f.lock().unwrap().push(r);
                Ok(())
            }
        },
    )
    .await
    .unwrap();
    assert!(seen.failed.lock().unwrap()[0].contains("cancelled"));
}
