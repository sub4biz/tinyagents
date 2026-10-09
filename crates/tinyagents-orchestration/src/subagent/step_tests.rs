//! `run_agent_step`: opaque host work goes through the real `SubagentDriver`.

use super::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use tinyagents_harness::retry::RetryPolicy;

use crate::subagent::{IncompleteKind, SpawnPolicy};

fn ident(task: &str) -> AgentStepIdentity {
    AgentStepIdentity::new("parent", task)
}

async fn ok_text(text: &'static str) -> Result<StepSuccess<u32>, StepWorkError> {
    Ok(StepSuccess::new(text, 7))
}

#[tokio::test]
async fn default_config_is_a_passthrough() {
    let result = run_agent_step(
        &AgentStepConfig::default(),
        ident("a"),
        CancellationToken::new(),
        |_| ok_text("hello"),
    )
    .await
    .unwrap();
    assert_eq!(result.outcome.status, SubagentOutcomeKind::Completed);
    assert_eq!(result.outcome.output, "hello");
    assert_eq!(result.value, Some(7));
}

#[tokio::test]
async fn result_policy_cap_trims_output() {
    let config =
        AgentStepConfig::default().with_result_policy(ResultPolicy::new().with_max_chars(5));
    let result = run_agent_step(&config, ident("a"), CancellationToken::new(), |_| {
        ok_text("0123456789abcdefghij")
    })
    .await
    .unwrap();
    assert!(result.outcome.output.chars().count() <= 5);
}

#[tokio::test]
async fn spawn_policy_denial_never_runs_the_worker() {
    let ran = Arc::new(AtomicU32::new(0));
    let config = AgentStepConfig::default().with_admission(SpawnAdmission::new(SpawnPolicy {
        max_total_per_root: Some(1),
        ..Default::default()
    }));
    for task in ["a", "b"] {
        let ran = ran.clone();
        let res = run_agent_step(&config, ident(task), CancellationToken::new(), move |_| {
            ran.fetch_add(1, Ordering::SeqCst);
            ok_text("x")
        })
        .await;
        if task == "b" {
            assert!(matches!(res, Err(AgentStepError::Rejected(_))));
        } else {
            assert!(res.is_ok());
        }
    }
    assert_eq!(ran.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn allowed_targets_check_the_identity_target() {
    let config = AgentStepConfig::default().with_admission(SpawnAdmission::new(SpawnPolicy {
        allowed_targets: Some(vec!["good".into()]),
        ..Default::default()
    }));
    let bad = run_agent_step(
        &config,
        ident("a").with_target("evil"),
        CancellationToken::new(),
        |_| ok_text("x"),
    )
    .await;
    assert!(matches!(bad, Err(AgentStepError::Rejected(_))));
    let good = run_agent_step(
        &config,
        ident("b").with_target("good"),
        CancellationToken::new(),
        |_| ok_text("x"),
    )
    .await;
    assert!(good.is_ok());
}

#[tokio::test]
async fn timeout_ends_incomplete_with_typed_kind() {
    let config = AgentStepConfig::default()
        .with_policy(SubAgentPolicy::default().with_timeout(Duration::from_millis(20)));
    let result = run_agent_step(&config, ident("a"), CancellationToken::new(), |_| async {
        std::future::pending::<()>().await;
        ok_text("never").await
    })
    .await
    .unwrap();
    match result.outcome.status {
        SubagentOutcomeKind::Incomplete(inc) => assert_eq!(inc.kind, IncompleteKind::Timeout),
        other => panic!("expected timeout, got {other:?}"),
    }
}

#[tokio::test]
async fn transient_failures_retry_per_policy() {
    let attempts = Arc::new(AtomicU32::new(0));
    let config = AgentStepConfig::default().with_policy(
        SubAgentPolicy::default().with_retry(
            RetryPolicy::default()
                .with_max_attempts(3)
                .with_backoff_sleep(false),
        ),
    );
    let seen = attempts.clone();
    let result = run_agent_step(&config, ident("a"), CancellationToken::new(), move |_| {
        let n = seen.fetch_add(1, Ordering::SeqCst);
        async move {
            if n < 2 {
                Err(StepWorkError::Transient {
                    error: anyhow::anyhow!("flaky"),
                    tools_ran: false,
                })
            } else {
                ok_text("recovered").await
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(result.outcome.output, "recovered");
    assert_eq!(attempts.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn fatal_worker_error_keeps_its_original_message() {
    let err = run_agent_step::<(), _, _>(
        &AgentStepConfig::default(),
        ident("a"),
        CancellationToken::new(),
        |_| async { Err(anyhow::anyhow!("boom 42").into()) },
    )
    .await
    .err()
    .unwrap();
    match err {
        AgentStepError::Worker(e) => assert_eq!(e.to_string(), "boom 42"),
        other => panic!("expected worker error, got {other}"),
    }
}

#[tokio::test]
async fn pre_cancelled_step_reports_cancelled_without_running() {
    let token = CancellationToken::new();
    token.cancel();
    let ran = Arc::new(AtomicU32::new(0));
    let seen = ran.clone();
    let err = run_agent_step(&AgentStepConfig::default(), ident("a"), token, move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
        ok_text("x")
    })
    .await
    .err()
    .unwrap();
    assert!(matches!(err, AgentStepError::Cancelled));
    assert_eq!(ran.load(Ordering::SeqCst), 0, "the worker never ran");
}
