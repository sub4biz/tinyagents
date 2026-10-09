use super::*;
use crate::subagent::SubAgentJobStatus as Job;
use tinyagents_session::run_ledger::AgentRunStatus as Run;
use tinyagents_tasks::OrchestrationTaskStatus as Task;

fn all() -> [DetachedSubagentStatus; 4] {
    [
        DetachedSubagentStatus::Running,
        DetachedSubagentStatus::Completed {
            output: "done".into(),
            iterations: 3,
        },
        DetachedSubagentStatus::AwaitingUser {
            question: "which one?".into(),
        },
        DetachedSubagentStatus::Failed {
            error: "boom".into(),
        },
    ]
}

#[test]
fn detached_to_task_covers_every_variant() {
    let expected = [Task::Running, Task::Completed, Task::Awaiting, Task::Failed];
    for (status, task) in all().iter().zip(expected) {
        assert_eq!(status.to_task_status(), task, "{}", status.label());
    }
}

#[test]
fn detached_to_run_covers_every_variant() {
    let expected = [Run::Running, Run::Completed, Run::AwaitingUser, Run::Failed];
    for (status, run) in all().iter().zip(expected) {
        assert_eq!(status.to_run_status(), run, "{}", status.label());
    }
}

#[test]
fn detached_to_job_covers_every_variant() {
    let expected = [
        Some(Job::Running),
        Some(Job::Completed),
        None,
        Some(Job::Failed),
    ];
    for (status, job) in all().iter().zip(expected) {
        assert_eq!(Job::try_from(status).ok(), job, "{}", status.label());
    }
}

#[test]
fn detached_to_job_error_names_the_source_status() {
    let paused = DetachedSubagentStatus::AwaitingUser {
        question: "q".into(),
    };
    let err = Job::try_from(&paused).unwrap_err();
    assert_eq!(
        err.to_string(),
        "status `awaiting_user` has no equivalent in SubAgentJobStatus"
    );
}

#[test]
fn run_ledger_is_terminal_agrees_except_for_awaiting() {
    // `DetachedSubagentStatus::is_terminal` treats an awaiting run as
    // terminal (it will not progress on its own); the task and ledger
    // vocabularies keep it live so a follow-up can resume it.
    for status in all() {
        let awaiting = matches!(status, DetachedSubagentStatus::AwaitingUser { .. });
        if awaiting {
            assert!(status.is_terminal());
            assert!(!status.to_task_status().is_terminal());
            assert!(!status.to_run_status().is_terminal());
        } else {
            assert_eq!(status.to_task_status().is_terminal(), status.is_terminal());
            assert_eq!(status.to_run_status().is_terminal(), status.is_terminal());
        }
    }
}

#[test]
fn from_impls_agree_with_the_named_methods() {
    for status in all() {
        assert_eq!(Task::from(&status), status.to_task_status());
        assert_eq!(Run::from(&status), status.to_run_status());
    }
}

#[test]
fn detached_to_completion_covers_every_variant() {
    use tinyagents_tasks::CompletionStatus as Done;
    let expected = [None, Some(Done::Success), None, Some(Done::Failed)];
    for (status, done) in all().iter().zip(expected) {
        assert_eq!(Done::try_from(status).ok(), done, "{}", status.label());
    }
    let err = Done::try_from(&DetachedSubagentStatus::Running).unwrap_err();
    assert_eq!(err.target(), "CompletionStatus");
}

#[test]
fn detached_label_is_the_stable_wire_label() {
    // `label()` is the only serialized form of a detached status (the enum
    // itself is a live, payload-carrying type with no serde impl).
    let labels: Vec<_> = all().iter().map(|s| s.label()).collect();
    assert_eq!(labels, ["running", "completed", "awaiting_user", "failed"]);
}
