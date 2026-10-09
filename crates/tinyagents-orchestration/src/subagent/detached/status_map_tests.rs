use super::*;
use crate::subagent::SubAgentJobStatus as Job;
use tinyagents_graph::orchestration::OrchestrationTaskStatus as Task;
use tinyagents_session::run_ledger::AgentRunStatus as Run;

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
