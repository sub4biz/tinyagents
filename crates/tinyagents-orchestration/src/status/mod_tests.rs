use super::*;
use tinyagents_graph::orchestration::OrchestrationTaskStatus as Task;
use tinyagents_session::run_ledger::AgentRunStatus as Run;

const ALL_TASKS: [Task; 9] = [
    Task::Pending,
    Task::Running,
    Task::Awaiting,
    Task::Completed,
    Task::Failed,
    Task::CancelRequested,
    Task::Cancelled,
    Task::TimedOut,
    Task::Abandoned,
];

const ALL_RUNS: [Run; 8] = [
    Run::Pending,
    Run::Running,
    Run::AwaitingUser,
    Run::Paused,
    Run::Completed,
    Run::Failed,
    Run::Cancelled,
    Run::Interrupted,
];

#[test]
fn task_to_run_covers_every_task_status() {
    let expected = [
        (Task::Pending, Run::Pending),
        (Task::Running, Run::Running),
        (Task::Awaiting, Run::AwaitingUser),
        (Task::Completed, Run::Completed),
        (Task::Failed, Run::Failed),
        (Task::CancelRequested, Run::Running),
        (Task::Cancelled, Run::Cancelled),
        (Task::TimedOut, Run::Failed),
        (Task::Abandoned, Run::Interrupted),
    ];
    assert_eq!(expected.len(), ALL_TASKS.len());
    for (task, run) in expected {
        assert_eq!(task_status_to_run_status(task), run, "{task:?}");
    }
}

#[test]
fn run_to_task_covers_every_run_status() {
    let expected = [
        (Run::Pending, Task::Pending),
        (Run::Running, Task::Running),
        (Run::AwaitingUser, Task::Awaiting),
        (Run::Paused, Task::Awaiting),
        (Run::Completed, Task::Completed),
        (Run::Failed, Task::Failed),
        (Run::Cancelled, Task::Cancelled),
        (Run::Interrupted, Task::Abandoned),
    ];
    assert_eq!(expected.len(), ALL_RUNS.len());
    for (run, task) in expected {
        assert_eq!(run_status_to_task_status(run), task, "{run:?}");
    }
}

#[test]
fn terminality_is_preserved_in_both_directions() {
    for task in ALL_TASKS {
        // A cancel request is still live work on both sides.
        assert_eq!(
            task_status_to_run_status(task).is_terminal(),
            task.is_terminal(),
            "{task:?}"
        );
    }
    for run in ALL_RUNS {
        assert_eq!(
            run_status_to_task_status(run).is_terminal(),
            run.is_terminal(),
            "{run:?}"
        );
    }
}

#[test]
fn run_to_task_to_run_only_loses_documented_distinctions() {
    for run in ALL_RUNS {
        let back = task_status_to_run_status(run_status_to_task_status(run));
        match run {
            // Paused and awaiting-user both read back as awaiting-user.
            Run::Paused => assert_eq!(back, Run::AwaitingUser),
            _ => assert_eq!(back, run, "{run:?}"),
        }
    }
}

#[test]
fn no_equivalent_status_error_names_both_sides() {
    let err = NoEquivalentStatus::new("awaiting", "SubAgentJobStatus");
    assert_eq!(
        err.to_string(),
        "status `awaiting` has no equivalent in SubAgentJobStatus"
    );
}
