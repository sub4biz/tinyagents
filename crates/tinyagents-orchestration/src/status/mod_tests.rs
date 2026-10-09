use super::*;
use tinyagents_session::run_ledger::AgentRunStatus as Run;
use tinyagents_tasks::OrchestrationTaskStatus as Task;

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

// The deprecated free functions are kept for source compatibility; this is the
// one place that still calls them, to prove they equal the `From` impls.
#[test]
#[allow(deprecated)]
fn deprecated_free_functions_delegate_to_the_from_impls() {
    for task in ALL_TASKS {
        assert_eq!(task_status_to_run_status(task), Run::from(task), "{task:?}");
    }
    for run in ALL_RUNS {
        assert_eq!(run_status_to_task_status(run), Task::from(run), "{run:?}");
    }
}

#[test]
fn no_equivalent_status_is_the_tasks_crate_type() {
    let err: tinyagents_tasks::NoEquivalentStatus = NoEquivalentStatus::new("awaiting", "X");
    assert_eq!(err.to_string(), "status `awaiting` has no equivalent in X");
}
