use super::*;
use serde_json::json;
use tinyagents_tasks::OrchestrationTaskStatus as Task;

const ALL_RUNS: [AgentRunStatus; 8] = [
    AgentRunStatus::Pending,
    AgentRunStatus::Running,
    AgentRunStatus::AwaitingUser,
    AgentRunStatus::Paused,
    AgentRunStatus::Completed,
    AgentRunStatus::Failed,
    AgentRunStatus::Cancelled,
    AgentRunStatus::Interrupted,
];

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

#[test]
fn agent_run_status_golden_wire_round_trips() {
    // Ledger rows and exported JSON carry these exact strings.
    let golden = [
        (AgentRunStatus::Pending, "pending"),
        (AgentRunStatus::Running, "running"),
        (AgentRunStatus::AwaitingUser, "awaiting_user"),
        (AgentRunStatus::Paused, "paused"),
        (AgentRunStatus::Completed, "completed"),
        (AgentRunStatus::Failed, "failed"),
        (AgentRunStatus::Cancelled, "cancelled"),
        (AgentRunStatus::Interrupted, "interrupted"),
    ];
    assert_eq!(golden.len(), ALL_RUNS.len());
    for (status, wire) in golden {
        assert_eq!(serde_json::to_value(status).unwrap(), json!(wire));
        let back: AgentRunStatus = serde_json::from_value(json!(wire)).unwrap();
        assert_eq!(back, status);
        assert_eq!(status.as_str(), wire);
        assert_eq!(AgentRunStatus::parse(wire), status);
    }
}

#[test]
fn run_to_task_covers_every_run_status() {
    let expected = [
        (AgentRunStatus::Pending, Task::Pending),
        (AgentRunStatus::Running, Task::Running),
        (AgentRunStatus::AwaitingUser, Task::Awaiting),
        (AgentRunStatus::Paused, Task::Awaiting),
        (AgentRunStatus::Completed, Task::Completed),
        (AgentRunStatus::Failed, Task::Failed),
        (AgentRunStatus::Cancelled, Task::Cancelled),
        (AgentRunStatus::Interrupted, Task::Abandoned),
    ];
    assert_eq!(expected.len(), ALL_RUNS.len());
    for (run, task) in expected {
        assert_eq!(Task::from(run), task, "{run:?}");
    }
}

#[test]
fn task_to_run_covers_every_task_status() {
    let expected = [
        (Task::Pending, AgentRunStatus::Pending),
        (Task::Running, AgentRunStatus::Running),
        (Task::Awaiting, AgentRunStatus::AwaitingUser),
        (Task::Completed, AgentRunStatus::Completed),
        (Task::Failed, AgentRunStatus::Failed),
        (Task::CancelRequested, AgentRunStatus::Running),
        (Task::Cancelled, AgentRunStatus::Cancelled),
        (Task::TimedOut, AgentRunStatus::Failed),
        (Task::Abandoned, AgentRunStatus::Interrupted),
    ];
    assert_eq!(expected.len(), ALL_TASKS.len());
    for (task, run) in expected {
        assert_eq!(AgentRunStatus::from(task), run, "{task:?}");
    }
}

#[test]
fn terminality_is_preserved_in_both_directions() {
    for task in ALL_TASKS {
        assert_eq!(
            AgentRunStatus::from(task).is_terminal(),
            task.is_terminal(),
            "{task:?}"
        );
    }
    for run in ALL_RUNS {
        assert_eq!(Task::from(run).is_terminal(), run.is_terminal(), "{run:?}");
    }
}

#[test]
fn run_task_run_only_loses_paused() {
    for run in ALL_RUNS {
        let back = AgentRunStatus::from(Task::from(run));
        match run {
            AgentRunStatus::Paused => assert_eq!(back, AgentRunStatus::AwaitingUser),
            _ => assert_eq!(back, run, "{run:?}"),
        }
    }
}

#[test]
fn task_run_task_only_loses_timeout_and_cancel_request() {
    for task in ALL_TASKS {
        let back = Task::from(AgentRunStatus::from(task));
        match task {
            Task::CancelRequested => assert_eq!(back, Task::Running),
            Task::TimedOut => assert_eq!(back, Task::Failed),
            _ => assert_eq!(back, task, "{task:?}"),
        }
    }
}
