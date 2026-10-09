use super::*;
use crate::status::NoEquivalentStatus;
use crate::subagent::SubAgentJobId;
use tinyagents_session::run_ledger::AgentRunStatus as Run;
use tinyagents_tasks::OrchestrationTaskStatus as Task;

use SubAgentJobStatus as Job;

const ALL_JOBS: [Job; 6] = [
    Job::Queued,
    Job::Running,
    Job::Completed,
    Job::Failed,
    Job::Incomplete,
    Job::Cancelled,
];

#[test]
fn job_to_task_covers_every_job_status() {
    let expected = [
        (Job::Queued, Task::Pending),
        (Job::Running, Task::Running),
        (Job::Completed, Task::Completed),
        (Job::Failed, Task::Failed),
        (Job::Incomplete, Task::Failed),
        (Job::Cancelled, Task::Cancelled),
    ];
    assert_eq!(expected.len(), ALL_JOBS.len());
    for (job, task) in expected {
        assert_eq!(job.to_task_status(), task, "{job:?}");
    }
}

#[test]
fn job_to_run_covers_every_job_status() {
    let expected = [
        (Job::Queued, Run::Pending),
        (Job::Running, Run::Running),
        (Job::Completed, Run::Completed),
        (Job::Failed, Run::Failed),
        (Job::Incomplete, Run::Failed),
        (Job::Cancelled, Run::Cancelled),
    ];
    assert_eq!(expected.len(), ALL_JOBS.len());
    for (job, run) in expected {
        assert_eq!(job.to_run_status(), run, "{job:?}");
    }
}

#[test]
fn job_mappings_preserve_terminality() {
    for job in ALL_JOBS {
        assert_eq!(job.to_task_status().is_terminal(), job.is_terminal());
        assert_eq!(job.to_run_status().is_terminal(), job.is_terminal());
    }
}

fn job_with(status: Job, kind: Option<IncompleteKind>) -> SubAgentJob {
    SubAgentJob {
        id: SubAgentJobId("job-1".into()),
        agent: "researcher".into(),
        status,
        output: None,
        error: None,
        subagent_run_id: None,
        parent_tool_call_id: None,
        incomplete_kind: kind,
        artifacts: Vec::new(),
        schema_error: None,
        artifact_error: None,
    }
}

#[test]
fn job_snapshot_refines_incomplete_by_kind() {
    let timed_out = job_with(Job::Incomplete, Some(IncompleteKind::Timeout));
    assert_eq!(timed_out.task_status(), Task::TimedOut);

    for kind in [
        Some(IncompleteKind::BudgetExceeded),
        Some(IncompleteKind::Unspecified),
        None,
    ] {
        assert_eq!(
            job_with(Job::Incomplete, kind).task_status(),
            Task::Failed,
            "{kind:?}"
        );
    }
}

#[test]
fn job_snapshot_ignores_kind_unless_incomplete() {
    for job in ALL_JOBS.into_iter().filter(|j| *j != Job::Incomplete) {
        let snapshot = job_with(job, Some(IncompleteKind::Timeout));
        assert_eq!(snapshot.task_status(), job.to_task_status(), "{job:?}");
    }
}

#[test]
fn task_to_job_covers_every_task_status() {
    let expected: [(Task, Result<Job, ()>); 9] = [
        (Task::Pending, Ok(Job::Queued)),
        (Task::Running, Ok(Job::Running)),
        (Task::Awaiting, Err(())),
        (Task::Completed, Ok(Job::Completed)),
        (Task::Failed, Ok(Job::Failed)),
        (Task::CancelRequested, Ok(Job::Running)),
        (Task::Cancelled, Ok(Job::Cancelled)),
        (Task::TimedOut, Ok(Job::Incomplete)),
        (Task::Abandoned, Err(())),
    ];
    for (task, job) in expected {
        let got: Result<Job, NoEquivalentStatus> = Job::try_from(task);
        assert_eq!(got.map_err(|_| ()), job, "{task:?}");
    }
}

#[test]
fn task_to_job_error_names_the_source_status() {
    let err = Job::try_from(Task::Awaiting).unwrap_err();
    assert_eq!(
        err.to_string(),
        "status `awaiting` has no equivalent in SubAgentJobStatus"
    );
}

#[test]
fn job_status_golden_wire_round_trips() {
    // `SubAgentJob` snapshots serialize these exact strings.
    let golden = [
        (Job::Queued, "queued"),
        (Job::Running, "running"),
        (Job::Completed, "completed"),
        (Job::Failed, "failed"),
        (Job::Incomplete, "incomplete"),
        (Job::Cancelled, "cancelled"),
    ];
    assert_eq!(golden.len(), ALL_JOBS.len());
    for (job, wire) in golden {
        assert_eq!(serde_json::to_value(job).unwrap(), serde_json::json!(wire));
        let back: Job = serde_json::from_value(serde_json::json!(wire)).unwrap();
        assert_eq!(back, job);
    }
}

#[test]
fn job_snapshot_golden_json_round_trips() {
    let old = serde_json::json!({
        "id": "job-1",
        "agent": "researcher",
        "status": "incomplete",
        "incomplete_kind": "timeout"
    });
    let job: SubAgentJob = serde_json::from_value(old).unwrap();
    assert_eq!(job.status, Job::Incomplete);
    let out = serde_json::to_value(&job).unwrap();
    assert_eq!(out["status"], "incomplete");
    assert_eq!(out["incomplete_kind"], "timeout");
    let again: SubAgentJob = serde_json::from_value(out).unwrap();
    assert_eq!(again, job);
}

#[test]
fn from_impls_agree_with_the_named_methods() {
    for job in ALL_JOBS {
        assert_eq!(Task::from(job), job.to_task_status());
        assert_eq!(Run::from(job), job.to_run_status());
    }
}

#[test]
fn run_to_job_covers_every_run_status() {
    let expected: [(Run, Option<Job>); 8] = [
        (Run::Pending, Some(Job::Queued)),
        (Run::Running, Some(Job::Running)),
        (Run::AwaitingUser, None),
        (Run::Paused, None),
        (Run::Completed, Some(Job::Completed)),
        (Run::Failed, Some(Job::Failed)),
        (Run::Cancelled, Some(Job::Cancelled)),
        (Run::Interrupted, None),
    ];
    for (run, job) in expected {
        assert_eq!(Job::try_from(run).ok(), job, "{run:?}");
    }
    let err = Job::try_from(Run::Paused).unwrap_err();
    assert_eq!(err.from_status(), "paused");
    assert_eq!(err.target(), "SubAgentJobStatus");
}

#[test]
fn job_run_job_round_trips_for_mappable_statuses() {
    for job in ALL_JOBS {
        let back = Job::try_from(Run::from(job)).unwrap();
        match job {
            Job::Incomplete => assert_eq!(back, Job::Failed),
            _ => assert_eq!(back, job),
        }
    }
}

#[test]
fn job_and_completion_status_convert_exhaustively() {
    use tinyagents_tasks::CompletionStatus as Done;
    let expected: [(Job, Option<Done>); 6] = [
        (Job::Queued, None),
        (Job::Running, None),
        (Job::Completed, Some(Done::Success)),
        (Job::Failed, Some(Done::Failed)),
        (Job::Incomplete, Some(Done::Incomplete)),
        (Job::Cancelled, Some(Done::Cancelled)),
    ];
    for (job, done) in expected {
        assert_eq!(Done::try_from(job).ok(), done, "{job:?}");
        assert_eq!(done.is_none(), !job.is_terminal(), "{job:?}");
    }
    // The reverse is total and lossless.
    for done in [
        Done::Success,
        Done::Failed,
        Done::Cancelled,
        Done::Incomplete,
    ] {
        assert_eq!(Done::try_from(Job::from(done)).unwrap(), done);
    }
}
