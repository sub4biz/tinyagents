use super::*;
use crate::status::NoEquivalentStatus;
use crate::subagent::SubAgentJobId;
use tinyagents_graph::orchestration::OrchestrationTaskStatus as Task;
use tinyagents_session::run_ledger::AgentRunStatus as Run;

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
