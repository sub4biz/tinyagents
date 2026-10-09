use super::*;
use crate::subagent::{
    SubAgentJobStatus as Job, SubagentIncomplete, SubagentPause, SubagentResume,
};
use tinyagents_session::run_ledger::AgentRunStatus as Run;
use tinyagents_tasks::{CompletionStatus as Done, OrchestrationTaskStatus as Task};

fn paused() -> SubagentOutcomeKind {
    SubagentOutcomeKind::AwaitingInput(SubagentPause {
        reason: "need input".into(),
        resume: SubagentResume::default(),
    })
}

fn incomplete(kind: IncompleteKind) -> SubagentOutcomeKind {
    SubagentOutcomeKind::Incomplete(SubagentIncomplete::new("stopped").with_kind(kind))
}

#[test]
fn outcome_to_task_covers_every_variant() {
    let expected = [
        (SubagentOutcomeKind::Completed, Task::Completed),
        (paused(), Task::Awaiting),
        (incomplete(IncompleteKind::Timeout), Task::TimedOut),
        (incomplete(IncompleteKind::BudgetExceeded), Task::Failed),
        (incomplete(IncompleteKind::Unspecified), Task::Failed),
        (SubagentOutcomeKind::Cancelled, Task::Cancelled),
    ];
    for (outcome, task) in expected {
        assert_eq!(Task::from(&outcome), task, "{outcome:?}");
    }
}

#[test]
fn outcome_to_run_follows_the_task_mapping() {
    let expected = [
        (SubagentOutcomeKind::Completed, Run::Completed),
        (paused(), Run::AwaitingUser),
        (incomplete(IncompleteKind::Timeout), Run::Failed),
        (incomplete(IncompleteKind::Unspecified), Run::Failed),
        (SubagentOutcomeKind::Cancelled, Run::Cancelled),
    ];
    for (outcome, run) in expected {
        assert_eq!(Run::from(&outcome), run, "{outcome:?}");
    }
}

#[test]
fn outcome_to_job_covers_every_variant() {
    let expected = [
        (SubagentOutcomeKind::Completed, Some(Job::Completed)),
        (paused(), None),
        (incomplete(IncompleteKind::Timeout), Some(Job::Incomplete)),
        (
            incomplete(IncompleteKind::Unspecified),
            Some(Job::Incomplete),
        ),
        (SubagentOutcomeKind::Cancelled, Some(Job::Cancelled)),
    ];
    for (outcome, job) in expected {
        assert_eq!(Job::try_from(&outcome).ok(), job, "{outcome:?}");
    }
    let err = Job::try_from(&paused()).unwrap_err();
    assert_eq!(err.from_status(), "awaiting_input");
    assert_eq!(err.target(), "SubAgentJobStatus");
}

#[test]
fn outcome_to_completion_covers_every_variant() {
    let expected = [
        (SubagentOutcomeKind::Completed, Some(Done::Success)),
        (paused(), None),
        (incomplete(IncompleteKind::Timeout), Some(Done::Incomplete)),
        (SubagentOutcomeKind::Cancelled, Some(Done::Cancelled)),
    ];
    for (outcome, done) in expected {
        assert_eq!(Done::try_from(&outcome).ok(), done, "{outcome:?}");
    }
}

#[test]
fn terminal_outcomes_map_to_terminal_tasks() {
    for outcome in [
        SubagentOutcomeKind::Completed,
        incomplete(IncompleteKind::BudgetExceeded),
        SubagentOutcomeKind::Cancelled,
    ] {
        assert!(Task::from(&outcome).is_terminal(), "{outcome:?}");
    }
    assert!(Task::from(&paused()).is_live());
}

#[test]
fn outcome_kind_golden_json_in_and_out() {
    // Records persisted before `kind` existed carry only a reason.
    let old = serde_json::json!({"Incomplete": {"reason": "stopped"}});
    let outcome: SubagentOutcomeKind = serde_json::from_value(old).unwrap();
    assert_eq!(
        outcome,
        SubagentOutcomeKind::Incomplete(SubagentIncomplete::new("stopped"))
    );
    assert_eq!(Task::from(&outcome), Task::Failed);
    assert_eq!(
        serde_json::to_value(&outcome).unwrap(),
        serde_json::json!({"Incomplete": {"reason": "stopped", "kind": "unspecified"}})
    );
}
