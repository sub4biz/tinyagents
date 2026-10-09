use super::*;
use serde_json::json;

const ALL_TASKS: [OrchestrationTaskStatus; 9] = [
    OrchestrationTaskStatus::Pending,
    OrchestrationTaskStatus::Running,
    OrchestrationTaskStatus::Awaiting,
    OrchestrationTaskStatus::Completed,
    OrchestrationTaskStatus::Failed,
    OrchestrationTaskStatus::CancelRequested,
    OrchestrationTaskStatus::Cancelled,
    OrchestrationTaskStatus::TimedOut,
    OrchestrationTaskStatus::Abandoned,
];

const ALL_COMPLETIONS: [CompletionStatus; 4] = [
    CompletionStatus::Success,
    CompletionStatus::Failed,
    CompletionStatus::Cancelled,
    CompletionStatus::Incomplete,
];

#[test]
fn task_status_golden_wire_round_trips() {
    // Persisted `tasks.jsonl` lines carry these exact strings.
    let golden = [
        (OrchestrationTaskStatus::Pending, "pending"),
        (OrchestrationTaskStatus::Running, "running"),
        (OrchestrationTaskStatus::Awaiting, "awaiting"),
        (OrchestrationTaskStatus::Completed, "completed"),
        (OrchestrationTaskStatus::Failed, "failed"),
        (OrchestrationTaskStatus::CancelRequested, "cancel_requested"),
        (OrchestrationTaskStatus::Cancelled, "cancelled"),
        (OrchestrationTaskStatus::TimedOut, "timed_out"),
        (OrchestrationTaskStatus::Abandoned, "abandoned"),
    ];
    assert_eq!(golden.len(), ALL_TASKS.len());
    for (status, wire) in golden {
        assert_eq!(serde_json::to_value(status).unwrap(), json!(wire));
        let back: OrchestrationTaskStatus = serde_json::from_value(json!(wire)).unwrap();
        assert_eq!(back, status);
        assert_eq!(crate::task_status_label(status), wire);
    }
}

#[test]
fn completion_status_golden_wire_round_trips() {
    // Persisted completion-store lines carry these exact strings.
    let golden = [
        (CompletionStatus::Success, "success"),
        (CompletionStatus::Failed, "failed"),
        (CompletionStatus::Cancelled, "cancelled"),
        (CompletionStatus::Incomplete, "incomplete"),
    ];
    assert_eq!(golden.len(), ALL_COMPLETIONS.len());
    for (status, wire) in golden {
        assert_eq!(serde_json::to_value(status).unwrap(), json!(wire));
        let back: CompletionStatus = serde_json::from_value(json!(wire)).unwrap();
        assert_eq!(back, status);
        assert_eq!(status.as_str(), wire);
    }
}

#[test]
fn completion_to_task_is_total() {
    let expected = [
        (
            CompletionStatus::Success,
            OrchestrationTaskStatus::Completed,
        ),
        (CompletionStatus::Failed, OrchestrationTaskStatus::Failed),
        (
            CompletionStatus::Cancelled,
            OrchestrationTaskStatus::Cancelled,
        ),
        (
            CompletionStatus::Incomplete,
            OrchestrationTaskStatus::Failed,
        ),
    ];
    assert_eq!(expected.len(), ALL_COMPLETIONS.len());
    for (completion, task) in expected {
        assert_eq!(OrchestrationTaskStatus::from(completion), task);
        assert!(OrchestrationTaskStatus::from(completion).is_terminal());
    }
}

#[test]
fn task_to_completion_covers_every_task_status() {
    use OrchestrationTaskStatus as T;
    let expected = [
        (T::Pending, None),
        (T::Running, None),
        (T::Awaiting, None),
        (T::CancelRequested, None),
        (T::Completed, Some(CompletionStatus::Success)),
        (T::Failed, Some(CompletionStatus::Failed)),
        (T::Cancelled, Some(CompletionStatus::Cancelled)),
        (T::TimedOut, Some(CompletionStatus::Incomplete)),
        (T::Abandoned, Some(CompletionStatus::Incomplete)),
    ];
    assert_eq!(expected.len(), ALL_TASKS.len());
    for (task, completion) in expected {
        match (CompletionStatus::try_from(task), completion) {
            (Ok(got), Some(want)) => assert_eq!(got, want, "{task:?}"),
            (Err(err), None) => {
                assert!(task.is_live(), "{task:?}");
                assert_eq!(err.target(), "CompletionStatus");
                assert_eq!(err.from_status(), crate::task_status_label(task));
            }
            other => panic!("{task:?}: unexpected {other:?}"),
        }
    }
}

#[test]
fn completion_round_trip_through_task_only_loses_incomplete_cause() {
    for completion in ALL_COMPLETIONS {
        let back = CompletionStatus::try_from(OrchestrationTaskStatus::from(completion)).unwrap();
        match completion {
            // `Incomplete` is carried as `Failed` (the cause is not in the status).
            CompletionStatus::Incomplete => assert_eq!(back, CompletionStatus::Failed),
            _ => assert_eq!(back, completion),
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
    let _: &dyn std::error::Error = &err;
}
