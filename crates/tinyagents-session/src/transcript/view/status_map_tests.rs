use super::*;
use serde_json::json;
use tinyagents_tasks::OrchestrationTaskStatus as Task;

const ALL: [TranscriptSubagentStatus; 5] = [
    TranscriptSubagentStatus::Completed,
    TranscriptSubagentStatus::Failed,
    TranscriptSubagentStatus::Incomplete,
    TranscriptSubagentStatus::Interrupted,
    TranscriptSubagentStatus::Running,
];

#[test]
fn transcript_status_golden_wire() {
    // The frontend reads these strings out of the projected transcript.
    let golden = [
        (TranscriptSubagentStatus::Completed, "completed"),
        (TranscriptSubagentStatus::Failed, "failed"),
        (TranscriptSubagentStatus::Incomplete, "incomplete"),
        (TranscriptSubagentStatus::Interrupted, "interrupted"),
        (TranscriptSubagentStatus::Running, "running"),
    ];
    assert_eq!(golden.len(), ALL.len());
    for (status, wire) in golden {
        assert_eq!(serde_json::to_value(status).unwrap(), json!(wire));
    }
}

#[test]
fn transcript_to_task_covers_every_variant() {
    let expected = [
        (TranscriptSubagentStatus::Completed, Task::Completed),
        (TranscriptSubagentStatus::Failed, Task::Failed),
        (TranscriptSubagentStatus::Incomplete, Task::Failed),
        (TranscriptSubagentStatus::Interrupted, Task::Abandoned),
        (TranscriptSubagentStatus::Running, Task::Running),
    ];
    assert_eq!(expected.len(), ALL.len());
    for (status, task) in expected {
        assert_eq!(Task::from(status), task, "{status:?}");
        assert_eq!(
            Task::from(status).is_terminal(),
            status != TranscriptSubagentStatus::Running
        );
    }
}

#[test]
fn task_to_transcript_covers_every_task_status() {
    use TranscriptSubagentStatus as S;
    let expected = [
        (Task::Pending, Some(S::Running)),
        (Task::Running, Some(S::Running)),
        (Task::Awaiting, Some(S::Running)),
        (Task::CancelRequested, Some(S::Running)),
        (Task::Completed, Some(S::Completed)),
        (Task::Failed, Some(S::Failed)),
        (Task::TimedOut, Some(S::Incomplete)),
        (Task::Abandoned, Some(S::Interrupted)),
        (Task::Cancelled, None),
    ];
    assert_eq!(expected.len(), 9); // one row per task status
    for (task, status) in expected {
        match (S::try_from(task), status) {
            (Ok(got), Some(want)) => assert_eq!(got, want, "{task:?}"),
            (Err(err), None) => {
                assert_eq!(err.from_status(), "cancelled");
                assert_eq!(err.target(), "TranscriptSubagentStatus");
            }
            other => panic!("{task:?}: {other:?}"),
        }
    }
}

#[test]
fn transcript_round_trip_only_loses_incomplete_cause() {
    for status in ALL {
        let back = TranscriptSubagentStatus::try_from(Task::from(status)).unwrap();
        match status {
            // `Incomplete` is carried as `Failed`.
            TranscriptSubagentStatus::Incomplete => {
                assert_eq!(back, TranscriptSubagentStatus::Failed)
            }
            _ => assert_eq!(back, status),
        }
    }
}
