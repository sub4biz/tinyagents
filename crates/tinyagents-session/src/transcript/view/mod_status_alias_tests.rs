use super::*;

#[test]
fn transcript_subagent_status_wire_format_is_snake_case() {
    for (status, wire) in [
        (TranscriptSubagentStatus::Completed, "completed"),
        (TranscriptSubagentStatus::Failed, "failed"),
        (TranscriptSubagentStatus::Incomplete, "incomplete"),
        (TranscriptSubagentStatus::Interrupted, "interrupted"),
        (TranscriptSubagentStatus::Running, "running"),
    ] {
        assert_eq!(serde_json::to_value(status).unwrap(), wire);
    }
}

#[test]
#[allow(deprecated)]
fn deprecated_subagent_status_alias_is_the_same_type() {
    let old: SubagentStatus = SubagentStatus::Incomplete;
    let new: TranscriptSubagentStatus = old;
    assert_eq!(new, TranscriptSubagentStatus::Incomplete);
}
