use super::*;
use serde_json::json;

fn all_kinds() -> Vec<(SubagentOutcomeKind, serde_json::Value)> {
    vec![
        (SubagentOutcomeKind::Completed, json!("Completed")),
        (SubagentOutcomeKind::Cancelled, json!("Cancelled")),
        (
            SubagentOutcomeKind::Incomplete(
                SubagentIncomplete::new("out of budget").with_kind(IncompleteKind::BudgetExceeded),
            ),
            json!({"Incomplete": {"reason": "out of budget", "kind": "budget_exceeded"}}),
        ),
    ]
}

#[test]
fn outcome_kind_wire_format_is_unchanged_by_the_rename() {
    for (kind, wire) in all_kinds() {
        assert_eq!(serde_json::to_value(&kind).unwrap(), wire);
        let back: SubagentOutcomeKind = serde_json::from_value(wire).unwrap();
        assert_eq!(back, kind);
    }
}

#[test]
fn awaiting_input_wire_key_is_unchanged_by_the_rename() {
    let kind = SubagentOutcomeKind::AwaitingInput(SubagentPause {
        reason: "need input".to_string(),
        resume: SubagentResume::default(),
    });
    let wire = serde_json::to_value(&kind).unwrap();
    assert!(wire.get("AwaitingInput").is_some(), "{wire}");
    let back: SubagentOutcomeKind = serde_json::from_value(wire).unwrap();
    assert_eq!(back, kind);
}

#[test]
#[allow(deprecated)]
fn deprecated_subagent_status_alias_is_the_same_type() {
    let old: SubagentStatus = SubagentStatus::Completed;
    let new: SubagentOutcomeKind = old.clone();
    assert_eq!(new, SubagentOutcomeKind::Completed);
    assert!(matches!(
        SubagentStatus::Incomplete(SubagentIncomplete::new("x")),
        SubagentOutcomeKind::Incomplete(_)
    ));
}
