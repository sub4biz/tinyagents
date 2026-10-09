use super::*;
use tinyinference_llm::message::{ContentBlock, Message, ToolMessage};

fn big(n: usize) -> String {
    "x".repeat(n)
}

fn tool_text(message: &Message) -> String {
    message.text()
}

#[test]
fn oversized_results_are_cut_and_say_so() {
    let mut messages = vec![Message::user("hi"), Message::tool("c1", big(10_000))];
    let outcome = truncate_tool_results(&mut messages, 1_000);
    assert_eq!(outcome.truncated, 1);
    assert!(outcome.saved_bytes > 7_000);
    let text = tool_text(&messages[1]);
    assert!(text.len() < 1_500, "len {}", text.len());
    assert!(text.contains("truncated by tool_result_budget"));
    assert_eq!(tool_text(&messages[0]), "hi");
}

#[test]
fn small_results_and_other_roles_are_untouched() {
    let original = vec![
        Message::user(big(10_000)),
        Message::assistant(big(10_000)),
        Message::tool("c1", big(500)),
    ];
    let mut messages = original.clone();
    let outcome = truncate_tool_results(&mut messages, 1_000);
    assert_eq!(outcome.truncated, 0);
    assert_eq!(messages, original);
}

#[test]
fn truncation_is_idempotent() {
    let mut messages = vec![Message::tool("c1", big(10_000))];
    truncate_tool_results(&mut messages, 1_000);
    let once = messages.clone();
    let outcome = truncate_tool_results(&mut messages, 1_000);
    assert_eq!(outcome.truncated, 0);
    assert_eq!(messages, once);
}

#[test]
fn trusted_verbatim_results_are_never_cut() {
    let mut message = Message::tool("c1", big(10_000));
    if let Message::Tool(ToolMessage {
        trusted_verbatim, ..
    }) = &mut message
    {
        *trusted_verbatim = true;
    }
    let mut messages = vec![message.clone()];
    assert_eq!(reducible_tool_result_bytes(&messages, 1_000), 0);
    truncate_tool_results(&mut messages, 1_000);
    assert_eq!(messages[0], message);
}

#[test]
fn reducible_bytes_matches_what_truncation_saves() {
    let mut messages = vec![
        Message::tool("a", big(10_000)),
        Message::tool("b", big(4_000)),
    ];
    let reducible = reducible_tool_result_bytes(&messages, 1_000);
    let outcome = truncate_tool_results(&mut messages, 1_000);
    assert_eq!(outcome.truncated, 2);
    // The estimate ignores the notice the real cut appends, so the two agree
    // to within one notice per block.
    assert!(reducible.abs_diff(outcome.saved_bytes) < 2 * 300);
}

#[test]
fn non_text_blocks_are_preserved() {
    let mut messages = vec![Message::Tool(ToolMessage {
        tool_call_id: "c1".into(),
        content: vec![
            ContentBlock::Json(serde_json::json!({"k": "v"})),
            ContentBlock::Text(big(10_000)),
        ],
        trusted_verbatim: false,
        artifact: Some(serde_json::json!({"a": 1})),
    })];
    truncate_tool_results(&mut messages, 1_000);
    let Message::Tool(tool) = &messages[0] else {
        panic!("tool message expected");
    };
    assert_eq!(tool.tool_call_id, "c1");
    assert!(matches!(tool.content[0], ContentBlock::Json(_)));
    assert!(tool.artifact.is_some());
}

#[test]
fn older_truncation_spares_results_after_the_last_assistant_message() {
    let mut messages = vec![
        Message::user("go"),
        Message::assistant("calling"),
        Message::tool("old", big(10_000)),
        Message::assistant("calling again"),
        Message::tool("new-a", big(10_000)),
        Message::tool("new-b", big(10_000)),
    ];
    let outcome = truncate_older_tool_results(&mut messages, 1_000);
    assert_eq!(outcome.truncated, 1);
    assert!(tool_text(&messages[2]).contains("truncated by tool_result_budget"));
    assert_eq!(tool_text(&messages[4]).len(), 10_000);
    assert_eq!(tool_text(&messages[5]).len(), 10_000);
}

#[test]
fn older_truncation_with_no_assistant_message_cuts_nothing() {
    let mut messages = vec![Message::tool("c", big(10_000))];
    assert_eq!(
        truncate_older_tool_results(&mut messages, 1_000).truncated,
        0
    );
}

#[test]
fn a_cap_below_the_notice_floor_is_normalized_for_the_estimate_too() {
    // Blocks no larger than the effective floor are neither cut nor counted.
    let floor = TRAILER_RESERVED + 1;
    let mut messages = vec![
        Message::tool("c1", big(floor)),
        Message::tool("c2", big(floor)),
    ];
    let before = messages.clone();
    assert_eq!(reducible_tool_result_bytes(&messages, 10), 0);
    assert_eq!(truncate_tool_results(&mut messages, 10).truncated, 0);
    assert_eq!(messages, before);
    // A block above the floor is counted against the floor, not the raw cap.
    let big_block = vec![Message::tool("c3", big(floor + 1_000))];
    assert_eq!(reducible_tool_result_bytes(&big_block, 10), 1_000);
}

#[test]
fn a_result_that_only_mentions_the_marker_is_still_cut() {
    let spoof = format!("truncated by tool_result_budget {}", big(10_000));
    let mut messages = vec![Message::tool("c1", spoof)];
    assert!(reducible_tool_result_bytes(&messages, 1_000) > 8_000);
    assert_eq!(truncate_tool_results(&mut messages, 1_000).truncated, 1);
}
