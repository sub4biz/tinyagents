//! Tests for the repeat-progress guard and its eviction observer.

use std::sync::Arc;

use serde_json::json;

use super::wrap_up::DEFAULT_CLEARED_PLACEHOLDER;
use super::*;
use crate::context::{RunConfig, RunContext};
use crate::middleware::{Middleware, ToolInvocationIdentity};
use crate::no_progress::{DEFAULT_REPEAT_CALL_THRESHOLD, DEFAULT_REPEAT_OUTPUT_THRESHOLD};
use crate::steering::{SteeringCommand, SteeringHandle};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message as TaMessage};
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall as TaToolCall;
use tinytools::ToolResult as TaToolResult;

fn ctx() -> RunContext {
    let mut ctx = RunContext::new(RunConfig::new("mw-test"), ());
    // Hook-level fixtures create a fresh wrapper for each phase; model those
    // phases as the same logical invocation, as the agent loop does in life.
    ctx.instance_id = 1;
    ctx
}

/// The polling tool the host exempts, as the OpenHuman host does.
fn exempt() -> RepeatExemption {
    Arc::new(|tool| tool == "wait_subagent")
}

/// These tests pin the halt mechanics at the first threshold, so they run with
/// the escalation stages off; `repeat_escalation_tests.rs` covers the staged
/// default.
fn new_mw(handle: SteeringHandle, summary: HaltSummarySlot) -> RepeatProgressMiddleware {
    RepeatProgressMiddleware::new(handle, summary, exempt())
        .with_config(crate::no_progress::RepeatProgressConfig::immediate_halt())
}

fn drain_pause_count(handle: &SteeringHandle) -> usize {
    handle
        .drain()
        .into_iter()
        .filter(|c| matches!(c, SteeringCommand::Pause))
        .count()
}

fn repeated_success_response(tool: &str, args: serde_json::Value) -> ModelResponse {
    let mut response = ModelResponse::assistant("working");
    response.message = AssistantMessage {
        id: None,
        content: vec![ContentBlock::Text("working".to_string())],
        tool_calls: vec![TaToolCall::new("repeat-1", tool, args)],
        usage: None,
        origin: None,
    };
    response.finish_reason = Some("tool_calls".to_string());
    response
}

async fn run_successful_repeat_cycle(
    mw: &RepeatProgressMiddleware,
    tool: &str,
    args: serde_json::Value,
    output: &str,
    error: Option<&str>,
) {
    let mut response = repeated_success_response(tool, args);
    mw.after_model(&mut ctx(), &(), &mut response)
        .await
        .unwrap();
    let mut result = match error {
        Some(error) => TaToolResult::error(error),
        None => TaToolResult::success(output),
    };
    let invocation = ToolInvocationIdentity::new("repeat-1", tool);
    mw.after_tool(&mut ctx(), &(), &invocation, &mut result)
        .await
        .unwrap();
}

// ── #6275: repeats that are not back to back ────────────────────────────

/// One A, B round of two different successful calls, each returning the given
/// output. The adjacent-batch streak restarts on every step of such a cycle.
async fn run_alternating_round(mw: &RepeatProgressMiddleware, a_output: &str, b_output: &str) {
    run_successful_repeat_cycle(mw, "use_skill", json!({"skill": "skills"}), a_output, None).await;
    run_successful_repeat_cycle(
        mw,
        "use_skill",
        json!({"skill": "skills", "tool": "skill_search"}),
        b_output,
        None,
    )
    .await;
}

#[tokio::test]
async fn alternating_identical_successful_calls_halt_on_recurrence() {
    let handle = SteeringHandle::allow_all();
    let summary = Arc::new(std::sync::Mutex::new(None));
    let mw = new_mw(handle.clone(), summary.clone());

    for _ in 0..DEFAULT_REPEAT_CALL_THRESHOLD - 1 {
        run_alternating_round(&mw, "doc", "hits").await;
    }
    assert_eq!(drain_pause_count(&handle), 0);

    run_successful_repeat_cycle(&mw, "use_skill", json!({"skill": "skills"}), "doc", None).await;
    assert_eq!(
        drain_pause_count(&handle),
        1,
        "an alternating call returning the identical result for the third time must halt the run"
    );
    assert!(
        summary
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|text| text.contains("identical result")),
        "the recurrence halt summary should reach the host turn result"
    );
}

#[tokio::test]
async fn alternating_calls_whose_output_changed_do_not_halt() {
    let handle = SteeringHandle::allow_all();
    let mw = new_mw(handle.clone(), Arc::new(std::sync::Mutex::new(None)));
    for i in 0..DEFAULT_REPEAT_CALL_THRESHOLD * 3 {
        run_alternating_round(&mw, &format!("doc-{i}"), &format!("hits-{i}")).await;
    }
    assert_eq!(
        drain_pause_count(&handle),
        0,
        "a re-read with identical arguments whose output changed is progress, not a repeat"
    );
}

#[tokio::test]
async fn alternating_exempt_polling_calls_do_not_halt() {
    let handle = SteeringHandle::allow_all();
    let mw = new_mw(handle.clone(), Arc::new(std::sync::Mutex::new(None)));
    for i in 0..DEFAULT_REPEAT_CALL_THRESHOLD * 3 {
        run_successful_repeat_cycle(
            &mw,
            "wait_subagent",
            json!({"task_id": "t"}),
            "still running",
            None,
        )
        .await;
        let output = format!("status-{i}");
        run_successful_repeat_cycle(&mw, "lookup", json!({"id": 1}), &output, None).await;
    }
    assert_eq!(
        drain_pause_count(&handle),
        0,
        "identical polling results must not feed the recurrence ledger"
    );
}

#[tokio::test]
async fn evicting_a_recorded_result_resets_the_recurrence_ledger() {
    // What the last `before_model` sees for the recorded result `repeat-1`:
    // untouched, blanked by microcompact, or dropped by compression / trim.
    let cases: [(Vec<TaMessage>, usize, &str); 3] = [
        (
            vec![TaMessage::tool("repeat-1", "doc")],
            1,
            "a result still in context keeps counting",
        ),
        (
            vec![TaMessage::tool("repeat-1", DEFAULT_CLEARED_PLACEHOLDER)],
            0,
            "a result blanked by compaction must not count toward a repeat",
        ),
        (
            vec![],
            0,
            "a result dropped by compaction must not count toward a repeat",
        ),
    ];
    for (compacted, expected_pauses, why) in cases {
        let handle = SteeringHandle::allow_all();
        let mw = new_mw(handle.clone(), Arc::new(std::sync::Mutex::new(None)));
        let observer = mw.eviction_observer();
        for _ in 0..DEFAULT_REPEAT_CALL_THRESHOLD - 1 {
            run_alternating_round(&mw, "doc", "hits").await;
        }

        // The next model call: the guard sees the request before the reduction
        // steps run, the observer after them.
        let mut request = ModelRequest::new(vec![TaMessage::tool("repeat-1", "doc")]);
        mw.before_model(&mut ctx(), &(), &mut request)
            .await
            .unwrap();
        let mut request = ModelRequest::new(compacted);
        observer
            .before_model(&mut ctx(), &(), &mut request)
            .await
            .unwrap();

        run_successful_repeat_cycle(&mw, "use_skill", json!({"skill": "skills"}), "doc", None)
            .await;
        assert_eq!(drain_pause_count(&handle), expected_pauses, "{why}");
    }
}

#[tokio::test]
async fn successful_repeat_tracker_halt_maps_to_summary_and_pause() {
    let handle = SteeringHandle::allow_all();
    let summary = Arc::new(std::sync::Mutex::new(None));
    let mw = new_mw(handle.clone(), summary.clone());

    for _ in 0..DEFAULT_REPEAT_CALL_THRESHOLD - 1 {
        run_successful_repeat_cycle(&mw, "lookup", json!({"id": 1}), "ok", None).await;
        assert_eq!(drain_pause_count(&handle), 0);
    }
    run_successful_repeat_cycle(&mw, "lookup", json!({"id": 1}), "ok", None).await;

    assert_eq!(drain_pause_count(&handle), 1);
    assert!(
        summary
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|text| text.contains("successful tool-call batch")),
        "crate halt summary should be preserved for the host turn result"
    );
}

#[tokio::test]
async fn successful_repeat_tracker_resets_failed_and_exempt_batches() {
    let handle = SteeringHandle::allow_all();
    let mw = new_mw(handle.clone(), Arc::new(std::sync::Mutex::new(None)));

    // Distinct outputs keep the run-wide recurrence ledger out of this test: it
    // pins the adjacent-batch streak, which a failure resets.
    for i in 0..DEFAULT_REPEAT_CALL_THRESHOLD - 1 {
        let output = format!("before-{i}");
        run_successful_repeat_cycle(&mw, "lookup", json!({"id": 1}), &output, None).await;
    }
    run_successful_repeat_cycle(
        &mw,
        "lookup",
        json!({"id": 1}),
        "ok",
        Some("temporary failure"),
    )
    .await;
    for i in 0..DEFAULT_REPEAT_CALL_THRESHOLD - 1 {
        let output = format!("after-{i}");
        run_successful_repeat_cycle(&mw, "lookup", json!({"id": 1}), &output, None).await;
    }
    assert_eq!(
        drain_pause_count(&handle),
        0,
        "a failed batch resets the successful-repeat streak"
    );

    for _ in 0..DEFAULT_REPEAT_OUTPUT_THRESHOLD + 1 {
        run_successful_repeat_cycle(&mw, "wait_subagent", json!({"task_id": "t"}), "ok", None)
            .await;
    }
    assert_eq!(
        drain_pause_count(&handle),
        0,
        "polling tools remain exempt from successful-repeat halts"
    );
}

// ── Volatility-aware outcome fingerprinting ─────────────────────────────

#[tokio::test]
async fn identical_results_with_fresh_timestamps_halt_on_recurrence() {
    let handle = SteeringHandle::allow_all();
    let summary = Arc::new(std::sync::Mutex::new(None));
    let mw = new_mw(handle.clone(), summary.clone());
    for i in 0..DEFAULT_REPEAT_CALL_THRESHOLD {
        run_alternating_round(
            &mw,
            &format!("doc fetched at 2026-10-06T12:00:{:02}Z in {}ms", i, 10 + i),
            &format!("hits-{i}"),
        )
        .await;
    }
    assert_eq!(
        drain_pause_count(&handle),
        1,
        "a result that differs only by timestamp and duration is the same result"
    );
}

#[tokio::test]
async fn custom_fingerprinter_is_honored_by_the_middleware() {
    struct Verbatim;
    impl crate::no_progress::OutcomeFingerprinter for Verbatim {
        fn fingerprint(&self, outcome: &str) -> String {
            outcome.to_string()
        }
    }
    let handle = SteeringHandle::allow_all();
    let mw = new_mw(handle.clone(), Arc::new(std::sync::Mutex::new(None)))
        .with_fingerprinter(Arc::new(Verbatim));
    for i in 0..DEFAULT_REPEAT_CALL_THRESHOLD * 2 {
        run_alternating_round(
            &mw,
            &format!("doc fetched at 2026-10-06T12:00:{i:02}Z"),
            &format!("hits-{i}"),
        )
        .await;
    }
    assert_eq!(
        drain_pause_count(&handle),
        0,
        "with a verbatim fingerprinter, fresh timestamps keep each result distinct"
    );
}

#[tokio::test]
async fn distinct_hex_only_results_do_not_halt_on_recurrence() {
    let handle = SteeringHandle::allow_all();
    let mw = new_mw(handle.clone(), Arc::new(std::sync::Mutex::new(None)));
    for (i, sha) in [
        "0123456789abcdef0123456789abcdef01234567",
        "fedcba9876543210fedcba9876543210fedcba98",
        "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        "99999999aaaaaaaa99999999aaaaaaaa99999999",
    ]
    .into_iter()
    .enumerate()
    {
        run_alternating_round(&mw, sha, &format!("hits-{i}")).await;
    }
    assert_eq!(
        drain_pause_count(&handle),
        0,
        "three different commit ids are three different results"
    );
}

#[tokio::test]
async fn a_repeat_warning_is_noted_on_the_run_context() {
    // The staged guard warns on a result before it blocks or halts; the
    // warning is also flagged on the run context so the agent loop can hand
    // reasoning back to a model that repeats itself without it.
    let handle = SteeringHandle::allow_all();
    let summary = Arc::new(std::sync::Mutex::new(None));
    let mw = RepeatProgressMiddleware::new(handle, summary, exempt());
    let mut ctx = ctx();
    assert!(!ctx.take_repeat_noted(), "nothing noted before any call");
    let mut noted_at = None;
    for cycle in 1..=6 {
        let mut response = repeated_success_response("use_skill", json!({"skill": "skills"}));
        mw.after_model(&mut ctx, &(), &mut response).await.unwrap();
        let mut result = TaToolResult::success("doc");
        let invocation = ToolInvocationIdentity::new("repeat-1", "use_skill");
        mw.after_tool(&mut ctx, &(), &invocation, &mut result)
            .await
            .unwrap();
        if ctx.take_repeat_noted() {
            assert!(
                result.output().contains("identical result"),
                "the flag is set on the result that carries the note: {}",
                result.output()
            );
            noted_at = Some(cycle);
            break;
        }
    }
    assert!(
        noted_at.is_some_and(|cycle| cycle > 1),
        "a repeat warning sets the flag once the repeat is noted: {noted_at:?}"
    );
}
