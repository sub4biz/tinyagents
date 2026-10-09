use super::*;

fn request_with(reasoning: Option<ReasoningConfig>) -> ModelRequest {
    ModelRequest {
        reasoning,
        ..ModelRequest::default()
    }
}

#[test]
fn inactive_until_a_dead_call() {
    let fallback = ReasoningFallback::default();
    assert!(!fallback.active());
    let mut request = request_with(Some(ReasoningConfig::effort(ReasoningEffort::High)));
    assert_eq!(fallback.apply(&mut request), None);
    assert_eq!(
        request.reasoning,
        Some(ReasoningConfig::effort(ReasoningEffort::High))
    );
}

#[test]
fn first_death_switches_reasoning_off_for_one_call() {
    let mut fallback = ReasoningFallback::default();
    assert_eq!(fallback.on_dead_call(), 1);
    let mut request = request_with(Some(ReasoningConfig::effort(ReasoningEffort::High)));
    assert_eq!(
        fallback.apply(&mut request),
        Some(Some(ReasoningConfig::effort(ReasoningEffort::High)))
    );
    assert_eq!(
        request.reasoning,
        Some(ReasoningConfig::effort(ReasoningEffort::None))
    );
    fallback.on_live_reply();
    assert!(
        !fallback.active(),
        "one live reply spends a hold-off of one"
    );
}

#[test]
fn hold_off_doubles_per_death_and_always_returns() {
    let mut fallback = ReasoningFallback::default();
    assert_eq!(fallback.on_dead_call(), 1);
    assert_eq!(fallback.on_dead_call(), 2);
    assert_eq!(fallback.on_dead_call(), 4);
    assert_eq!(fallback.on_dead_call(), 8);
    assert_eq!(fallback.on_dead_call(), 16);
    for _ in 0..16 {
        assert!(fallback.active());
        fallback.on_live_reply();
    }
    assert!(
        !fallback.active(),
        "every hold-off is spent by live replies; reasoning always returns"
    );
}

#[test]
fn a_request_with_no_reasoning_config_is_switched_off_too() {
    // A dead call is itself the evidence that the provider reasons on this
    // model, whatever the request said.
    let mut fallback = ReasoningFallback::default();
    fallback.on_dead_call();
    let mut request = request_with(None);
    assert_eq!(fallback.apply(&mut request), Some(None));
    assert_eq!(
        request.reasoning,
        Some(ReasoningConfig::effort(ReasoningEffort::None))
    );
}

#[test]
fn a_request_already_without_reasoning_is_left_alone() {
    let mut fallback = ReasoningFallback::default();
    fallback.on_dead_call();
    let mut request = request_with(Some(ReasoningConfig::effort(ReasoningEffort::None)));
    assert_eq!(fallback.apply(&mut request), None);
}

#[test]
fn a_budget_only_config_is_replaced() {
    let mut fallback = ReasoningFallback::default();
    fallback.on_dead_call();
    let mut request = request_with(Some(ReasoningConfig {
        effort: None,
        budget_tokens: Some(4096),
        summary: None,
    }));
    assert!(fallback.apply(&mut request).is_some());
    assert_eq!(
        request.reasoning,
        Some(ReasoningConfig::effort(ReasoningEffort::None))
    );
}

#[test]
fn a_repeat_note_hands_reasoning_back_mid_hold_off() {
    let mut fallback = ReasoningFallback::default();
    assert!(
        !fallback.on_repeat_note(),
        "nothing to restore while reasoning is on"
    );
    for _ in 0..4 {
        fallback.on_dead_call();
    }
    assert!(
        fallback.active(),
        "eight calls of hold-off after the fourth death"
    );
    assert!(fallback.on_repeat_note());
    assert!(!fallback.active(), "reasoning is back for the next call");
    // The backoff scale stands: the next death holds off for sixteen.
    assert_eq!(fallback.on_dead_call(), 16);
}
