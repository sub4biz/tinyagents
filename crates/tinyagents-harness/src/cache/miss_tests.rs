use super::*;

fn usage(input: u64, cached: u64) -> Usage {
    Usage {
        input_tokens: input,
        cache_read_tokens: cached,
        ..Usage::default()
    }
}

fn tracker() -> PromptCacheTracker {
    PromptCacheTracker::new(1_000)
}

#[test]
fn the_first_call_of_a_key_has_nothing_to_compare() {
    assert_eq!(tracker().observe("t", "m", &usage(10_000, 0)), None);
}

#[test]
fn a_warm_cache_reports_no_miss() {
    let mut t = tracker();
    t.observe("t", "m", &usage(10_000, 0));
    // The whole previous prompt was read back.
    assert_eq!(t.observe("t", "m", &usage(12_000, 10_000)), None);
    assert_eq!(t.observe("t", "m", &usage(13_000, 12_000)), None);
}

#[test]
fn a_sharp_drop_reports_the_wasted_tokens() {
    let mut t = tracker();
    t.observe("t", "m", &usage(10_000, 9_000));
    let miss = t.observe("t", "m", &usage(12_000, 2_000)).expect("a miss");
    assert_eq!(
        miss,
        PromptCacheMiss {
            expected_cached_tokens: 10_000,
            cached_tokens: 2_000,
            wasted_input_tokens: 8_000,
        }
    );
}

#[test]
fn expected_is_capped_at_the_new_prompt_after_a_shrink() {
    let mut t = tracker();
    t.observe("t", "m", &usage(20_000, 15_000));
    let miss = t.observe("t", "m", &usage(5_000, 0)).expect("a miss");
    assert_eq!(miss.expected_cached_tokens, 5_000);
    assert_eq!(miss.wasted_input_tokens, 5_000);
}

#[test]
fn a_shortfall_within_the_noise_floor_is_ignored() {
    let mut t = tracker();
    t.observe("t", "m", &usage(10_000, 9_000));
    // 1_000 short of the previous prompt: breakpoint granularity, not a miss.
    assert_eq!(t.observe("t", "m", &usage(11_000, 9_000)), None);
    // Exactly at the floor still counts as noise; one token over is a miss.
    assert_eq!(t.observe("t", "m", &usage(12_000, 10_000)), None);
    assert!(t.observe("t", "m", &usage(13_000, 10_999)).is_some());
}

#[test]
fn a_provider_that_never_reports_caching_never_misses() {
    let mut t = tracker();
    t.observe("t", "m", &usage(10_000, 0));
    assert_eq!(t.observe("t", "m", &usage(12_000, 0)), None);
}

#[test]
fn a_total_miss_counts_once_the_provider_has_reported_caching() {
    let mut t = tracker();
    t.observe("t", "m", &usage(10_000, 0));
    t.observe("t", "m", &usage(11_000, 10_000));
    assert!(t.observe("t", "m", &usage(12_000, 0)).is_some());
}

#[test]
fn reset_forgets_a_legitimately_changed_prefix() {
    let mut t = tracker();
    t.observe("t", "m", &usage(10_000, 9_000));
    t.reset("t");
    assert_eq!(t.observe("t", "m", &usage(4_000, 0)), None);
}

#[test]
fn keys_are_independent() {
    let mut t = tracker();
    t.observe("a", "m", &usage(10_000, 9_000));
    assert_eq!(t.observe("b", "m", &usage(12_000, 0)), None);
    assert!(t.observe("a", "m", &usage(12_000, 0)).is_some());
}

#[test]
fn a_response_without_input_tokens_is_not_a_baseline() {
    let mut t = tracker();
    t.observe("t", "m", &usage(10_000, 9_000));
    assert_eq!(t.observe("t", "m", &usage(0, 0)), None);
    assert!(
        t.observe("t", "m", &usage(12_000, 0)).is_some(),
        "the old baseline stands"
    );
}

#[test]
fn the_number_of_tracked_keys_is_bounded() {
    let mut t = tracker();
    for i in 0..(MAX_TRACKED_KEYS + 10) {
        t.observe(&format!("k{i}"), "m", &usage(10_000, 9_000));
    }
    assert!(t.len() <= MAX_TRACKED_KEYS);
}

#[test]
fn a_model_switch_starts_a_fresh_baseline() {
    let mut t = tracker();
    t.observe("t", "model-a", &usage(10_000, 9_000));
    // A different model has its own cache: nothing to compare against.
    assert_eq!(t.observe("t", "model-b", &usage(11_000, 5_000)), None);
    // And the new model's own baseline is then tracked.
    assert!(t.observe("t", "model-b", &usage(12_000, 0)).is_some());
}

#[test]
fn a_first_creation_only_call_is_a_baseline_not_a_miss() {
    let mut t = tracker();
    t.observe("t", "m", &usage(10_000, 0));
    let creation = Usage {
        input_tokens: 12_000,
        cache_creation_tokens: 10_000,
        ..Usage::default()
    };
    assert_eq!(t.observe("t", "m", &creation), None);
    // The baseline is now established: a later total drop is a miss.
    assert!(t.observe("t", "m", &usage(13_000, 0)).is_some());
}
