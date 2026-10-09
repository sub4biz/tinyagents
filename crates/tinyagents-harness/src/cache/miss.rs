//! Prompt-cache miss accounting.
//!
//! A provider prompt cache should read back the whole previous prompt on the
//! next call of the same conversation. When it reads back far less, the
//! difference was re-billed as fresh input: wasted spend that no error ever
//! reports. [`PromptCacheTracker`] compares each call's `cache_read_tokens`
//! with the previous call's prompt size per conversation key and reports the
//! shortfall. Port of pi's `cache-stats.ts` `detectMiss`.
//!
//! Reporting only: nothing here warms, pins or retries a cache.
//!
//! ## Noise floor
//!
//! Providers cache in blocks and place breakpoints at fixed granularity, so a
//! shortfall of up to about a thousand tokens is normal. Anything at or below
//! the floor (default [`DEFAULT_CACHE_MISS_NOISE_FLOOR_TOKENS`]) is ignored.
//!
//! ## What is not a miss
//!
//! * The first call of a key (nothing to compare).
//! * A provider that has never reported cache activity for the key (it does
//!   not cache, or does not say so).
//! * A call answered by a different model than the previous one (each model
//!   has its own cache).
//! * A call after [`PromptCacheTracker::reset`], which the caller uses when the
//!   prefix legitimately changed (compaction, edited system prompt).

use std::collections::{HashMap, VecDeque};

use tinyinference_llm::usage::Usage;

/// Shortfalls at or below this many tokens are breakpoint-granularity noise.
pub const DEFAULT_CACHE_MISS_NOISE_FLOOR_TOKENS: u64 = 1_024;

/// Most conversation keys tracked at once; the oldest is evicted past this.
pub(crate) const MAX_TRACKED_KEYS: usize = 256;

/// A cache read that fell short of the previous prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptCacheMiss {
    /// Prompt tokens that should have been read from cache: the smaller of
    /// the previous call's prompt and this call's prompt.
    pub expected_cached_tokens: u64,
    /// Tokens the provider actually read from cache.
    pub cached_tokens: u64,
    /// `expected_cached_tokens - cached_tokens`: input re-billed at the
    /// uncached rate.
    pub wasted_input_tokens: u64,
}

#[derive(Clone, Debug)]
struct Previous {
    /// The model that answered; a different one has a different cache.
    model: String,
    prompt_tokens: u64,
    /// Sticky: some call of this key reported cache activity.
    reported_cache: bool,
}

/// Per-conversation prompt-cache miss detector. See the module docs.
#[derive(Clone, Debug)]
pub struct PromptCacheTracker {
    noise_floor: u64,
    entries: HashMap<String, Previous>,
    order: VecDeque<String>,
}

impl Default for PromptCacheTracker {
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_MISS_NOISE_FLOOR_TOKENS)
    }
}

impl PromptCacheTracker {
    /// A tracker ignoring shortfalls of `noise_floor` tokens or fewer.
    pub fn new(noise_floor: u64) -> Self {
        Self {
            noise_floor,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Number of keys currently tracked.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no key is tracked.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Forgets `key`'s baseline: its prefix legitimately changed, so the next
    /// call's uncached tokens are new content, not a miss.
    pub fn reset(&mut self, key: &str) {
        if self.entries.remove(key).is_some() {
            self.order.retain(|k| k != key);
        }
    }

    /// Records `usage` (answered by `model`) for `key` and returns the miss it
    /// shows against the key's previous call, if any. A call answered by a
    /// different model than the previous one has no baseline. `usage.input_tokens` is the whole prompt
    /// (cache reads are a subset of it); a call reporting none is ignored and
    /// leaves the baseline alone.
    pub fn observe(&mut self, key: &str, model: &str, usage: &Usage) -> Option<PromptCacheMiss> {
        let prompt_tokens = usage.input_tokens.max(usage.cache_read_tokens);
        if prompt_tokens == 0 {
            return None;
        }
        let cache_activity = usage.cache_read_tokens + usage.cache_creation_tokens > 0;
        let previous = self
            .entries
            .get(key)
            .filter(|prev| prev.model == model)
            .cloned();
        let miss = previous.as_ref().and_then(|prev| {
            // Only a baseline that reported cache activity can be missed: a
            // first creation-only call establishes it, it does not break it.
            if !prev.reported_cache {
                return None;
            }
            let expected = prev.prompt_tokens.min(prompt_tokens);
            let wasted = expected.saturating_sub(usage.cache_read_tokens);
            (wasted > self.noise_floor).then_some(PromptCacheMiss {
                expected_cached_tokens: expected,
                cached_tokens: usage.cache_read_tokens,
                wasted_input_tokens: wasted,
            })
        });
        let entry = Previous {
            model: model.to_string(),
            prompt_tokens,
            reported_cache: cache_activity
                || previous.as_ref().is_some_and(|prev| prev.reported_cache),
        };
        if self.entries.insert(key.to_string(), entry).is_none() {
            self.order.push_back(key.to_string());
            if self.order.len() > MAX_TRACKED_KEYS
                && let Some(oldest) = self.order.pop_front()
            {
                self.entries.remove(&oldest);
            }
        }
        miss
    }
}

#[cfg(test)]
#[path = "miss_tests.rs"]
mod tests;
