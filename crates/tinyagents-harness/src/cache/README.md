# harness::cache

Response caching, provider prompt/KV-cache layout protection, and stampede
protection for the recursive harness.

## Why this exists

In the recursive runtime the same request can recur many times — a sub-agent
re-asked an identical sub-question, a graph node replayed during recovery, or
a deterministic test driving the loop twice. This module makes that recursion
cheap and deterministic:

1. **Local response cache** — skips a provider call entirely when the harness
   has already seen an identical request.
2. **Provider prompt/KV-cache layout protection** — tracks and reports whether
   a middleware pass preserved the byte-prefix a *provider's own* prompt cache
   depends on.
3. **Stampede protection** — collapses concurrent identical cache misses into
   one provider call.

## Public surface

### Response cache

- [`ResponseCache`] (`types.rs`) — the async trait; only `get`/`put` are
  required, `put_with_ttl`/`clear`/`stats` have sensible defaults.
- [`InMemoryResponseCache`] (`types.rs` + `memory.rs`) — LRU, dual-bounded
  (entry count and approximate byte budget) in-process implementation. Not
  durable across restarts.
- `SqliteResponseCache` (`sqlite.rs`, feature `sqlite`) — durable,
  namespace-scoped SQLite-backed implementation with lazy TTL purge.
- [`CacheStats`], [`CacheSkipReason`] (`types.rs`) — point-in-time counters and
  the reasons a call skipped the cache lookup entirely, so a 0% hit rate is
  always explainable.

### Cache key derivation (`key.rs`)

- [`cache_key`] — request-half key: an explicit allowlist projection of the
  request's behaviour-affecting fields, hashed incrementally per-component.
- [`scoped_cache_key`] — folds in the *resolved* model's cache identity,
  streaming mode, and policy namespace. Always compose the two — a request-half
  key alone carries no model/provider identity.
- [`model_cache_identity`], [`credential_fingerprint`] — build a model's cache
  identity string without ever hashing or logging a raw credential.
- [`PROMPT_CACHE_KEY_OPTION`], [`prompt_cache_key`],
  [`apply_prompt_cache_breakpoints`] — derive and inject a provider-side
  `prompt_cache_key` routing hint from the request's stable prefix. This is not
  the local response-cache identity: [`cache_key`] still hashes the changing
  conversation history, so two turns with one stable provider route cannot
  return the same cached answer by mistake.

### Prompt-cache layout protection

- [`PromptCacheLayout`] (`types.rs` + `layout.rs`) — snapshot of a request's
  ordered cacheable prefix plus a content-aware fingerprint and per-message
  digest chain, so a middleware that rewrites a stable segment's *text* is
  caught, not just one that reorders segment ids.
- A middleware that prepends a System instruction to a request with an explicit
  canonical cache layout must extend and renumber its `cache_segments` (or clear
  the annotation to take the conservative full-request path). The built-in
  dynamic-prompt and prompted structured-output paths share
  `prepend_system_message` for this. Segment ids alone cannot distinguish a
  prepended instruction from a later volatile System history summary.
  A durable session supplies its frozen System-tier count through `RunContext`
  when the harness rebuilds a request on the next invocation; standalone runs
  retain leading-System inference. A recorded zero-prefix session with no tools
  uses a noncacheable marker so its leading System history summary is not
  silently promoted into a reusable prompt tier. If a built-in dynamic or
  structured instruction is later prepended, `prepend_system_message` replaces
  that marker with one canonical stable tier and leaves the summary in history.
  If tools are added instead, dispatch promotes only their schemas to a
  canonical tools-only tier.
- [`CacheLayoutEvent`] (`types.rs` + `layout.rs`) — describes a before/after
  layout change; `under_policy` evaluates it against a `CachePolicy` and is
  what makes `CachePolicy::protect_prompt_prefix` load-bearing.

### Stampede protection

- [`SingleFlight`] (`singleflight.rs`) — collapses concurrent identical model
  calls sharing a cache key into one leader call; followers get a clone of the
  leader's success, but a leader's error is never shared (each follower falls
  back to its own call and its own retry budget).

## Files

| File             | Role                                                               |
| ---------------- | -------------------------------------------------------------------- |
| `mod.rs`         | Module overview and re-exports.                                     |
| `types.rs`       | All public types: `ResponseCache`, `InMemoryResponseCache`, `CacheStats`, `CacheSkipReason`, `PromptCacheLayout`, `CacheLayoutEvent`. |
| `hash.rs`        | Shared deterministic hashing primitives (SHA-256 canonical folding, FNV-1a). |
| `key.rs`         | Response-cache key derivation and provider prompt-cache breakpoint injection. |
| `layout.rs`      | `PromptCacheLayout`/`CacheLayoutEvent` construction and comparison logic. |
| `memory.rs`      | `InMemoryResponseCache` LRU implementation.                         |
| `singleflight.rs`| `SingleFlight` stampede protection.                                  |
| `sqlite.rs`      | `SqliteResponseCache` (feature `sqlite`).                            |
| `mod_tests.rs`   | Unit tests (see its module doc for coverage).                        |

## Operational constraints

- A cache key must always be the two-part composition
  `scoped_cache_key(cache_key(request), model.cache_identity(), streaming, ns)`
  — never the request hash alone. Without the identity half, one
  `Arc<InMemoryResponseCache>` shared between two differently-configured
  harnesses (different provider, model, or credential) can serve one
  harness's answer to the other.
- `cache_key`'s envelope is an **exhaustive** destructure of `ModelRequest`
  (no `..`), so adding a request field is a compile error here until someone
  decides whether it belongs in the key. Preserve that when touching this
  file — it is what stops a new field from silently affecting or silently
  missing cache identity.
- Never fold a raw credential into a cache key, log line, or event — only
  [`credential_fingerprint`]'s truncated digest.
- `PromptCacheLayout` comparisons must use the content-aware fingerprint and
  message-digest chain, not just `prefix_ids` equality — id-only comparison is
  exactly the bug this module exists to prevent (a middleware editing a stable
  segment's text while keeping its id).
- `SingleFlight::run`'s internal lock is never held across an `await`; keep
  any future edit to `claim`/`run` preserving that, or the future becomes
  `!Send` and unusable from `tokio::spawn`.
- `InMemoryResponseCache` is not durable; `SqliteResponseCache` is, and is the
  only backend safe to rely on across a process restart.
