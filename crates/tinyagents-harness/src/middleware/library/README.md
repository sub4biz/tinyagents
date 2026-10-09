# harness::middleware::library

The built-in middleware catalog that ships with the harness. Every type here
is re-exported through `crate::middleware`, so callers write
`tinyagents_harness::middleware::RetryMiddleware`, not
`middleware::library::RetryMiddleware`.

See the parent [`middleware/README.md`](../README.md) for the lifecycle and
around-agent/model/tool extension shapes and the onion-ordering rules these
implementations build on. A handful of other
built-in middleware (`LoggingMiddleware`, `MessageTrimMiddleware`,
`ContextCompressionMiddleware`, `MicrocompactMiddleware`,
`PromptCacheGuardMiddleware`, `UsageAccountingMiddleware`) live directly in
`middleware/types.rs` instead of here — see that module's README for why.

## Public surface

Grouped by extension shape:

### Resilience (`ModelMiddleware`, wrap around the real model call)

- `RetryMiddleware` — retries the wrapped model call on a
  [retryable][crate::retry::is_retryable] error while the configured
  `RetryPolicy` still permits another attempt. Computes backoff but does not
  sleep by default (see `RetryMiddleware::backoff_for_attempt`), keeping tests
  deterministic.
- `TimeoutMiddleware` — races the wrapped call against `tokio::time::timeout`;
  returns `TinyAgentsError::Timeout` on elapse.
- `ModelFallbackMiddleware` — on a retryable failure, retries against an
  ordered chain of fallback model names, emitting `AgentEvent::FallbackSelected`
  before each attempt.
- `RateLimitMiddleware` — gates calls through a shared `RateLimiter`
  (token bucket); either fails fast (`RateLimitBehavior::Error`) or polls until
  capacity frees up (`RateLimitBehavior::Wait`), with an injectable clock for
  deterministic tests.

### Tool policy / selection (`Middleware` lifecycle hooks)

- `ToolAllowlistMiddleware` — rejects `before_tool` calls whose name is not on
  a fixed allowlist.
- `ToolPolicyMiddleware` — enforces the structured `ToolPolicy` metadata each
  tool advertises (classification, side-effect denylist, background-safety,
  sandbox requirement, approval requirement, result-byte cap) at both
  model-visible exposure time (`before_model`) and execution time
  (`before_tool`). `ToolPolicyMiddleware::strict` gives a fail-closed baseline.
- `DynamicToolSelectionMiddleware` — filters model-visible tools via a plain
  `ToolPredicate(&ToolSchema) -> bool`.
- `ContextualToolSelectionMiddleware` — filters model-visible tools via a
  `ContextualToolPredicate` that also sees run context (depth, tags, requested
  model). `from_lists`/`inheriting` build one from allow/deny name lists, with
  `inheriting` composing a child policy on top of a parent's so a delegated
  sub-agent can only narrow, never widen, tool exposure.
- `HumanApprovalMiddleware` — raises `TinyAgentsError::Interrupted` from
  `before_tool` for flagged tools unless an `ApprovalFn` admits the call; the
  harness-native half of a human-in-the-loop gate.

### Budget (`Middleware` lifecycle hooks)

- `BudgetLimits` / `BudgetTracker` / `BudgetSpend` — declarative token/cost
  limits, a shared accumulating spend tracker (clone to roll up spend across a
  parent and its sub-agents), and a point-in-time snapshot.
- `BudgetMiddleware` — preflights each call in `before_model` (reserving
  estimated input tokens under a single lock so concurrent runs on a shared
  tracker cannot collectively overshoot), reconciles actual usage/cost in
  `after_model`, and releases the reservation in `on_error` so a failed call
  never leaks it. Emits `AgentEvent::BudgetReserved`, `BudgetReconciled`,
  `UsageRecorded`, `CostRecorded`, `BudgetWarning`, and `BudgetExceeded`.
- `TurnClock` / `TurnClockMiddleware` — a reading of the run's wall-clock
  budget (the tighter of the run's own deadline and the host's
  `RunPolicy::limits.max_wall_clock_ms`), and an `after_tool` hook that appends
  `[turn budget: 32m elapsed / 28m remaining]` to a tool result once per tenth
  of the budget past half of it. The note rides a tool result (durable,
  never rewritten) rather than a per-request system message, so it does not
  churn the prompt-prefix cache. Pair with `RunPolicy::tool_result_durations`,
  which appends `[took 12.3s]` to each executed tool row.

### Observation (`Middleware` lifecycle hooks)

- `StructuredOutputValidatorMiddleware` — validates an `after_model` response
  against an expected `ResponseFormat` (JSON object or provider-schema
  extraction), failing with `TinyAgentsError::StructuredOutput`.
- `DynamicPromptMiddleware<State, Ctx>` — derives an optional system prompt
  from application state and `RunConfig` on each call via a `PromptFn`,
  inserting it at the front of `request.messages`.
- `RedactionMiddleware` — replaces configured literal-substring patterns with
  a mask string across model response text/JSON, tool-call arguments, raw
  provider payloads, tool results, and tool errors. Single-pass, idempotent
  (never matches inside its own mask output).
- `TracingMiddleware` — implements every lifecycle hook, recording a bounded
  ring buffer of `PhaseTrace` begin/end records plus per-phase `TraceCounts`,
  independent of the event stream.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: middleware structs, their config/outcome types (`RateLimitBehavior`, `BudgetLimits`, `ToolSelectionContext`, `PhaseTrace`, ...), and type aliases (`NowFn`, `ToolPredicate`, `ContextualToolPredicate`, `ApprovalFn`, `PromptFn`). |
| `mod.rs` | Re-exports `types`, declares the `resilience`/`budget`/`tool_policy`/`context`/`observe` submodules, and documents cross-cutting testability guarantees. |
| `resilience.rs` | Constructors and `ModelMiddleware` impls for `RetryMiddleware`, `TimeoutMiddleware`, `ModelFallbackMiddleware`, `RateLimitMiddleware`. |
| `budget.rs` | Constructors and `Middleware` impl for `BudgetMiddleware`, plus `BudgetTracker`/`BudgetLimits` helper methods and the shared input-token estimator. |
| `tool_policy.rs` | Constructors and `Middleware` impls for `ToolAllowlistMiddleware`, `ToolPolicyMiddleware`, `DynamicToolSelectionMiddleware`, `ContextualToolSelectionMiddleware`, `HumanApprovalMiddleware`. |
| `context.rs` | Constructors and `Middleware` impls for `MessageTrimMiddleware`, `ContextCompressionMiddleware`, `MicrocompactMiddleware`, `PromptCacheGuardMiddleware` — note these structs are *defined* in `middleware/types.rs`, not here; this file only holds their behavior. See `middleware/README.md`. |
| `context/overflow.rs`, `context/summary.rs` | Child modules of `context.rs`: overflow-recovery attempts (compact, route, truncate) and how a compaction batch becomes a summary (split-turn prefix, file lists). Behaviour documented in `docs/modules/harness/compaction.md`. |
| `observe.rs` | Constructors and `Middleware` impls for `StructuredOutputValidatorMiddleware`, `DynamicPromptMiddleware`, `RedactionMiddleware`, `TracingMiddleware`, and (also structs defined in `middleware/types.rs`) `LoggingMiddleware`/`UsageAccountingMiddleware`. |
| `mod_tests.rs` | Unit tests for every middleware in this directory (construction, hook behavior, event emission, edge cases like poisoned mutexes and concurrent budget reservations). |

## Operational constraints

- **No middleware here sleeps uncontrollably on the wall clock in tests.**
  `RetryMiddleware` only sleeps when its `RetryPolicy` opts in via
  `with_backoff_sleep` (off by default); `TimeoutMiddleware` is exercised under
  `tokio::time` paused-time tests; `RateLimitMiddleware` takes an injectable
  clock and configurable poll interval. Preserve this when adding new
  middleware: prefer computing a delay over unconditionally awaiting one.
- **Retry engines are coordinated, not yet unified (R-3, partial).**
  `invoke_model_resolving` (the loop's own base-call attempt engine),
  `RetryMiddleware`, and `ModelFallbackMiddleware` each still implement their
  own attempt loop. Double-retrying is prevented by
  `ModelMiddleware::overrides_retry` — a registered `RetryMiddleware` (or any
  middleware overriding it) makes the base call skip its own retry loop
  entirely, so attempts are bounded by whichever engine actually runs, never
  by their product (`middleware::library::test::
  retry_middleware_correlates_retry_scheduled_with_the_loops_call_id` and
  `agent_loop::test::
  retry_middleware_and_run_policy_retry_do_not_multiply_attempts` cover this).
  `RetryMiddleware` also mirrors the loop's own call id onto
  `RunContext::active_model_call` so its `RetryScheduled` events correlate
  with the `ModelStarted`/`ModelCompleted` pair for the same attempt. Making
  the loop's engine the *only* one and turning these middlewares into pure
  policy overrides (`code-review-harness.md` R-3) is still open — it changes
  their semantics from "execute" to "configure" and touches every test that
  exercises them directly via `MiddlewareStack::run_wrapped_model`.
- **Budget reservations are keyed by `RunContext::instance_id`, not `run_id`.**
  Concurrent runs sharing a `BudgetTracker` may share a caller-supplied
  `run_id`; only the process-unique instance id keeps each run releasing
  exactly what it reserved.
- **Tool-policy and tool-selection middleware only change what the model
  *sees* in `before_model`.** They do not by themselves guard execution — pair
  them with `ToolAllowlistMiddleware`/`ToolPolicyMiddleware`'s `before_tool`
  enforcement (or your own) if a model can call a tool it was never shown.
- **Wrap middleware in `resilience.rs` are responsible for their own retry
  budget.** `MiddlewareStack` does not cap how many times a `ModelMiddleware`
  calls `next.run(..)`.
- **Per-call guidance goes through `push_ephemeral_instruction`.** Middleware
  that adds a note to one request (a contents list, a nudge) should not push a
  raw `Message::system`: on a model whose profile sets
  `hoists_system_messages` (DeepSeek moves every system turn to the prompt
  head) a new system message rewrites the cached prefix. The helper reads the
  target profile the loop previews onto `RunContext::model_profile` and, for
  such a model, appends the text to the tail tool result or user turn under a
  `[harness note]` header instead.
