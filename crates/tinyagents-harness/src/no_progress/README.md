# harness::no_progress

Foundational no-progress detector: recognises when an agent turn is stuck
re-issuing the same failing (or successful) tool call and hands back a
structured verdict a driver can turn into a corrective nudge or a halt.

## Why this exists

A model that hits an unproductive tool result tends to retry the *identical*
strategy — same tool, same arguments — instead of adapting. Left unchecked,
that only stops when a coarse limit like `RunLimits::max_tool_calls` trips,
burning dozens of wasted round trips first. This module is the reusable,
harness-type-free detector that breaks the pattern early: it tracks recent
`(tool, args) → outcome` across a turn and, on each failure, escalates through
an ordered ladder — keep going, **nudge** the model to change approach, or
**halt** once same-strategy retries are exhausted.

A sibling tracker, [`SuccessfulRepeatTracker`], covers the complementary shape
— a model that keeps *succeeding* at the same no-op call (or cycles through a
short repeating sequence) without making progress.

The successful-repeat side escalates in stages: the first threshold **warns**
(a note for the host to attach to the tool result), a call that keeps
repeating is **blocked** before it executes ([`SuccessfulRepeatTracker::pre_call`]
-> [`CallGate`]), and a second block *of the same call* **halts** (blocks are
counted per call signature and cleared when the call returns a new result).
A block predicts that the call would return what it returned last, so a host
calls `invalidate_predictions_except` (through `RepeatMonitor::record_call`'s
`read_only` flag) after a state-changing call. [`RepeatEscalation`]
sets the gaps; a tracker built with `new` and no escalation halts at the first
threshold, as it always did. Two warning-only detectors ([`PingPongDetector`],
[`ArgumentChurnDetector`]) and a [`PostCompactionGuard`] sit beside it, and
[`RepeatMonitor`] composes all of them for one run behind
[`RepeatProgressConfig`].

All trackers are free of harness types (no `RunContext`, no `Message`) so they
can be unit-tested in isolation. The agent loop drives
`StreamTextStallDetector` on visible streaming output and ends the model call
with non-retryable `GenerationStalled` when it fires. Tool-call trackers remain
host-wired through `after_tool`; see the "Driving this from an `after_tool`
hook" section in `mod.rs` for that contract.

## Public surface

- [`NoProgressTracker`] — holds the identical-failure and any-failure ladder
  state for one turn. `new(identical_halt_threshold)` builds it,
  `record(step, &ToolAttempt) -> NoProgress` feeds one outcome and returns the
  verdict, `reset()` clears all counters (called internally after a halt).
- [`OutcomeFingerprinter`] — pluggable reduction of a tool outcome to the
  identity the trackers compare. The default, [`VolatileSpanNormalizer`],
  blanks timestamps, clock times, 10/13-digit epochs that are the value of a
  time-like key (`ts=`, `"timestamp":`, `updated_at:`; a bare 10-digit number
  is a byte count or an id as often as a clock), measurement durations (`took
  123ms`, `duration=1.2s`), `attempt N` / `retry N of M`, diagnostic `pid N`
  values (while preserving PIDs in process-creation results), and UUIDs in
  request/trace-style fields, and leaves every other
  number alone, including long hex ids such as commit SHAs and checksums, which
  are usually the result itself. An outcome that is *only*
  volatile (a bare commit id or checksum: fewer than four alphanumeric
  characters left outside the spans) is compared verbatim. The normalizer is
  deliberately identity-preserving for explicit state timestamps in
  `event_at`, `eventat`, `created_at`, `updated_at`, and `timestamp` fields;
  this includes quoted JSON forms and whitespace before the colon.
  text-based and does not parse JSON, so a volatile field is only blanked when
  its value matches one of those patterns (a counter or opaque message id
  under an arbitrary key is not). `NoProgressTracker` uses it on the first
  error line (identical-failure rung) and `SuccessfulRepeatTracker` on the
  result; both take a replacement through `with_fingerprinter`, and a host
  that wants the previous byte-for-byte behavior passes a verbatim
  fingerprinter. `NoProgressTracker` fingerprints the complete error message,
  including multiline tails, so changing diagnostic detail on a later line is
  treated as a different failure. For example:

  ```rust
  struct VerbatimFingerprinter;

  impl OutcomeFingerprinter for VerbatimFingerprinter {
      fn fingerprint(&self, outcome: &str) -> String {
          outcome.to_string()
      }
  }

  let tracker = NoProgressTracker::new(3).with_fingerprinter(
      std::sync::Arc::new(VerbatimFingerprinter),
  );
  ```
- [`ClassifiedFailureTracker`] — an additive ledger for equivalent failures
  keyed by class, operation, and resource or permission scope. `record` accepts
  a class-specific recovery budget; `clear` removes one group only after an
  observation shows its blocker changed. Intervening tool calls leave it intact.
  Hold one tracker per turn, or call `reset()` at a new turn boundary. A
  `NoProgress::Halt` verdict does not reset classified counts: retrying the same
  unchanged blocker in a resumed turn would halt again.
- [`ToolAttempt`] — one observed outcome, built with `success`/`failure` plus
  the `hard_reject()`/`recoverable_miss()` modifiers.
- [`NoProgress`] — the verdict enum: `Continue`, `Nudge(String)`,
  `Halt(String)`, with `message()`/`is_nudge()`/`is_halt()`/`as_str()` helpers.
- [`fingerprint_arguments`] — the canonical (key-order-independent) argument
  hash every driver must use so the identical-repeat rung compares correctly.
- [`SuccessfulRepeatTracker`] / [`SuccessfulRepeat`] — the successful-repeat
  counterpart: `record_output`, `record_call_batch`, `record_call_outcome`,
  and `reset`.
- [`RepeatEscalation`] / [`CallGate`] — staged escalation settings (defaults:
  block 2 repeats after the warning, halt on the 2nd block) and the
  before-execution verdict (`Allow` / `Block` / `Halt`);
  `SuccessfulRepeat::Warn` is the first-stage verdict.
- [`PingPongDetector`] (warn at 6 alternating calls) and
  [`ArgumentChurnDetector`] (3 variants x 3 calls, one result) — warning-only.
- [`PostCompactionGuard`] — warning-only: remembers the last 3 calls that were
  already repeating (2+ identical results) before a compaction and flags a
  repeat of one within the next 3 calls. A single re-read of evicted content
  is correct and is never warned about or blocked.
- [`RepeatMonitor`] / [`RepeatProgressConfig`] — one run's composed repeat
  accounting; `RepeatProgressConfig::immediate_halt()` is the legacy preset.
  `SuccessfulRepeat`, `CallGate` and `RepeatProgressConfig` are
  `#[non_exhaustive]`; configure the latter with its `with_*` builders.

- [`StreamTextStallDetector`] — consumes visible text fragments during one
  model call and flags a long run of similarly opened sentences before the
  provider stream finishes.
- Threshold constants: [`DEFAULT_IDENTICAL_HALT_THRESHOLD`],
  [`DEFAULT_REPEAT_OUTPUT_THRESHOLD`], [`DEFAULT_REPEAT_CALL_THRESHOLD`] (all
  re-exported from `crate`).

## Marker on guard-answered results

`RepeatProgressMiddleware` answers a blocked call (and the halting call) without
running the tool. Those results carry `"tinyagents.repeat_guard": "blocked"` or
`"halted"` in `ToolResult::metadata` (`REPEAT_GUARD_METADATA_KEY`,
`REPEAT_GUARD_BLOCKED`, `REPEAT_GUARD_HALTED`; read it with
`repeat_guard_marker(&result)`). Metadata never reaches the model. A host's
repeated-failure middleware should skip results where the marker is present.
The marker is queued at refusal time (`RunContext::set_refusal_metadata`) and
stamped by the loop when it builds the error result, so it is present for
every `after_tool` hook, whatever its registration order. They are the
guard's answer, not the tool failing, and counting them would double-escalate.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | The identical/any-failure escalation ladder (`NoProgressTracker::record`), argument fingerprinting, and the nudge/halt message builders. |
| `successful_repeat.rs` | The successful-repeat streak tracker (`SuccessfulRepeatTracker`) and its private `Streak` helper. |
| `escalation.rs` | `RepeatEscalation`: the block/halt gaps for staged escalation. |
| `loop_patterns.rs` | Warning-only `PingPongDetector` and `ArgumentChurnDetector`. |
| `post_compaction.rs` | `PostCompactionGuard`. |
| `util.rs` | Shared hashing and poison-tolerant locking helpers. |
| `monitor.rs` | `RepeatMonitor` and `RepeatProgressConfig`: the per-run composition a middleware drives. |
| `stream_text/` | Chunk-independent streamed-text stall detector and focused tests. |
| `types.rs` | Public and crate-private type definitions shared by both trackers. |
| `mod_tests.rs`, `escalation_tests.rs` and sibling `*_tests.rs` | Unit tests for the escalation ladder and the trackers. |

## Operational constraints

- `identical_halt_threshold` passed to `NoProgressTracker::new` is clamped so
  it always sits strictly above the nudge threshold — a driver cannot
  accidentally configure a halt that fires before any nudge is given.
- A hard policy rejection (`ToolAttempt::hard_reject`) trips the ladder
  fastest (`HARD_REJECT_HALT_THRESHOLD = 2`), since a blocked call re-issued
  unchanged can never succeed.
- The unknown-tool recovery sentinel (`ToolAttempt::recoverable_miss`) feeds
  the identical-repeat counter but **not** the any-failure backstop, so a
  model that recovers from one bad tool name and then legitimately exhausts
  its budget does not trip the generic backstop early.
- On `NoProgress::Halt`, the tracker resets its own state, so a resumed run
  does not immediately re-trip on latched counters.
