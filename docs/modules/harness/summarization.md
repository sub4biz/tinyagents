# Harness Summarization

`tinyagents_harness::summarization` (`crates/tinyagents-harness/src/summarization/`)
keeps long-running conversations and agent loops inside model context limits
while preserving useful provenance. It is the harness's direct answer to
"context rot": context-window-aware gating (`SummarizationPolicy`) decides
*when* a run's transcript has grown large enough to compress, and the
trimming/summarization primitives decide *what* to keep verbatim versus fold
into a summary.

Every policy decision is an explicit, inspectable data type. Nothing here runs
implicitly inside the agent loop — a middleware (`ContextCompressionMiddleware`,
`MicrocompactMiddleware`) or a host calls into this module explicitly and
decides what to do with the result.

For the durable, rule-driven compaction layer built on top of this — cut
points, split-turn summarization, iterative summaries, the durable
`CompactionRecord`, and overflow → compact → retry — see
[`compaction.md`](./compaction.md).

## Source Inspiration

LangChain v1's summarization middleware and short-term-memory docs:

- <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py>
- <https://docs.langchain.com/oss/python/langchain/short-term-memory>
- <https://docs.langchain.com/oss/python/langchain/middleware/built-in>

pi's `compaction.ts`/`overflow.ts` (see `docs/runtime-comparison/pi.md` §4.5)
for the cut-point/overflow-recovery design `compaction.md` documents.

## Core building blocks

- **Token estimation** — `estimate_tokens(text) -> u64` and
  `crate::token_estimation::{estimate_message_tokens, estimate_slice_tokens,
  count_tokens_approximately}` are cheap `chars / 4`-family heuristics, not a
  real tokenizer: good for budget checks with a ~30% error margin, not exact
  accounting.
- **Trimming** (`trim.rs`) is synchronous and LLM-free: `trim_messages`/
  `trim_messages_with` drop messages from a slice according to a
  `TrimStrategy` (`KeepLast`, `KeepFirstAndLast`, `MaxTokens`). Every strategy
  is pairing-safe by default via `TrimOptions`.
- **Summarization** (`Summarizer` trait, `ConcatSummarizer`,
  `SummarizationPolicy`) is async and typically LLM-backed.
  `SummarizationPolicy::plan` decides a count-based split between
  `to_summarize` and `to_keep`; a `Summarizer` condenses the former into a
  `SummaryRecord` carrying `CompressionProvenance`. `Summarizer` also has two
  default methods used by the compaction layer: `summarize_request` (threads
  a previous summary for iterative refinement) and `merge` (reconciles a
  split turn's two half-summaries).
- **Tool-call pairing** (`pairing.rs`) is the structural safety net every cut
  point in this module routes through: a naive length- or token-based cut
  routinely separates an assistant tool-call turn from the tool results
  answering it, producing a transcript OpenAI/Anthropic reject outright.
  `find_safe_cutoff_point` and friends move the cut to preserve pairing (or
  repair an already-broken one).
- **Rendering** (`render.rs`) turns any `Message` — including tool calls,
  tool results, and reasoning blocks `Message::text()` drops — into
  summarizable text, so `ConcatSummarizer`'s default output isn't a column of
  bare role labels for a tool-driven run.
- **Compaction** (`compaction.rs`, `pub mod compaction`) — token-budget cut
  points, split-turn summarization, `before_compaction` hook types,
  `OverflowClassifier`. See [`compaction.md`](./compaction.md).

## Public surface

### Token estimation
- `estimate_tokens(text) -> u64` / `TokenEstimate`.

### Trimming (`trim.rs`)
- `trim_messages(messages, strategy) -> Vec<Message>`.
- `trim_messages_with` / `trim_messages_to_token_budget_with`.
- `TrimStrategy`, `TrimOptions`, `TokenTrimPolicy`, `MessageRole`.

### Summarization
- `Summarizer` (trait, object-safe): `summarize`, `summarize_request`
  (default delegates to `summarize`), `merge` (default: deterministic
  concatenation with a numbered header).
- `SummaryRequest { messages, previous_summary }` — input to
  `summarize_request`.
- `ConcatSummarizer` — deterministic, LLM-free default.
- `SummarizationPolicy` — `from_profile`/`with_context_window`/
  `with_threshold_fraction`; `trigger_budget()`, `should_summarize(messages)`,
  `plan(messages) -> (to_summarize, to_keep)`.
- `SummaryRecord` — `summary: Message` + `provenance: CompressionProvenance`.
- `CompressionProvenance` — `source_ids`, `original_token_estimate`,
  `summary_token_estimate`, `reason`.

### Tool-call pairing (`pairing.rs`, `pub mod pairing`)
- `find_safe_cutoff_point`, `tool_pairing_is_intact`,
  `retract_orphan_tool_calls`, `advance_past_orphan_tools`,
  `is_tool_calling_assistant`.

### Rendering
- `render_message_for_summary(&Message) -> String`.

### Compaction (`compaction.rs`, `pub mod compaction`)
- `find_cut_point`, `CutPoint`, `summarize_with_split`.
- `CompactionContext`, `CompactionDecision`.
- `OverflowClassifier`, `OverflowInfo`, `OverflowProbe`.
- `CompactionReason`, `CompactionRecord`, `CompactionSink` (in `types.rs`,
  re-exported at `summarization::*`).

See [`compaction.md`](./compaction.md) for the full contract.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `estimate_tokens`, `Summarizer for ConcatSummarizer`, `SummarizationPolicy` impls; wires the submodules together. |
| `types.rs` | Public struct/enum/trait definitions, including `Summarizer`, `SummaryRequest`, `CompactionReason`, `CompactionRecord`, `CompactionSink`. |
| `pairing.rs` | Tool-call-pairing-safe cut-point repair. |
| `render.rs` | `render_message_for_summary`. |
| `trim.rs` | `trim_messages`/`trim_messages_with`/`trim_messages_to_token_budget_with`. |
| `compaction.rs` | `find_cut_point`, `summarize_with_split`, `OverflowClassifier`, `CompactionContext`/`CompactionDecision`. |
| `mod_tests.rs`, `compaction/compaction_tests.rs` | Unit test coverage. |

## Key invariants

- **`repair_tool_pairs` (on `TrimOptions`) should stay on.** Turning it off
  restores transcripts providers reject outright.
- **Every cut point in this module — `SummarizationPolicy::plan`'s and
  `find_cut_point`'s — routes through `find_safe_cutoff_point`.** A
  `keep_last` count or a `keep_recent_tokens` budget is therefore a
  *minimum*, not an exact value; `to_keep` is always a provider-acceptable
  slice.
- **`pin_turn_user_message` keeps the turn's assignment out of the
  summary.** When the kept tail has no user message, the most recent one is
  moved to its front verbatim (size-capped), and the persisted
  `CompactionRecord` names it in `details.pinned_user_index`.
- **System messages are never placed in `to_summarize`** and are (by
  default) never dropped by trimming either.
- **`ConcatSummarizer` renders through `render_message_for_summary`, not
  `Message::text()`.**
- This module never calls into the agent loop or decides *when* to invoke a
  `Summarizer` on its own — the caller (the agent loop, a middleware) acts on
  `should_summarize`/`plan`/`find_cut_point`.

## Relation to neighbouring modules

`SummarizationPolicy::from_profile` reads
`tinyinference_llm::model::ModelProfile::max_input_tokens`.
`token_estimation::estimate_slice_tokens` backs whole-slice estimates.
Consumers are `middleware::library::context` (`ContextCompressionMiddleware`,
`MicrocompactMiddleware`) and, when a session is attached, a
`tinyagents_session::entry_tree::SessionCompactionSink` persisting
`CompactionRecord`s as `CompactionEntry`s.
