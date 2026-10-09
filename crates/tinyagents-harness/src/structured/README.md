# harness::structured

Structured output: how a caller gets *typed values* back out of a model call
instead of free-form prose. In the recursive architecture this is the
boundary that turns a model's output into a program input — it underpins
agents that return machine-checkable results or schema-conformant data for the
runtime to consume next.

## Design

Two extraction strategies (`StructuredStrategy`):

| Strategy | How it works |
| --- | --- |
| `ProviderSchema` | Provider returns JSON text (native structured-output/response-format API); parsed from the raw response text. |
| `ToolCall` | An artificial tool is exposed to the model; the structured value is read from that tool call's `arguments`. |

`StructuredStrategy::for_profile` picks between them for `ResponseFormat::Auto`
based on a `ModelProfile`'s advertised capabilities — critically, it only
selects `ToolCall` for a model that can actually call tools, since a
`tool_calling: false` profile falls back to prompt-guided calling where the
wire `tools` array is empty and a forced tool choice can never be answered.

Extraction is not a bare `serde_json::from_str`. Three things happen around
it, each owned by its own submodule:

- **`repair`** climbs a conservative repair ladder (strict → code-fence
  strip → prose slice → relaxed JSON → truncation close) so a fenced, chatty,
  or cut-off answer is recovered instead of ending a run. A rung is accepted
  only when the repaired text parses strictly — it never invents structure.
- **`validate`** checks the parsed value against the declared JSON Schema
  (a supported subset: `type`/union types, `properties`, `required`,
  `additionalProperties: false`, `items`, `enum`), reporting the failing
  instance path rather than silently accepting the wrong shape.
- **`StructuredExtractor::extract_outcome`** returns a `StructuredOutcome`
  instead of `Result`, so a failure is data a caller can repair, re-ask, or
  fall back on, rather than an error that discards the run — mirrors
  LangChain's `include_raw=True`.

## Public surface

- `StructuredStrategy` — `ProviderSchema` / `ToolCall`; `for_profile` picks
  one for `ResponseFormat::Auto`.
- `StructuredExtractor` (`types.rs`, impl in `mod.rs`) — configured with a
  strategy, schema name, and JSON Schema; `extract` (fails the caller) and
  `extract_outcome` (never fails, records the error as data) are the two
  entry points; `schema()` exposes the configured schema.
- `StructuredOutput` — the successful result: the extracted `value` plus,
  when available, the `raw_text` that was parsed. `as_value()` / `parse::<T>()`
  for typed deserialization.
- `StructuredOutcome` — the non-fatal result of `extract_outcome`: `value`
  (`Option`), the preserved `raw: ModelResponse`, and `error: Option<String>`.
- `response_format_for_strategy` — the `ResponseFormat` a `ModelRequest`
  should carry for a given strategy (`JsonSchema` for `ProviderSchema`,
  `Text` for `ToolCall`, since the structure arrives via tool arguments there).
- `JsonRepair` (`repair.rs`) — which rung of the repair ladder produced a
  value (`Strict`/`CodeFence`/`Slice`/`Relaxed`/`Closed`); `as_str()` for
  logging, `is_repaired()` to tell a clean parse from a recovered one.
- `validate::validate_value` — standalone entry point for schema validation,
  used internally by every `extract` call and available for a caller that
  wants to validate independently.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Strategy selection, `StructuredExtractor`/`StructuredOutput` impls, `response_format_for_strategy`. |
| `types.rs` | `StructuredStrategy`, `StructuredOutput`, `StructuredOutcome`, `StructuredExtractor` struct definitions. |
| `repair.rs` | `JsonRepair`, `parse_lenient` and the repair-ladder rungs. |
| `validate.rs` | Local, non-`$ref` JSON Schema subset validator. |
| `mod_tests.rs` | Extraction, strategy-selection, and empty-response tests. `repair.rs` and `validate.rs` have their own `repair_test_tests.rs` and `validate_test_tests.rs`. |

## Key invariants

- **Validation always runs before a value is returned**, for both
  strategies — see `StructuredExtractor::extract`.
- **Repair never invents structure.** Every rung is tried only after a
  strict parse has failed, and a rung is accepted only when its output
  itself parses strictly.
- **`extract` and `extract_outcome` differ only in failure handling**, not in
  what they attempt: `extract_outcome` calls `extract` and turns `Err` into
  `StructuredOutcome { value: None, error: Some(..) }`.
- This module intentionally does not implement general JSON Schema (`$ref`,
  `allOf`/`anyOf`/`oneOf`, numeric/string facets); provider-side validation
  and `harness::tool::schema` handle the request-shaping side, this module
  only validates the response.

## Relation to neighbouring modules

`response_format_for_strategy` supplies the `ResponseFormat` for a
`tinyinference_llm::model::ModelRequest`; `repair::parse_lenient` reuses
`tinytools_agent::repair::json::recover_object`, the same relaxed-JSON repair
tool-call argument parsing already uses, rather than a second divergent
implementation.
