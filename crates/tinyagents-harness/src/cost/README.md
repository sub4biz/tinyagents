# harness::cost

Per-call cost estimation and additive cost accumulation.

## Why this exists

Cost composes the same way the runtime recurses: each model call's cost folds
into its run, and a parent run rolls up the cost of every nested sub-agent and
sub-graph beneath it into one total. [`CostTotals`] is additive
(`+`/`+=`) specifically so that roll-up is a plain sum rather than bespoke
aggregation logic at every recursion boundary.

## Public surface

- [`ModelPricing`] (`types.rs`) — per-token pricing for a model. Every field
  is `Option<f64>`; `None` means the price is unknown, not that the token
  class is free.
- [`CostTotals`] (`types.rs` + `mod.rs`) — an accumulating cost breakdown
  (input/output/cache/reasoning/total). Implements `Add`/`AddAssign` so
  totals can be summed across calls or rolled up a run tree; `total_cost` is
  always recomputed from the components on accumulation, never summed
  independently.
- [`estimate_cost`] (`mod.rs`) — prices a `tinyinference_llm::usage::Usage`
  record against a `ModelPricing` entry, returning a `CostTotals`.

## Files

| File       | Role                                                       |
| ---------- | --------------------------------------------------------------- |
| `types.rs` | `ModelPricing`, `CostTotals`.                                   |
| `mod.rs`   | `CostTotals` arithmetic impls, `estimate_cost`.                  |
| `mod_tests.rs` | Unit tests: per-token-class pricing, missing-price handling, accumulation. |

## Operational constraints

- Providers report `cache_read_tokens` as a *subset* of `input_tokens` (and
  `reasoning_tokens` as a subset of `output_tokens`), never an addition to
  them. `estimate_cost` prices the standard input/output rate only on the
  non-cached/non-reasoning remainder — pricing the full counts *and* the
  cached/reasoning subset separately would double-charge those tokens. Any
  change to the pricing formula must preserve that subtraction.
- `CostTotals::total_cost` must never be set directly outside
  `recompute_total`; every mutation path (`AddAssign`, `estimate_cost`) routes
  through it so `total_cost` cannot drift out of sync with the components.
