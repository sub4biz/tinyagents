# tool::select

Prompt-driven tool selection: rank a large tool catalogue against a task
prompt and keep only the handful of tools that plausibly matter, so a
sub-agent bound to a large third-party catalogue (GitHub's action catalogue
alone runs to ~500 entries) doesn't get every entry advertised in its prompt
on every turn.

The ranker is pure CPU, stdlib-only, no model load — see the module doc on
`mod.rs` for the full five-stage pipeline (verb detection, verb gate, query
token expansion, weighted token overlap, verb-alignment boost). It is
deliberately explainable rather than clever, since its output decides what a
model is allowed to see.

## Public surface

- `rank_tools_by_prompt(prompt, tools, max_results) -> Vec<usize>` — the
  entry point. Ranks `tools` against `prompt` and returns indices into
  `tools`, best match first. Returns an empty `Vec` for a blank prompt, an
  empty catalogue, or no token hits.
- `MIN_CONFIDENT_HITS` — the minimum hit count a caller should require before
  trusting the ranked result; below it, callers should fall back to the
  unfiltered catalogue rather than risk starving the agent of a needed tool.
- `SelectableTool<'a>` (`types.rs`) — a borrowed `{ name, description }` view
  a host adapts its own tool descriptor into. Named fields on purpose: name
  hits are weighted 3x over description hits, so a transposed tuple would
  silently change the ranking.
- `ToolVerb` (`types.rs`) — the small, stable detected-intent enum
  (`Create`/`Send`/`Read`/`List`/`Update`/`Delete`/`Merge`).

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `rank_tools_by_prompt`, `MIN_CONFIDENT_HITS`, and the full scoring pipeline (verb detection/gate, tokenization, abbreviation expansion, weighted overlap, verb bonus). |
| `types.rs` | `SelectableTool`, `ToolVerb`. |
| `mod_tests.rs` | Unit tests: verb classification, abbreviation expansion, stopword filtering, ranked overlap scoring. |

## Key invariants

- **Verb gating never produces zero results by itself when it can be
  avoided.** A tool with a neutral (unrecognized) name prefix is kept as
  ambiguous rather than dropped, so a catalogue with unconventional naming
  degrades to token overlap instead of returning nothing.
- **`Read`/`List` are treated as compatible verbs** (`verbs_are_compatible`)
  so a search-flavored prompt ("find the emails about X") can still reach a
  `FETCH_*`/`GET_*` action instead of only enumerating ids.
- **Send is inferred from resource nouns only when no conflicting verb is
  already present** — see the `SEND_NOUN_ALIASES` handling in `detect_verbs`;
  this is pinned by a ranking snapshot in the tests and must not be
  simplified to "noun overrides any detected verb."
- Callers own the fallback decision: this module never decides that a
  narrowed selection is "too thin" on its own — it just returns whatever it
  found, and exposes `MIN_CONFIDENT_HITS` as the threshold callers should
  apply.

## Relation to neighbouring modules

`tool::select` is re-exported from `tool/mod.rs` (`pub mod select` / `pub use
select::*`) and has no dependency on the rest of `tool/` or on `tinytools`
beyond the `SelectableTool` view a caller constructs. Its intended consumer
is a host that binds a large third-party tool catalogue (e.g. a sub-agent
scoped to hundreds of external actions) and needs to narrow it before
advertising it to a model; the `tinyagents-graph` crate's todo-dispatch
module (`crates/tinyagents-graph/src/todos/dispatch/`) is the current
caller.
