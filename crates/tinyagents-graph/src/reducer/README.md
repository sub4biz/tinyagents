# graph::reducer

The deterministic fan-in point of the durable executor: how updates from one
or more active nodes in a superstep are folded back into committed state.

When a superstep runs several active nodes — including concurrent branches
(`GraphBuilder::with_parallel`) or a node that merged results from a subgraph
or sub-agent — `compiled::executor` folds their `NodeResult` updates through
the configured reducer in deterministic active-set order at the step
boundary, so the merged state is reproducible regardless of which branch
finished first (see `compiled::README.md`'s "Sequential vs. parallel steps").
This module supplies the trait definitions and the built-in reducer set;
`channel` builds a richer per-field reducer on top of the same idea for
channel-style state.

## Public surface

### Traits

- `Reducer<T>` — merges two values of the same channel type:
  `reduce(&self, current: T, update: T) -> Result<T>`. Used for channel-style
  state where each key has its own merge policy (see `channel`).
- `StateReducer<State, Update>` — merges a partial `Update` into the whole
  `State`: `apply(&self, state: State, update: Update) -> Result<State>`. This
  is the contract `CompiledGraph` executes against; `GraphBuilder::set_reducer`
  configures it, and `GraphBuilder::overwrite` is shorthand for the
  whole-state-overwrite default.

### Built-in `Reducer<T>` markers

- `OverwriteReducer` — last-value semantics (`update` replaces `current`).
- `AppendReducer` — appends the update vector onto the current vector.
- `SetUnionReducer` — unions the update vector into the current vector,
  skipping duplicates and preserving first-seen order (requires `T: Eq + Hash
  + Clone`).
- `MinReducer` / `MaxReducer` — keeps the smaller/larger of the two values
  (requires `T: PartialOrd`).
- `ClosureReducer<T, F>` — a custom binary reducer backed by an `Fn(T, T) ->
  Result<T>` closure; construct with `ClosureReducer::new`.

### `StateReducer` implementations

- `OverwriteStateReducer` — overwrites whole state with the update; the
  default for `State == Update` graphs (`GraphBuilder::overwrite`).
- `ClosureStateReducer<State, Update, F>` — a custom state reducer backed by
  an `Fn(State, Update) -> Result<State>` closure; construct with
  `ClosureStateReducer::new`.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `Reducer`, `StateReducer` traits; the built-in marker structs. |
| `mod.rs` | `impl` blocks wiring each marker/closure type to its trait. |
| `mod_tests.rs` | Unit tests for every built-in reducer and the closure-backed ones. |

## How it fits together

- `builder::GraphBuilder::set_reducer` accepts any `StateReducer<State,
  Update>`; `compile()` stores it as `Arc<dyn StateReducer<...>>` on the
  frozen `CompiledGraph`.
- `compiled::executor` is the sole caller of `StateReducer::apply`, invoked
  once per completed activation's update at each superstep boundary, in
  active-set order.
- `channel` is a parallel, richer reducer surface: it applies a per-field
  `Reducer<T>` to each named channel and detects same-step concurrent writes
  a plain `StateReducer` would silently overwrite — see `channel/README.md`.

## Operational constraints

- A `Reducer`/`StateReducer` implementation must be a pure, order-sensitive
  merge over exactly the two values it is given — the executor relies on
  applying updates in deterministic active-set order to make parallel-step
  results reproducible; a reducer with side effects or hidden ordering
  dependencies breaks that guarantee.
- `SetUnionReducer`'s duplicate detection and `Min`/`MaxReducer`'s ordering
  both depend on the element type's `Eq`/`Hash`/`PartialOrd` impls matching
  the domain's notion of equality/ordering — a type with a `PartialOrd` that
  disagrees with intended domain order will silently keep the "wrong" value.
