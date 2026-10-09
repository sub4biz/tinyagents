# harness::cancel

Cooperative, runtime-agnostic run cancellation for the recursive harness.

## Why this exists

In the recursive runtime a single [`CancellationToken`] can be shared down a
tree of nested runs — a parent agent, its sub-agents, and their sub-graphs —
so an orchestrator cancels an entire recursion with one `cancel()` call and
every level unwinds at its next safe checkpoint with
`crate::error::TinyAgentsError::Cancelled`.

Cancellation is **cooperative, not preemptive**: a token never aborts a
running future. The agent loop polls `CancellationToken::is_cancelled` at the
same safe checkpoints it uses for steering (before each model call and before
each tool call), and the streaming pipeline races `CancellationToken::cancelled`
against the provider stream. This guarantees a token is never observed in the
middle of a side-effecting tool call or a partially consumed stream chunk.

The module deliberately avoids a heavier dependency such as `tokio-util`: it
needs only `new`/`cancel`/`is_cancelled` plus an async `cancelled().await`
future, implemented directly over an `Arc<AtomicBool>` paired with a
`tokio::sync::Notify` — a few lines over the `tokio` `sync` feature already in
the dependency tree.

## Public surface

- [`CancellationToken`] — the cheap, clonable handle. `new`/`Default` create a
  fresh, never-cancelled token; `cancel` latches it into the cancelled state
  (idempotent, irreversible); `is_cancelled` is a lock-free poll;
  `cancelled().await` resolves once cancellation is (or becomes) requested and
  is cancel-safe for use in a `select!` arm; `child_token` mints a linked
  child (parent cancel cascades down, child cancel never propagates up or
  sideways). `RunContext::child` hands each child run such a linked token.

`CancelState` (the `Arc`-shared inner state) is private; callers only ever
hold a `CancellationToken`.

## Files

| File       | Role                                                              |
| ---------- | -------------------------------------------------------------------- |
| `types.rs` | `CancellationToken` and its private `CancelState` data layout.       |
| `mod.rs`   | `CancellationToken` method implementations (`new`, `cancel`, `is_cancelled`, `cancelled`), `Default`/`Debug` impls. |
| `mod_tests.rs` | Unit tests for construction, cancellation, and cross-clone visibility. |

## Operational constraints

- A token attaches to a run through
  `crate::context::RunContext::with_cancellation`; the default `RunContext`
  carries a fresh, never-cancelled token, so cancellation is strictly opt-in.
- Cancellation is latching: once `cancel()` is called, `is_cancelled()` never
  reverts to `false`. There is no way to "un-cancel" a token — construct a new
  one instead.
- `cancel()` uses `Release` ordering on the flag store and calls
  `notify_waiters()` after the store, so any waiter that wakes and re-checks
  the flag is guaranteed to observe `true`; `is_cancelled()` reads with
  `Acquire`. Preserve that ordering pair if this file is ever touched — it is
  what prevents a missed wake-up race.
- `cancelled()` registers its `Notify` interest *before* re-checking the flag,
  which is what closes the race where a `cancel()` lands between the initial
  check and the park. Any reimplementation of the wait loop must preserve that
  ordering too.
