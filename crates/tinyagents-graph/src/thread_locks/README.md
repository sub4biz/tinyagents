# graph::thread_locks

Crate-private per-thread async lock map used to serialize `load → mutate →
put` cycles against durable stores.

## Why this exists

Several stores in this crate (goals, todos, todo runs) key their durable
state by `thread_id` and need read-modify-write cycles on one thread's state
to be atomic even though the store itself has no transactions. The natural
fix is one async mutex per thread id, but a plain
`HashMap<String, Arc<Mutex<()>>>` never releases an entry once a thread id has
been touched, so a long-lived process leaks one mutex (and its key) per
thread id ever seen.

`ThreadLockMap` is that per-thread mutex map with the leak fixed: it stores
[`Weak`](std::sync::Weak) handles instead of owning `Arc`s, so a thread's
mutex is freed as soon as every caller holding it drops its `Arc`. Dead
entries are reclaimed by an amortized sweep on insertion (see the `# Sweep`
note below), so map size tracks recently active threads, not every thread id
ever seen.

## Public surface

This module is `pub(crate)` — it has no crate-external API. Within
`tinyagents-graph`:

- `ThreadLockMap` — the weak-value lock map.
  - `ThreadLockMap::new(what: &'static str)` — creates an empty map; `what`
    names the owner in poisoned-lock panic messages (e.g. `"goal lock
    map"`).
  - `ThreadLockMap::lock_for(&self, thread_id: &str) -> Arc<tokio::sync::Mutex<()>>`
    — returns the mutex for `thread_id`, creating one if none is currently
    live. All concurrent callers for the same `thread_id` receive the same
    `Arc` for as long as at least one of them keeps it alive.

Callers hold the returned `Arc<Mutex<()>>`'s guard for the duration of their
critical section (typically a `load`, mutate, `put` sequence against a
store); the map itself never keeps a mutex alive past the last caller.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `ThreadLockMap` and its private `Inner` bookkeeping (weak-value map + sweep threshold). |
| `mod_tests.rs` | Unit tests: identity of the mutex handed out while held, reclamation of dropped locks, and mutual exclusion under concurrent access. |

## Invariants and operational constraints

- **Correctness under concurrency**: lookup-and-insert is atomic under one
  internal `std::sync::Mutex`, so a caller either finds a live `Weak` it can
  upgrade or is the one that inserts a fresh `Arc` — never both, and never a
  window where two callers each believe they minted the map's only mutex for
  a given `thread_id`.
- **Sweep**: an insertion that brings the map to `sweep_at` entries (starting
  at 16, then roughly doubling the live count after each sweep) retains only
  entries whose `Weak` still upgrades. This keeps the amortized cost of
  reclaiming dead entries O(1) per insertion rather than sweeping on every
  call.
- The map does not itself provide fairness or ordering guarantees beyond
  those of `tokio::sync::Mutex`; it only guarantees that two callers passing
  the same `thread_id` contend on the same mutex.

## Relation to neighbouring modules

`goals::store`, `todos::store`, and `delegation::run`
each hold their own `ThreadLockMap` (constructed with a store-specific `what`
label) and call `lock_for(thread_id)` around their `load → mutate → put`
sequences. This module has no dependency on any of them — it is a generic
concurrency primitive, not aware of goals/todos/runs data shapes.
