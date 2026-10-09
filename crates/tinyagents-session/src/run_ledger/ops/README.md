# Run-ledger operations

This directory implements the durable run-ledger query and mutation surface.

- `../ops.rs` owns agent-run, workflow-run, telemetry, and event operations,
  and re-exports the agent-team API so existing callers retain the same path.
- `team.rs` owns team, member, task, claim, completion, and release lifecycle
  operations.
- `rows.rs` owns SQLite row decoding and connection-scoped lookup helpers used
  by transactional operations.

The ledger's public types live in `../types.rs`; schema initialization remains
in `../store.rs`. Tests remain in `../mod_tests.rs` because they exercise the
public ledger surface across these implementation modules.
