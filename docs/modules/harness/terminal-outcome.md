# Terminal outcomes and lifecycle events

`TerminalOutcome` is the typed explanation for how an agent run ended. It is
attached to completed and failed run events and is available on `AgentRun`.
The `reason` identifies the specific cause, while `class` provides a stable
coarse-grained category for hosts that need only success, timeout,
cancellation, failure, or suspension.

Outcomes preserve the legacy human-readable message. Timeout outcomes also
record whether the deadline landed before a provider call, during one, or
after a call. Provider-start state is tracked separately so a host can tell a
first-call failure from a preflight failure.

The direct and graph loop drivers publish the outcome on `AgentRun` before
running `after_agent` middleware. This lets middleware observe the same
classification as the terminal event. Session hooks receive the typed outcome
before the legacy terminal notification.

Turn lifecycle events announce appended messages and retract only messages
that were previously announced. Initial input messages are treated as a seed
prefix and are not re-announced on the first turn.

Mutations that are not appends are explicit. `MessageRetracted { index }` is
emitted (highest index first) when an announced message is popped, for example
an unusable assistant reply dropped before a retry; it always precedes the
`MessageAppended` of its replacement. `TranscriptRewritten { len, reason }` is
emitted when the transcript is rewritten in place (a tool-set change folded
into the leading system message); a mirror should resynchronise and count later
`MessageAppended` indices from `len`. Nested tool calls (a tool calling another
tool) produce `ToolStarted` / `ToolCompleted` events with a `parent_call_id`
but no transcript rows, so they never emit `MessageAppended`.

On `RunCompleted` / `RunFailed` the `outcome` field is `Some` for every run this
crate ends; it is `None` only when deserializing journals written before the
field existed.

Scope: the typed outcome is published by both the direct and the graph loop
driver, but the turn and message lifecycle events (`TurnStarted`,
`MessageAppended`, `MessageRetracted`, ...) are emitted by the direct loop only.
The graph driver does not emit them yet, so graph-engine hosts should not rely
on them.
