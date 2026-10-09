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

Scope: the typed outcome and the turn and message lifecycle events
(`TurnStarted`, `TurnCompleted`, `MessageAppended`) are published by both the
direct loop and the graph loop driver, with the same semantics: input messages
are never announced, each appended message is announced exactly once, and a
nested tool call never produces a `MessageAppended`. The graph driver announces
appends at the same points as the direct loop (before a turn's `ModelStarted`,
after the assistant reply, after a tool batch, and at the end of the run,
including after `after_agent`). `MessageRetracted` and `TranscriptRewritten`
come from direct-loop recovery paths the graph rendition does not implement. The
step-by-step `LoopIter` and `compile_loop` graph announce the same events and
treat the transcript present at their first node activation (any node) as the seed. A node that
interrupts discards its state and re-runs on resume, so the appends it announced
are retracted with `MessageRetracted` first. A model node's turn is closed on an
interrupt; a tools node leaves its turn open, and the re-run closes it with the
real results. A fresh runtime resuming from a checkpoint continues the turn
numbering from the checkpoint and re-opens the in-flight tool turn (without a
second `TurnStarted`).

Known differences from the direct loop: the graph rendition closes a turn before
an output-retry prompt is announced (the direct loop announces the prompt first),
and `LoopIter` / `compile_loop` do not close an open turn when a node errors
(`GraphLoopDriver` does, on every exit).

## `provider_started` and summarizers

`provider_started` is true once any provider call was dispatched, including a
context-window summarizer's. Summarizer calls bypass the run context's dispatch
marker, so the compaction middleware scopes each summarization with a dispatch
tracker (`summarization::dispatch`): the built-in model-backed summarizers (`ModelSummarizer`, `TaskStateSummarizer`)
mark it right before they call their model (host-defined `Summarizer` impls
cannot, since `mark_dispatched` is crate-private; they still count when they
fail with usage), and the middleware then sets the run-wide flag. A
summarizer that fails with no usage after dispatching (a transport error) still
reports `provider_started: true`; one rejected before dispatch (empty input,
validation) does not. Only the run-wide flag is set, so a timeout during the main
model call is still classified by that call's own dispatch.
