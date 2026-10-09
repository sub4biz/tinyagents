# harness::steering

Policy-checked, observable orchestrator → sub-agent steering: how a *parent*
in the run tree exerts typed control over a *child* it is currently running,
without killing or restarting it.

This is the mid-run counterpart to `tinyagents_orchestration::subagent::SubAgentSession` reuse
(which resumes a *completed* child) — together they cover both ways an
orchestrator keeps a sub-agent "in play". An orchestrating agent, a human UI,
a graph supervisor, or a test harness attaches a `SteeringHandle` to a run's
`RunContext` and enqueues `SteeringCommand`s on it; the agent loop drains and
applies them at a safe checkpoint (before each model call), never mid-stream
or mid-tool-call.

## Public surface

- `SteeringCommand` — the typed instruction sent to a running loop: `Pause`,
  `PauseWith { reason }`, `Resume`, `Cancel`, `InjectMessage(Message)`,
  `Redirect { instruction }`, `SetMetadata { metadata }`,
  `SwitchModel { model }` (opt-in, see below).
  `Serialize`/`Deserialize` so commands can be logged, transported, and
  replayed. `.kind()` returns the payload-free `SteeringCommandKind`.
- `SteeringCommandKind` — the policy-relevant discriminant of a command;
  `ALL` lists every kind, `as_str()` gives a stable lower-snake-case label for
  logging/events.
- `SteeringPolicy` — an allowlist of permitted `SteeringCommandKind`s.
  `SteeringPolicy::new()` permits nothing (fail-closed default);
  `allow_all()` / `.allow(kind)` grant kinds explicitly. `allow_all()` does
  **not** include `SwitchModel`; use `allow_all_with_model_switch()` (or
  `.allow(SteeringCommandKind::SwitchModel)`).
- `SteeringHandle` — a cloneable, `Arc`-backed handle shared by the sender
  (orchestrator) and receiver (agent loop). `send` enqueues; `drain` empties
  the FIFO queue; `pending`/`is_empty` inspect it; `pause_state`/`is_paused`/
  `resume` read and clear the latched pause independent of the queue.
- `PauseState` — the latched state behind a `SteeringOutcome::Pause`: an
  optional human-readable `reason` and the zero-based `paused_at_checkpoint`
  index.
- `SteeringOutcome` — the control-flow decision from one checkpoint:
  `Continue`, `Pause` (a pause is latched — see the docs on this variant for
  the loop's required follow-up), `Cancel`. `.is_pause()` is a convenience
  check.
- `apply_pending_steering(ctx, messages)` — the single steering checkpoint:
  drains `ctx`'s handle (if any), validates the whole batch against the run's
  policy before applying anything, applies permitted commands to `messages`
  and `ctx.config`, and returns the resulting `SteeringOutcome`.

## Live model switch (`SwitchModel`)

`SteeringCommand::SwitchModel { model }` re-points the run at another model of
the harness's registry. Because it changes cost, rate limits and which provider
receives the transcript, it is **opt-in**: `SteeringPolicy::allow_all()` and
`SteeringHandle::allow_all()` withhold it.

- The checkpoint records the name on the run's handle
  (`SteeringHandle::model_override`); the agent loop applies it as
  `request.model` at the next model-call boundary, **before** the binding is
  resolved. `ModelStarted`/`ModelFailed`, `ctx.model_profile`, the
  cross-provider handoff transform, the host budget estimate and the dialect
  decision therefore all use the new model. It wins over a model a
  `before_model` middleware selected.
- Sticky for the rest of the run; the latest accepted switch wins. The
  in-flight call is never interrupted.
- An unknown, capability-ineligible or retired name, a blank name, or a
  host-routed run is rejected: `Steered { accepted: false }` (plus
  `ModelOverrideSkipped` from the loop) and the run continues on its current
  model; the rejected name is dropped so it is reported once.
- One outcome per command: queuing a switch emits nothing; the model call
  emits `Steered { accepted: true }` once when it first applies the switch, or
  the `accepted: false` above when it rejects it. Replacing an unreported
  switch rejects the superseded switch with `accepted: false`; a switch
  replaced after it was reported, or a switch whose run ends before validation,
  emits no additional outcome.
- Fallback: when the switched model is in `RunPolicy::fallback`, the walk
  continues from its position; when it is not, a failure falls back through the
  whole chain from its head (the original primary's chain). Each fallback
  call's `request.model` is retargeted to the fallback's own name, so adapters
  that honour it never re-ask the model that just failed.
- Precedence: wins over a `before_model` middleware's `request.model`; a
  rejection after middleware restores the request's earlier `request.model`
  (one rejection event, the rejected name never reaches an adapter). The
  `ModelMiddleware` wrap layer can still replace `request.model` afterwards.
- Sticky means sticky: a failing switched model is tried first on every turn
  (a fallback answers one call). A revoked credential is written off for the
  run, so later calls skip it. `PromptCacheGuardMiddleware` does not record a
  layout change at a switch. Host-routed runs always reject.
- Binding a handle to a new root run (`with_steering`) clears a leftover
  switch. `SteeringCommand`/`SteeringCommandKind` are `#[non_exhaustive]`.
- Per-run state: a child's handle (`for_child`) has its own override, so a
  parent's switch never reaches a child and a child's rejection never clears
  the parent's.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: `SteeringCommand`, `SteeringCommandKind`, `SteeringPolicy`, `PauseState`, `SteeringOutcome`, `SteeringHandle` (and its private `SteeringInner`). |
| `mod.rs` | Behavioral code: `SteeringPolicy`/`SteeringHandle` methods and the `apply_pending_steering` checkpoint function. |
| `mod_tests.rs` | Unit tests against `apply_pending_steering` directly, plus integration-style tests driving a full `AgentHarness` run with a `SteeringHandle` attached and asserting both transcript outcome and `AgentEvent::Steered` events. |

## Key invariants

- **Fail-closed by default.** A fresh `SteeringPolicy` permits nothing; a run
  that wants steering must opt in per command kind.
- **Batch validation is atomic.** `apply_pending_steering` checks every
  drained command against the policy *before* applying any of them. A
  disallowed command anywhere in the batch aborts the whole checkpoint with
  `TinyAgentsError::Steering` and leaves the transcript/metadata untouched —
  it does not partially apply commands `0..n` and drop the rest.
- **`Cancel` takes precedence** over every other command in the same batch: it
  is applied and the function returns immediately, ignoring anything queued
  after it.
- **A pause is latched on the `SteeringHandle`, not scoped to a batch.** Once
  latched it survives across checkpoints until a `Resume` arrives — in the
  same batch or any later one. A caller distinguishes "paused" from "model
  produced an empty answer" via `SteeringHandle::pause_state`/`is_paused`
  (`AgentRun::paused` on the loop side, see `crate::middleware::AgentRun`).
- **Delivery is pull-based and checkpoint-scoped.** Commands enqueued via
  `SteeringHandle::send` become visible only at the next checkpoint (before a
  model call) — never mid-stream or mid-tool-call — so steering cannot
  interrupt a side-effecting operation partway through.

## Relation to neighbouring modules

- `crate::context::RunContext::with_steering` attaches a `SteeringHandle` to a
  run; the agent loop calls `apply_pending_steering` at its checkpoint.
- Applied commands emit `crate::events::AgentEvent::Steered` through the
  `RunContext`'s event sink, so steering activity is observable the same way
  as middleware and retry events.
- `SteeringCommand::InjectMessage`/`Redirect` operate on the same
  `tinyinference_llm::message::Message` transcript that `crate::prompt`
  assembles into a `ModelRequest`.
