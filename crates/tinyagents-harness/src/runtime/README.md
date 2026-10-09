# harness::runtime

The harness runtime facade: `AgentHarness`, `RunPolicy`, and the host-driven
invocation entry points that sit on top of the default agent loop
(`crate::agent_loop`).

`AgentHarness<State, Ctx>` is the durable, reusable object a caller builds
once — it owns the model registry, tool registry, middleware stack, and
run policy. The agent loop itself (model ⇄ tool) is implemented as inherent
methods on `AgentHarness` in the sibling `agent_loop` module; this module
owns construction/registration (`mod.rs`), the type definitions (`types.rs`),
and the separate host-driven invocation surface (`agent.rs`) that wires a
`crate::host::HostCapabilities<State>` bundle into a turn.

## Two entry points, kept deliberately separate

- **Explicit-model path** (`AgentHarness::invoke` /
  `invoke_with_status` / `invoke_streaming*`, defined in `agent_loop`): the
  caller already knows which model and tools to use. No host is consulted.
- **Host-driven path** (`AgentHarness::invoke_agent` / `invoke_agent_stream`,
  defined in `agent.rs`): a product host resolves the agent definition,
  composes/screens context, and authorizes tools/model routing before the
  loop runs. This is `runtime::agent`'s reason to exist as a separate module
  rather than folding into `agent_loop`: an embedding SDK call must not
  accidentally acquire product policy merely because a harness happens to
  also be wired for hosted turns.

Both paths converge on the same underlying loop, so the two kinds of runs
share identical event vocabulary, retry/fallback behavior, and limits
enforcement — only how the turn gets *set up* differs.

## Public surface

| Item | File | Role |
| --- | --- | --- |
| `AgentHarness<State, Ctx>` | `types.rs` (impl in `mod.rs`, `agent.rs`) | The builder/facade: registries, middleware, policy, response cache. |
| `RunPolicy` | `types.rs` | Declarative cross-cutting policy: limits, unknown-tool/invalid-args handling, retry, fallback, response format, payload capture, cache policy, empty-response handling. |
| `UnknownToolPolicy`, `InvalidArgsPolicy` | `types.rs` | How the loop recovers from a hallucinated tool name or schema-invalid arguments. |
| `PayloadCapture` | `types.rs` | Opt-in capture of model/tool payloads onto observability events. |
| `InvocationRuntime<State, Ctx>` | `types.rs` | Invocation-local model/tool/middleware overlay a hosted root can attach without mutating the durable harness. |
| `AgentInvocation<State, Ctx>`, `AgentTurnRequest`, `AgentStream` | `agent.rs` | The host-driven turn request, its live execution handle, and the caller-consumable hosted stream. |

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Builder/registration/accessor methods on `AgentHarness` (`register_model`, `push_agent_middleware`, `push_middleware`, `with_response_cache`, …). |
| `types.rs` | Public type definitions: `RunPolicy` and its sub-policies, `PayloadCapture`, `AgentHarness`'s fields, `InvocationRuntime`, and the crate-private `HostInvocationBinding`. |
| `agent.rs` | Host-driven invocation: definition resolution, security screening, context composition, memory/experience recall, the `invoke_agent*` entry points, and post-turn memory/learning/experience finalization. |
| `mod_tests.rs` | Tests for `AgentHarness` construction/registration and `RunPolicy` defaults. |

## Key invariants

- **A hosted child never substitutes its own host policy.** Recursive
  sub-agent calls inherit the parent's exact `HostCapabilities` bundle and,
  when present, its exact `InvocationRuntime` overlay — a missing overlay on
  an authorized child is rejected rather than silently falling back to that
  child's own durable harness.
- **`InvocationRuntime` is invocation-local and never durable.** It lives only
  on a `RunContext` for one invocation tree; it is never stored on the
  reusable `AgentHarness`, keyed by run id, or written to graph/checkpoint
  state (see `types.rs`'s doc on `HostInvocationBinding`).
- **Hosted stream output is sanitized.** `AgentStream` strips raw provider,
  middleware, and tool diagnostics from every error it yields to a hosted
  caller (`sanitize_hosted_stream_item` / `sanitize_hosted_event` in
  `agent.rs`) — the full typed detail still reaches the run's own event sink
  and observability pipeline, just not the public stream.
- **Dropping an `AgentStream` early still finalizes the turn.** Its `Drop`
  impl cancels the run and drives the terminal observer with a partial
  result, so memory/learning/experience sinks still see a (failed) turn
  rather than silently leaking one.
- **Progress delivery to a host is best-effort and bounded.** `ProgressSender`
  in `agent.rs` caps outstanding nonterminal events and always reserves a slot
  for the turn's one terminal event, so a saturated UI socket can drop
  intermediate updates but never the final outcome.
- **`RunPolicy::default()` enables response caching but is inert without a
  cache.** Caching only activates once a `ResponseCache` is attached via
  `AgentHarness::with_response_cache`; the default policy is safe to use
  as-is on a harness with no cache configured.

## Relationship to neighbouring modules

- `crate::agent_loop` implements the actual model⇄tool loop as further
  inherent methods on `AgentHarness`; this module supplies the type it is
  implemented on plus the policy it reads every iteration.
- `crate::host` supplies the capability traits (`ContextComposer`,
  `SecurityGate`, `ModelResolver`, …) that `agent.rs` consults when preparing
  a hosted turn.
- `crate::context::RunContext` carries the crate-private
  `HostInvocationAuthority` for the duration of one invocation tree; only
  `runtime::agent` installs or reads it.
- `crate::subagent` is the recursive delegation boundary: a sub-agent tool
  invocation re-enters `invoke_agent_with_capabilities` /
  `invoke_agent_streaming_with_capabilities` with the parent's inherited
  authority rather than the plain `invoke_agent` entry point.
