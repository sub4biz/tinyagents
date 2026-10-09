# harness::host

The host capability traits: the seams a product host implements to supply
identity, memory, security, and routing policy to the generic agent runtime.
See `docs/spec/host-capability-traits-rfc.md` for the design rationale in
full; this README covers the public surface and how the pieces fit together.

The agent runtime (`runtime`/`agent_loop`) is deliberately generic over its
host. It runs the model⇄tool loop but has no opinion on what a memory is,
whether a tool call is permitted, which model a role should use, or what a
connected integration means — those are product decisions, and this module is
where the runtime asks for each of them instead of assuming an answer. It is
the same extension-trait pattern the crate already uses for `ChatModel`,
`Tool`, and middleware, applied to the host boundary rather than the
provider boundary.

## Required vs. optional capabilities

Four capabilities are **required** — without them there is no coherent turn
to run:

- `ContextComposer` — builds the system prompt and any preamble messages.
- `tinyagents_definition::DefinitionRegistry` — resolves agent ids to
  definitions (owned by the `tinyagents-definition` crate, bundled here).
- `SecurityGate` — authorizes tool calls and screens untrusted input.
- `ModelResolver<State>` — chooses which model answers a turn.

The other six are **optional**, expressed as `Option<Arc<dyn …>>` on
`HostCapabilities`:

- `AgentMemory` — durable, host-scoped user memory.
- `BudgetGate` — capacity back-pressure and spend accounting.
- `ProgressSink` — coarse, fire-and-forget turn progress for a UI.
- `LearningSink` — post-turn reflection/distillation hook.
- `ToolOutcomeClassifier` — product judgement on whether a tool result failed.
- `ExperienceStore` — procedural memory of how an agent has performed before.

**Absence is always modelled as `None`, never as an implementation that
errors.** A registered-but-failing capability teaches a model the capability
exists and makes it retry; an absent one is simply not offered. The no-op
implementations shipped alongside most traits (`NoopProgressSink`,
`UnlimitedBudgetGate`, `AllowAllSecurityGate`, …) exist for tests and
embedding, not as a way to fake "unconfigured."

## Public surface

| Trait / type | Module | One-line role |
| --- | --- | --- |
| `HostCapabilities<State>` | `mod.rs` | The bundle handed to a session: four required + six optional capabilities. |
| `ContextComposer`, `StaticContextComposer`, `TurnContextRequest` | `context_composer` | Builds a turn's system prompt and preamble (required). |
| `SecurityGate`, `AllowAllSecurityGate`, `GateDecision`, `ScreenOutcome`, `ToolCallRequest`, `ContentOrigin` | `security_gate` | Authorizes tool calls, screens untrusted text (required). |
| `ModelResolver<State>`, `FixedModelResolver`, `ModelResolveRequest` | `model_resolver` | Chooses the model for a turn (required). |
| `AgentMemory`, `InMemoryAgentMemory`, `MemoryId`, `MemoryItem`, `NewMemory`, `RecallRequest` | `agent_memory` | Durable user memory (optional). |
| `BudgetGate`, `UnlimitedBudgetGate`, `Permit`, `CallEstimate`, `ContextState`, `CompressionHint` | `budget_gate` | Admission, spend accounting, compression advice (optional). |
| `ProgressSink`, `NoopProgressSink`, `RecordingProgressSink`, `ProgressEvent` | `progress_sink` | Coarse turn progress for a UI (optional). |
| `LearningSink`, `NoopLearningSink`, `TurnSummary` | `learning_sink` | Post-turn reflection hook (optional). |
| `ToolOutcomeClassifier`, `ErrorFieldClassifier`, `OutcomeClass` | `tool_outcome_classifier` | Classifies a tool result as success/retryable/permanent (optional). |
| `ExperienceStore`, `InMemoryExperienceStore`, `Experience` | `experience_store` | Procedural memory of prior task attempts (optional). |

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `HostCapabilities<State>` bundle, its builder methods, and the family-wide module doc. |
| `agent_memory.rs` | `AgentMemory` trait, value types, `InMemoryAgentMemory` test double. |
| `budget_gate.rs` | `BudgetGate` trait, `Permit`, estimate/state/hint value types, `UnlimitedBudgetGate`. |
| `context_composer.rs` | `ContextComposer` trait, `TurnContextRequest`, `StaticContextComposer`. |
| `experience_store.rs` | `ExperienceStore` trait, `Experience`, `InMemoryExperienceStore`. |
| `learning_sink.rs` | `LearningSink` trait, `TurnSummary`, `NoopLearningSink`. |
| `model_resolver.rs` | `ModelResolver<State>` trait, `ModelResolveRequest`, `FixedModelResolver`. |
| `progress_sink.rs` | `ProgressEvent`, `ProgressSink` trait, `NoopProgressSink`, `RecordingProgressSink`. |
| `security_gate.rs` | `SecurityGate` trait, `ToolCallRequest`, `GateDecision`, `ScreenOutcome`, `ContentOrigin`. |
| `tool_outcome_classifier.rs` | `ToolOutcomeClassifier` trait, `OutcomeClass`, `ErrorFieldClassifier`. |
| `mod_tests.rs` | Bundle-level tests: the required/optional split and the hand-written `Clone`. |

## Value types are inert

Every value type in this family is `serde` + `std` only — no tokio types, no
storage engines, no host-defined schema types. A host can depend on this
module to write an adapter against these structs without pulling in the rest
of the crate; only the traits themselves are `async_trait` and require an
async runtime to implement.

## Key invariants

- **The runtime never re-ranks or re-filters what a capability returns.**
  Recalled memory (`AgentMemory::recall`), recalled experience
  (`ExperienceStore::recall_for`), and composed preamble
  (`ContextComposer::preamble`) are injected in the order the host returns
  them. Re-ranking would silently override a host's own ordering/access
  decision.
- **No provenance, trust, or taint field flows from the runtime.** Types like
  `NewMemory` and `TurnSummary` carry no such field on purpose — only the host
  knows the transport a piece of text arrived over, and a runtime-supplied
  value would be self-attested and worthless as a security signal.
- **`SecurityGate` is consulted, never cached.** The same tool call may be
  permitted now and refused later because the host's tier, quota, or user
  consent changed; the runtime must ask again every time.
- **`AgentMemory` and `ExperienceStore` are not interchangeable** even though
  they look similar — the former is the *user's* declarative knowledge, the
  latter is the *agent's* procedural performance history. See
  `experience_store.rs`'s module doc for the full comparison table.

## Relationship to neighbouring modules

`runtime::agent` is the sole caller of this family in the crate: it resolves
a `HostInvocationBinding` from a `HostCapabilities<State>`, drives context
composition, security screening, model resolution, and (after the turn
completes) the optional memory/learning/experience sinks. `agent_loop`, by
contrast, knows nothing about hosts — it only sees the resolved model, the
composed transcript, and a security-authorized tool set, which is what keeps
the loop reusable for both hosted and unhosted (explicit-model) callers.
