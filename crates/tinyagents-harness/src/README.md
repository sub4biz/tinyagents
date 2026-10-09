# harness

Provider-neutral model calls, tools, middleware, and streaming — the harness
runtime is where a single model call becomes a recursive system: it runs the
agent loop (model ⇄ tools), and because a whole harness agent can be wrapped
as a tool (the wrapper, `SubAgent`, lives in `tinyagents-orchestration`), an
agent calling a tool *is* an agent calling another agent. Parent/child run
lineage, depth limits, usage/cost roll-up, steering, and cancellation all flow
through the harness.

The crate is intentionally split by feature: each module directory owns one
substantial part of model/tool orchestration. Within a module, type
definitions live in `types.rs`, behavior in `mod.rs`, unit tests in a sibling
`<module>_tests.rs` (`mod_tests.rs` beside a `mod.rs`), and complex modules additionally carry their own `README.md` — see
the module map below.

## Module map

| Module | Concern |
| --- | --- |
| `agent_loop` | The default model-tool-model agent loop — one model call driven to completion, plus tool dispatch. |
| `artifacts` | Filesystem offload for oversized worker artifacts on long-horizon runs. |
| `blocking` (private) | `run_blocking`, the helper that moves synchronous file/DB work off the tokio runtime. |
| `cache` | Prompt, response, and layout caches for repeated/replayed requests. |
| `cancel` | Cooperative, runtime-agnostic run cancellation (`CancellationToken`). See [`cancel/README.md`](cancel/README.md). |
| `capability` | `Capability` bundles (instructions, toolset, middleware, model defaults, exposure, `defer_loading`) composed as one named unit, plus `CapabilityToolSet` / `LoadCapabilityTool` for on-demand loading. |
| `config` | Crate-owned session configuration, decoupled from any host's config schema. |
| `context` | `RunContext` — the unit of recursion: run configuration and runtime context threaded through every nested layer. |
| `cost` | Additive cost accounting (`CostTotals`) that rolls a child run's cost up into its parent. |
| `error` | The crate-wide `TinyAgentsError` type and `Result` alias every fallible surface funnels through. |
| `events` | The typed observability layer (`AgentEvent`, sinks, listeners) — the live in-process event surface. |
| `finish_reason` (private) | Shared helper that recognises provider spellings of an output-cap stop (`length`, `max_tokens`, `MAX_TOKENS`). |
| `handoff` | Progressive-disclosure handoff cache for oversized tool results, keeping bloated payloads out of sub-agent history. |
| `host` | Host capability traits — the seams a host implements to supply product-specific behavior to a generic runtime. |
| `ids` | Identifier newtypes (`RunId`, `CallId`, …) and lifecycle enums used to correlate a recursive run tree. |
| `limits` | Run-scoped limit enforcement (model/tool call caps, wall clock) that keeps recursion bounded. |
| `media` (feature `media`) | `GenerateImageTool` / `GenerateVideoTool`, tools over the `tinyinference-image` / `tinyinference-video` generators. |
| `middleware` | The middleware stack wrapping every level of the recursion identically. See [`middleware/README.md`](middleware/README.md). |
| `model_registry` | Runtime-owned executable model registry, name resolution, and fallback ordering. |
| `multimodal` (feature `multimodal`) | Attachment resolution for `[IMAGE:…]` / `[FILE:…]` markers into model-readable bytes. |
| `no_progress` | Detects a stuck turn (identical failing/successful tool calls) and escalates through a nudge/halt ladder. See [`no_progress/README.md`](no_progress/README.md). |
| `observability` | Durable observability — journals, status stores, sinks — making the live event stream persistent. See [`observability/README.md`](observability/README.md). |
| `prompt` | Prompt assembly — templates and `PromptBuilder` turning runtime values into the final request. |
| `providers` | Model adapters whose behavior depends on TinyAgents-specific prompt dialects (e.g. Claude Code/Agent SDK). |
| `retriever` | Provider-neutral retrieval contracts (`Retriever`) for injecting ranked context into a prompt. See [`retriever/README.md`](retriever/README.md). |
| `retry` | Retry/backoff, model fallback, and rate-limiting policies applied uniformly to every model call. See [`retry/README.md`](retry/README.md). |
| `run_queue` | A generic multi-lane FIFO queue (steer/followup/collect) for messages arriving during an active run. See [`run_queue/README.md`](run_queue/README.md). |
| `runtime` | The harness runtime facade (`AgentHarness`) and invocation-local runtime wiring. See [`runtime/README.md`](runtime/README.md). |
| `steering` | Policy-checked, observable orchestrator → sub-agent steering commands. See [`steering/README.md`](steering/README.md). |
| `store` | Long-term key-value and append-only stream storage backends (`Store`, `AppendStore`, `NamespacedStore`). See [`store/README.md`](store/README.md). |
| `stream` | Higher-level streaming projections (`StreamChunk`, `StreamSink`) from the raw event stream. See [`stream/README.md`](stream/README.md). |
| `structured` | Structured (typed, JSON-schema-validated) output extraction from a model call. |
| `summarization` | Explicit message trimming, summarization, and compression policies (the harness's answer to context rot). |
| `terminal` | `TerminalOutcome`, the typed reason a run ended (completed, halted, limit reached, timeout phase, ...). |
| `testkit` | Deterministic model/tool doubles, event recorder, and trajectory assertions for testing without a live provider. See [`testkit/README.md`](testkit/README.md). |
| `title` | Pure helpers for generating, sanitising and validating conversation thread titles. |
| `token_estimation` | The crate's shared, structurally-complete token estimator (a port of LangChain's `count_tokens_approximately`). |
| `tool` | Harness-side registration and execution support for canonical (`tinytools`) tools. See [`tool/README.md`](tool/README.md). |
| `tools` (feature `builtin-tools`; `tools` is a deprecated alias) | Optional builtin harness tools implementing the canonical `tinytools::Tool` interface. See [`tools/README.md`](tools/README.md). |
| `workspace` | Workspace isolation and sandbox hooks (`WorkspaceIsolation`, `SharedRootWorkspace`, path policy, git helpers). See [`workspace/README.md`](workspace/README.md). |

## How the pieces fit together

1. **Configure**: a host builds a `RunContext` (`context`) carrying its
   `StoreRegistry` (`store`), `ModelRegistry` (`model_registry`),
   `ToolRegistry` (`tool`), limits (`limits`), cancellation token (`cancel`),
   and event sink (`events`).
2. **Run**: `agent_loop` drives the model ⇄ tool cycle for one turn,
   consulting `retry`/`model_registry` on provider failure, `no_progress` and
   `summarization` to keep the loop productive and in-budget, `handoff` to
   keep oversized tool results out of history, and `middleware` around every
   step.
3. **Recurse**: a tool can wrap a whole nested `AgentHarness` invocation
   (`SubAgent` / `SubAgentTool` in `tinyagents-orchestration`), so the same
   loop runs again one level deeper, tracked by `ids`/`limits`/`cost` roll-up.
4. **Observe**: every step emits an `AgentEvent` (`events`); `stream` projects
   those onto consumer-facing `StreamChunk`s, and `observability` durably
   journals them.
5. **Extend**: host-owned `AgentMiddleware` wraps the complete run to load and
   save memory, prepare and clean up workspaces, or attach other product policy.
   The harness keeps only the execution seam.
6. **Test**: `testkit` supplies deterministic doubles for every seam above so
   the whole loop is testable without a live provider.

## Feature flags

Cargo features are crate-local (see `Cargo.toml`): `sqlite`, `storage-drivers`,
`builtin-tools` (with `tools` as a deprecated alias), `multimodal`,
`png-optimize`, `media`, `claude-code` and `langfuse` (both on by default), and
`tracing`. Tracing instrumentation is always compiled in; the `tracing` feature
is a retained no-op for downstream feature forwards.
