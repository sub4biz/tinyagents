//! Compiled-graph rendition of the harness agent loop (A5).
//!
//! `docs/runtime-comparison/langgraph.md` §4 ("Agent loop as a graph") and
//! `docs/runtime-comparison/pydantic-ai.md` §4 (the `iter` API) both ask the
//! same question of tinyagents: can the default `plan -> model -> tools`
//! agent loop — normally a monolithic Rust function
//! (`tinyagents_harness::agent_loop::run_loop`) — also be expressed as an
//! ordinary [`crate::CompiledGraph`], so it gets the graph runtime's
//! checkpoint/resume, step-by-step `iter()`, and interrupt machinery for
//! free, instead of each of those being reinvented (or left unavailable) on
//! the harness's own loop?
//!
//! This module is "yes": [`compile_loop`] builds a real
//! `CompiledGraph<LoopState, LoopUpdate>` with four nodes
//! (`plan`/`model`/`tools`/`settle`, see [`types::node`]), and
//! [`AgentLoopGraphExt::iter`] drives it step-by-step through [`LoopIter`].
//! [`GraphLoopDriver`] additionally plugs the same phase logic into
//! [`tinyagents_harness::runtime::AgentHarness::invoke`] (and friends) via
//! [`tinyagents_harness::agent_loop::phases::LoopDriver`] +
//! [`tinyagents_harness::runtime::AgentHarness::with_loop_driver`], selected
//! by [`tinyagents_harness::runtime::RunPolicy::execution`]`::Graph`.
//!
//! # Dependency direction
//!
//! `tinyagents-graph` depends on `tinyagents-harness`, never the reverse, so
//! this compiled-graph rendition lives here rather than in the harness
//! crate — the harness only exposes the seam
//! ([`tinyagents_harness::agent_loop::phases`]) this module plugs into. See
//! that module's doc comment for the harness-side half of the boundary.
//!
//! # Scope
//!
//! This is a **subset** of `run_loop`'s behavior, not a byte-for-byte
//! reimplementation. It covers the common path exercised by
//! `crates/tinyagents-integration-tests/tests/loop_as_graph.rs`: tool
//! calling, structured output (`ResponseFormat::Auto`/`JsonSchema`, provider-
//! schema and tool-call-fallback strategies), the output-validation retry
//! loop (A3), run limits, `MiddlewareControl` routing (`JumpTo`,
//! `StopWithFinal`, `Interrupt`, `UpdateState`), and steering
//! (cancel/pause/inject). It intentionally does **not** cover: host-model
//! routing (`HostCapabilities`), cross-provider handoff transforms
//! (`agent_loop::handoff_transform`), the deferred-tool discovery bridge,
//! truncated-empty-response recovery/retry, `RunPolicy::retry`/`fallback`
//! (a registered `ModelMiddleware` still runs, so a retry-on-error
//! middleware still applies, but the built-in retry/fallback loop the direct
//! model call performs does not), response caching, and
//! `StructuredStrategy::Prompted`/`ToolCallUnion` (A6's
//! `structured_strategy_override`) or `EndStrategy::Early`/`Exhaustive` (A6);
//! `resolve_structured_plan` also resolves the profile-driven `Auto` choice
//! against the *default* model binding rather than the turn's actually-
//! resolved model (see that function's docs). [`tinyagents_harness::runtime::RunPolicy::execution`]
//! defaults to [`tinyagents_harness::runtime::LoopExecution::Direct`], so
//! every existing caller is unaffected unless it opts in.
//!
//! # Lifecycle events
//!
//! The node bodies announce `TurnStarted`, `TurnCompleted` and
//! `MessageAppended` at the same points as the direct loop, through
//! [`tinyagents_harness::agent_loop::phases`]'s `lifecycle_*` functions: pending
//! appends and the turn open right before `ModelStarted`, the assistant reply
//! after the model call, the tool results (and the turn close) after a tool
//! batch, and the final close in `settle`. The transcript present at the first
//! `plan` activation is the seed and is never announced. Nested tool calls never
//! reach the transcript, so they never produce `MessageAppended`.
//!
//! # Tool batch execution shape
//!
//! The `tools` node runs a turn's whole tool-call batch in one node
//! activation via [`tinyagents_harness::agent_loop::phases::execute_tool_batch`]
//! rather than one graph node (or one `Send` fan-out branch) per tool call.
//! This was the deliberate choice over a per-call fan-out: the direct loop's
//! serial-admission / serial-or-concurrent-dispatch decision (see
//! `tinyagents_harness::agent_loop::tools`'s module docs) is genuinely
//! call-count- and middleware-dependent, and re-deriving it as graph
//! topology would either have to duplicate that decision as a router (two
//! sources of truth to keep in sync) or lose the exact ordering/budget
//! guarantees the direct loop promises. Reusing the harness function as-is
//! keeps the tool batch's ordering, concurrency-eligibility, and budget/limit
//! semantics identical to the direct loop by construction, at the cost of a
//! coarser graph (a `tools` node activation is opaque to the executor's own
//! per-task checkpointing — a batch is atomic from the graph's point of
//! view, resuming re-runs the whole batch rather than only its unfinished
//! calls).
//!
//! # `GraphLoopDriver` vs. `compile_loop`/`LoopIter`
//!
//! These are two different integration points sharing the same node bodies
//! (`runtime::plan_node`/`model_node`/`tools_node`/`settle_node`), not one
//! layered on the other:
//!
//! - [`compile_loop`]/[`LoopIter`] build and step a real
//!   `CompiledGraph<LoopState, LoopUpdate>` over an owned [`runtime::LoopRuntime`]
//!   (`Arc`'d harness/state, owned `RunContext`) — this is the path that gets
//!   checkpointing, `resume`, and `iter()`.
//! - [`GraphLoopDriver`] is handed only a transient `&mut RunContext`/`&mut
//!   AgentRun`/`&mut HarnessRunStatus` bundle by
//!   [`tinyagents_harness::agent_loop::phases::LoopDriver::drive`] (that
//!   trait's signature mirrors `run_loop`'s own borrowed-state contract) and
//!   an `&AgentHarness`/`&State` it does not own. Building a `LoopRuntime`
//!   (which needs `Arc<AgentHarness<State, Ctx>>` to satisfy the `'static`
//!   bound `GraphBuilder::add_node`'s closures require) from a bare `&AgentHarness`
//!   is not possible in safe Rust without either forcing every caller of
//!   `with_loop_driver` to already hold an `Arc<AgentHarness>` (which would
//!   make installing a driver on a not-yet-`Arc`'d harness impossible — a
//!   real usability regression) or unsafely extending the borrow's lifetime
//!   (which this workspace denies via `unsafe_code = "deny"`). So
//!   [`GraphLoopDriver`]'s `drive` instead calls the exact same node bodies
//!   directly, in a hand-rolled loop, against the real borrowed `&mut`
//!   state — no `Arc`, no `Mutex`, no `CompiledGraph` involved — which is
//!   sound with zero unsafe code precisely because a borrowed async call
//!   needs no `'static` bound. The two paths are behaviorally identical
//!   (same node functions) but the `GraphLoopDriver` path does not get graph
//!   checkpointing: an interrupt it raises still pauses the run (mirroring
//!   `run.paused`, exactly like a steering pause on the direct loop) but is
//!   not itself a resumable `CompiledGraph` checkpoint — resume it the same
//!   way a paused direct-loop run is resumed (feed `run.messages` back in as
//!   the next call's `input`), not via `CompiledGraph::resume`. A caller that
//!   wants graph-level checkpoint/resume across the interrupt should drive
//!   the loop through [`compile_loop`]/[`LoopIter`] directly instead of
//!   through `AgentHarness::invoke`.

pub mod driver;
pub mod iter;
pub mod runtime;
pub mod types;

mod compile;

pub use compile::compile_loop;
pub use driver::GraphLoopDriver;
pub use iter::{AgentLoopGraphExt, LoopIter, LoopStep};
pub use runtime::LoopRuntime;
pub use types::{LoopState, LoopUpdate, node};
