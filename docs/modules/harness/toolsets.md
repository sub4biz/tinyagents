# Composable Toolsets (gap B3)

`tinyagents_harness::tool::toolset` gives tool visibility a value-level
composition model instead of leaving it entirely to middleware ordering. See
`docs/runtime-comparison/pydantic-ai.md` §3.4/§4 and `docs/sdk-gaps/tools.md` §9 for
the design lessons and acceptance criteria this closes.

## `ToolSet<State, Ctx>`

```rust
#[async_trait]
pub trait ToolSet<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>>;
    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult>;
    fn instructions(&self) -> Option<String> { None }
    async fn for_run(&self, ctx: &RunContext<Ctx>) -> Result<()> { Ok(()) }
}
```

`tools` is called once per turn by whatever is building the model-visible
catalogue (a bare toolset, or an adaptor wrapping one), so exposure may
legitimately vary by `RunContext` every turn. `call` must return
`TinyAgentsError::ToolNotFound` for a name it does not currently expose so a
caller chaining adaptors can distinguish "not mine" from "mine, and it
failed".

`tinyagents_harness::tool::ToolRegistry<State, Ctx>` implements `ToolSet`
directly (`State: Default` is required because `ToolSet::call` deliberately
carries no `&State`; the registry's own `ToolDispatch::execute` does, and the
impl supplies `State::default()`). Existing code that builds a registry keeps
working unchanged while gaining the ability to be wrapped by any adaptor
below.

## Adaptors

| Adaptor | Module | Purpose |
| --- | --- | --- |
| `CombinedToolSet` | `tool::toolset::combined` | Merges multiple toolsets; `call` dispatches to whichever member currently owns the name. |
| `FilteredToolSet` | `tool::toolset::filtered` | Keeps only the tools a predicate accepts. |
| `PrefixedToolSet` | `tool::toolset::prefixed` | Prefixes every advertised name (collision avoidance when combining toolsets with overlapping names) and strips the prefix again before delegating a call. |
| `RenamedToolSet` | `tool::toolset::renamed` | Renames tools per an explicit map. |
| `PreparedToolSet` | `tool::toolset::prepared` | Applies a per-step schema transform, consulted every turn so it can vary by `RunContext` — Pydantic AI's per-tool `prepare`. |
| `ApprovalRequiredToolSet` | `tool::toolset::approval_required` | Marks matching tools as requiring human approval via `ToolPolicy::access`. |
| `ExternalToolSet` | `tool::toolset::external` | Schema-only tools the *host* executes — see below. |

Every adaptor is a plain value: constructible, inspectable, and testable on
its own, which is the concrete answer to `docs/sdk-gaps/tools.md` §9's "why was
this tool hidden" requirement. Each adaptor that changes or withholds a tool
records a `ToolExposureExplanation` (`FilteredOut`, `Renamed { from, to }`,
`Prefixed { from, to }`, `Prepared`, `ApprovalRequired`, `Deferred`, `Hidden`)
which is additive on `AgentEvent::ToolsFiltered`, so existing consumers of
that event are unaffected.

## `ExternalToolSet` — the OpenHuman MCP seam

`ExternalToolSet` is intentionally schema-only: it advertises tool
declarations to the model but never executes them itself. Every declaration
carries the same host-extension marker as `ToolRegistry::register_external`,
so a tool bridged into a harness's registry (`ToolSetDispatchBridge` +
`AgentHarness::register_tool_dispatch`, see below) is recognized at
admission and deferred through the agent loop's ordinary A2 exit
(`DeferredToolRequests`/`AgentRun::deferred`, or a registered
`DeferredToolHandler`) exactly like a directly registered external tool —
`ToolSet::call` is never reached for the admitted case. A caller driving the
toolset chain directly instead still gets `TinyAgentsError::CallDeferred`
from `ToolSet::call` to catch itself. This is the seam OpenHuman's MCP layer
plugs into — TinyAgents does not ship an MCP client or
`ServerToolUse`/`ServerToolResult` content parts (see
`docs/runtime-comparison/pydantic-ai.md` §3.5 and
`docs/runtime-comparison/plan.md`); those live in OpenHuman, built against
`ExternalToolSet` and `TinyAgentsError::CallDeferred`.

## Wiring into `AgentHarness`

`AgentHarness::with_toolset(toolset)` installs a toolset chain that is
resolved into the per-turn advertised catalogue automatically. Composing
adaptors — e.g. wrapping two colliding member toolsets in their own
`PrefixedToolSet::new(member, "prefix")` before combining them with
`CombinedToolSet::new(vec![...])`, then wrapping the result in
`FilteredToolSet::new(combined, predicate)` — builds the exact per-run
exposure policy declaratively instead of through middleware ordering.

Dispatch (as opposed to advertisement) for a toolset-only tool — one not
already reachable through `ToolRegistry::model_dispatch` — requires an
explicit `ToolSetDispatchBridge`, because bridging into
`Arc<dyn ToolDispatch<State, Ctx>>` needs `State: 'static, Ctx: 'static`, a
bound the agent loop's generic admission path does not otherwise carry
(recursive sub-agent dispatch stays callable with borrowed, non-`'static`
state). See `tool::toolset::ToolSetDispatchBridge` for the explicit
registration call.

## Middleware rebased onto toolsets

`ToolAllowlistMiddleware` and `DynamicToolSelectionMiddleware`
(`middleware::library::tool_policy`) are kept as public types — existing
hosts do not need to migrate — but are now thin wrappers over the same
predicate logic the `ToolSet` adaptors use:

- `ToolAllowlistMiddleware::allows` delegates to
  `tool::toolset::tool_name_allowed`, the exact membership test
  `FilteredToolSet::allowing` uses, so the middleware and its `ToolSet`
  counterpart cannot drift.
- `DynamicToolSelectionMiddleware` shares `tool::toolset::retain_matching_schemas`
  with `PreparedToolSet`.

Prefer the `ToolSet` adaptors for new code — they compose and are
independently testable — and keep the middleware for hosts that already
depend on the `Middleware` hook ordering.

## Testing

Each adaptor has a dedicated `mod_tests.rs` next to its `mod.rs`
(`tool/toolset/{combined,filtered,prefixed,renamed,prepared,approval_required,external}/mod_tests.rs`)
covering: basic wrapping behavior, `ToolExposureExplanation` correctness,
prefix collision handling (`prefixed/mod_tests.rs`), and per-step `prepare`
varying by `RunContext` (`prepared/mod_tests.rs`). `tool/toolset/mod_tests.rs` covers
chain composition end to end.
