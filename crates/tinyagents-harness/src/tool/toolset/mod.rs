//! Composable toolsets (gap B3): a `ToolSet` is a *value* a caller can wrap,
//! filter, rename, prefix, and combine, instead of tool visibility living
//! only as ordering-sensitive middleware.
//!
//! Mirrors Pydantic AI's `AbstractToolset` (`docs/runtime-comparison/
//! pydantic-ai.md` §3.4): `get_tools`/`call_tool`/`get_instructions`/
//! `for_run` plus `.filtered()`, `.prefixed()`, `.renamed()`, `.prepared()`,
//! `.approval_required()`, a `CombinedToolset`, and a schema-only external
//! toolset. [`crate::tool::ToolRegistry`] implements [`ToolSet`] directly, so
//! existing harness code that builds a registry keeps working unchanged while
//! gaining the ability to be wrapped by any adaptor here.
//!
//! # Adaptors
//!
//! - [`CombinedToolSet`] — merges multiple toolsets; `call` dispatches to
//!   whichever member currently owns the name.
//! - [`FilteredToolSet`] — keeps only the tools a predicate accepts.
//! - [`PrefixedToolSet`] — prefixes every advertised name (collision
//!   avoidance when combining toolsets with overlapping names) and strips
//!   the prefix again before delegating a call.
//! - [`RenamedToolSet`] — renames tools per an explicit map.
//! - [`PreparedToolSet`] — applies a per-step schema transform, the same
//!   seam [`crate::tool::SchemaPreparation`] uses for provider projection,
//!   but caller-supplied and consulted every turn (so it can vary by
//!   [`RunContext`]).
//! - [`ApprovalRequiredToolSet`] — marks matching tools as requiring human
//!   approval via [`tinytools::ToolPolicy::access`].
//! - [`ExternalToolSet`] — schema-only tools the *host* executes; see
//!   [`crate::error::TinyAgentsError::CallDeferred`].
//!
//! # Why `ToolRegistry` needs `State: Default` to implement `ToolSet`
//!
//! [`ToolSet::call`] deliberately carries no `&State` parameter — a toolset
//! is meant to be composable without threading the harness's application
//! state through every adaptor. [`crate::tool::ToolDispatch::execute`], the
//! mechanism a [`crate::tool::ToolRegistry`] dispatches through, does take
//! one (it exists for the rare recursive sub-agent tool that needs the full
//! typed parent run). The [`ToolSet`] impl on [`crate::tool::ToolRegistry`]
//! below satisfies that with `State::default()`, which is exactly right for
//! the overwhelming majority of tools (they ignore `state`) and is a real,
//! documented limitation for a [`crate::tool::ToolDispatch`] that actually
//! needs the caller's live state: such a dispatcher must keep being invoked
//! through [`crate::tool::ToolRegistry`] directly (or a host-owned bridge),
//! not through the `ToolSet` chain.

mod approval_required;
mod combined;
mod external;
mod filtered;
mod prefixed;
mod prepared;
mod renamed;
mod types;

#[cfg(test)]
#[path = "mod_tests.rs"]
pub(crate) mod test;

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinytools::{
    PermissionLevel, Tool, ToolCallOptions, ToolCategory, ToolExposure, ToolInjectedArgument,
    ToolPolicy, ToolResult, ToolScope, ToolTimeout,
};

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};

pub use approval_required::ApprovalRequiredToolSet;
pub use combined::CombinedToolSet;
pub use external::ExternalToolSet;
pub use filtered::FilteredToolSet;
pub use prefixed::PrefixedToolSet;
pub use prepared::PreparedToolSet;
pub use renamed::RenamedToolSet;
pub use types::ToolExposureExplanation;

// Crate-internal reuse seams: the pure predicate logic behind
// `FilteredToolSet`/`PreparedToolSet` is shared with the older
// `ToolAllowlistMiddleware`/`DynamicToolSelectionMiddleware` (see those
// modules) instead of each keeping its own copy.
pub(crate) use filtered::tool_name_allowed;
pub(crate) use prepared::retain_matching_schemas;

/// A composable source of tools, generic over the harness's application
/// `State` and run-context data `Ctx` — the same split
/// [`crate::tool::ToolRegistry`] and [`crate::runtime::AgentHarness`] use.
///
/// This is the unit of composition Pydantic AI's `AbstractToolset` occupies
/// (`docs/runtime-comparison/pydantic-ai.md` §3.4/§4): a value that knows
/// its own tools, can execute them, can carry its own instructions, and can
/// be wrapped by any of the adaptors in this module. A
/// [`crate::tool::ToolRegistry`] is one `ToolSet`; an MCP client, an
/// authenticated per-user tool source, or a capability bundle
/// (`docs/runtime-comparison/plan.md`'s Phase 6 `Capability`) is meant to be
/// another.
#[async_trait]
pub trait ToolSet<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Returns the tools this toolset currently exposes for `ctx`.
    ///
    /// Called once per turn by a caller building the model-visible catalogue
    /// (or by an adaptor wrapping this toolset), so it may legitimately vary
    /// by run context — this is what makes [`PreparedToolSet`] and
    /// [`ApprovalRequiredToolSet`] meaningful per-step rather than only at
    /// construction time.
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>>;

    /// Executes the named tool with `args`.
    ///
    /// Implementations should return
    /// [`TinyAgentsError::ToolNotFound`] for a name this toolset does not
    /// currently expose (including one that [`Self::tools`] would have
    /// filtered out this turn), so a caller chaining adaptors can tell "not
    /// mine" apart from "mine, and it failed".
    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult>;

    /// Instructions this toolset contributes to the system prompt.
    ///
    /// `None` by default. A toolset backed by an MCP server or an
    /// authenticated capability can surface server-provided usage guidance
    /// here instead of requiring the host to know about it out of band.
    fn instructions(&self) -> Option<String> {
        None
    }

    /// Lifecycle hook invoked once when a run that will use this toolset
    /// starts (Pydantic AI's `for_run`). The default is a no-op; a toolset
    /// with per-run setup (opening a connection, priming a cache) overrides
    /// it.
    async fn for_run(&self, _ctx: &RunContext<Ctx>) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl<State, Ctx> ToolSet<State, Ctx> for crate::tool::ToolRegistry<State, Ctx>
where
    State: Default + Send + Sync,
    Ctx: Send + Sync,
{
    async fn tools(&self, _ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(self
            .model_callable_names()
            .into_iter()
            .filter_map(|name| self.get(&name))
            .collect())
    }

    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        let dispatch = self
            .model_dispatch(name)
            .ok_or_else(|| TinyAgentsError::ToolNotFound(name.to_string()))?;
        let tool = dispatch.tool();
        let options = ToolCallOptions {
            prefer_markdown: tool.supports_markdown(),
        };
        let state = State::default();
        // `ToolSet::call` carries no admitted call id (B1's real
        // correlation id lives on `agent_loop`'s admission path, which does
        // not reach here — this seam is a `ToolRegistry` consulted directly
        // as a `ToolSet`, e.g. from a `CombinedToolSet`), so a fresh
        // process-unique id is minted for the `ToolExecutionContext` the
        // dispatch builds instead of a transcript-correlated one.
        dispatch
            .execute(&state, crate::ids::new_call_id(), args, options, ctx)
            .await
            .map_err(|err| TinyAgentsError::Tool(err.to_string()))
    }
}

/// Bridges one [`ToolSet`]-owned tool into the [`crate::tool::ToolDispatch`]
/// the agent loop's admission path already speaks, so a name only a toolset
/// chain exposes (not [`crate::tool::ToolRegistry::model_dispatch`]) can be
/// admitted and executed through the exact same call path as a directly
/// registered tool — timeout policy, injected-argument handling, schema
/// validation, and every other admission step in `agent_loop/tools.rs` apply
/// identically.
///
/// # Why this is not wired automatically
///
/// [`crate::runtime::AgentHarness::with_toolset`] wires the toolset chain
/// into per-turn **advertisement** automatically. Dispatch is different:
/// coercing this bridge into `Arc<dyn ToolDispatch<State, Ctx>>` requires
/// `State: 'static, Ctx: 'static`, a bound the agent loop's generic admission
/// path deliberately does not carry (recursive sub-agent dispatch stays
/// callable with a borrowed, non-`'static` `State`/`Ctx` — see
/// `runtime/agent.rs`'s `host_invocation_binding`). A concrete application's
/// `State`/`Ctx` are `'static` in the overwhelming majority of cases, so a
/// caller wanting a toolset-only tool to be callable (not just advertised)
/// registers a bridge for it explicitly:
///
/// ```ignore
/// let tool = toolset.tools(&ctx).await?.into_iter().find(|t| t.name() == "search").unwrap();
/// harness.register_tool_dispatch(Arc::new(ToolSetDispatchBridge::new(toolset.clone(), tool)));
/// ```
pub struct ToolSetDispatchBridge<State: Send + Sync, Ctx: Send + Sync> {
    toolset: Arc<dyn ToolSet<State, Ctx>>,
    tool: Arc<dyn Tool>,
}

impl<State: Send + Sync, Ctx: Send + Sync> ToolSetDispatchBridge<State, Ctx> {
    /// Builds a dispatcher for `tool` (a declaration `toolset` currently
    /// exposes) that executes it through [`ToolSet::call`].
    pub fn new(toolset: Arc<dyn ToolSet<State, Ctx>>, tool: Arc<dyn Tool>) -> Self {
        Self { toolset, tool }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> crate::tool::ToolDispatch<State, Ctx>
    for ToolSetDispatchBridge<State, Ctx>
{
    fn tool(&self) -> Arc<dyn Tool> {
        self.tool.clone()
    }

    async fn execute(
        &self,
        _state: &State,
        // `ToolSet::call` (below) has no call-id slot yet, so the real
        // admitted call id this trait method receives (B1) cannot be
        // threaded any further than this bridge; the wrapped toolset builds
        // whatever `ToolExecutionContext` it needs on its own terms (see
        // `ToolRegistry::call`'s synthetic id above for the one case that
        // does).
        _call_id: crate::ids::CallId,
        arguments: Value,
        _options: ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<ToolResult> {
        // Preserve the typed `TinyAgentsError` (not a flattened string) so a
        // deferral (`ApprovalRequired`/`CallDeferred`, A2) the wrapped
        // toolset's `ToolSet::call` raises still downcasts cleanly in
        // `agent_loop/tools.rs`'s `execute_tool_recovering_model_retry` and
        // reaches the loop's ordinary `AgentRun::deferred` exit, exactly as
        // for a directly registered tool. `From<TinyAgentsError>` for
        // `anyhow::Error` keeps the original error available to
        // `Error::downcast`.
        self.toolset
            .call(self.tool.name(), arguments, parent)
            .await
            .map_err(anyhow::Error::from)
    }
}

/// Internal helper shared by every renaming/prefixing/prepared/approval
/// adaptor: a [`Tool`] that forwards everything to `inner` except the fields
/// explicitly overridden here.
///
/// Every method is delegated explicitly rather than relying on
/// [`Tool`]'s trait defaults: those defaults are the *crate's* conservative
/// fallback (for example [`Tool::policy`] defaulting to
/// [`ToolPolicy::default`]), not "ask `inner`" — leaving any of them
/// undelegated would silently reset that declaration for every wrapped tool.
pub(crate) struct OverrideTool {
    pub(crate) inner: Arc<dyn Tool>,
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) parameters: Option<Value>,
    pub(crate) policy_transform: Option<Arc<dyn Fn(ToolPolicy) -> ToolPolicy + Send + Sync>>,
    /// Overrides [`Tool::exposure`] instead of delegating to `inner`. Used by
    /// [`crate::capability::CapabilityToolSet`] (gap G3) to apply a
    /// [`crate::capability::Capability::exposure`] uniformly to every tool a
    /// capability's inner [`ToolSet`] contributes, regardless of what each
    /// individual tool declares for itself.
    pub(crate) exposure_override: Option<ToolExposure>,
}

impl OverrideTool {
    pub(crate) fn new(inner: Arc<dyn Tool>) -> Self {
        Self {
            inner,
            name: None,
            description: None,
            parameters: None,
            policy_transform: None,
            exposure_override: None,
        }
    }

    /// Overrides [`Tool::exposure`] with `exposure` instead of delegating to
    /// `inner`.
    pub(crate) fn with_exposure(mut self, exposure: ToolExposure) -> Self {
        self.exposure_override = Some(exposure);
        self
    }

    pub(crate) fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub(crate) fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub(crate) fn with_parameters(mut self, parameters: Value) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub(crate) fn with_policy_transform(
        mut self,
        transform: Arc<dyn Fn(ToolPolicy) -> ToolPolicy + Send + Sync>,
    ) -> Self {
        self.policy_transform = Some(transform);
        self
    }
}

#[async_trait]
impl Tool for OverrideTool {
    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or_else(|| self.inner.name())
    }

    fn description(&self) -> &str {
        self.description
            .as_deref()
            .unwrap_or_else(|| self.inner.description())
    }

    fn parameters_schema(&self) -> Value {
        self.parameters
            .clone()
            .unwrap_or_else(|| self.inner.parameters_schema())
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        self.inner.execute(args).await
    }

    async fn execute_with_options(
        &self,
        args: Value,
        options: ToolCallOptions,
    ) -> anyhow::Result<ToolResult> {
        self.inner.execute_with_options(args, options).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        options: ToolCallOptions,
        context: Option<&dyn tinytools::ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        self.inner
            .execute_with_context(args, options, context)
            .await
    }

    fn injected_arguments(&self) -> Vec<ToolInjectedArgument> {
        self.inner.injected_arguments()
    }

    fn supports_markdown(&self) -> bool {
        self.inner.supports_markdown()
    }

    fn permission_level(&self) -> PermissionLevel {
        self.inner.permission_level()
    }

    fn permission_level_with_args(&self, args: &Value) -> PermissionLevel {
        self.inner.permission_level_with_args(args)
    }

    fn scope(&self) -> ToolScope {
        self.inner.scope()
    }

    fn category(&self) -> ToolCategory {
        self.inner.category()
    }

    fn exposure(&self) -> ToolExposure {
        self.exposure_override
            .unwrap_or_else(|| self.inner.exposure())
    }

    // A renamed or prefixed tool is still the same tool to the host's tool
    // rules: keep its family, tags and indirect target.
    fn family(&self) -> Option<&str> {
        self.inner.family()
    }

    fn tags(&self) -> Vec<String> {
        self.inner.tags()
    }

    fn indirect_target(&self, args: &Value) -> Option<tinytools::IndirectCall> {
        self.inner.indirect_target(args)
    }

    fn is_concurrency_safe(&self, args: &Value) -> bool {
        self.inner.is_concurrency_safe(args)
    }

    fn external_effect(&self) -> bool {
        self.inner.external_effect()
    }

    fn external_effect_with_args(&self, args: &Value) -> bool {
        self.inner.external_effect_with_args(args)
    }

    fn max_result_size_chars(&self) -> Option<usize> {
        self.inner.max_result_size_chars()
    }

    fn timeout_policy(&self, args: &Value) -> ToolTimeout {
        self.inner.timeout_policy(args)
    }

    fn host_extension(&self) -> Option<&(dyn Any + Send + Sync)> {
        self.inner.host_extension()
    }

    fn host_call_extension(&self, args: &Value) -> Option<Box<dyn Any + Send + Sync>> {
        self.inner.host_call_extension(args)
    }

    fn policy(&self) -> ToolPolicy {
        let base = self.inner.policy();
        match &self.policy_transform {
            Some(transform) => transform(base),
            None => base,
        }
    }

    fn display_label(&self, args: &Value) -> Option<String> {
        self.inner.display_label(args)
    }

    fn display_detail(&self, args: &Value) -> Option<String> {
        self.inner.display_detail(args)
    }

    fn return_direct(&self) -> bool {
        self.inner.return_direct()
    }
}
