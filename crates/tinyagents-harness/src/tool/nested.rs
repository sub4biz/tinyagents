//! Nested tool calls: one tool calling another through the harness (C9).
//!
//! A tool that needs another tool's result (a planner calling `search`, a
//! batch tool fanning out `read`s) calls it with
//! [`ToolExecutionContext::call_tool`](super::ToolExecutionContext::call_tool)
//! instead of reaching into the registry. The harness installs a
//! [`NestedToolRunner`] on the context of every call it executes; the runner
//! sends the nested call through the admission and execution path of a
//! model-issued call, with the exceptions listed below.
//!
//! Applies to a nested call: tool lookup and allow-list, argument validation,
//! the approval refusal, `Middleware::check_nested_tool` (the nested-call form
//! of `before_tool` enforcement), host authorization, the tool-wrap onion,
//! timeouts, the run budget, and `Middleware::observe_nested_result`. Does
//! **not** apply: `before_tool` / `after_tool` proper (they need `&mut
//! RunContext`), the progress gate, host output screening and `ToolControl`.
//! A middleware that enforces policy only in `before_tool` does not bind
//! nested calls until it implements `check_nested_tool`. See
//! `docs/modules/harness/nested-tool-calls.md`.
//!
//! A [`ToolDispatch`](super::ToolDispatch) that builds a
//! [`ToolExecutionContext`](super::ToolExecutionContext) must do so inside the
//! future the loop polls (the scoped runner is read when the context is built).
//!
//! This module holds only the seam. The runner itself lives in the agent loop
//! (`agent_loop::nested`), because it needs the harness, the run state and the
//! run context that a tool future cannot own.

use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::error::Result;
use crate::ids::CallId;

/// Executes a nested tool call on behalf of the tool that is currently running.
///
/// Installed per call by the harness; a host does not normally implement it.
/// It is a trait object so [`super::ToolExecutionContext`] stays free of the
/// harness's `State`/`Ctx` type parameters.
pub trait NestedToolRunner: Send + Sync {
    /// Runs the tool `name` with `arguments` as a nested call and returns its
    /// result. A refusal (unknown tool, invalid arguments, a spent limit, a
    /// call that would need approval) is an `Err` carrying a clear message.
    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        arguments: serde_json::Value,
    ) -> BoxFuture<'a, Result<tinytools::ToolResult>>;
}

tokio::task_local! {
    /// The runner of the call currently being driven, with the id of that call.
    static CURRENT: (CallId, Arc<dyn NestedToolRunner>);
}

/// Scopes `runner` to `future`, the execution of the call `call_id`.
///
/// Read back by [`current_for`] when the call's [`super::ToolExecutionContext`]
/// is built; the id check keeps a nested call's own context from picking up its
/// parent's runner while the parent's scope is still on the stack.
pub(crate) fn scope<F: Future>(
    call_id: CallId,
    runner: Arc<dyn NestedToolRunner>,
    future: F,
) -> impl Future<Output = F::Output> {
    CURRENT.scope((call_id, runner), future)
}

/// The runner for exactly the call `call_id`, when one is in scope.
pub(crate) fn current_for(call_id: &CallId) -> Option<Arc<dyn NestedToolRunner>> {
    CURRENT
        .try_with(|(scoped, runner)| (scoped == call_id).then(|| Arc::clone(runner)))
        .ok()
        .flatten()
}
