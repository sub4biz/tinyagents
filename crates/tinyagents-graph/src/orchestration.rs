//! Compatibility shim: the detached-task machinery moved to `tinyagents-tasks`.
//!
//! Every item formerly defined here is re-exported unchanged so existing
//! `tinyagents_graph::orchestration::*` paths keep compiling. The shim is
//! permanent for now; prefer `tinyagents_tasks` in new code.

pub use tinyagents_tasks::*;

#[cfg(test)]
#[path = "orchestration_tests.rs"]
mod tests;
