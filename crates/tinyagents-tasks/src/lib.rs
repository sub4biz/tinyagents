//! Graph-independent detached-task machinery for TinyAgents.
//!
//! This crate is the managed child-work surface: task model, `TaskStore`
//! implementations, the detached task registry, restart reconciliation and the
//! orchestration control tools. It does not depend on the graph engine, so
//! `tinyagents-graph` and `tinyagents-orchestration` both build on it.
//!
//! It gives
//! language-model orchestrators stable task ids and typed controls (`spawn`,
//! `await`, `cancel`, `kill`, `status`, `list`, `timeout`, `race`, `yield`, and
//! `steer`) without exposing raw executor handles such as `tokio::JoinHandle`.
//!
//! The controls are ordinary harness tools. Use [`OrchestrationTool`] directly,
//! call [`orchestration_tools`] to build the full set, or call
//! [`register_orchestration_tools`] to insert them into a
//! [`tinyagents_harness::tool::ToolRegistry`] alongside any other tools.

mod reconcile;
mod recovery;
mod runtime;
mod store;
mod store_registry;
mod tool;
mod types;

pub use reconcile::{
    ReconcileOutcome, ReconcileReport, ReconciledTask, reconcile_orphaned_tasks, task_status_label,
};
pub use recovery::{
    MAX_RECOVERY_CHILDREN, MAX_RECOVERY_LABEL_CHARS, RESTART_RECOVERY_INSTRUCTION, RecoveryChild,
    build_restart_recovery_note, recovery_children,
};
pub use runtime::DetachedTaskRegistry;
pub use store::{InMemoryTaskStore, JsonlTaskStore, TaskStore};
pub use store_registry::{
    TaskStoreRegistry, TaskStoreRegistryError, open_jsonl_task_store_or_memory,
};
pub use tool::{
    OrchestrationTool, SteeringRegistry, orchestration_tool_schema, orchestration_tool_schemas,
    orchestration_tools, orchestration_tools_with_steering, register_orchestration_tools,
};
pub use types::*;

#[cfg(test)]
#[path = "lib_tests.rs"]
mod test;
