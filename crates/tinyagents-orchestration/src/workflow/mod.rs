//! Durable, host-neutral workflow definitions and execution.
//!
//! The workflow engine owns phase scheduling, bounded fan-out, cancellation,
//! resume semantics, and the JSON phase-state projection.  A host supplies a
//! [`WorkflowStore`] and [`WorkflowExecutor`]; therefore this module has no
//! knowledge of credentials, model selection, policy, progress, or RPC.

mod child_step;
mod engine;
mod graph;
mod state;
mod types;
mod validate;

pub use engine::{
    OrchestrationError, SessionWorkflowStore, WorkflowChildRegistration, WorkflowChildRequest,
    WorkflowChildResult, WorkflowEngine, WorkflowExecutor, WorkflowStore,
};
pub use graph::scheduler_topology_preview;
pub use state::{
    PhaseStatus, all_phases_completed, init_phase_states, next_runnable_phase, phase_prompt,
    phase_status, reset_running_phases, synthesize_summary, upstream_outputs,
};
pub use types::{
    DefinitionError, WorkflowDefinition, WorkflowDefinitionListResponse, WorkflowPhase,
};
pub use validate::{validate_agents, validate_structure};

#[cfg(test)]
mod tests;
