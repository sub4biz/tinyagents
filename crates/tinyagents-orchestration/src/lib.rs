//! Host-neutral subagent invocation and lifecycle orchestration.
//!
//! This crate owns TinyAgents' subagent-facing surfaces: direct child-agent
//! invocation, the typed tool adapter, reusable child sessions, and the
//! durable lifecycle driver. Hosts supply persistence and concrete execution;
//! policy, credentials, progress, and RPC remain host concerns.
//!
//! The [`status`] module and the `to_*_status` methods map the job, detached,
//! task and run-ledger status vocabularies onto each other.
//!
//! Dependency direction is deliberately one way:
//! `orchestration -> {harness, runtime}`. The lower-level crates never
//! depend on this composition layer.

pub mod status;
pub mod subagent;
pub mod teams;
pub mod workflow;

#[cfg(test)]
#[path = "lib_boundary_tests.rs"]
mod boundary_tests;
