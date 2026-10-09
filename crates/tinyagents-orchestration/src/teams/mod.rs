//! Durable, dependency-aware agent-team composition.
//!
//! A **team** is a durable group of worker agents (members) who claim and
//! complete collaborative tasks. This module owns team creation, member
//! lifecycle, task assignment and completion, and the event log that records
//! all durable state changes.
//!
//! [`TeamService`] is the primary API: it validates team structure, manages
//! member and task persistence (via the [`TeamLedger`] trait), and enforces
//! coordination invariants (no duplicate names, no cycles in task dependencies,
//! no dangling member or task references). `runtime` handles the per-member
//! details: reading undelivered messages from the event log and composing the
//! prompt a worker should receive. `graph` executes a member's work as a
//! generic execute → complete/fail → done DAG, bridging the graph layer and
//! durable team state.

mod graph;
mod runtime;
mod service;
mod types;

pub use graph::{MemberOutcome, member_graph_topology, run_member_graph, run_member_graph_with};
pub use runtime::{
    DeliveredMessages, EVENT_PAGE_SIZE, MESSAGE_DELIVERED_EVENT, TEAM_MESSAGE_EVENT,
    build_member_prompt, deliver_pending_messages, drain_run_events, truncate_chars,
};
pub use service::{SessionTeamLedger, TeamLedger, TeamService, claimable_task};
pub use types::{LEAD_SENDER, MemberShutdown, MemberStep, NewMember, TeamError, TeamView};

#[cfg(test)]
mod tests;
