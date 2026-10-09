//! Tool rules in the agent loop: one gate for the catalogue, tool search and
//! every call.
//!
//! [`tinytools::ToolRules`] is the vocabulary; this module is where the loop
//! applies it. A run's rules come from two places and stack as layers of one
//! [`tinytools::ToolRuleSet`], so neither can widen the other:
//!
//! - [`RunPolicy::tool_rules`](crate::runtime::RunPolicy::tool_rules) — the
//!   harness-wide rules and the [`tinytools::RuleContext`] they are evaluated
//!   in (the channel, the agent, the origin of the turn);
//! - the hosted agent definition's
//!   [`tool_rules`](tinyagents_definition::AgentDefinition::tool_rules).
//!
//! The definition's exact `tools` allowlist still applies first, unchanged.
//!
//! The [`ToolGate`] answers three questions with the same rules, so a tool a
//! rule removes from the catalogue cannot be found through `tool_search` or
//! called by a name the model guessed:
//!
//! | Site | Surface |
//! |---|---|
//! | direct schemas on the request (and mid-run toolset changes) | `catalog` |
//! | deferred catalogue, `tool_search` answers, replayed promotions | `search` |
//! | admission of a model call, and of a nested call | `call` |
//!
//! At call time the rules also yield an [`tinytools::ApprovalDirective`]: a
//! `require_approval` rule defers the call for approval exactly like a tool
//! that declares `approval_required`, and an `auto_approve` rule waives that
//! declaration.

mod types;

pub use types::ToolRulePolicy;
pub(crate) use types::{CallGate, ToolGate};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
