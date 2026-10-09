//! Host security gate — the runtime asks, it never decides.
//!
//! This crate is a redistributed agent runtime. It executes tools and feeds
//! text into a model context, but it holds **no authority** over whether either
//! is permitted: permission is a property of the embedding product (its user's
//! consent, its autonomy tier, its trusted roots, its approval UX), none of
//! which can be expressed here without dragging a product schema into a library.
//!
//! So the runtime consults [`SecurityGate`] at two points and obeys the answer:
//!
//! 1. **Before every tool invocation** — [`SecurityGate::authorize_tool`].
//! 2. **Before untrusted text enters the model context** —
//!    [`SecurityGate::screen_input`], for tool output, fetched web pages,
//!    inbound channel messages, and anything else the runtime did not author.
//!
//! Unlike the optional capabilities in this module family, a host **must**
//! supply a gate: there is no meaningful "no security" default other than
//! [`AllowAllSecurityGate`], which exists for tests and is deliberately loud
//! about it. Absence here cannot be modelled as `None`, because "nobody was
//! asked" and "everybody said yes" must not look the same to a reader of the
//! call site.
//!
//! **The gate is a gate, not a hint.** The runtime may not cache a decision
//! across calls, may not infer one call's answer from another's, and may not
//! substitute its own judgement when the host is slow or errors — an `Err` from
//! either method is a failure to obtain permission and must be treated as a
//! refusal by the caller, never as a fallback allow.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::ids::CallId;
use tinyinference_llm::tool::ToolCall;

// ── ContentOrigin ─────────────────────────────────────────────────────────────

/// Where a piece of text came from, so the host can apply its own
/// origin-dependent screening.
///
/// This enum carries **provenance only, never trust**. There is deliberately no
/// `is_trusted()` helper and no ordering: ranking origins by trustworthiness is
/// exactly the policy decision this crate must not make. One host treats
/// channel messages as hostile and tool output as safe; another inverts that
/// because its tools reach the open internet. Both are correct, and both belong
/// on the host side of [`SecurityGate::screen_input`].
///
/// Modelled as a closed enum rather than a free-form string so a host's
/// screening `match` is exhaustive — a new origin becomes a compile error at the
/// host, not text that quietly falls through an `_ => allow` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContentOrigin {
    /// Typed or spoken by the human driving this turn.
    #[default]
    User,
    /// Returned by a tool the runtime invoked.
    Tool,
    /// Fetched from the open web (page content, search results, API bodies).
    Web,
    /// Received over an external messaging channel from a third party.
    Channel,
    /// Produced by another agent — a subagent result or a delegated turn.
    ///
    /// Distinct from [`Tool`](Self::Tool) because a subagent's output may itself
    /// contain unscreened text it read from the web, and some hosts screen it
    /// again on that basis.
    Agent,
    /// Read back from durable storage the host owns (recalled memory, a stored
    /// experience, a prior transcript).
    Stored,
}

// ── ToolCallRequest ───────────────────────────────────────────────────────────

/// The subject of an authorization question: one pending tool invocation.
///
/// Inert by construction — serde and std only — so a host can build its policy
/// mapper against this struct without linking the agent loop.
///
/// It carries the *proposed* call, not an executed one. The runtime constructs
/// it after parsing the model's request and before any side effect, which is
/// why `arguments` is raw [`Value`]: the gate must see what the model actually
/// asked for, including shapes the tool would have rejected. Sanitizing before
/// asking would hide the interesting cases from the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Name of the tool the model wants to invoke.
    pub tool_name: String,
    /// Arguments exactly as the model supplied them.
    ///
    /// [`Value::Null`] for a no-argument call. When the model emitted
    /// unparseable JSON this holds the raw string the provider preserved, so a
    /// host that blocks on argument content still sees the original text rather
    /// than an empty object.
    #[serde(default)]
    pub arguments: Value,
    /// Id of the agent making the call.
    ///
    /// A plain `String`: this crate has no `AgentId` newtype, and the RFC's
    /// signature does not require one. Hosts key their own registries however
    /// they like, so an opaque string is both sufficient and non-committal.
    pub agent_id: String,
    /// Provider-assigned call id, when the invocation has one.
    ///
    /// `None` for calls the runtime synthesizes internally (a replayed step, a
    /// host-injected call). Hosts should treat it as a correlation handle for
    /// their audit log, never as an identity to authorize against — it is
    /// attacker-influenced in exactly the same way the arguments are.
    #[serde(default)]
    pub call_id: Option<CallId>,
    /// The call this one is nested under, when a running tool made it through
    /// `ToolExecutionContext::call_tool` rather than the model.
    ///
    /// `None` for a model-issued call. A nested call cannot be deferred or
    /// prompted for: the parent is mid-execution. A host gate that would
    /// normally ask a human should **fail closed** when this is `Some`, and
    /// authorize only what its policy allows without asking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_call_id: Option<CallId>,
}

impl ToolCallRequest {
    /// A request to run `tool_name` with `arguments` on behalf of `agent_id`,
    /// with no provider call id.
    pub fn new(
        tool_name: impl Into<String>,
        arguments: Value,
        agent_id: impl Into<String>,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
            agent_id: agent_id.into(),
            call_id: None,
            parent_call_id: None,
        }
    }

    /// Lifts a parsed [`ToolCall`] into an authorization request for
    /// `agent_id`.
    ///
    /// Copies the arguments verbatim, including the raw-string form a provider
    /// leaves behind when the model emitted malformed argument JSON. A
    /// malformed call is precisely the kind the host most wants to see, so it is
    /// never filtered out here — the caller decides separately whether to run it
    /// once the gate has answered.
    pub fn from_tool_call(call: &ToolCall, agent_id: impl Into<String>) -> Self {
        Self {
            tool_name: call.name.clone(),
            arguments: call.arguments.clone(),
            agent_id: agent_id.into(),
            call_id: Some(CallId::new(call.id.clone())),
            parent_call_id: None,
        }
    }

    /// Marks this request as a nested call made by a tool of `parent`.
    #[must_use]
    pub fn with_parent_call_id(mut self, parent: impl Into<CallId>) -> Self {
        self.parent_call_id = Some(parent.into());
        self
    }

    /// Whether a running tool, rather than the model, made this call.
    pub fn is_nested(&self) -> bool {
        self.parent_call_id.is_some()
    }

    /// Attaches `call_id` to this request.
    pub fn with_call_id(mut self, call_id: impl Into<CallId>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }
}

// ── GateDecision ──────────────────────────────────────────────────────────────

/// The host's answer to [`SecurityGate::authorize_tool`].
///
/// Callers inside the runtime should branch on [`is_allowed`](Self::is_allowed)
/// rather than matching the variants. That is not a style preference: it is what
/// keeps the runtime from developing behaviour that depends on *how* permission
/// was obtained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateDecision {
    /// The call may proceed.
    Allow,
    /// The call must not proceed. `reason` is host-authored text safe to show a
    /// model as a tool error, so the model can adapt instead of retrying the
    /// same blocked call forever.
    Deny {
        /// Why the call was refused.
        reason: String,
    },
    /// The host ran an out-of-band approval flow and it resolved to `approved`.
    ///
    /// This variant exists so an interactive approval gate — parking a turn,
    /// showing a prompt, timing out — stays **entirely** host-side. The runtime
    /// receives only the settled answer. It must not treat `Prompted` as a
    /// signal that a human is available, must not surface "a human approved
    /// this" to the model, and must not retry a `Prompted { approved: false }`
    /// in the hope of catching someone at the keyboard. The variant is
    /// distinguishable from [`Allow`](Self::Allow) / [`Deny`](Self::Deny) only
    /// so a host's own audit log can record which path produced the answer.
    Prompted {
        /// Whether the host's approval flow resolved in favour of the call.
        approved: bool,
    },
}

impl GateDecision {
    /// A denial carrying `reason`.
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// Whether the call may proceed.
    ///
    /// The one predicate runtime code should use. `Prompted { approved: true }`
    /// is indistinguishable from [`Allow`](Self::Allow) here by design.
    pub fn is_allowed(&self) -> bool {
        match self {
            Self::Allow => true,
            Self::Deny { .. } => false,
            Self::Prompted { approved } => *approved,
        }
    }

    /// Host-authored refusal text, when the decision was a refusal.
    ///
    /// `None` for any allowed decision, and also for
    /// `Prompted { approved: false }` — a declined approval has no host-written
    /// explanation, and the runtime must not invent one that implies a human
    /// said no. Callers should fall back to neutral wording of their own.
    pub fn denial_reason(&self) -> Option<&str> {
        match self {
            Self::Deny { reason } => Some(reason.as_str()),
            _ => None,
        }
    }
}

// ── ScreenOutcome ─────────────────────────────────────────────────────────────

/// The host's answer to [`SecurityGate::screen_input`].
///
/// Note the asymmetry with [`GateDecision`]: screening may *rewrite* its
/// subject. The host owns redaction because only it knows which substrings are
/// secrets; the runtime's job is to use the returned text and never the
/// original.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenOutcome {
    /// The text is admissible unchanged.
    ///
    /// Carries no payload deliberately: the runtime already holds the original,
    /// and returning a copy would invite a host to "pass" a *modified* string
    /// through a variant whose name promises it did not.
    Pass,
    /// The text is admissible in the rewritten form carried here.
    ///
    /// The runtime must inject this string and drop the original. Redaction is
    /// not advisory.
    Redacted(String),
    /// The text must not enter the model context at all.
    ///
    /// `reason` is host-authored and safe to surface. Blocking discards the
    /// content entirely — there is no partial-admit path, because a host that
    /// wanted one would have returned [`Redacted`](Self::Redacted).
    Block {
        /// Why the content was rejected.
        reason: String,
    },
}

impl ScreenOutcome {
    /// A rejection carrying `reason`.
    pub fn block(reason: impl Into<String>) -> Self {
        Self::Block {
            reason: reason.into(),
        }
    }

    /// Whether the content may enter the model context in some form.
    pub fn is_admissible(&self) -> bool {
        !matches!(self, Self::Block { .. })
    }

    /// The text the runtime must actually use, given the `original` that was
    /// screened.
    ///
    /// Returns `None` when the content was blocked — the single call that keeps
    /// a caller from accidentally falling back to `original` on a block, which
    /// is the one mistake this whole trait exists to prevent. Borrowing rather
    /// than cloning also means there is no second copy of a redacted string
    /// floating around for a later caller to pick the wrong one of.
    pub fn effective_text<'a>(&'a self, original: &'a str) -> Option<&'a str> {
        match self {
            Self::Pass => Some(original),
            Self::Redacted(text) => Some(text.as_str()),
            Self::Block { .. } => None,
        }
    }

    /// Host-authored rejection text, when the content was blocked.
    pub fn block_reason(&self) -> Option<&str> {
        match self {
            Self::Block { reason } => Some(reason.as_str()),
            _ => None,
        }
    }
}

// ── SecurityGate ──────────────────────────────────────────────────────────────

/// Host-supplied authority over tool execution and untrusted text.
///
/// Required, not optional: every session has a gate. See the module header for
/// why absence is not modelled as `None` here.
///
/// # Implementor contract
/// - **Answer, do not act.** Returning [`GateDecision::Allow`] is a statement
///   about permission; performing the side effect yourself is not this trait's
///   job and the runtime will still run the tool.
/// - **Be fast, or be async about it.** Both methods sit on the critical path of
///   every tool call and every injected block. A host that needs to park a turn
///   for human approval should do so inside `authorize_tool` and return
///   [`GateDecision::Prompted`] once it resolves — including on timeout, where
///   `approved: false` is the safe resolution.
/// - **Reserve `Err` for genuine failure.** A policy refusal is `Deny` /
///   `Block`, not an error. `Err` means the gate could not reach a verdict
///   (storage down, config unreadable), and callers must fail closed on it.
#[async_trait]
pub trait SecurityGate: Send + Sync {
    /// Consulted before every tool invocation.
    ///
    /// Called once per call, immediately before execution and after argument
    /// parsing, so the host sees the arguments the tool would actually receive.
    /// The runtime must not cache the verdict: the same call may be permitted
    /// now and refused a minute later because the host's tier, quota, or user
    /// consent changed underneath it.
    async fn authorize_tool(&self, call: &ToolCallRequest) -> Result<GateDecision>;

    /// Consulted before untrusted `text` enters the model context.
    ///
    /// `origin` tells the host where the text came from; see [`ContentOrigin`]
    /// for why that is provenance and not trust. The runtime must use
    /// [`ScreenOutcome::effective_text`] on the result rather than `text` — a
    /// [`ScreenOutcome::Redacted`] answer is binding, and a
    /// [`ScreenOutcome::Block`] means the content is dropped, not summarized or
    /// truncated into the prompt anyway.
    async fn screen_input(&self, text: &str, origin: ContentOrigin) -> Result<ScreenOutcome>;
}

// ── AllowAllSecurityGate ──────────────────────────────────────────────────────

/// A gate that permits every tool call and admits every input unchanged.
///
/// **For tests and local experiments only.** It performs no checks whatsoever:
/// with this gate installed, any tool the model can name it can run, with any
/// arguments, and any text from any origin reaches the model context verbatim —
/// including prompt-injection payloads fetched from the open web.
///
/// It exists so the crate's own tests and a host's first integration spike do
/// not have to hand-roll a stub, and so that landing [`SecurityGate`] requires
/// no host change. Shipping it in a product build defeats the entire point of
/// the trait. If you find yourself reaching for it outside `#[cfg(test)]`,
/// implement the trait instead — even a hard-coded allowlist is a real gate.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAllSecurityGate;

impl AllowAllSecurityGate {
    /// Creates the permissive test gate.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl SecurityGate for AllowAllSecurityGate {
    /// Always [`GateDecision::Allow`].
    ///
    /// Returns `Allow` rather than `Prompted { approved: true }` so nothing
    /// downstream can mistake a stub for a host that actually asked someone.
    async fn authorize_tool(&self, _call: &ToolCallRequest) -> Result<GateDecision> {
        Ok(GateDecision::Allow)
    }

    /// Always [`ScreenOutcome::Pass`] — the text is injected verbatim.
    async fn screen_input(&self, _text: &str, _origin: ContentOrigin) -> Result<ScreenOutcome> {
        Ok(ScreenOutcome::Pass)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "security_gate_tests.rs"]
mod tests;
