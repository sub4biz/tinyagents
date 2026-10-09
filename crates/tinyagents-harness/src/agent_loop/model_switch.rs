//! Live model switching: applies a pending
//! [`SteeringCommand::SwitchModel`][crate::steering::SteeringCommand::SwitchModel]
//! to the turn's request *before* the model binding is resolved.
//!
//! The steering checkpoint only records the requested name on the run's
//! [`SteeringHandle`][crate::steering::SteeringHandle]. This module turns it
//! into `request.model` -- the same per-request override an SDK caller or a
//! `before_model` middleware uses -- so the ordinary resolution path then
//! produces the binding and everything downstream (`ModelStarted`/`ModelFailed`
//! names, `ctx.model_profile`, the handoff transform, the host budget
//! estimate, the tool-dialect decision) follows from the new model with no
//! special cases. An override that resolution would have to skip is rejected
//! up front instead, through the existing
//! [`AgentEvent::ModelOverrideSkipped`] diagnostic.

use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// The run's sticky steered model name, when one is pending and the run is
    /// not host-routed (a host resolver owns routing there, so a switch can
    /// never apply). Used to point the pre-middleware profile preview at the
    /// model the call is about to use.
    pub(super) fn steered_model(&self, ctx: &RunContext<Ctx>) -> Option<String> {
        let requested = ctx.steering.as_ref()?.model_override()?;
        matches!(
            crate::runtime::host_invocation_binding::<State, Ctx>(ctx),
            Ok(None)
        )
        .then_some(requested)
    }

    /// Applies the run's pending model switch to `request`.
    ///
    /// A switch whose name is registered and passes the request's
    /// capability/lifecycle gate sets `request.model` (winning over a model a
    /// middleware picked: an operator's switch is the stronger instruction).
    /// Otherwise -- unknown name, ineligible model, or a host-routed run -- the
    /// switch is dropped from the handle (so it is reported once, not on every
    /// call), [`AgentEvent::ModelOverrideSkipped`] names the model the call
    /// resolves to instead, and a rejected
    /// [`AgentEvent::Steered`] is emitted. Never fails the run.
    ///
    /// Called both before and after `before_model` middleware: the second
    /// call re-validates against the capabilities middleware added and
    /// re-asserts the switch over a model a middleware selected. Neither call
    /// reports the switch as applied; [`Self::announce_applied_model_switch`]
    /// does that once the request is actually about to be dispatched.
    /// `model_before_switch` is what `request.model` held before the first
    /// call; a rejection puts it back when `request.model` still carries the
    /// rejected name, so that name never reaches resolution or an adapter that
    /// honours `request.model`. A model a middleware picked is left alone.
    pub(super) fn apply_steered_model_switch(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &mut ModelRequest,
        model_before_switch: &Option<String>,
    ) {
        let Some(handle) = ctx.steering.clone() else {
            return;
        };
        let Some(requested) = handle.model_override() else {
            return;
        };
        let hosted = !matches!(
            crate::runtime::host_invocation_binding::<State, Ctx>(ctx),
            Ok(None)
        );
        let eligible = !hosted
            && self.models.get(&requested).is_some_and(|model| {
                model_eligible(
                    model.as_ref(),
                    request.required_capabilities.as_ref(),
                    false,
                )
            });
        if eligible {
            if request.model.as_deref() != Some(requested.as_str()) {
                match request.model.as_deref() {
                    Some(chosen) => tracing::debug!(
                        target: "tinyagents::steering",
                        run_id = %ctx.run_id(),
                        chosen = %chosen,
                        to = %requested,
                        "[steering] model switch overrides a model chosen by the request or middleware"
                    ),
                    None => tracing::debug!(
                        target: "tinyagents::steering",
                        run_id = %ctx.run_id(),
                        to = %requested,
                        "[steering] model switch applied at the model-call boundary"
                    ),
                }
                request.model = Some(requested);
            }
            return;
        }
        if request.model.as_deref() == Some(requested.as_str()) {
            request.model = model_before_switch.clone();
        }
        let resolved = self
            .models
            .resolve_request(request, None, None)
            .map_or_else(|| "<default>".to_string(), |binding| binding.resolved.name);
        tracing::warn!(
            target: "tinyagents::steering",
            run_id = %ctx.run_id(),
            requested = %requested,
            current = %resolved,
            hosted,
            "[steering] model switch rejected; keeping the current model"
        );
        let already_reported = handle.reject_model_override();
        ctx.emit(AgentEvent::ModelOverrideSkipped {
            requested,
            resolved,
        });
        // A switch already reported as applied keeps its single outcome.
        if !already_reported {
            ctx.emit(AgentEvent::Steered {
                command_kind: crate::steering::SteeringCommandKind::SwitchModel
                    .as_str()
                    .to_string(),
                accepted: false,
            });
        }
    }

    /// Reports a steered switch as applied (`Steered { accepted: true }`, once
    /// per switch) when `request` carries the switched model and is about to be
    /// dispatched. Called after the pre-call control checkpoint, so a switch
    /// whose request never reaches a model call produces no outcome.
    pub(super) fn announce_applied_model_switch(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
    ) {
        let (Some(handle), Some(model)) = (ctx.steering.clone(), request.model.as_deref()) else {
            return;
        };
        if handle.announce_model_override(model) {
            ctx.emit(AgentEvent::Steered {
                command_kind: crate::steering::SteeringCommandKind::SwitchModel
                    .as_str()
                    .to_string(),
                accepted: true,
            });
        }
    }

    /// Whether a fallback walk starting at `cursor` should begin at the head of
    /// the fallback chain rather than after `cursor`.
    ///
    /// True only when `cursor` is the run's steered model and that model is not
    /// in the chain: the chain then describes the *original* primary's
    /// fallbacks, and every entry is still untried, so the walk starts from the
    /// first one. A steered model that is in the chain simply continues from its
    /// own position.
    pub(super) fn fallback_starts_at_chain_head(
        &self,
        ctx: &RunContext<Ctx>,
        cursor: &str,
    ) -> bool {
        let steered = ctx
            .steering
            .as_ref()
            .and_then(|handle| handle.model_override())
            .is_some_and(|name| name == cursor);
        steered
            && self
                .policy
                .fallback
                .as_ref()
                .is_some_and(|chain| !chain.models.iter().any(|name| name == cursor))
    }
}
