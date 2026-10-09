//! Turn and message lifecycle events for the superstep loop.
//!
//! The loop pushes to its working transcript from many places (assistant
//! replies, tool results, recovery nudges, steering, queued messages). Rather
//! than instrument each push, [`TurnTracker`] watches the transcript length and
//! announces whatever was appended since the last look, in order, at the points
//! the loop already treats as boundaries. That keeps every present and future
//! push site covered with no per-site code. Mutations that are *not* appends
//! must say so explicitly: [`TurnTracker::retract_to`] for pops and
//! [`TurnTracker::rebase`] for in-place rewrites.

use crate::context::RunContext;
use crate::events::{AgentEvent, EventSink};
use crate::ids::CallId;
use crate::runtime::PayloadCapture;
use tinyinference_llm::message::Message;

/// Tracks which transcript messages have been announced and which turn is open.
///
/// Lives on the [`RunContext`](crate::context::RunContext) so every site that
/// mutates the transcript can reach it.
#[derive(Debug, Default)]
pub(crate) struct TurnTracker {
    /// Number of initial input messages that were never announced.
    seed_len: usize,
    /// Messages `[0, announced)` have been announced (or are the seed input).
    announced: usize,
    /// Number of the most recently opened turn.
    turn: u32,
    /// The open turn and the transcript index it started at.
    open: Option<(u32, usize)>,
    /// Whether the seed has been fixed. A context that never went through
    /// [`TurnTracker::new`] (a graph run entered mid-flight) is seeded on first
    /// use by [`TurnTracker::ensure_seeded`].
    seeded: bool,
}

pub(crate) fn role_of(message: &Message) -> &'static str {
    match message {
        Message::System(_) => "system",
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::Tool(_) => "tool",
        Message::Custom(_) => "custom",
    }
}

impl TurnTracker {
    /// A tracker for a transcript that starts with `seed_len` input messages,
    /// which are not announced.
    pub(crate) fn new(seed_len: usize) -> Self {
        Self {
            announced: seed_len,
            seed_len,
            turn: 0,
            open: None,
            seeded: true,
        }
    }

    /// Continues numbering from `completed_turns` (a resumed run in a fresh
    /// runtime starts its tracker at zero) and, when `open_from` is given and no
    /// turn is open, re-opens the in-flight turn that began at that transcript
    /// index, without announcing it again.
    pub(crate) fn adopt(&mut self, completed_turns: u32, open_from: Option<usize>) {
        self.turn = self.turn.max(completed_turns);
        if let Some(start) = open_from
            && self.open.is_none()
        {
            tracing::debug!(
                target: "tinyagents::agent_loop",
                turn = self.turn,
                start,
                "[agent_loop] re-opened the in-flight turn after a resume"
            );
            self.open = Some((self.turn, start));
        }
    }

    /// Treats the first `len` messages as seed input if no seed was fixed yet.
    /// A no-op once seeded, so repeated node entries never swallow appends.
    pub(crate) fn ensure_seeded(&mut self, len: usize) {
        if !self.seeded {
            tracing::debug!(
                target: "tinyagents::agent_loop",
                seed_len = len,
                "[agent_loop] lifecycle tracker seeded on first use"
            );
            *self = Self::new(len);
        }
    }

    /// Announces every message appended since the last call, in order.
    pub(crate) fn flush(
        &mut self,
        events: &EventSink,
        capture: PayloadCapture,
        messages: &[Message],
    ) {
        // A shrunk transcript must have been reported through `retract_to` or
        // `rebase`; clamp defensively rather than index out of range.
        if messages.len() < self.announced {
            tracing::warn!(
                target: "tinyagents::agent_loop",
                announced = self.announced,
                len = messages.len(),
                "[agent_loop] transcript shrank without retract_to/rebase; re-basing the lifecycle cursor"
            );
            self.announced = messages.len();
        }
        for (index, message) in messages.iter().enumerate().skip(self.announced) {
            let (call_id, captured) = match message {
                Message::Tool(tool) => (
                    Some(CallId::new(tool.tool_call_id.clone())),
                    capture.tool_io,
                ),
                _ => (None, capture.model_io),
            };
            events.emit(AgentEvent::MessageAppended {
                role: role_of(message).to_string(),
                index,
                call_id,
                message: captured.then(|| to_value_logged(message)),
            });
        }
        self.announced = messages.len();
    }

    /// Reports that the transcript was truncated to `new_len` messages (a pop).
    /// Emits [`AgentEvent::MessageRetracted`] for each *announced* message
    /// removed, highest index first; removing a message that was never
    /// announced is silent.
    pub(crate) fn retract_to(&mut self, events: &EventSink, new_len: usize) {
        for index in (new_len.max(self.seed_len)..self.announced).rev() {
            events.emit(AgentEvent::MessageRetracted { index });
        }
        self.announced = self.announced.min(new_len.max(self.seed_len));
        if let Some((_, start)) = self.open.as_mut() {
            *start = (*start).min(new_len);
        }
    }

    /// Reports that the transcript was rewritten in place and now holds
    /// `new_len` messages. Call [`Self::flush`] *before* the mutation so
    /// pending appends are announced against the old transcript.
    pub(crate) fn rebase(&mut self, events: &EventSink, new_len: usize, reason: &str) {
        events.emit(AgentEvent::TranscriptRewritten {
            len: new_len,
            reason: reason.to_string(),
        });
        self.announced = new_len;
        if let Some((_, start)) = self.open.as_mut() {
            *start = (*start).min(new_len);
        }
    }

    /// Opens the next turn, first announcing pending messages and closing any
    /// turn still open (a recovery retry re-enters the model call without
    /// finishing its predecessor). Returns the new turn number.
    pub(crate) fn start_turn(
        &mut self,
        events: &EventSink,
        capture: PayloadCapture,
        messages: &[Message],
    ) -> u32 {
        self.close_turn(events, capture, messages);
        self.turn += 1;
        self.open = Some((self.turn, messages.len()));
        events.emit(AgentEvent::TurnStarted { turn: self.turn });
        self.turn
    }

    /// Announces pending messages and closes the open turn, if any, reporting
    /// the tool results it added to the transcript.
    pub(crate) fn close_turn(
        &mut self,
        events: &EventSink,
        capture: PayloadCapture,
        messages: &[Message],
    ) {
        self.flush(events, capture, messages);
        let Some((turn, start)) = self.open.take() else {
            return;
        };
        let tool_call_ids: Vec<CallId> = messages
            .get(start..)
            .unwrap_or_default()
            .iter()
            .filter_map(|message| match message {
                Message::Tool(tool) => Some(CallId::new(tool.tool_call_id.clone())),
                _ => None,
            })
            .collect();
        tracing::debug!(
            target: "tinyagents::agent_loop",
            turn,
            tool_results = tool_call_ids.len(),
            "[agent_loop] turn completed"
        );
        events.emit(AgentEvent::TurnCompleted {
            turn,
            tool_result_count: tool_call_ids.len(),
            tool_call_ids,
        });
    }
}

/// Serializes a value for an event payload, logging instead of silently
/// substituting `null` when serialization fails.
pub(crate) fn to_value_logged<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).unwrap_or_else(|error| {
        tracing::warn!(
            target: "tinyagents::agent_loop",
            %error,
            "[agent_loop] could not serialize an event payload; emitting null"
        );
        serde_json::Value::Null
    })
}

impl<Ctx> RunContext<Ctx> {
    /// Announces transcript appends not yet announced.
    pub(crate) fn flush_transcript(&mut self, capture: PayloadCapture, messages: &[Message]) {
        self.turns.flush(&self.events, capture, messages);
    }

    /// Opens the next turn (see [`TurnTracker::start_turn`]).
    pub(crate) fn start_turn(&mut self, capture: PayloadCapture, messages: &[Message]) -> u32 {
        self.turns.start_turn(&self.events, capture, messages)
    }

    /// Closes the open turn (see [`TurnTracker::close_turn`]).
    pub(crate) fn close_turn(&mut self, capture: PayloadCapture, messages: &[Message]) {
        self.turns.close_turn(&self.events, capture, messages);
    }

    /// Reports a pop: the transcript now holds `new_len` messages.
    pub(crate) fn retract_transcript(&mut self, new_len: usize) {
        self.turns.retract_to(&self.events, new_len);
    }

    /// Reports an in-place rewrite: the transcript now holds `new_len` messages.
    pub(crate) fn rebase_transcript(&mut self, new_len: usize, reason: &str) {
        self.turns.rebase(&self.events, new_len, reason);
    }
}

impl<Ctx> RunContext<Ctx> {
    /// See [`TurnTracker::adopt`].
    pub(crate) fn adopt_turn_state(&mut self, completed_turns: u32, open_from: Option<usize>) {
        self.turns.adopt(completed_turns, open_from);
    }

    /// Fixes the lifecycle seed at `len` messages unless one is already set.
    pub(crate) fn ensure_turn_tracker_seeded(&mut self, len: usize) {
        self.turns.ensure_seeded(len);
    }
}
