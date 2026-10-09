//! Live mid-call progress from a running tool, and the guard that keeps it
//! ordered.
//!
//! A tool reports through [`tinytools::ToolRunContext::report_progress`]. The
//! loop gives every executing call a [`ToolProgressGate`]: the sink behind
//! that method. The gate turns each accepted update into an
//! [`AgentEvent::ToolProgress`] **immediately** (so a UI sees activity while
//! the tool is still running) and queues a matching
//! [`ToolDelta`] for the middleware stack's `on_tool_delta` hook.
//!
//! # Ordering
//!
//! The gate is the single authority on whether an update is still wanted:
//!
//! - Updates are accepted only while the call is *open*. The loop
//!   [closes](ToolProgressGate::close) the gate the moment the tool's future
//!   resolves (return, error, timeout, or cancellation) and **before** it
//!   emits that call's terminal `ToolCompleted` / `ToolFailed`. An update a
//!   detached task reports afterwards is dropped, never emitted late. This is
//!   the same `acceptingUpdates` guard pi's agent loop uses around `onUpdate`.
//! - The open check and the event emission happen under one lock, and so does
//!   closing. There is no window in which a late update passes the check, the
//!   terminal event is emitted, and the progress event then lands after it.
//! - Calls in one concurrent batch each have their own gate, so progress
//!   interleaves across calls but each call's progress precedes its own
//!   terminal event.
//!
//! # Middleware
//!
//! `on_tool_delta` needs `&mut RunContext`, which the loop lends to the
//! executing tool (serial path) or shares read-only with its siblings
//! (concurrent path), so the hook cannot run *during* the call. The gate
//! therefore queues deltas and the loop replays them, in order, to the stack
//! right after the call settles and before its terminal event. Middleware
//! observes progress; the live event is already out, so a rewrite of the delta
//! is not reflected in it.
//!
//! # Flooding
//!
//! A tool in a tight loop must not drown the event stream. Each gate admits at
//! most [`ToolProgressLimits::max_per_window`] events per
//! [`ToolProgressLimits::window`] (default 32 per second). Beyond that, updates
//! are **coalesced**: the newest value of each field replaces the held one.
//! There is no timer: the held update is emitted on the first accepted update
//! of the next window, or when the gate closes (the call settles or its
//! future is dropped; that final flush is exempt from the window limit), so the
//! final state is never lost but a tool that goes
//! quiet right after a burst shows its last state only at settle. Coalesced-away
//! updates produce no event and no middleware delta. The limits are fixed at
//! the defaults; they are crate-private rather than a policy knob.
//!
//! The middleware replay queue is bounded too: at most
//! [`MAX_PENDING_DELTAS`] deltas are retained (the newest win, the dropped
//! count is logged), each delta's `content` is capped at
//! [`MAX_DELTA_CONTENT_BYTES`] so a huge `partial` is never serialized in full,
//! and nothing is queued at all when the run has no middleware.
//!
//! # Listeners
//!
//! Events are emitted while the gate's lock is held (that is what makes the
//! drop-after-settle guarantee race-free). An [`EventListener`](crate::events::EventListener)
//! must therefore not call `report_progress` re-entrantly: it would deadlock on
//! that call's own gate.
//!
//! # Reaching the gate
//!
//! The loop does not hand the gate to a `ToolDispatch` — that trait is
//! implemented outside this crate. It scopes the gate in a task-local around
//! the dispatch future instead, and [`ToolExecutionContext::from_run_context`]
//! picks it up when the dispatch builds the tool's context for the matching
//! call id. A dispatch must therefore build its context **inside**
//! `ToolDispatch::execute`'s future (not ahead of time or on another task), or
//! the context carries no sink. The scope also closes the gate when the future
//! is *dropped* (run cancelled mid-call), so a task the tool spawned cannot
//! emit afterwards. The sink then lives in the context, so a tool may move it into a
//! spawned task; the gate's `open` flag, not the task-local, is what silences
//! it later.

use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use tinyinference_llm::tool::ToolDelta;
use tinytools::{ProgressSink, ToolProgress};

use crate::events::{AgentEvent, EventSink};
use crate::ids::CallId;
mod types;

use self::types::GateState;
pub(crate) use self::types::{ToolProgressGate, ToolProgressLimits};

const MAX_PENDING_DELTAS: usize = 64;
const MAX_DELTA_CONTENT_BYTES: usize = 4096;

tokio::task_local! {
    static CURRENT: Arc<ToolProgressGate>;
}

impl ToolProgressGate {
    pub(crate) fn new(
        call_id: CallId,
        tool_name: impl Into<String>,
        events: EventSink,
        limits: ToolProgressLimits,
        queue_deltas: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            call_id,
            tool_name: tool_name.into(),
            events,
            limits,
            queue_deltas,
            state: Mutex::new(GateState::default()),
        })
    }

    /// The sink a tool reports through. Holds the gate alive, and stays safe
    /// to call after the gate is closed.
    pub(crate) fn sink(self: &Arc<Self>) -> ProgressSink {
        let gate = Arc::clone(self);
        ProgressSink::new(move |update| gate.accept(update))
    }

    /// Runs `future` with this gate visible to
    /// [`ToolExecutionContext::from_run_context`](super::ToolExecutionContext::from_run_context),
    /// and closes the gate when the future finishes **or is dropped**.
    pub(crate) fn scope<F: Future>(self: &Arc<Self>, future: F) -> impl Future<Output = F::Output> {
        let guard = CloseOnDrop(Arc::clone(self));
        CURRENT.scope(Arc::clone(self), async move {
            let _guard = guard;
            future.await
        })
    }

    /// The sink for `call_id`, when a gate for exactly that call is in scope.
    pub(crate) fn current_sink_for(call_id: &CallId) -> Option<ProgressSink> {
        CURRENT
            .try_with(|gate| {
                if &gate.call_id == call_id {
                    Some(gate.sink())
                } else {
                    tracing::trace!(
                        target: "tinyagents::tool_progress",
                        wanted = %call_id,
                        scoped = %gate.call_id,
                        "[tool_progress] scoped gate belongs to another call; no sink"
                    );
                    None
                }
            })
            .ok()
            .flatten()
    }

    /// Stops accepting updates, first emitting any coalesced one so the
    /// call's final reported state is never lost. Idempotent.
    pub(crate) fn close(&self) {
        let mut state = self.lock();
        if state.closed {
            return;
        }
        if let Some(held) = state.held.take() {
            // The terminal flush is exempt from the per-window limit: it is at
            // most one extra event, and it is what keeps the final state.
            self.emit(&mut state, held);
        }
        state.closed = true;
    }

    /// Drains the deltas for events already emitted, for the middleware replay.
    pub(crate) fn take_pending(&self) -> Vec<ToolDelta> {
        let mut state = self.lock();
        if state.evicted > 0 {
            tracing::debug!(
                target: "tinyagents::tool_progress",
                call_id = %self.call_id,
                evicted = state.evicted,
                "[tool_progress] replay queue overflowed; oldest deltas dropped"
            );
            state.evicted = 0;
        }
        std::mem::take(&mut state.pending).into()
    }

    fn accept(&self, update: ToolProgress) {
        if update.is_empty() {
            return;
        }
        // Bound before retaining: a held (coalesced) update must not pin an
        // arbitrarily large allocation until the call settles.
        let update = ToolProgress {
            message: update.message.as_deref().map(bounded_text),
            fraction: update.fraction,
            partial: update.partial.as_ref().map(bounded_value),
        };
        let mut state = self.lock();
        if state.closed {
            tracing::debug!(
                target: "tinyagents::tool_progress",
                call_id = %self.call_id,
                tool = %self.tool_name,
                "[tool_progress] dropped update reported after the call settled"
            );
            return;
        }
        let now = Instant::now();
        let window_open = state
            .window_start
            .is_some_and(|start| now.duration_since(start) < self.limits.window);
        if !window_open {
            state.window_start = Some(now);
            state.emitted_in_window = 0;
            if let Some(held) = state.held.take() {
                self.emit(&mut state, held);
            }
        }
        if state.emitted_in_window < self.limits.max_per_window {
            self.emit(&mut state, update);
        } else {
            tracing::trace!(
                target: "tinyagents::tool_progress",
                call_id = %self.call_id,
                tool = %self.tool_name,
                "[tool_progress] coalescing update past the per-window limit"
            );
            state.held = Some(match state.held.take() {
                Some(older) => merge(older, update),
                None => update,
            });
        }
    }

    fn emit(&self, state: &mut GateState, update: ToolProgress) {
        state.emitted_in_window += 1;
        // `ToolProgress::fraction` is a public field, so do not trust it.
        let fraction = update
            .fraction
            .filter(|f| !f.is_nan())
            .map(|f| f.clamp(0.0, 1.0));
        if self.queue_deltas {
            if state.pending.len() >= MAX_PENDING_DELTAS {
                state.pending.pop_front();
                state.evicted += 1;
            }
            state.pending.push_back(ToolDelta {
                call_id: self.call_id.as_str().to_string(),
                content: delta_content(&update, fraction),
                tool_name: Some(self.tool_name.clone()),
                ..ToolDelta::default()
            });
        }
        let message = update
            .message
            .map(|message| bounded_text(&message))
            .unwrap_or_default();
        let partial = update.partial.map(|partial| bounded_value(&partial));
        self.events.emit(AgentEvent::ToolProgressDetail {
            call_id: self.call_id.clone(),
            message,
            fraction,
            partial,
        });
    }

    fn lock(&self) -> MutexGuard<'_, GateState> {
        // A poisoned lock only means a listener panicked mid-emit; the state
        // is still coherent, and progress must not take the run down.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Newer fields replace older ones; a field the newer update omits is kept.
fn merge(older: ToolProgress, newer: ToolProgress) -> ToolProgress {
    ToolProgress {
        message: newer.message.or(older.message),
        fraction: newer.fraction.or(older.fraction),
        partial: newer.partial.or(older.partial),
    }
}

/// Closes its gate when dropped, so cancelling a run mid-call silences a
/// sink a spawned task still holds.
struct CloseOnDrop(Arc<ToolProgressGate>);

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// The text the middleware sees: the status line, else the partial output,
/// else the fraction as a percentage. Capped at [`MAX_DELTA_CONTENT_BYTES`];
/// a partial is serialized through a writer that stops at the cap, so a huge
/// value costs a bounded amount of work.
fn delta_content(update: &ToolProgress, fraction: Option<f32>) -> String {
    let mut content = if let Some(message) = &update.message {
        bounded_text(message)
    } else if let Some(partial) = &update.partial {
        bounded_json(partial)
    } else {
        fraction
            .map(|fraction| format!("{:.0}%", fraction * 100.0))
            .unwrap_or_default()
    };
    truncate_at_char_boundary(&mut content, MAX_DELTA_CONTENT_BYTES);
    content
}

fn truncate_at_char_boundary(text: &mut String, max: usize) {
    if text.len() > max {
        let mut end = max;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
}

/// Copies at most [`MAX_DELTA_CONTENT_BYTES`] of `text`, never the whole of it.
fn bounded_text(text: &str) -> String {
    let mut end = text.len().min(MAX_DELTA_CONTENT_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

fn bounded_value(value: &serde_json::Value) -> serde_json::Value {
    let encoded = bounded_json(value);
    serde_json::from_str(&encoded).unwrap_or_else(|_| {
        // Truncated JSON is not valid JSON: carry it as a string, trimmed so
        // the *escaped* form (quotes and backslashes grow) still fits the cap.
        let mut budget = MAX_DELTA_CONTENT_BYTES.saturating_sub(2);
        let mut text = String::new();
        for ch in encoded.chars() {
            let cost = serde_json::to_string(&ch.to_string()).map_or(budget + 1, |s| s.len() - 2);
            if cost > budget {
                break;
            }
            budget -= cost;
            text.push(ch);
        }
        serde_json::Value::String(text)
    })
}

/// Serializes `value`, stopping once [`MAX_DELTA_CONTENT_BYTES`] are written.
fn bounded_json(value: &serde_json::Value) -> String {
    struct Bounded(Vec<u8>);
    impl std::io::Write for Bounded {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let room = MAX_DELTA_CONTENT_BYTES.saturating_sub(self.0.len());
            if room == 0 {
                return Err(std::io::ErrorKind::WriteZero.into());
            }
            let take = room.min(buf.len());
            self.0.extend_from_slice(&buf[..take]);
            Ok(take)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut out = Bounded(Vec::new());
    // An error here is just the cap being hit; keep what was written.
    let _ = serde_json::to_writer(&mut out, value);
    String::from_utf8_lossy(&out.0).into_owned()
}

#[cfg(test)]
#[path = "progress_tests.rs"]
mod tests;
