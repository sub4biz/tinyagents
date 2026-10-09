use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tinyinference_llm::tool::ToolDelta;
use tinytools::ToolProgress;

use crate::events::EventSink;
use crate::ids::CallId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ToolProgressLimits {
    pub(crate) max_per_window: usize,
    pub(crate) window: Duration,
}

impl Default for ToolProgressLimits {
    fn default() -> Self {
        Self {
            max_per_window: 32,
            window: Duration::from_secs(1),
        }
    }
}

#[derive(Default)]
pub(super) struct GateState {
    pub(super) closed: bool,
    pub(super) pending: VecDeque<ToolDelta>,
    pub(super) evicted: usize,
    pub(super) window_start: Option<Instant>,
    pub(super) emitted_in_window: usize,
    pub(super) held: Option<ToolProgress>,
}

pub(crate) struct ToolProgressGate {
    pub(super) call_id: CallId,
    pub(super) tool_name: String,
    pub(super) events: EventSink,
    pub(super) limits: ToolProgressLimits,
    pub(super) queue_deltas: bool,
    pub(super) state: Mutex<GateState>,
}
