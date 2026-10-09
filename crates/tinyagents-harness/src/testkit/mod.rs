//! Test-support toolkit for the harness.
//!
//! In the recursive architecture this is how nested, model-driven behaviour is
//! made *deterministically testable*: scripted/streaming model doubles, fake
//! tools, controllable clocks/ids, and a [`Trajectory`] over recorded
//! [`AgentEvent`]s let tests assert exactly what an agent — and the sub-agents
//! and sub-graphs it spawns — did, all without a live provider. The same
//! [`EventRecorder`] that observes a top-level run also captures child-run
//! events fanned onto a shared sink, so recursion is observable in tests.
//!
//! Provides deterministic doubles and trajectory assertions that make it
//! possible to test model-and-tool workflows without live providers.
//!
//! # Contents
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`ScriptedModel`] | Pre-loaded `ChatModel` returning queued responses |
//! | [`SchemaDrivenModel`] | `ChatModel` that calls every declared tool once with schema-generated args |
//! | [`SlowModel`] | `ChatModel` that sleeps before replying (timeout testing) |
//! | [`FakeTool`] | Configurable `Tool` recording invocations |
//! | [`DeterministicClock`] | Controllable millisecond clock |
//! | [`DeterministicIds`] | Monotonic `"{prefix}-N"` id generator |
//! | [`EventRecorder`] | Captures `AgentEvent`s from an `EventSink` |
//! | [`Trajectory`] | Structural assertions over a sequence of events |
//! | [`text_response`], [`tool_call_response`] | Canned model responses for scripted doubles |

mod types;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use crate::error::{Result, TinyAgentsError};
use crate::events::{AgentEvent, EventSink, RecordingListener};
use serde_json::Value;
use tinyinference_llm::message::{AssistantMessage, MessageDelta};
use tinyinference_llm::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem, StreamAccumulator,
};
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolResult};

pub use types::*;

// ---------------------------------------------------------------------------
// Canned responses
// ---------------------------------------------------------------------------

/// A plain-text assistant response that finished with `stop`.
///
/// Chain [`ModelResponse::with_usage`] for token counts.
pub fn text_response(text: impl Into<String>) -> ModelResponse {
    ModelResponse::assistant(text).with_finish_reason("stop")
}

/// An assistant response that makes exactly `call` and says nothing, finishing
/// with `tool_calls`. The message id is `msg-{call id}`, so each scripted turn
/// is distinguishable in a transcript.
///
/// Chain [`ModelResponse::with_usage`] for token counts.
pub fn tool_call_response(call: ToolCall) -> ModelResponse {
    let mut response = ModelResponse::assistant(String::new()).with_finish_reason("tool_calls");
    response.message.id = Some(format!("msg-{}", call.id));
    response.message.content = Vec::new();
    response.message.tool_calls = vec![call];
    response
}

// ---------------------------------------------------------------------------
// StreamingMock
// ---------------------------------------------------------------------------

impl StreamingMock {
    /// Creates a streaming mock that replays the given scripted items verbatim.
    ///
    /// The items should follow the streaming contract: a leading
    /// [`ModelStreamItem::Started`], any number of delta items, and a terminal
    /// [`ModelStreamItem::Completed`] or [`ModelStreamItem::Failed`].
    pub fn new(items: Vec<ModelStreamItem>) -> Self {
        Self {
            items,
            calls: Mutex::new(0),
            profile: None,
        }
    }

    /// Attaches a capability profile, returned by
    /// [`tinyinference_llm::model::ChatModel::profile`] for the rest of this
    /// mock's lifetime.
    ///
    /// Use this to exercise profile-driven streaming normalization (thinking
    /// tags, leading-whitespace stripping) against a scripted stream.
    #[must_use]
    pub fn with_profile(mut self, profile: tinyinference_llm::model::ModelProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Builds a streaming mock from text chunks.
    ///
    /// Produces a [`ModelStreamItem::Started`], one
    /// [`ModelStreamItem::MessageDelta`] per chunk, and a terminal
    /// [`ModelStreamItem::Completed`] carrying the concatenated text as the
    /// merged assistant response.
    pub fn from_text_chunks<S: AsRef<str>>(chunks: impl IntoIterator<Item = S>) -> Self {
        let mut items = vec![ModelStreamItem::Started];
        let mut full = String::new();
        for chunk in chunks {
            let text = chunk.as_ref().to_string();
            full.push_str(&text);
            items.push(ModelStreamItem::MessageDelta(MessageDelta {
                text,
                reasoning: String::new(),
                tool_call: None,
            }));
        }
        items.push(ModelStreamItem::Completed(ModelResponse::assistant(full)));
        Self::new(items)
    }

    /// Returns the number of `stream`/`invoke` calls made so far.
    pub fn call_count(&self) -> u64 {
        *self
            .calls
            .lock()
            .expect("StreamingMock calls lock poisoned")
    }

    /// Folds the scripted items into the response they merge to.
    fn merged_response(&self) -> tinyinference_llm::Result<ModelResponse> {
        let mut accumulator = StreamAccumulator::new();
        for item in &self.items {
            accumulator.push(item);
        }
        accumulator.finish()
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for StreamingMock {
    /// Returns the merged response the scripted stream folds into.
    async fn invoke(
        &self,
        _state: &State,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self
            .calls
            .lock()
            .expect("StreamingMock calls lock poisoned") += 1;
        self.merged_response()
    }

    /// Replays the scripted items as a real [`ModelStream`].
    async fn stream(
        &self,
        _state: &State,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelStream> {
        *self
            .calls
            .lock()
            .expect("StreamingMock calls lock poisoned") += 1;
        let items = self.items.clone();
        Ok(ModelStream::new(Box::pin(futures::stream::iter(items))))
    }

    fn profile(&self) -> Option<&tinyinference_llm::model::ModelProfile> {
        self.profile.as_ref()
    }
}

// ---------------------------------------------------------------------------
// SlowModel
// ---------------------------------------------------------------------------

impl SlowModel {
    /// Creates a slow model that sleeps `delay` before returning `reply`.
    pub fn new(delay: Duration, reply: impl Into<String>) -> Self {
        Self {
            delay,
            reply: reply.into(),
            calls: Mutex::new(0),
        }
    }

    /// Returns the number of `invoke` calls made so far.
    pub fn call_count(&self) -> u64 {
        *self.calls.lock().expect("SlowModel calls lock poisoned")
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for SlowModel {
    /// Sleeps for the configured delay, then returns the fixed reply.
    async fn invoke(
        &self,
        _state: &State,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        {
            // Bump the counter in its own scope so the guard is dropped before
            // the `.await` (a `MutexGuard` is not `Send`).
            *self.calls.lock().expect("SlowModel calls lock poisoned") += 1;
        }
        tokio::time::sleep(self.delay).await;
        Ok(ModelResponse::assistant(self.reply.clone()))
    }
}

// ---------------------------------------------------------------------------
// ScriptedModel
// ---------------------------------------------------------------------------

impl ScriptedModel {
    /// Creates a scripted model that will return `responses` in order.
    ///
    /// The first element is returned on the first `invoke`, the second on the
    /// second call, and so on. When the queue is drained, subsequent calls
    /// return [`TinyAgentsError::Model`].
    pub fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            queue: Mutex::new(VecDeque::from(responses)),
            received: Mutex::new(Vec::new()),
            profile: None,
        }
    }

    /// Attaches a capability profile, returned by
    /// [`tinyinference_llm::model::ChatModel::profile`] for the rest of this
    /// model's lifetime.
    ///
    /// Use this to exercise profile-driven harness behavior (schema
    /// transforms, structured-output mode selection, thinking-tag
    /// extraction, reasoning-level mapping) against a scripted response.
    #[must_use]
    pub fn with_profile(mut self, profile: tinyinference_llm::model::ModelProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Creates a scripted model from a list of plain text replies.
    ///
    /// Each string in `texts` becomes a [`ModelResponse::assistant`] wrapping
    /// that text. This is the most concise constructor for text-only tests.
    pub fn replies<S: AsRef<str>>(texts: Vec<S>) -> Self {
        let responses = texts
            .into_iter()
            .map(|t| ModelResponse::assistant(t.as_ref()))
            .collect();
        Self::new(responses)
    }

    /// Returns a snapshot of every [`ModelRequest`] received by `invoke`, in
    /// call order.
    ///
    /// Use this to assert on the exact messages, tools, or parameters passed to
    /// the model by the component under test.
    pub fn requests(&self) -> Vec<ModelRequest> {
        self.received
            .lock()
            .expect("ScriptedModel received lock poisoned")
            .clone()
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for ScriptedModel {
    /// Pops the next response from the queue and records the received request.
    ///
    /// Returns [`TinyAgentsError::Model`] when the queue is exhausted so the
    /// test gets a clear message rather than a thread panic.
    async fn invoke(
        &self,
        _state: &State,
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.received
            .lock()
            .expect("ScriptedModel received lock poisoned")
            .push(request);

        self.queue
            .lock()
            .expect("ScriptedModel queue lock poisoned")
            .pop_front()
            .ok_or_else(|| {
                tinyinference_llm::Error::Model(
                    "ScriptedModel: response queue is exhausted; no more scripted responses"
                        .to_string(),
                )
            })
    }

    fn profile(&self) -> Option<&tinyinference_llm::model::ModelProfile> {
        self.profile.as_ref()
    }
}

// ---------------------------------------------------------------------------
// SchemaDrivenModel
// ---------------------------------------------------------------------------

impl SchemaDrivenModel {
    /// Creates a model that calls every tool declared on a request once,
    /// with schema-generated arguments, then returns `final_response`.
    pub fn new(final_response: ModelResponse) -> Self {
        Self {
            final_response,
            calls: Mutex::new(0),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Creates a model whose configured final response is plain assistant
    /// text.
    pub fn with_final_text(text: impl Into<String>) -> Self {
        Self::new(ModelResponse::assistant(text))
    }

    /// Number of `invoke`/`stream` calls made so far.
    pub fn call_count(&self) -> u64 {
        *self
            .calls
            .lock()
            .expect("SchemaDrivenModel calls lock poisoned")
    }

    /// Every request received by `invoke`, in call order.
    pub fn requests(&self) -> Vec<ModelRequest> {
        self.received
            .lock()
            .expect("SchemaDrivenModel received lock poisoned")
            .clone()
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for SchemaDrivenModel {
    async fn invoke(
        &self,
        _state: &State,
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let index = {
            let mut calls = self
                .calls
                .lock()
                .expect("SchemaDrivenModel calls lock poisoned");
            let index = *calls;
            *calls += 1;
            index as usize
        };
        self.received
            .lock()
            .expect("SchemaDrivenModel received lock poisoned")
            .push(request.clone());

        let Some(tool) = request.tools.get(index) else {
            return Ok(self.final_response.clone());
        };
        let arguments = generate_args_from_schema(&tool.parameters);
        Ok(ModelResponse {
            message: AssistantMessage {
                id: None,
                content: Vec::new(),
                tool_calls: vec![ToolCall::new(
                    format!("schema-driven-{index}"),
                    tool.name.clone(),
                    arguments,
                )],
                usage: None,
                origin: None,
            },
            usage: None,
            finish_reason: None,
            raw: None,
            resolved_model: None,
            continue_turn: None,
            served_from_cache: false,
            correlation: request.correlation,
            resolved_route: None,
        })
    }
}

/// Synthesizes a JSON value satisfying `schema`'s declared shape.
///
/// A minimal, deterministic JSON-Schema-to-value generator: an `object`
/// schema recurses into every declared property, an `array` schema produces
/// a single-element array from its `items` schema, `string`/`integer`/
/// `number`/`boolean` produce a fixed placeholder of that type, and anything
/// unrecognized (including a schema with no `type`) produces `null` — except
/// a top-level object with `properties` and no explicit `type`, which is
/// still treated as an object. This is intended for
/// [`SchemaDrivenModel`], not as a general JSON Schema example generator: it
/// does not honor `enum`, `const`, `minimum`/`maximum`, `pattern`, or other
/// constraining keywords, and always fills every declared property
/// regardless of `required`.
pub fn generate_args_from_schema(schema: &Value) -> Value {
    let object_like = schema.get("type").and_then(Value::as_str) == Some("object")
        || (schema.get("type").is_none() && schema.get("properties").is_some());
    if object_like {
        let mut object = serde_json::Map::new();
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (name, property_schema) in properties {
                object.insert(name.clone(), generate_args_from_schema(property_schema));
            }
        }
        return Value::Object(object);
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => Value::String("test".to_string()),
        Some("integer") => Value::from(0),
        Some("number") => Value::from(0.0),
        Some("boolean") => Value::Bool(false),
        Some("array") => {
            let items_schema = schema.get("items").cloned().unwrap_or(Value::Null);
            Value::Array(vec![generate_args_from_schema(&items_schema)])
        }
        _ => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// FakeTool
// ---------------------------------------------------------------------------

impl FakeTool {
    /// Creates a `FakeTool` with the given name that returns an empty string
    /// result on every invocation.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            tool_description: format!("Fake tool: {name}"),
            tool_name: name,
            behavior: FakeToolBehavior::Return(String::new()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Creates a `FakeTool` that returns `content` as plain text on every
    /// successful invocation.
    pub fn returning(name: impl Into<String>, content: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            tool_description: format!("Fake tool: {name}"),
            tool_name: name,
            behavior: FakeToolBehavior::Return(content.into()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Creates a `FakeTool` that always returns
    /// a foreign `anyhow` error carrying `message`.
    pub fn failing(name: impl Into<String>, message: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            tool_description: format!("Fake tool: {name}"),
            tool_name: name,
            behavior: FakeToolBehavior::Fail(message.into()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Returns a snapshot of every [`ToolCall`](tinytools::ToolCall) received by this tool, in
    /// invocation order.
    pub fn calls(&self) -> Vec<serde_json::Value> {
        self.received
            .lock()
            .expect("FakeTool received lock poisoned")
            .clone()
    }
}

#[async_trait]
impl Tool for FakeTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    /// Returns a minimal schema advertising no required parameters.
    fn parameters_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {}, "required": [] })
    }

    /// Records the call and then either returns a fixed result or an error,
    /// depending on how the tool was constructed.
    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.received
            .lock()
            .expect("FakeTool received lock poisoned")
            .push(arguments);

        match &self.behavior {
            FakeToolBehavior::Return(content) => Ok(ToolResult::success(content.clone())),
            FakeToolBehavior::Fail(message) => Err(anyhow::anyhow!(message.clone())),
        }
    }
}

// ---------------------------------------------------------------------------
// DeterministicClock
// ---------------------------------------------------------------------------

impl DeterministicClock {
    /// Creates a new clock starting at `start_millis` milliseconds.
    pub fn new(start_millis: u64) -> Self {
        Self {
            millis: Mutex::new(start_millis),
        }
    }

    /// Returns the current clock time in milliseconds.
    pub fn now_millis(&self) -> u64 {
        *self
            .millis
            .lock()
            .expect("DeterministicClock lock poisoned")
    }

    /// Advances the clock forward by `ms` milliseconds.
    ///
    /// The clock never advances on its own; this is the only way to move it
    /// forward, keeping test timing fully deterministic.
    pub fn advance(&self, ms: u64) {
        *self
            .millis
            .lock()
            .expect("DeterministicClock lock poisoned") += ms;
    }
}

impl Default for DeterministicClock {
    /// Creates a clock starting at epoch zero (0 ms).
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// DeterministicIds
// ---------------------------------------------------------------------------

impl DeterministicIds {
    /// Creates a new generator with the given `prefix`.
    ///
    /// The first call to [`DeterministicIds::next`] returns `"{prefix}-0"`, the
    /// second returns `"{prefix}-1"`, and so on.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            counter: Mutex::new(0),
        }
    }

    /// Returns the next id in the sequence and increments the internal counter.
    pub fn next(&self) -> String {
        let mut counter = self.counter.lock().expect("DeterministicIds lock poisoned");
        let id = format!("{}-{}", self.prefix, *counter);
        *counter += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// EventRecorder
// ---------------------------------------------------------------------------

impl EventRecorder {
    /// Creates a new recorder with an empty buffer.
    ///
    /// The internal [`RecordingListener`] is subscribed to the internal
    /// [`EventSink`] immediately; callers only need to obtain the sink via
    /// [`EventRecorder::sink`] and pass it to the component under test.
    pub fn new() -> Self {
        let listener = Arc::new(RecordingListener::new());
        let sink = EventSink::new();
        sink.subscribe(listener.clone());
        Self { listener, sink }
    }

    /// Returns a clone of the internal [`EventSink`] that the recorder is
    /// listening to.
    ///
    /// Pass this sink to the component under test so its emitted events are
    /// captured.
    pub fn sink(&self) -> EventSink {
        self.sink.clone()
    }

    /// Returns a snapshot of the raw [`AgentEvent`] payloads captured so far,
    /// in arrival order.
    pub fn events(&self) -> Vec<AgentEvent> {
        self.listener
            .events()
            .into_iter()
            .map(|r| r.event)
            .collect()
    }

    /// Returns the `kind()` string for each captured event, in arrival order.
    ///
    /// Useful for quick assertions like:
    ///
    /// ```rust
    /// # use tinyagents_harness::testkit::EventRecorder;
    /// # use tinyagents_harness::events::AgentEvent;
    /// # use tinyagents_harness::ids::RunId;
    /// let recorder = EventRecorder::new();
    /// recorder.sink().emit(AgentEvent::RunStarted {
    ///     run_id: RunId::new("r1"),
    ///     thread_id: None,
    /// });
    /// assert_eq!(recorder.kinds(), vec!["run.started"]);
    /// ```
    pub fn kinds(&self) -> Vec<String> {
        self.listener
            .events()
            .into_iter()
            .map(|r| r.event.kind().to_string())
            .collect()
    }
}

impl Default for EventRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

impl Trajectory {
    /// Constructs a `Trajectory` from an owned sequence of [`AgentEvent`]s.
    pub fn from_events(events: Vec<AgentEvent>) -> Self {
        Self { events }
    }

    // ── Tool assertions ──────────────────────────────────────────────────────

    /// Returns `true` when at least one [`AgentEvent::ToolStarted`] with the
    /// given `name` is present in the trajectory.
    pub fn tool_was_called(&self, name: &str) -> bool {
        self.tool_call_count(name) > 0
    }

    /// Panics with a descriptive message when the named tool was not called.
    ///
    /// Use in tests for ergonomic assertions:
    ///
    /// ```rust
    /// # use tinyagents_harness::testkit::Trajectory;
    /// # use tinyagents_harness::events::AgentEvent;
    /// # use tinyagents_harness::ids::CallId;
    /// let events = vec![
    ///     AgentEvent::ToolStarted {
    ///         call_id: CallId::new("c1"),
    ///         tool_name: "search".into(),
    ///         input: None,
    ///         parent_call_id: None,
    ///     },
    /// ];
    /// Trajectory::from_events(events).assert_tool_called("search");
    /// ```
    pub fn assert_tool_called(&self, name: &str) {
        assert!(
            self.tool_was_called(name),
            "Trajectory: expected tool '{name}' to have been called, but it was not found in the \
             event sequence"
        );
    }

    /// Returns the number of times [`AgentEvent::ToolStarted`] with the given
    /// `name` appears in the trajectory.
    pub fn tool_call_count(&self, name: &str) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolStarted { tool_name, .. } if tool_name == name))
            .count()
    }

    // ── Model assertions ─────────────────────────────────────────────────────

    /// Returns the number of [`AgentEvent::ModelStarted`] events in the
    /// trajectory.
    pub fn model_call_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ModelStarted { .. }))
            .count()
    }

    /// Panics when the number of model calls does not equal `n`.
    pub fn assert_model_called_times(&self, n: usize) {
        let actual = self.model_call_count();
        assert_eq!(
            actual, n,
            "Trajectory: expected {n} model call(s) but found {actual}"
        );
    }

    // ── Ordering assertions ──────────────────────────────────────────────────

    /// Asserts that `labels` appear as a subsequence of the trajectory events.
    ///
    /// Each label is matched against the events in order. A label matches the
    /// first unmatched event for which *either*:
    ///
    /// - the event's [`AgentEvent::kind()`] equals the label (e.g.
    ///   `"tool.started"`, `"model.completed"`), **or**
    /// - the event is a `ToolStarted` or `ToolCompleted` whose `tool_name`
    ///   equals the label.
    ///
    /// The check is a *subsequence* match: there may be other events between
    /// the matched ones.
    ///
    /// Returns [`TinyAgentsError::Validation`] with a descriptive message on
    /// failure.
    pub fn assert_order(&self, labels: &[&str]) -> Result<()> {
        let mut event_iter = self.events.iter();
        for &label in labels {
            let found = event_iter.any(|e| Self::event_matches_label(e, label));
            if !found {
                return Err(TinyAgentsError::Validation(format!(
                    "Trajectory: expected label '{label}' in order but it was not found after the \
                     previous matched label"
                )));
            }
        }
        Ok(())
    }

    /// Returns `true` when the trajectory contains a [`AgentEvent::RunCompleted`]
    /// event.
    pub fn completed(&self) -> bool {
        self.events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunCompleted { .. }))
    }

    /// Panics when the trajectory does not contain a `RunCompleted` event.
    pub fn assert_completed(&self) {
        assert!(
            self.completed(),
            "Trajectory: expected RunCompleted event but none was found"
        );
    }

    /// Returns `true` when the trajectory contains at least one
    /// [`AgentEvent::RunFailed`] event.
    pub fn failed(&self) -> bool {
        self.events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunFailed { .. }))
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Returns `true` when `event` should be counted as a match for `label`.
    ///
    /// Matches on the event kind string (e.g. `"tool.started"`) **or** on the
    /// `tool_name` field of `ToolStarted`/`ToolCompleted` events.
    fn event_matches_label(event: &AgentEvent, label: &str) -> bool {
        if event.kind() == label {
            return true;
        }
        match event {
            AgentEvent::ToolStarted { tool_name, .. }
            | AgentEvent::ToolCompleted { tool_name, .. } => tool_name == label,
            AgentEvent::RouteSelected { route } => route == label,
            _ => false,
        }
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
