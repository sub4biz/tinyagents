//! JSONL line shapes (`_meta`, message, and compaction records) and the
//! conversions between them and the public [`TranscriptMessage`] /
//! [`TranscriptMeta`] / [`DisplayMessage`] types.

use super::types::{
    BackgroundOrigin, DisplayMessage, MessageUsage, TRANSCRIPT_SCHEMA_VERSION, TranscriptMeta,
    TurnUsage,
};
use super::types::{ToolFailure, TranscriptMessage, TranscriptPart, TranscriptToolCall};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tinytools_agent::dialect::{
    ContentPart, NativeToolCall, encode_assistant_envelope, encode_tool_envelope, join_image_parts,
    parse_canonical_assistant_envelope, parse_canonical_tool_envelope, split_image_parts,
};

/// Discriminator value for a compaction record's `kind` field.
pub(super) const COMPACTION_KIND: &str = "compaction";

/// Discriminator value for a tool-declaration record's `kind` field.
pub(super) const TOOLS_KIND: &str = "tools";

/// `v` of a typed row (see [`MessageLine::row_version`]).
pub(super) const TYPED_ROW_VERSION: u32 = 2;

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// The structure a typed row stores as fields instead of inside `content`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TypedShape {
    /// Assistant row that made native tool calls: `content` is the visible
    /// text, `tool_calls` the calls.
    AssistantCalls,
    /// Native tool result: `content` is the output, `tool_call_id` the call.
    ToolResult,
    /// User row with inline images: `parts` is the ordered text/image list.
    UserParts,
}

impl TypedShape {
    const fn as_str(self) -> &'static str {
        match self {
            Self::AssistantCalls => "assistant_calls",
            Self::ToolResult => "tool_result",
            Self::UserParts => "user_parts",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "assistant_calls" => Some(Self::AssistantCalls),
            "tool_result" => Some(Self::ToolResult),
            "user_parts" => Some(Self::UserParts),
            _ => None,
        }
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

// ── Internal JSONL types ─────────────────────────────────────────────

/// The `_meta` line serialisation shape.
#[derive(Serialize, Deserialize)]
pub(super) struct MetaLine {
    #[serde(rename = "_meta")]
    pub(super) meta: MetaPayload,
}

#[derive(Serialize, Deserialize)]
pub(super) struct MetaPayload {
    /// Schema version of the transcript record format (see
    /// [`TRANSCRIPT_SCHEMA_VERSION`]). Absent (deserialises to `0`) on files
    /// written before the append-only migration.
    #[serde(default)]
    pub(super) version: u32,
    pub(super) agent: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) agent_type: Option<String>,
    pub(super) dispatcher: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) model: Option<String>,
    pub(super) created: String,
    pub(super) updated: String,
    pub(super) turn_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prefix_message_count: Option<usize>,
    pub(super) input_tokens: u64,
    pub(super) output_tokens: u64,
    pub(super) cached_input_tokens: u64,
    pub(super) charged_amount_usd: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) parent_session_id: Option<String>,
}

/// One message line in the JSONL — only `role` and `content` are required.
/// All other fields are optional; unknown fields are flattened to preserve
/// forward-compatibility.
#[derive(Serialize, Deserialize)]
pub(super) struct MessageLine {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) id: Option<String>,
    pub(super) role: String,
    pub(super) content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) extra_metadata: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) cache_breakpoints: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) usage: Option<MessageUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_calls: Option<Vec<TranscriptToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) iteration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ts: Option<String>,
    /// Turn boundary marker: the caller-provided `request_id` this message
    /// belongs to, when available. Stamped on every line of a turn so the
    /// display projection can group a turn's messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) request_id: Option<String>,
    /// `true` when this line is a *partial* assistant answer captured because
    /// the turn was interrupted/cancelled mid-stream. Present for **display
    /// only** — the model-context reader skips these so a resumed context never
    /// carries a truncated answer.
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) interrupted: bool,
    /// `true` when this tool-result line's tool call **failed**
    /// (`ToolResult::is_error`). Additive + optional: legacy lines and every
    /// non-tool line omit it and default to success. Lifted from the tool
    /// message's failure metadata by [`build_message_line`]; consumed by the
    /// display projection to render an error tool row instead of success.
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) failure: bool,
    /// Optional short, single-line reason for a failed tool call (the head of
    /// the error output). Present only alongside `failure: true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) failure_detail: Option<String>,
    /// Row form marker. Absent (`0`) on every row written before typed rows;
    /// [`TYPED_ROW_VERSION`] on a row whose tool-call / tool-result / image
    /// structure is stored as fields rather than encoded into `content`.
    #[serde(default, rename = "v", skip_serializing_if = "is_zero")]
    pub(super) row_version: u32,
    /// Which structure a typed row carries (`assistant_calls`, `tool_result`
    /// or `user_parts`); see [`TypedShape`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) shape: Option<String>,
    /// The answered call id of a `tool_result` row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) tool_call_id: Option<String>,
    /// The ordered text/image parts of a `user_parts` row. A part kind this
    /// reader does not know (a future typed-row version) reads as `None`, so
    /// the row keeps its stored `content` instead of being dropped.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_parts"
    )]
    pub(super) parts: Option<Vec<TranscriptPart>>,
    /// Set only on a line written out of band by
    /// [`append_background_message`](super::append_background_message): who
    /// delivered it and its idempotency key. Additive — a reader that predates
    /// it keeps the line as an ordinary message — and lenient, so a malformed
    /// value reads as `None` instead of costing the whole message.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_background"
    )]
    pub(super) background: Option<BackgroundOrigin>,
    /// Absorb any unknown fields so forward-compat reads don't error.
    #[serde(flatten)]
    pub(super) _extra: HashMap<String, serde_json::Value>,
}

/// Deserialises `background`, mapping a shape this reader cannot decode to
/// `None` rather than failing the whole line.
fn lenient_background<'de, D>(deserializer: D) -> Result<Option<BackgroundOrigin>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(raw.and_then(|value| serde_json::from_value(value).ok()))
}

/// Deserialises `parts`, mapping any shape this reader cannot decode (an
/// unknown part variant) to `None` rather than failing the whole line.
fn lenient_parts<'de, D>(deserializer: D) -> Result<Option<Vec<TranscriptPart>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(raw.and_then(|value| serde_json::from_value(value).ok()))
}

/// A compaction record: `{"kind":"compaction","replacement":[…]}`.
///
/// Appended when the harness reduces context (post-compaction / trim) so the
/// model-context reader can reconstruct the reduced set without the file being
/// destructively rewritten. `replacement` is the **full** logical message set
/// that supersedes everything before it — an explicit replacement list
/// (mirroring Codex's `Compacted { replacement_history }`) rather than
/// surviving-message ids, because our writer already holds the reduced
/// `messages` slice on each persist call and message ids are optional, so an
/// id-reference scheme would be less robust for no gain.
#[derive(Serialize, Deserialize)]
pub(super) struct CompactionLine {
    pub(super) kind: String,
    pub(super) replacement: Vec<MessageLine>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) ts: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) request_id: Option<String>,
    #[serde(flatten)]
    pub(super) _extra: HashMap<String, serde_json::Value>,
}

/// A tool-declaration record: `{"kind":"tools","tools":[…]}`.
///
/// Written whenever the model-visible tool set a turn was sent with differs
/// from the one last recorded in this file, so a resumed session can send the
/// same declarations again instead of rebuilding them from whatever the new
/// process happens to have registered. Last record wins. Neither reader puts
/// it into the message stream.
#[derive(Serialize, Deserialize)]
pub(super) struct ToolsLine {
    pub(super) kind: String,
    pub(super) tools: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) ts: Option<String>,
    #[serde(flatten)]
    pub(super) _extra: HashMap<String, serde_json::Value>,
}

/// Serialises `tools` as one `{"kind":"tools"}` record line (no trailing newline).
pub(super) fn tools_line_json(tools: &serde_json::Value) -> Result<String> {
    serde_json::to_string(&ToolsLine {
        kind: TOOLS_KIND.to_string(),
        tools: tools.clone(),
        ts: Some(chrono::Utc::now().to_rfc3339()),
        _extra: HashMap::new(),
    })
    .context("serialise transcript tools record")
}

/// Build the serialised `_meta` header line for `meta`, stamping the current
/// [`TRANSCRIPT_SCHEMA_VERSION`].
fn meta_payload_from(meta: &TranscriptMeta) -> MetaPayload {
    MetaPayload {
        version: TRANSCRIPT_SCHEMA_VERSION,
        agent: meta.agent_name.clone(),
        agent_id: meta.agent_id.clone(),
        session_id: meta.session_id.clone(),
        parent_session_id: meta.parent_session_id.clone(),
        agent_type: meta.agent_type.clone(),
        dispatcher: meta.dispatcher.clone(),
        provider: meta.provider.clone(),
        model: meta.model.clone(),
        created: meta.created.clone(),
        updated: meta.updated.clone(),
        turn_count: meta.turn_count,
        prefix_message_count: meta.prefix_message_count,
        input_tokens: meta.input_tokens,
        output_tokens: meta.output_tokens,
        cached_input_tokens: meta.cached_input_tokens,
        charged_amount_usd: meta.charged_amount_usd,
        thread_id: meta.thread_id.clone(),
        task_id: meta.task_id.clone(),
    }
}

/// Serialises `meta` as the JSON `_meta` header line (no trailing newline).
pub(super) fn meta_line_json(meta: &TranscriptMeta) -> Result<String> {
    let meta_line = MetaLine {
        meta: meta_payload_from(meta),
    };
    serde_json::to_string(&meta_line).context("serialise transcript meta header")
}

/// Build a [`MessageLine`] for `msg`, folding in `turn_usage` (assistant rows)
/// and stamping the `request_id` turn boundary when supplied.
pub(super) fn build_message_line(
    msg: &TranscriptMessage,
    turn_usage: Option<&TurnUsage>,
    request_id: Option<&str>,
    interrupted: bool,
) -> MessageLine {
    // A row lifted from a legacy string and not edited since is stored from
    // that string, so a non-canonical envelope is never re-encoded lossily.
    let unlifted = msg.unlifted();
    let msg = unlifted.as_ref().unwrap_or(msg);
    let assistant_usage = if msg.role == "assistant" {
        turn_usage
    } else {
        None
    };
    let extra_metadata = msg.extra_metadata.clone();
    let failure = msg
        .tool_failure
        .as_ref()
        .is_some_and(|failure| failure.failed);
    let failure_detail = msg
        .tool_failure
        .as_ref()
        .and_then(|failure| failure.detail.clone());
    // A row read from a transcript owns its recorded correlation id; only a
    // newly-created row takes the opaque id supplied for this append.
    let request_id = if msg.preserve_request_id {
        msg.request_id.clone()
    } else {
        request_id.map(str::to_string)
    };
    let message_reasoning = (msg.role == "assistant")
        .then(|| {
            extra_metadata
                .as_ref()
                .and_then(|meta| meta.get("reasoning_content"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .flatten();
    let native_envelope = (msg.role == "assistant")
        .then(|| serde_json::from_str::<serde_json::Value>(&msg.content).ok())
        .flatten();
    let envelope_reasoning = native_envelope
        .as_ref()
        .and_then(|value| value.get("reasoning_content"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let envelope_tool_calls = native_envelope
        .as_ref()
        .and_then(|value| value.get("tool_calls"))
        .and_then(|value| serde_json::from_value::<Vec<TranscriptToolCall>>(value.clone()).ok())
        .filter(|calls| !calls.is_empty());
    let typed = typed_form(msg);
    let (content, tool_calls_override) = match &typed {
        Some(typed) => (typed.content.clone(), typed.tool_calls.clone()),
        None => (msg.content.clone(), None),
    };
    MessageLine {
        id: msg.id.clone(),
        role: msg.role.clone(),
        content,
        extra_metadata,
        cache_breakpoints: msg.cache_breakpoints.clone(),
        provider: assistant_usage.map(|tu| tu.provider.clone()),
        model: assistant_usage.map(|tu| tu.model.clone()),
        usage: assistant_usage.map(|tu| tu.usage.clone()),
        reasoning_content: message_reasoning
            .or(envelope_reasoning)
            .or_else(|| assistant_usage.and_then(|tu| tu.reasoning_content.clone())),
        tool_calls: tool_calls_override.or(envelope_tool_calls).or_else(|| {
            assistant_usage.and_then(|tu| {
                if tu.tool_calls.is_empty() {
                    None
                } else {
                    Some(tu.tool_calls.clone())
                }
            })
        }),
        iteration: assistant_usage.map(|tu| tu.iteration),
        ts: assistant_usage.map(|tu| tu.ts.clone()),
        request_id,
        interrupted,
        failure,
        failure_detail,
        row_version: typed.as_ref().map_or(0, |_| TYPED_ROW_VERSION),
        shape: typed.as_ref().map(|typed| typed.shape.as_str().to_string()),
        tool_call_id: typed.as_ref().and_then(|typed| typed.tool_call_id.clone()),
        parts: typed.and_then(|typed| typed.parts),
        background: None,
        _extra: HashMap::new(),
    }
}

/// The storage form of a row that already carries typed fields. `None` when
/// the fields do not fit the row's role (they are then stored as the legacy
/// string, which a reader normalizes back).
fn direct_form(msg: &TranscriptMessage) -> Option<TypedForm> {
    match msg.role.as_str() {
        "assistant" if !msg.tool_calls.is_empty() => Some(TypedForm {
            shape: TypedShape::AssistantCalls,
            content: msg.content.clone(),
            tool_calls: Some(msg.tool_calls.clone()),
            tool_call_id: None,
            parts: None,
        }),
        "tool" if msg.tool_call_id.is_some() => Some(TypedForm {
            shape: TypedShape::ToolResult,
            content: msg.content.clone(),
            tool_calls: None,
            tool_call_id: msg.tool_call_id.clone(),
            parts: None,
        }),
        "user" if msg.parts.is_some() => Some(TypedForm {
            shape: TypedShape::UserParts,
            content: msg.content.clone(),
            tool_calls: None,
            tool_call_id: None,
            parts: msg.parts.clone(),
        }),
        _ => None,
    }
}

/// The typed storage form of one row.
struct TypedForm {
    shape: TypedShape,
    content: String,
    tool_calls: Option<Vec<TranscriptToolCall>>,
    tool_call_id: Option<String>,
    parts: Option<Vec<TranscriptPart>>,
}

/// Lift a row's string-encoded structure into typed fields, **only** when
/// [`rebuild_content`] reproduces the row's `content` byte for byte. Anything
/// that is not exactly a canonical native envelope or marker string stays a
/// legacy row, so a read can never return a different string than was written.
fn typed_form(msg: &TranscriptMessage) -> Option<TypedForm> {
    if msg.is_typed() {
        return direct_form(msg);
    }
    let form = match msg.role.as_str() {
        "assistant" => {
            let envelope = parse_canonical_assistant_envelope(&msg.content)?;
            TypedForm {
                shape: TypedShape::AssistantCalls,
                content: envelope.content,
                tool_calls: Some(
                    envelope
                        .tool_calls
                        .into_iter()
                        .map(|call| TranscriptToolCall {
                            id: call.id,
                            name: call.name,
                            arguments: call.arguments,
                            extra_content: call.extra_content,
                        })
                        .collect(),
                ),
                tool_call_id: None,
                parts: None,
            }
        }
        "tool" => {
            let (tool_call_id, content) = parse_canonical_tool_envelope(&msg.content)?;
            TypedForm {
                shape: TypedShape::ToolResult,
                content,
                tool_calls: None,
                tool_call_id: Some(tool_call_id),
                parts: None,
            }
        }
        "user" => {
            let parts = split_image_parts(&msg.content);
            if !parts
                .iter()
                .any(|part| matches!(part, ContentPart::Image(_)))
            {
                return None;
            }
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(text.as_str()),
                    ContentPart::Image(_) => None,
                })
                .collect::<String>();
            TypedForm {
                shape: TypedShape::UserParts,
                content: text,
                tool_calls: None,
                tool_call_id: None,
                parts: Some(
                    parts
                        .into_iter()
                        .map(|part| match part {
                            ContentPart::Text(text) => TranscriptPart::Text { text },
                            ContentPart::Image(url) => TranscriptPart::Image { url },
                        })
                        .collect(),
                ),
            }
        }
        _ => return None,
    };
    let rebuilt = rebuild_content(
        form.shape,
        &form.content,
        form.tool_calls.as_deref(),
        form.tool_call_id.as_deref(),
        form.parts.as_deref(),
    )?;
    (rebuilt == msg.content).then_some(form)
}

/// Rebuild the legacy string `content` of a typed row. `None` when the row's
/// typed fields do not carry what its `shape` needs (the caller then keeps
/// `content` as read).
fn rebuild_content(
    shape: TypedShape,
    content: &str,
    tool_calls: Option<&[TranscriptToolCall]>,
    tool_call_id: Option<&str>,
    parts: Option<&[TranscriptPart]>,
) -> Option<String> {
    match shape {
        TypedShape::AssistantCalls => {
            let calls: Vec<NativeToolCall> = tool_calls?
                .iter()
                .map(|call| NativeToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    extra_content: call.extra_content.clone(),
                })
                .collect();
            (!calls.is_empty()).then(|| encode_assistant_envelope(Some(content), &calls, None))
        }
        TypedShape::ToolResult => Some(encode_tool_envelope(tool_call_id?, content)),
        TypedShape::UserParts => {
            let original = parts?;
            if original.iter().any(|part| {
                matches!(
                    part,
                    TranscriptPart::Audio { .. }
                        | TranscriptPart::Video { .. }
                        | TranscriptPart::Document { .. }
                )
            }) {
                return Some(serde_json::json!({"_tinyagents_media_parts": original}).to_string());
            }
            let parts: Vec<ContentPart> = original
                .iter()
                .map(|part| match part {
                    TranscriptPart::Text { text } => ContentPart::Text(text.clone()),
                    TranscriptPart::Image { url } => ContentPart::Image(url.clone()),
                    TranscriptPart::Audio { .. }
                    | TranscriptPart::Video { .. }
                    | TranscriptPart::Document { .. } => {
                        unreachable!("new media uses the typed compatibility envelope")
                    }
                })
                .collect();
            Some(join_image_parts(&parts))
        }
    }
}

/// The in-memory row a message line stands for: plain `content` plus the typed
/// structure. A typed (`"v":2`) line of a known shape supplies its fields
/// directly; any other line (every row written before typed rows, an unknown
/// future shape, a typed line missing the fields its shape needs) is a legacy
/// string row and is normalized by lifting whatever envelope or marker its
/// `content` holds.
fn row_from_line(ml: &MessageLine) -> TranscriptMessage {
    let mut row = TranscriptMessage::new(ml.role.clone(), ml.content.clone());
    row.id = ml.id.clone();
    if ml.row_version == TYPED_ROW_VERSION {
        let typed = match ml.shape.as_deref().and_then(TypedShape::parse) {
            Some(TypedShape::AssistantCalls) => ml
                .tool_calls
                .clone()
                .filter(|calls| !calls.is_empty())
                .map(|calls| row.tool_calls = calls),
            Some(TypedShape::ToolResult) => ml
                .tool_call_id
                .clone()
                .map(|tool_call_id| row.tool_call_id = Some(tool_call_id)),
            Some(TypedShape::UserParts) => ml.parts.clone().map(|parts| row.parts = Some(parts)),
            None => None,
        };
        if typed.is_some() {
            return row;
        }
    }
    // A row from a newer writer belongs to a schema this reader does not
    // know: its stored content stays opaque rather than being re-read through
    // today's legacy conventions.
    if ml.row_version > TYPED_ROW_VERSION {
        return row;
    }
    row.normalized()
}

/// Serialise `messages` into JSONL message lines, attributing
/// `last_assistant_turn_usage` (or per-message embedded usage) to the last
/// assistant row and stamping `request_id` on every line.
pub(super) fn serialise_message_lines(
    messages: &[TranscriptMessage],
    last_assistant_turn_usage: Option<&TurnUsage>,
    request_id: Option<&str>,
    buf: &mut String,
) -> Result<()> {
    for (i, line) in stamped_lines(messages, last_assistant_turn_usage, request_id)
        .into_iter()
        .enumerate()
    {
        let line_json =
            serde_json::to_string(&line).with_context(|| format!("serialise message line {i}"))?;
        buf.push_str(&line_json);
        buf.push('\n');
    }
    Ok(())
}

/// The lines [`serialise_message_lines`] writes for `messages`: the turn's
/// usage on its last assistant row, `request_id` on every fresh row, and the
/// per-step `(iteration, ts)` stamps.
fn stamped_lines(
    messages: &[TranscriptMessage],
    last_assistant_turn_usage: Option<&TurnUsage>,
    request_id: Option<&str>,
) -> Vec<MessageLine> {
    let last_assistant_idx = messages.iter().rposition(|m| m.role == "assistant");
    let turn_stamps = turn_step_stamps(messages, last_assistant_idx, last_assistant_turn_usage);
    messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            let turn_usage = if Some(i) == last_assistant_idx {
                last_assistant_turn_usage
                    .cloned()
                    .or_else(|| msg.turn_usage.clone())
            } else {
                msg.turn_usage.clone()
            };
            let mut line = build_message_line(msg, turn_usage.as_ref(), request_id, false);
            if let Some((iteration, ts)) = turn_stamps.get(&i) {
                line.iteration = line.iteration.or(Some(*iteration));
                if line.ts.is_none() && !ts.is_empty() {
                    line.ts = Some(ts.clone());
                }
            }
            line
        })
        .collect()
}

/// `messages` exactly as the JSONL writer would record them for one turn and
/// a reader would return them: what a non-file transcript backend stores so
/// its replay carries the same per-turn provenance (usage, request ids,
/// step stamps) as a transcript file.
#[cfg(feature = "storage-drivers")]
pub(crate) fn stamped_rows(
    messages: &[TranscriptMessage],
    last_assistant_turn_usage: Option<&TurnUsage>,
    request_id: Option<&str>,
) -> Vec<TranscriptMessage> {
    stamped_lines(messages, last_assistant_turn_usage, request_id)
        .into_iter()
        .map(message_from_line)
        .collect()
}

/// Per-step `(iteration, ts)` stamps for the intermediate assistant rows of the
/// turn being written.
///
/// A turn's [`TurnUsage`] lands on its final assistant row only, so without
/// this every earlier step of a multi-step turn (the tool-calling rows) was
/// written with no `iteration` and no `ts`, and a reader could not tell which
/// model call a row — or the reasoning on it — belonged to. The turn's own
/// rows are the fresh (not replayed, `preserve_request_id == false`) assistant
/// rows after the last `user` row; they are the turn's model calls in order,
/// the last being call `turn_usage.iteration`. Earlier rows count back from
/// it (never below 1). `ts` is the turn's commit stamp: the only clock the
/// writer has, but enough to place the row in time.
///
/// Empty when there is no turn usage, the usage records no iteration, or the
/// final assistant row is not itself one of the turn's fresh rows. The final
/// row is excluded: it already carries both through its usage.
fn turn_step_stamps(
    messages: &[TranscriptMessage],
    last_assistant_idx: Option<usize>,
    turn_usage: Option<&TurnUsage>,
) -> HashMap<usize, (u32, String)> {
    let mut stamps = HashMap::new();
    let (Some(last), Some(usage)) = (last_assistant_idx, turn_usage) else {
        return stamps;
    };
    if usage.iteration == 0 || messages[last].preserve_request_id {
        return stamps;
    }
    let turn_start = messages[..last]
        .iter()
        .rposition(|m| m.role == "user")
        .map_or(0, |idx| idx + 1);
    let steps: Vec<usize> = (turn_start..last)
        .filter(|&i| messages[i].role == "assistant" && !messages[i].preserve_request_id)
        .collect();
    let total = steps.len() as u32;
    for (k, idx) in steps.into_iter().enumerate() {
        let back = total - k as u32;
        let iteration = usage.iteration.saturating_sub(back).max(1);
        stamps.insert(idx, (iteration, usage.ts.clone()));
    }
    stamps
}

/// Convert a parsed `MetaPayload` into the public [`TranscriptMeta`].
pub(super) fn meta_from_payload(mp: MetaPayload) -> TranscriptMeta {
    TranscriptMeta {
        session_id: mp.session_id,
        parent_session_id: mp.parent_session_id,
        agent_name: mp.agent,
        agent_id: mp.agent_id,
        agent_type: mp.agent_type,
        dispatcher: mp.dispatcher,
        provider: mp.provider,
        model: mp.model,
        created: mp.created,
        updated: mp.updated,
        turn_count: mp.turn_count,
        prefix_message_count: mp.prefix_message_count,
        input_tokens: mp.input_tokens,
        output_tokens: mp.output_tokens,
        cached_input_tokens: mp.cached_input_tokens,
        charged_amount_usd: mp.charged_amount_usd,
        thread_id: mp.thread_id,
        task_id: mp.task_id,
    }
}

/// Recover the [`TurnUsage`] a message line carried (assistant rows only).
fn turn_usage_from_line(ml: &MessageLine) -> Option<TurnUsage> {
    match (
        ml.provider.clone(),
        ml.model.clone(),
        ml.usage.clone(),
        ml.ts.clone(),
    ) {
        (Some(provider), Some(model), Some(usage), Some(ts)) if ml.role == "assistant" => {
            Some(TurnUsage {
                provider,
                model,
                usage,
                ts,
                reasoning_content: ml.reasoning_content.clone(),
                tool_calls: ml.tool_calls.clone().unwrap_or_default(),
                iteration: ml.iteration.unwrap_or_default(),
            })
        }
        _ => None,
    }
}

/// Reconstruct a [`TranscriptMessage`] from a message line, re-attaching turn-usage
/// metadata so the round-trip is lossless for the model-context path.
pub(super) fn message_from_line(ml: MessageLine) -> TranscriptMessage {
    let turn_usage = turn_usage_from_line(&ml);
    let failure_detail = ml.failure.then(|| ml.failure_detail.clone());
    let typed = row_from_line(&ml);
    TranscriptMessage {
        id: typed.id,
        role: typed.role,
        content: typed.content,
        tool_calls: typed.tool_calls,
        tool_call_id: typed.tool_call_id,
        parts: typed.parts,
        legacy: typed.legacy,
        extra_metadata: ml.extra_metadata,
        cache_breakpoints: ml.cache_breakpoints,
        turn_usage: turn_usage.clone(),
        request_id: ml.request_id,
        preserve_request_id: true,
        interrupted: ml.interrupted,
        tool_failure: failure_detail.map(|detail| ToolFailure {
            failed: true,
            detail,
        }),
    }
}

/// Classification of one non-empty JSONL line.
pub(super) enum LineKind {
    Meta(MetaLine),
    Compaction(CompactionLine),
    Tools(ToolsLine),
    Message(MessageLine),
}

/// Classify a raw line: a `_meta` header/update, a `compaction` record, or a
/// message line. Returns `Err` only when the line is malformed for its
/// apparent kind; the caller decides whether that is fatal (first line) or a
/// skippable warning (later lines).
pub(super) fn classify_line(line: &str) -> Result<LineKind, serde_json::Error> {
    // Cheap structural peek. Unknown/other shapes fall through to MessageLine,
    // whose required `role`/`content` gate rejects genuinely foreign lines.
    let value: serde_json::Value = serde_json::from_str(line)?;
    if value.get("_meta").is_some() {
        return serde_json::from_str::<MetaLine>(line).map(LineKind::Meta);
    }
    if value.get("kind").and_then(|k| k.as_str()) == Some(COMPACTION_KIND) {
        return serde_json::from_str::<CompactionLine>(line).map(LineKind::Compaction);
    }
    if value.get("kind").and_then(|k| k.as_str()) == Some(TOOLS_KIND) {
        return serde_json::from_str::<ToolsLine>(line).map(LineKind::Tools);
    }
    serde_json::from_str::<MessageLine>(line).map(LineKind::Message)
}

// ── Display read ──────────────────────────────────────────────────────

/// Reconstruct a [`DisplayMessage`] from a message line, preserving the
/// turn-boundary + partial flags the model-context path discards.
pub(super) fn display_message_from_line(ml: MessageLine) -> DisplayMessage {
    let turn_usage = turn_usage_from_line(&ml);
    let reasoning_content = ml.reasoning_content.clone().or_else(|| {
        turn_usage
            .as_ref()
            .and_then(|tu| tu.reasoning_content.clone())
    });
    let typed = row_from_line(&ml);
    DisplayMessage {
        interrupted: ml.interrupted,
        request_id: ml.request_id.clone(),
        iteration: ml.iteration,
        ts: ml.ts.clone(),
        turn_usage: turn_usage.clone(),
        reasoning_content,
        failure: ml.failure,
        failure_detail: ml.failure_detail.clone(),
        background: ml.background.clone(),
        message: TranscriptMessage {
            id: typed.id,
            role: typed.role,
            content: typed.content,
            tool_calls: typed.tool_calls,
            tool_call_id: typed.tool_call_id,
            parts: typed.parts,
            legacy: typed.legacy,
            extra_metadata: ml.extra_metadata,
            cache_breakpoints: ml.cache_breakpoints,
            turn_usage,
            request_id: ml.request_id,
            preserve_request_id: true,
            interrupted: ml.interrupted,
            tool_failure: ml.failure.then(|| ToolFailure {
                failed: true,
                detail: ml.failure_detail.clone(),
            }),
        },
    }
}
