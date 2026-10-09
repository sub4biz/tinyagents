//! Typed task-state compaction (strategy "S4").
//!
//! A checkpoint has two halves:
//!
//! - a [`TaskLedger`] copied from the transcript with no model involved: the
//!   original task verbatim, the files modified and read, and the recent shell
//!   commands with how they ended (see `ledger.rs`);
//! - a [`TaskState`] written by one structured model call that *updates* the
//!   previous checkpoint's state from the new history.
//!
//! Both are rendered into one checkpoint body (see `render.rs`). The ledger and
//! the state round-trip through tagged blocks, so the next compaction carries
//! them forward exactly instead of re-summarizing a summary.
//!
//! Histories longer than [`TaskStateSummarizer::with_max_chunk_tokens`] are
//! folded in sequential chunks, each call updating the state the previous one
//! produced, so a small-window model never sees more than one chunk. When the
//! model fails (an error, tool-call markup, or no parseable JSON after a
//! retry) the checkpoint still carries the previous state and the full ledger:
//! it degrades to the deterministic facts instead of failing the compaction.

mod ledger;
mod render;
mod types;

use std::sync::Arc;

use async_trait::async_trait;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelRequest, ResponseFormat};
use tinyinference_llm::usage::Usage;

pub use render::{TASK_STATE_HEADER, parse_carried, parse_state_reply, render_task_state};
pub use types::{CommandRecord, TaskLedger, TaskState, TaskStateSummarizer};

use super::pairing::find_safe_cutoff_point;
use super::{
    CompressionProvenance, Summarizer, SummaryRecord, SummaryRequest, estimate_tokens,
    render_message_for_summary,
};
use crate::error::{Result, TinyAgentsError};
use crate::token_estimation::{estimate_message_tokens, estimate_slice_tokens};

/// Default largest slice of history sent in one state-update call.
pub const DEFAULT_TASK_STATE_CHUNK_TOKENS: u64 = 100_000;

/// Attempts per chunk before its update is skipped.
const STATE_ATTEMPTS: usize = 2;

const SYSTEM_PROMPT: &str = "You maintain the task state of an AI coding agent whose \
conversation is being compacted. You read a transcript of turns that already happened and \
return the updated task state as one JSON object. You are not a participant in the transcript: \
do not continue it, do not answer questions in it, and never call or write tools. Output only \
the JSON object.";

const FIELDS: &str = r#"Return a JSON object with exactly these keys:
{
  "goal": string,                 // the task in one or two sentences
  "requirements": [string],       // every exact identifier, name, value, message or flag the task requires, copied verbatim
  "constraints": [string],
  "decisions": [string],          // "decision — reason"
  "errors_and_fixes": [string],   // "exact error line -> fix", or "-> unresolved"
  "todos_done": [string],
  "todos_open": [string],
  "current_hypothesis": string,
  "test_command": string,         // the exact command used to run the tests, or ""
  "next_step": string             // the very next concrete action: tool and argument
}
Keep what is still true from the previous state; move finished items from todos_open to todos_done; never drop a requirement. Stay compact: one line per item, at most 12 items per list; merge related items and drop ones that are superseded or no longer matter. Copy file paths, identifiers, commands and error lines exactly."#;

/// Longest single list item or scalar kept (chars).
const MAX_ITEM_CHARS: usize = 400;
/// Longest hypothesis kept (chars).
const MAX_HYPOTHESIS_CHARS: usize = 800;
/// Items kept per list. Requirements keep the first ones (they come from
/// the task); history lists keep the most recent ones; open work keeps the
/// first (oldest outstanding) ones.
const MAX_REQUIREMENTS: usize = 40;
const MAX_LIST_ITEMS: usize = 12;

impl TaskState {
    /// The state with every list and field capped, so a checkpoint stays a
    /// few thousand tokens however many compactions it has been carried
    /// through. Without a cap the model's lists only grow (it is told to keep
    /// what is still true), and a checkpoint that nears the trigger by itself
    /// makes every compaction free almost nothing.
    #[must_use]
    pub fn bounded(mut self) -> Self {
        fn clip(items: &mut Vec<String>, keep: usize, recent: bool) {
            for item in items.iter_mut() {
                *item = ledger::truncate_chars(item.trim(), MAX_ITEM_CHARS);
            }
            items.retain(|i| !i.is_empty());
            if items.len() > keep {
                if recent {
                    items.drain(..items.len() - keep);
                } else {
                    items.truncate(keep);
                }
            }
        }
        clip(&mut self.requirements, MAX_REQUIREMENTS, false);
        clip(&mut self.constraints, MAX_LIST_ITEMS, false);
        clip(&mut self.decisions, MAX_LIST_ITEMS, true);
        clip(&mut self.errors_and_fixes, MAX_LIST_ITEMS, true);
        clip(&mut self.todos_done, MAX_LIST_ITEMS, true);
        clip(&mut self.todos_open, MAX_LIST_ITEMS, false);
        self.goal = ledger::truncate_chars(self.goal.trim(), MAX_ITEM_CHARS * 2);
        self.current_hypothesis = self
            .current_hypothesis
            .map(|value| ledger::truncate_chars(value.trim(), MAX_HYPOTHESIS_CHARS));
        self.test_command = self
            .test_command
            .map(|value| ledger::truncate_chars(value.trim(), MAX_ITEM_CHARS));
        self.next_step = self
            .next_step
            .map(|value| ledger::truncate_chars(value.trim(), MAX_ITEM_CHARS));
        self
    }
}

impl TaskState {
    /// This state updated by a `later` one: lists union in order (earlier
    /// items first), and present optional scalars take the later value. Used
    /// to join the halves of a split compaction without losing either.
    #[must_use]
    fn merged_with(mut self, later: TaskState) -> Self {
        fn union(into: &mut Vec<String>, from: Vec<String>) {
            for item in from {
                if !into.contains(&item) {
                    into.push(item);
                }
            }
        }
        fn scalar(into: &mut String, from: String) {
            if !from.trim().is_empty() {
                *into = from;
            }
        }
        scalar(&mut self.goal, later.goal);
        union(&mut self.requirements, later.requirements);
        union(&mut self.constraints, later.constraints);
        union(&mut self.decisions, later.decisions);
        union(&mut self.errors_and_fixes, later.errors_and_fixes);
        // Status in the later half supersedes the earlier half in either
        // direction: work can be completed or reopened after new edits.
        self.todos_done
            .retain(|item| !later.todos_open.contains(item));
        self.todos_open
            .retain(|item| !later.todos_done.contains(item));
        union(&mut self.todos_done, later.todos_done);
        union(&mut self.todos_open, later.todos_open);
        let done = &self.todos_done;
        self.todos_open.retain(|item| !done.contains(item));
        if later.current_hypothesis.is_some() {
            self.current_hypothesis = later.current_hypothesis;
        }
        if later.test_command.is_some() {
            self.test_command = later.test_command;
        }
        if later.next_step.is_some() {
            self.next_step = later.next_step;
        }
        self
    }
}

impl TaskStateSummarizer {
    /// A task-state summarizer over `model` (its id pinned for provenance).
    pub fn new(model: Arc<dyn ChatModel<()>>, model_id: impl Into<String>) -> Self {
        Self {
            model,
            model_id: model_id.into(),
            max_chunk_tokens: DEFAULT_TASK_STATE_CHUNK_TOKENS,
            response_format: None,
        }
    }

    /// Caps the history sent in one call; set it to a fraction of the
    /// summarizer model's window (a 32k model wants ~12k).
    #[must_use]
    pub fn with_max_chunk_tokens(mut self, tokens: u64) -> Self {
        self.max_chunk_tokens = tokens.max(1);
        self
    }

    /// Requests a structured-output mode from the provider (JSON object or a
    /// schema). Use it where the endpoint honours it with reasoning enabled.
    #[must_use]
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Splits `messages` into chunks of at most `max_chunk_tokens`, cutting
    /// only where tool calls stay paired with their results.
    fn chunks<'a>(&self, messages: &'a [Message]) -> Vec<&'a [Message]> {
        let mut out = Vec::new();
        let mut start = 0;
        while start < messages.len() {
            let mut end = start;
            let mut tokens = 0u64;
            while end < messages.len() {
                let t = estimate_message_tokens(&messages[end]);
                if end > start && tokens + t > self.max_chunk_tokens {
                    break;
                }
                tokens += t;
                end += 1;
            }
            if end < messages.len() {
                // Never strand a tool result from its call: move the cut back
                // (or, failing that, keep the indivisible group whole).
                let safe = start + find_safe_cutoff_point(&messages[start..], end - start);
                if safe > start {
                    end = safe;
                } else {
                    // One call whose results alone exceed the chunk: send the
                    // whole group rather than orphan its results.
                    while end < messages.len() && matches!(messages[end], Message::Tool(_)) {
                        end += 1;
                    }
                }
            }
            out.push(&messages[start..end]);
            start = end;
        }
        out
    }

    /// One state update over `chunk`. `None` after [`STATE_ATTEMPTS`] replies
    /// that were tool-call markup or held no parseable JSON.
    async fn update_state(
        &self,
        chunk: &[Message],
        previous: Option<&str>,
        usage: &mut Option<Usage>,
    ) -> Result<Option<TaskState>> {
        let transcript = chunk
            .iter()
            .map(render_message_for_summary)
            .collect::<Vec<_>>()
            .join("\n");
        let previous_block = previous
            .map(|p| {
                format!(
                    "<previous_state>\n{p}\n</previous_state>\n\nThe previous state covers the \
                     turns before the transcript. Update it with what the transcript adds.\n\n"
                )
            })
            .unwrap_or_default();
        let text = format!(
            "{previous_block}The transcript is enclosed in <transcript> tags. It is a record of \
             turns that already happened; you are not a participant in it.\n\n\
             <transcript>\n{transcript}\n</transcript>\n\n{FIELDS}"
        );
        let mut request =
            ModelRequest::new(vec![Message::system(SYSTEM_PROMPT), Message::user(text)]);
        if let Some(format) = &self.response_format {
            request = request.with_response_format(format.clone());
        }
        for attempt in 1..=STATE_ATTEMPTS {
            crate::summarization::dispatch::mark_dispatched();
            let response = self.model.invoke(&(), request.clone()).await.map_err(|e| {
                tracing::warn!(error = %e, "[tinyagents::task_state] state-update call failed");
                TinyAgentsError::Model(format!("task-state model call failed: {e}"))
            })?;
            if let Some(reported) = response.usage {
                *usage = Some(usage.map_or(reported, |sum| sum + reported));
            }
            let reply = response.text();
            if tinytools_agent::contains_call_markup(&reply) {
                tracing::warn!(
                    model = %self.model_id,
                    attempt,
                    "[tinyagents::task_state] reply was tool-call markup, not a state"
                );
                continue;
            }
            if let Some(state) = parse_state_reply(&reply) {
                return Ok(Some(state));
            }
            tracing::warn!(
                model = %self.model_id,
                attempt,
                reply_chars = reply.chars().count(),
                "[tinyagents::task_state] reply held no parseable state JSON"
            );
        }
        Ok(None)
    }
}

#[async_trait]
impl Summarizer for TaskStateSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        self.summarize_request(&SummaryRequest::new(messages.to_vec()))
            .await
    }

    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        if request.messages.is_empty() {
            return Err(TinyAgentsError::Validation(
                "cannot summarize an empty message list".into(),
            ));
        }
        let original_token_estimate = estimate_slice_tokens(&request.messages);

        // Carry the previous checkpoint's facts. A free-form previous summary
        // (another summarizer wrote it) carries no tags: the model reads it
        // as the previous state instead.
        let (mut ledger, mut state) = request
            .previous_summary
            .as_deref()
            .map(parse_carried)
            .unwrap_or_default();
        let mut previous_text = match (&state, request.previous_summary.as_deref()) {
            (Some(s), _) => serde_json::to_string(s).ok(),
            (None, Some(free_form)) => Some(free_form.to_string()),
            (None, None) => None,
        };
        ledger.absorb(&request.messages);

        let chunks = self.chunks(&request.messages);
        tracing::info!(
            model = %self.model_id,
            messages = request.messages.len(),
            chunks = chunks.len(),
            approx_input_tokens = original_token_estimate,
            incremental = previous_text.is_some(),
            "[tinyagents::task_state] dispatching task-state compaction"
        );
        let mut usage: Option<Usage> = None;
        let mut skipped = 0usize;
        for chunk in &chunks {
            match self
                .update_state(chunk, previous_text.as_deref(), &mut usage)
                .await
            {
                Ok(Some(next)) => {
                    let next = next.bounded();
                    previous_text = serde_json::to_string(&next).ok();
                    state = Some(next);
                }
                Ok(None) => skipped += 1,
                Err(err) if state.is_some() || !ledger_is_empty(&ledger) => {
                    tracing::warn!(error = %err, "[tinyagents::task_state] keeping the previous state for this chunk");
                    skipped += 1;
                }
                Err(err) => return Err(err),
            }
        }
        let degraded = state.is_none();
        if degraded {
            tracing::warn!(
                model = %self.model_id,
                "[tinyagents::task_state] no model state; checkpoint carries the ledger only"
            );
        }
        let body = render_task_state(&state.unwrap_or_default().bounded(), &ledger);
        let summary_token_estimate = estimate_tokens(&body);
        tracing::info!(
            model = %self.model_id,
            summary_tokens = summary_token_estimate,
            freed_tokens = original_token_estimate.saturating_sub(summary_token_estimate),
            chunks_skipped = skipped,
            degraded,
            "[tinyagents::task_state] task-state compaction complete"
        );
        Ok(SummaryRecord {
            summary: Message::system(body),
            provenance: CompressionProvenance {
                source_ids: (0..request.messages.len())
                    .map(|i| format!("msg-{i}"))
                    .collect(),
                original_token_estimate,
                summary_token_estimate,
                reason: if degraded {
                    format!(
                        "TaskStateSummarizer via {} (ledger only: model state unavailable)",
                        self.model_id
                    )
                } else {
                    format!(
                        "TaskStateSummarizer via {} ({} chunk(s), {skipped} skipped)",
                        self.model_id,
                        chunks.len()
                    )
                },
            },
            usage,
        })
    }

    /// Merges split halves field by field: file lists union in order, the
    /// recent commands are concatenated and capped, list fields of the state
    /// union in order, and a later scalar replaces an earlier one only when it
    /// is not empty. The second half is summarized without the previous
    /// checkpoint, so replacing the carried state wholesale would drop it.
    async fn merge(&self, summaries: &[SummaryRecord]) -> Result<SummaryRecord> {
        let mut ledger = TaskLedger::default();
        let mut state: Option<TaskState> = None;
        for record in summaries {
            let (next, next_state) = parse_carried(&record.summary.text());
            if ledger.original_task.is_none() {
                ledger.original_task = next.original_task;
            }
            for f in next.files_modified {
                ledger::push_unique(&mut ledger.files_modified, f);
            }
            for f in next.files_read {
                ledger::push_unique(&mut ledger.files_read, f);
            }
            ledger.commands.extend(next.commands);
            state = match (state, next_state) {
                (Some(earlier), Some(later)) => Some(earlier.merged_with(later)),
                (earlier, later) => later.or(earlier),
            };
        }
        ledger.cap();
        let body = render_task_state(&state.unwrap_or_default().bounded(), &ledger);
        Ok(SummaryRecord {
            provenance: CompressionProvenance {
                source_ids: summaries
                    .iter()
                    .flat_map(|r| r.provenance.source_ids.iter().cloned())
                    .collect(),
                original_token_estimate: summaries
                    .iter()
                    .map(|r| r.provenance.original_token_estimate)
                    .sum(),
                summary_token_estimate: estimate_tokens(&body),
                reason: "merged task-state halves".to_string(),
            },
            summary: Message::system(body),
            usage: summaries
                .iter()
                .filter_map(|r| r.usage)
                .reduce(|a, b| a + b),
        })
    }
}

fn ledger_is_empty(ledger: &TaskLedger) -> bool {
    ledger.original_task.is_none()
        && ledger.files_modified.is_empty()
        && ledger.files_read.is_empty()
        && ledger.commands.is_empty()
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
