//! Turning completions into model-visible text.
//!
//! The harness carries no product wording. A host supplies a
//! [`CompletionFormatter`] with its own framing; [`NeutralCompletionFormatter`]
//! is the default.

use serde::Serialize;
use tinyinference_llm::message::Message;

use super::types::CompletionRecord;

/// Renders a batch of completions for the parent model.
pub trait CompletionFormatter: Send + Sync {
    /// The text for `records`. Child output is untrusted: a formatter must not
    /// let it close or forge the framing around it.
    fn format_batch(&self, records: &[CompletionRecord]) -> String;

    /// The message pushed onto a parent's queue lane. Defaults to a user
    /// message; a host that frames completions as system context overrides it.
    fn to_message(&self, records: &[CompletionRecord]) -> Message {
        Message::user(self.format_batch(records))
    }
}

/// Framing used when the host supplies none: a JSON roster in a
/// `<completed_child_tasks>` block with every `<` escaped as `<`, so child
/// text cannot close the block.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeutralCompletionFormatter;

#[derive(Serialize)]
struct Row<'a> {
    task_id: &'a str,
    agent_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<&'a str>,
    status: &'static str,
    text: &'a str,
    #[serde(skip_serializing_if = "is_zero")]
    omitted_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_id: Option<&'a str>,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

impl CompletionFormatter for NeutralCompletionFormatter {
    fn format_batch(&self, records: &[CompletionRecord]) -> String {
        if records.is_empty() {
            return String::new();
        }
        let rows: Vec<Row<'_>> = records
            .iter()
            .map(|r| Row {
                task_id: &r.task_id,
                agent_id: &r.agent_id,
                label: r.label.as_deref(),
                status: r.status.as_str(),
                text: &r.result.text,
                omitted_chars: r.result.omitted_chars,
                artifact_id: r.result.artifact.as_ref().map(|a| a.id.as_str()),
            })
            .collect();
        let json = serde_json::to_string_pretty(&rows)
            .expect("completion rows serialize")
            .replace('<', "\\u003c");
        format!("Child tasks finished:\n<completed_child_tasks>\n{json}\n</completed_child_tasks>")
    }
}
