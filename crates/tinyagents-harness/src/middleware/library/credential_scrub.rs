//! [`CredentialScrubMiddleware`]: scrub credential-shaped secrets out of every
//! tool result before it leaves the tool boundary.

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::RunContext;
use crate::error::Result as TaResult;
use crate::middleware::{MiddlewareToolOutcome, ToolHandler, ToolMiddleware};
use tinyinference_llm::tool::ToolCall as TaToolCall;

/// The placeholder `scrub_credentials` emits once per redacted value. Counted
/// to tell the model *how many* values went, which is the difference between
/// "something was withheld" and a number it can relay.
pub const REDACTION_PLACEHOLDER: &str = "*[REDACTED]";

/// Appended to a scrubbed tool result so the **model** learns what the log
/// already knew.
///
/// Without it the model receives a result that silently differs from what the
/// tool returned: it cannot find the content it was asked for, re-runs the same
/// call, gets an identically scrubbed result, and never converges — until the
/// successful-repeat tracker halts the run and the user is told "Incomplete"
/// with no reason (#6416). The redaction itself is correct and unchanged; only
/// its silence was the defect.
///
/// The "do not retry" clause is load-bearing: a retry is guaranteed to be
/// scrubbed identically, so it is the one action that cannot help.
///
/// Deliberately contains no `<keyword>: <value>` shape, so it cannot match
/// `SENSITIVE_KV_REGEX` and scrub itself on a second pass — pinned by
/// `the_notice_does_not_scrub_itself`.
pub fn redaction_notice(count: usize) -> String {
    format!(
        "[credential_scrub] {count} value(s) in this result were redacted as credentials. \
         Re-running this tool returns the same redaction, so do not retry — tell the user \
         which values were withheld and that they can view them directly in the source app."
    )
}

/// Scrub `content`, returning the replacement text **and** how many values
/// went — or `None` when nothing was credential-shaped.
///
/// Split out of `wrap_tool` so the decision and the composed result are
/// directly testable. Exercising this is the difference between proving the
/// notice text is well-formed and proving a scrubbed result actually carries
/// it; only `replace_tool_result_text` plumbing stays untested.
pub fn scrub_with_notice(content: &str) -> Option<(String, usize)> {
    let scrubbed = tinyinference_core::sanitize::scrub_credentials(content);
    if scrubbed == content {
        return None;
    }
    // Count what this pass removed, not what the text already carried — a
    // result may legitimately contain the placeholder already.
    let redactions = scrubbed
        .matches(REDACTION_PLACEHOLDER)
        .count()
        .saturating_sub(content.matches(REDACTION_PLACEHOLDER).count());
    Some((
        format!("{scrubbed}\n\n{}", redaction_notice(redactions)),
        redactions,
    ))
}

/// Decides, per tool, how a result's text is scrubbed: `(tool_name, content)`
/// to the replacement text and redaction count, or `None` when nothing needs
/// to change. The default is [`scrub_with_notice`] for every tool; a host that
/// must protect a host-minted field in one tool's output supplies its own and
/// falls back to [`scrub_with_notice`] for everything else.
pub type ToolScrubber = Arc<dyn Fn(&str, &str) -> Option<(String, usize)> + Send + Sync>;

/// `wrap_tool`: scrub credential-shaped secrets out of every tool result before
/// it leaves the tool boundary. Secrets in tool output (env dumps, config
/// reads, API responses, shell output) would otherwise reach model context,
/// on-disk transcripts, worker-thread mirrors, and any tool-outcome capture
/// sink.
///
/// Install it as the **innermost** tool wrap (pushed last), so it observes the
/// RAW tool result first and scrubs it before any outer wrap, the `after_tool`
/// chain (summarization/caps), the transcript push, or an outcome-capture sink
/// can see the unredacted content. Generic over the run-context payload.
pub struct CredentialScrubMiddleware {
    scrubber: ToolScrubber,
}

impl Default for CredentialScrubMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl CredentialScrubMiddleware {
    /// Scrub every tool's result with [`scrub_with_notice`].
    pub fn new() -> Self {
        Self {
            scrubber: Arc::new(|_tool, content| scrub_with_notice(content)),
        }
    }

    /// Scrub with a host-supplied per-tool scrubber (see [`ToolScrubber`]).
    pub fn with_scrubber(scrubber: ToolScrubber) -> Self {
        Self { scrubber }
    }
}

#[async_trait]
impl<C: Send + Sync> ToolMiddleware<(), C> for CredentialScrubMiddleware {
    fn name(&self) -> &str {
        "credential_scrub"
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext<C>,
        state: &(),
        call: TaToolCall,
        next: ToolHandler<'_, (), C>,
    ) -> TaResult<MiddlewareToolOutcome> {
        let tool_name = call.name.clone();
        let outcome = next.run(ctx, state, call).await?;
        // `MiddlewareToolOutcome` is `#[non_exhaustive]`; today it only carries a
        // `Result`, but match rather than irrefutable-let so a future variant
        // fails loud instead of silently bypassing scrubbing.
        let mut result = match outcome {
            MiddlewareToolOutcome::Result(result) => result,
            other => return Ok(other),
        };

        let mut redactions = 0usize;
        for block in result.content.iter_mut().chain(result.follow_up.iter_mut()) {
            if let tinytools::ToolContent::Image {
                data: tinytools::ImageData::Url(url),
                ..
            } = block
            {
                if let Some((scrubbed, count)) = (self.scrubber)(&tool_name, url) {
                    // The default scrubber appends an explanatory notice for
                    // text. Keep that notice out of the URL; it is added to
                    // the result text below.
                    *url = match scrubbed.split_once("\n\n[credential_scrub]") {
                        Some((url, _)) => url.to_owned(),
                        None => scrubbed,
                    };
                    redactions = redactions.saturating_add(count);
                }
                continue;
            }
            let (text, is_json) = match block {
                tinytools::ToolContent::Text { text } => (text.clone(), false),
                tinytools::ToolContent::Json { data } => (data.to_string(), true),
                tinytools::ToolContent::Image { .. } | tinytools::ToolContent::File { .. } => {
                    continue;
                }
            };
            if let Some((scrubbed, count)) = (self.scrubber)(&tool_name, &text) {
                if is_json {
                    // The default scrubber appends its model-facing notice after
                    // the JSON. Parse only the scrubbed payload so structured
                    // tool results remain structured.
                    let json_payload = scrubbed
                        .split_once("\n\n[credential_scrub]")
                        .map_or(scrubbed.as_str(), |(payload, _)| payload);
                    if let Ok(value) = serde_json::from_str(json_payload) {
                        if let tinytools::ToolContent::Json { data } = block {
                            *data = value;
                        }
                    } else if let tinytools::ToolContent::Json { data } = block {
                        *data = serde_json::Value::String(scrubbed);
                    }
                } else if let tinytools::ToolContent::Text { text } = block {
                    *text = scrubbed;
                }
                redactions = redactions.saturating_add(count);
            }
        }
        if let Some(markdown) = &mut result.markdown_formatted
            && let Some((scrubbed, count)) = (self.scrubber)(&tool_name, markdown)
        {
            *markdown = scrubbed;
            redactions = redactions.saturating_add(count);
        }
        if redactions > 0 {
            tracing::warn!(
                tool = %tool_name,
                redactions,
                "[tinyagents::mw] credential_scrub redacted secret(s) from tool result content"
            );
            // The notice goes to the model, in the result itself. The warning
            // above goes to the log, where no model will ever read it.
            result.content.push(tinytools::ToolContent::Text {
                text: redaction_notice(redactions),
            });
            result.markdown_formatted = None;
        }

        Ok(MiddlewareToolOutcome::Result(result))
    }
}
