//! How [`ContextCompressionMiddleware`] turns the messages a compaction folds
//! into one summary: split-turn aware, with file-operation lists appended.

use super::*;
use crate::summarization::{
    append_file_sections, extract_file_operations, split_file_sections, split_turn_start,
    summarize_split_turn,
};

impl ContextCompressionMiddleware {
    /// A summary a `before_compaction` hook supplied (already built on the
    /// previous summary's text), with this compaction's file lists added to
    /// the ones it carried.
    pub(in crate::middleware::library) fn hook_summary_text(
        &self,
        to_summarize: &[Message],
        text: String,
    ) -> String {
        let Some(extractor) = &self.file_ops else {
            return text;
        };
        let (body, mut ops) = split_file_sections(&text);
        ops.merge(&extract_file_operations(to_summarize, extractor.as_ref()));
        append_file_sections(&body, &ops)
    }

    /// Summarizes `to_summarize` (the messages folded away) for a compaction
    /// that keeps `to_keep` verbatim.
    ///
    /// * A cut inside a turn gives that turn's prefix its own summary request
    ///   ([`summarize_split_turn`]) instead of size-halving.
    /// * Unless disabled, the files read and modified by the folded tool calls
    ///   are appended as `<read-files>` / `<modified-files>` sections, unioned
    ///   with the lists the previous summary carried. The previous summary
    ///   reaches the summarizer without its lists, and any the summarizer
    ///   echoes are dropped, so each list appears exactly once.
    pub(in crate::middleware::library) async fn summarize_batch<Ctx: Send + Sync>(
        &self,
        ctx: &mut RunContext<Ctx>,
        to_summarize: &[Message],
        to_keep: &[Message],
        previous_summary: Option<String>,
    ) -> Result<SummaryRecord> {
        let (result, dispatched) = crate::summarization::dispatch::track_dispatch(
            self.summarize_batch_inner(to_summarize, to_keep, previous_summary),
        )
        .await;
        if dispatched {
            ctx.mark_summarizer_dispatched();
        }
        result
    }

    async fn summarize_batch_inner(
        &self,
        to_summarize: &[Message],
        to_keep: &[Message],
        previous_summary: Option<String>,
    ) -> Result<SummaryRecord> {
        let mut ops = crate::summarization::FileOperations::default();
        let previous_summary = match (&self.file_ops, previous_summary) {
            (Some(_), Some(previous)) => {
                let (body, carried) = split_file_sections(&previous);
                ops = carried;
                Some(body)
            }
            (_, previous) => previous,
        };
        let mut record = summarize_split_turn(
            self.summarizer.as_ref(),
            to_summarize,
            self.split_turn_prefix
                .then(|| split_turn_start(to_summarize, to_keep))
                .flatten(),
            self.max_turn_tokens.unwrap_or(u64::MAX),
            previous_summary,
            crate::token_estimation::estimate_message_tokens,
        )
        .await?;
        if let Some(extractor) = &self.file_ops {
            ops.merge(&extract_file_operations(to_summarize, extractor.as_ref()));
            let (body, _) = split_file_sections(&record.summary.text());
            record.summary = Message::system(append_file_sections(&body, &ops));
        }
        Ok(record)
    }
}
