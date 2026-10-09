//! Conversions into and out of [`OrchestrationTaskStatus`], the one durable
//! lifecycle status for a subagent or task run.
//!
//! Other layers keep their own enums because their serialized forms are
//! persisted (run ledger rows, completion records, job snapshots). Each such
//! enum is expressed in terms of the canonical one through `From` (total) or
//! `TryFrom` (fallible, [`NoEquivalentStatus`]) impls. This module owns the
//! error type and the [`CompletionStatus`] pair; the mapping table covering
//! every vocabulary is in this crate's README.

use std::fmt;

use crate::{CompletionStatus, OrchestrationTaskStatus};

/// A status has no counterpart in the target vocabulary.
///
/// Returned by the fallible (`TryFrom`) conversions, never for a status that
/// has a lossy-but-sensible mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoEquivalentStatus {
    from: &'static str,
    target: &'static str,
}

impl NoEquivalentStatus {
    /// Builds the error for a status labelled `from` that has no equivalent in
    /// the vocabulary named `target`.
    pub fn new(from: &'static str, target: &'static str) -> Self {
        Self { from, target }
    }

    /// Wire label of the status that could not be converted.
    pub fn from_status(&self) -> &'static str {
        self.from
    }

    /// Name of the vocabulary that has no equivalent.
    pub fn target(&self) -> &'static str {
        self.target
    }
}

impl fmt::Display for NoEquivalentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "status `{}` has no equivalent in {}",
            self.from, self.target
        )
    }
}

impl std::error::Error for NoEquivalentStatus {}

/// Lossy: `Incomplete` becomes `Failed`, because the completion status alone
/// does not say whether the child timed out or exhausted a budget.
impl From<CompletionStatus> for OrchestrationTaskStatus {
    fn from(status: CompletionStatus) -> Self {
        match status {
            CompletionStatus::Success => Self::Completed,
            CompletionStatus::Failed | CompletionStatus::Incomplete => Self::Failed,
            CompletionStatus::Cancelled => Self::Cancelled,
        }
    }
}

/// Only terminal task statuses are completions; a live status fails with
/// [`NoEquivalentStatus`]. `TimedOut` and `Abandoned` both read as
/// `Incomplete` (stopped without a complete result).
impl TryFrom<OrchestrationTaskStatus> for CompletionStatus {
    type Error = NoEquivalentStatus;

    fn try_from(status: OrchestrationTaskStatus) -> Result<Self, Self::Error> {
        use OrchestrationTaskStatus as T;
        match status {
            T::Completed => Ok(Self::Success),
            T::Failed => Ok(Self::Failed),
            T::Cancelled => Ok(Self::Cancelled),
            T::TimedOut | T::Abandoned => Ok(Self::Incomplete),
            T::Pending | T::Running | T::Awaiting | T::CancelRequested => Err(
                NoEquivalentStatus::new(crate::task_status_label(status), "CompletionStatus"),
            ),
        }
    }
}

#[cfg(test)]
#[path = "status_tests.rs"]
mod tests;
