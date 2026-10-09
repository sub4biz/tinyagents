//! Conversions between [`TranscriptSubagentStatus`] (a display projection read
//! back from a transcript) and [`OrchestrationTaskStatus`] (the canonical
//! lifecycle status). The projection's serde output is unchanged.

use tinyagents_tasks::{NoEquivalentStatus, OrchestrationTaskStatus};

use super::types::TranscriptSubagentStatus;

/// Lossy: `Incomplete` becomes `Failed` (the projection does not keep whether
/// the cause was a timeout or a budget); `Interrupted` becomes `Abandoned`.
/// Terminality is preserved (`Running` is the only live projection).
impl From<TranscriptSubagentStatus> for OrchestrationTaskStatus {
    fn from(status: TranscriptSubagentStatus) -> Self {
        match status {
            TranscriptSubagentStatus::Completed => Self::Completed,
            TranscriptSubagentStatus::Failed | TranscriptSubagentStatus::Incomplete => {
                Self::Failed
            }
            TranscriptSubagentStatus::Interrupted => Self::Abandoned,
            TranscriptSubagentStatus::Running => Self::Running,
        }
    }
}

/// Projects a task status onto the transcript view.
///
/// Every live status (`Pending`, `Running`, `Awaiting`, `CancelRequested`)
/// reads as `Running` (no terminal record yet); `TimedOut` reads as
/// `Incomplete` and `Abandoned` as `Interrupted`. `Cancelled` has no
/// transcript projection and fails with [`NoEquivalentStatus`].
impl TryFrom<OrchestrationTaskStatus> for TranscriptSubagentStatus {
    type Error = NoEquivalentStatus;

    fn try_from(status: OrchestrationTaskStatus) -> Result<Self, Self::Error> {
        use OrchestrationTaskStatus as T;
        match status {
            T::Pending | T::Running | T::Awaiting | T::CancelRequested => Ok(Self::Running),
            T::Completed => Ok(Self::Completed),
            T::Failed => Ok(Self::Failed),
            T::TimedOut => Ok(Self::Incomplete),
            T::Abandoned => Ok(Self::Interrupted),
            T::Cancelled => Err(NoEquivalentStatus::new(
                "cancelled",
                "TranscriptSubagentStatus",
            )),
        }
    }
}

#[cfg(test)]
#[path = "status_map_tests.rs"]
mod tests;
