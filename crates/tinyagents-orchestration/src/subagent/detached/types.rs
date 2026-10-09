//! Data definitions for the detached subagent runtime.

use tinyagents_tasks::DetachedTaskRegistryError;

/// Terminal/transient state of a detached subagent, published by the
/// spawner's background task and observed by waiters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetachedSubagentStatus {
    /// Still executing its inner tool-call loop.
    Running,
    /// Finished normally with a final response.
    Completed {
        /// Final response text.
        output: String,
        /// Loop iterations the run used (0 when recovered from a durable record).
        iterations: usize,
    },
    /// Paused on a clarification request; resumed by a follow-up.
    AwaitingUser {
        /// The question the child is waiting on.
        question: String,
    },
    /// The run errored out.
    Failed {
        /// Failure description.
        error: String,
    },
}

/// The terminal outcome of a run that finished before its cancel arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishedOutcome {
    /// The run completed.
    Completed,
    /// The run failed.
    Failed,
}

/// Why a wait or resolution could not be set up.
#[derive(Debug, PartialEq, Eq)]
pub enum WaitError {
    /// No such subagent.
    Unknown,
    /// The caller does not own it.
    NotOwned,
    /// The process-local registry lock was poisoned by a panicking operation.
    RegistryPoisoned,
}

/// Result of waiting on a subagent.
#[derive(Debug)]
pub enum WaitOutcome {
    /// Reached a terminal status (the registry entry is pruned).
    Terminal(DetachedSubagentStatus),
    /// The timeout elapsed first; the entry is intact so the caller can wait
    /// again. Carries the latest non-terminal snapshot.
    TimedOut(DetachedSubagentStatus),
}

/// What a host records when a detached subagent is spawned.
#[derive(Debug, Clone, Copy)]
pub struct SpawnedSubagent<'a> {
    /// Transient task id (the registry key).
    pub task_id: &'a str,
    /// Worker type.
    pub agent_id: &'a str,
    /// Owning parent session.
    pub parent_session: &'a str,
    /// Session-parent prefix; its first `__`-delimited segment names the root run.
    pub session_parent_prefix: Option<&'a str>,
    /// Durable per-worker reference.
    pub subagent_session_id: Option<&'a str>,
    /// Workspace the run belongs to (recorded as a display string).
    pub workspace_dir: &'a str,
    /// Originating parent thread.
    pub parent_thread_id: Option<&'a str>,
}

/// What a host's registry metadata must expose for roster and resolution.
pub trait SubagentIdentity {
    /// Worker type (not unique across parallel workers).
    fn agent_id(&self) -> &str;
    /// Durable, stable per-worker reference, if any.
    fn subagent_session_id(&self) -> Option<&str>;
    /// Chat thread that spawned the worker, if any. Used to abort workers when
    /// their thread is deleted or stopped; the default is "no parent thread".
    fn parent_thread_id(&self) -> Option<&str> {
        None
    }
}

/// Compact, read-only view of one registered subagent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentSnapshot {
    /// Worker type.
    pub agent_id: String,
    /// Durable per-worker reference.
    pub subagent_session_id: Option<String>,
    /// Transient registry key.
    pub task_id: String,
    /// Stable status label (see [`DetachedSubagentStatus::label`]).
    pub status: &'static str,
}

/// A subagent addressed for resumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentResumeRef {
    /// Transient task id.
    pub task_id: String,
    /// Worker type.
    pub agent_id: String,
    /// Durable per-worker reference.
    pub subagent_session_id: Option<String>,
}

impl From<DetachedTaskRegistryError> for WaitError {
    /// Map a registry error onto [`WaitError`]: ownership and poisoning keep
    /// their own variants, everything else is unknown.
    fn from(error: DetachedTaskRegistryError) -> Self {
        match error {
            DetachedTaskRegistryError::NotOwned => Self::NotOwned,
            DetachedTaskRegistryError::LockPoisoned => Self::RegistryPoisoned,
            _ => Self::Unknown,
        }
    }
}
