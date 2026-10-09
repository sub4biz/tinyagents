use std::{ops::Range, path::PathBuf, sync::Arc};

use tinyagents_harness::{
    CancellationToken,
    context::{RunConfig, RunContext},
};
use tinyagents_session::transcript::{
    SessionRef, TranscriptLocator, TranscriptMessage, TranscriptMeta, session_stem,
};
use tinyinference_llm::message::Message;

use crate::{PrefixSnapshot, ToolSnapshot};

/// Selects the durable transcript a turn should load before execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResumeMode {
    /// Keep this session's current in-memory history.
    #[default]
    Never,
    /// Load the most recent transcript for the configured agent/stem key.
    LatestForAgent,
    /// Load the most recent root transcript matching `TurnOptions::thread_id`.
    Thread,
    /// Load the head generation of the session bound to this target.
    ///
    /// Unlike [`Self::Thread`] this is an exact lookup rather than a
    /// newest-wins scan, and the file it reads is the file the turn then
    /// appends to. That identity between read and write is what keeps one
    /// conversation in one transcript across restarts and across processes.
    Session,
}

/// Explicit runtime controls for one session turn.
pub struct TurnOptions<C = ()> {
    /// Opaque correlation identifier persisted with transcript rows.
    pub request_id: Option<String>,
    /// Optional conversation thread identifier used for resume and metadata.
    pub thread_id: Option<String>,
    /// Whether the driver should use its streaming invocation path.
    pub stream: bool,
    /// Transcript resume behavior requested for this turn.
    pub resume: ResumeMode,
    /// Durable session to resume under [`ResumeMode::Session`]. When absent,
    /// the bound target's own session is used.
    pub session: Option<SessionRef>,
    /// Cooperative cancellation shared with the caller.
    pub cancellation: CancellationToken,
    /// Explicit live execution context consumed by the driver.
    pub run_context: RunContext<C>,
}

/// The codec-visible, durable subset of one turn's explicit options.
///
/// `RunContext` itself is live and consumed by the driver. A clone of its host
/// context is captured before that handoff so transcript reconciliation can
/// stamp host-owned data after the driver returns without relying on task-local
/// state or a lossy default context.
#[derive(Clone, Debug)]
pub struct TranscriptTurnOptions<C = ()> {
    /// Opaque correlation identifier for the current turn.
    pub request_id: Option<String>,
    /// Conversation thread selected for this turn.
    pub thread_id: Option<String>,
    /// Whether this turn used the streaming driver path.
    pub stream: bool,
    /// Resume mode selected before execution.
    pub resume: ResumeMode,
    /// Host-owned context cloned from `TurnOptions::run_context.data`.
    pub context: C,
}

/// A transcript destination selected lazily by a host for a session.
///
/// Constructing a target performs no I/O. The runtime opens it only when a
/// requested resume or the first append needs a bound history handle.
#[derive(Clone)]
pub struct TranscriptTarget {
    pub locator: Arc<dyn TranscriptLocator>,
    /// The durable stem used for every append and write.
    pub stem: String,
    /// Optional agent key used only by `ResumeMode::LatestForAgent` lookup.
    /// When absent, the write stem is also the resume lookup key.
    pub resume_agent: Option<String>,
    /// Durable session identity, when the host binds one. Present means
    /// `ResumeMode::Session` can resolve, and that a compaction opens the next
    /// generation instead of rewriting this one.
    pub session: Option<SessionRef>,
    pub meta: TranscriptMeta,
}

impl TranscriptTarget {
    pub fn new(
        locator: Arc<dyn TranscriptLocator>,
        stem: impl Into<String>,
        meta: TranscriptMeta,
    ) -> Self {
        Self {
            locator,
            stem: stem.into(),
            resume_agent: None,
            session: None,
            meta,
        }
    }

    /// A target addressed by durable session identity rather than a raw stem.
    ///
    /// The stem is derived from the session, so it is stable across processes
    /// and launches — the property a `{unix_ts}_{agent}` stem never had.
    ///
    /// `meta.session_id`/`parent_session_id` are populated from `session`
    /// here, the same way [`Self::rebind_session`] keeps them in sync after a
    /// compaction. Leaving them as whatever the caller passed in (typically
    /// `None`, since a newly bound target usually has no opinion on session
    /// identity yet) would otherwise let a session-addressed transcript carry
    /// metadata that does not name its own session — metadata-based session
    /// discovery would then fail to recognise it.
    pub fn for_session(
        locator: Arc<dyn TranscriptLocator>,
        session: SessionRef,
        mut meta: TranscriptMeta,
    ) -> Self {
        meta.session_id = Some(session.session_id());
        meta.parent_session_id = session.parent_session_id();
        Self {
            locator,
            stem: session_stem(&session),
            resume_agent: None,
            session: Some(session),
            meta,
        }
    }

    /// Rebinds this target onto `session` after a compaction opened it.
    pub(crate) fn rebind_session(&mut self, session: SessionRef) {
        self.stem = session_stem(&session);
        self.meta.session_id = Some(session.session_id());
        self.meta.parent_session_id = session.parent_session_id();
        self.session = Some(session);
    }

    /// Uses a distinct agent key when looking up the latest transcript.
    pub fn with_resume_agent(mut self, resume_agent: impl Into<String>) -> Self {
        self.resume_agent = Some(resume_agent.into());
        self
    }

    /// Whether `other` addresses the same durable destination as `self`.
    ///
    /// For a session-bound target this compares `first_generation()` rather
    /// than the `SessionRef`s (or stems) directly: `before_resume` runs on
    /// every turn and is expected to keep returning the *same* logical
    /// target, but `resume`/`persist` call [`Self::rebind_session`] on it as
    /// soon as a later generation is discovered or a compaction opens one.
    /// Comparing the raw `session`/`stem` fields would then reject that
    /// still-identical target the moment its generation advanced, and
    /// `apply_resume_preparation` would fail every subsequent turn with
    /// `InvalidSessionState`. Generation 0 is the one identity that never
    /// changes across a session's lifetime, so it is what identifies "the
    /// same session" here. Non-session targets have no generation to anchor
    /// on, so they keep comparing the raw stem.
    pub(crate) fn same_binding(&self, other: &Self) -> bool {
        let same_destination = match (&self.session, &other.session) {
            (Some(a), Some(b)) => a.first_generation() == b.first_generation(),
            (None, None) => self.stem == other.stem,
            _ => false,
        };
        same_destination && self.resume_agent == other.resume_agent && self.same_locator(other)
    }

    /// Whether both targets' locators resolve to the same durable destination.
    ///
    /// Allocation identity is the fast path, not the answer: `TranscriptLocator`
    /// documents that a host builds one lazily from its *current*
    /// `workspace_dir` and never freezes it, so a host that follows that
    /// instruction hands `before_resume` a fresh `Arc` every turn. Judging it
    /// by pointer rejected exactly those hosts on their second turn — the same
    /// too-strict comparison [`Self::same_binding`] already had to abandon for
    /// the session field. A locator that cannot name its destination still
    /// only matches itself, so no binding that is rejected today starts being
    /// accepted.
    fn same_locator(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.locator, &other.locator) {
            return true;
        }
        match (
            self.locator.destination_key(),
            other.locator.destination_key(),
        ) {
            (Some(ours), Some(theirs)) => ours == theirs,
            _ => false,
        }
    }
}

/// Values prepared by `SessionHooks::before_resume` before transcript loading.
#[derive(Clone, Default)]
pub struct ResumePreparation {
    /// A lazy transcript destination. It can be selected or replaced before
    /// the first history handle is bound, but cannot be redirected afterwards.
    pub transcript: Option<TranscriptTarget>,
}

/// Values prepared by `SessionHooks::before_turn` for exactly one driver call.
#[derive(Clone, Default)]
pub struct TurnPreparation {
    /// A replacement prefix, normally allowed only before the first committed turn.
    /// Opt into later replacement with [`PrefixSnapshot::refreshing`]. Later refreshes
    /// preserve conversation rows and become committed only after persistence succeeds;
    /// identical prefixes are no-ops. Session targets record a successor generation,
    /// while stem targets use compaction. Initial prefixes reconcile resumed history.
    pub prefix: Option<PrefixSnapshot>,
    /// The immutable tool declarations for this driver request. `None` uses
    /// the builder's compatibility default and is never retained from a prior
    /// preparation.
    pub tools: Option<ToolSnapshot>,
}

impl TurnPreparation {
    pub fn with_tools(tools: ToolSnapshot) -> Self {
        Self {
            tools: Some(tools),
            ..Self::default()
        }
    }
}

/// Read-only session state supplied to `before_turn`.
#[derive(Clone, Copy)]
pub struct SessionStateView<'a> {
    pub history: &'a [Message],
    pub raw_history: &'a [TranscriptMessage],
    pub prefix: &'a PrefixSnapshot,
    pub transcript_target: Option<&'a TranscriptTarget>,
    pub committed_turns: usize,
    /// `true` only when this call loaded and decoded a durable transcript
    /// before `before_turn` ran.
    pub resumed: bool,
}

/// The shape of a successful logical transcript transition.
///
/// An append extends the prior logical rows. Any rewrite, including a context
/// compaction with a longer replacement, is reported as `Replace` rather than
/// pretending that a suffix range was appended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptDelta {
    Append {
        previous_len: usize,
        appended: Range<usize>,
    },
    Replace {
        previous_len: usize,
        next_len: usize,
    },
}

/// Durable transcript information supplied after a successful append.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptCommitReceipt {
    pub path: PathBuf,
    pub delta: TranscriptDelta,
}

/// Exactly-once post-durability observation data.
#[derive(Clone, Debug)]
pub struct CommitReceipt<C = ()> {
    pub outcome: SessionTurnOutcome,
    pub options: TranscriptTurnOptions<C>,
    /// `None` when the host selected no durable transcript target.
    pub transcript: Option<TranscriptCommitReceipt>,
}

impl<C: Clone> TurnOptions<C> {
    pub(crate) fn transcript_options(&self) -> TranscriptTurnOptions<C> {
        TranscriptTurnOptions {
            request_id: self.request_id.clone(),
            thread_id: self.thread_id.clone(),
            stream: self.stream,
            resume: self.resume,
            context: self.run_context.data.clone(),
        }
    }
}

impl Default for TurnOptions<()> {
    fn default() -> Self {
        let cancellation = CancellationToken::new();
        Self {
            request_id: None,
            thread_id: None,
            stream: false,
            resume: ResumeMode::Never,
            session: None,
            run_context: RunContext::new(RunConfig::new("session"), ())
                .with_cancellation(cancellation.clone()),
            cancellation,
        }
    }
}

/// The input a host asks a session to execute.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurnRequest {
    /// The next user/application message. Hooks may replace it before the
    /// runtime performs trailing-input deduplication.
    pub input: Message,
}

impl SessionTurnRequest {
    /// Creates a request with one next input message.
    pub fn new(input: Message) -> Self {
        Self { input }
    }
}

/// A committed turn result.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurnOutcome {
    /// The full logical history after this turn.
    pub history: Vec<Message>,
    /// The driver's final visible output, when it produced one.
    pub output: Option<String>,
    /// `true` when the driver intentionally ended at an interruptible point.
    pub interrupted: bool,
}

/// The result of loading a transcript into a session.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionResume {
    /// Whether a transcript was found and decoded.
    pub loaded: bool,
    /// The loaded model history, or the existing history when none was found.
    pub history: Vec<Message>,
}

/// The one terminal observation emitted for each call to [`crate::Session::turn`].
#[derive(Clone, Debug, PartialEq)]
pub enum SessionTerminal {
    /// The turn committed. The outcome supplies durable finalization data.
    Completed(SessionTurnOutcome),
    /// The turn was cooperatively cancelled.
    Cancelled,
    /// The turn ended with an error after any recoverable partial persistence.
    Failed(String),
}

impl SessionTerminal {
    /// A best-effort typed outcome derived from this terminal alone.
    ///
    /// A failure carries only its message here, so it classifies as
    /// `Internal`. The precise outcome (timeout, provider failure, ...) is
    /// delivered separately through
    /// [`SessionHooks::on_terminal_outcome`][crate::SessionHooks::on_terminal_outcome].
    pub fn outcome(&self) -> tinyagents_harness::terminal::TerminalOutcome {
        use tinyagents_harness::terminal::{TerminalOutcome, TerminalReason};
        match self {
            Self::Completed(turn) if turn.interrupted => {
                TerminalOutcome::new(TerminalReason::Paused, "turn interrupted before completion")
            }
            Self::Completed(_) => TerminalOutcome::completed(),
            Self::Cancelled => TerminalOutcome::new(TerminalReason::Cancelled, "turn cancelled"),
            Self::Failed(message) => {
                TerminalOutcome::new(TerminalReason::Internal, message.clone())
            }
        }
    }
}
