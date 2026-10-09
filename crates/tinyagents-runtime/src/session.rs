use std::{future::Future, sync::Arc};

use tinyagents_harness::CancellationToken;
use tinyagents_harness::terminal::{TerminalOutcome, TerminalReason};
use tinyagents_session::transcript::{
    SessionRef, SessionTurnGuard, TranscriptHistory, TranscriptMessage, TranscriptPartial,
    TranscriptTurn, TurnUsage, lock_session_turn, session_stem,
};
use tinyinference_llm::message::Message;

use crate::{
    CommitReceipt, DriverRequest, PrefixSnapshot, ResumeMode, ResumePreparation, RuntimeError,
    SessionDriver, SessionHooks, SessionResume, SessionStateView, SessionTerminal,
    SessionTurnOutcome, SessionTurnRequest, ToolSnapshot, TranscriptCodec, TranscriptCommitReceipt,
    TranscriptDelta, TranscriptTarget, TranscriptTurnOptions, TurnOptions, TurnPreparation,
};

/// Host-neutral mutable state for one conversation session.
pub struct Session<C: Clone + Send + Sync + 'static = ()> {
    driver: Arc<dyn SessionDriver<C>>,
    codec: Option<Arc<dyn TranscriptCodec<C>>>,
    hooks: Arc<dyn SessionHooks<C>>,
    prefix: PrefixSnapshot,
    default_tools: ToolSnapshot,
    history: Vec<Message>,
    persisted: Vec<TranscriptMessage>,
    target: Option<TranscriptTarget>,
    transcript: Option<Arc<dyn TranscriptHistory>>,
    committed_turns: usize,
    /// Number of leading rows in the currently bound durable transcript that
    /// belong to its stored prefix. This can differ from a replacement
    /// `self.prefix` and must survive repeated resume calls before a commit.
    persisted_prefix_len: Option<usize>,
    /// Tool declarations this session last sent, restored from the transcript
    /// on resume and updated after every recorded turn.
    recorded_tools: Option<ToolSnapshot>,
    /// The `tools` record currently in force in the bound transcript file.
    recorded_tools_json: Option<serde_json::Value>,
    retain_recorded_tools: bool,
}

impl<C: Clone + Send + Sync + 'static> Session<C> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        driver: Arc<dyn SessionDriver<C>>,
        codec: Option<Arc<dyn TranscriptCodec<C>>>,
        hooks: Arc<dyn SessionHooks<C>>,
        prefix: PrefixSnapshot,
        default_tools: ToolSnapshot,
        target: Option<TranscriptTarget>,
    ) -> Self {
        Self {
            driver,
            codec,
            hooks,
            history: prefix.messages().to_vec(),
            prefix: prefix.frozen(),
            default_tools,
            persisted: Vec::new(),
            target,
            transcript: None,
            committed_turns: 0,
            persisted_prefix_len: None,
            recorded_tools: None,
            recorded_tools_json: None,
            retain_recorded_tools: false,
        }
    }

    pub(crate) fn set_retain_recorded_tools(&mut self, retain: bool) {
        self.retain_recorded_tools = retain;
    }

    /// Tool declarations this session last sent (restored on resume).
    pub fn recorded_tools(&self) -> Option<&ToolSnapshot> {
        self.recorded_tools.as_ref()
    }

    /// Returns the currently committed model history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns the stable prefix currently applied to this session.
    pub fn prefix_snapshot(&self) -> &PrefixSnapshot {
        &self.prefix
    }

    /// Returns the builder compatibility default used only when a preparation
    /// supplies no per-turn tool snapshot.
    pub fn tool_snapshot(&self) -> &ToolSnapshot {
        &self.default_tools
    }

    /// Seeds an uncommitted session from an explicit, lossless host snapshot.
    ///
    /// This replaces neither the host's raw rows nor their metadata. It is the
    /// supported alternative to a host keeping a shadow history beside the
    /// runtime. Seeding after any durable transition is rejected.
    pub fn seed_history(
        &mut self,
        history: Vec<Message>,
        raw: Vec<TranscriptMessage>,
    ) -> Result<(), RuntimeError> {
        if self.committed_turns != 0 {
            return Err(RuntimeError::InvalidSessionState(
                "cannot seed history after a committed turn".into(),
            ));
        }
        self.history = self.with_prefix(history);
        self.persisted = raw;
        Ok(())
    }

    /// Loads the selected durable transcript, retaining its lossless raw rows
    /// as the base for the next append-only delta.
    pub async fn resume(
        &mut self,
        options: &TurnOptions<C>,
    ) -> Result<SessionResume, RuntimeError> {
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let Some(target) = self.target.as_ref() else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        // Captured before the scanned transcript's metadata overwrites
        // `target.meta` below, so a session-bound target resumed through
        // `Thread`/`LatestForAgent` can fall back to its own pre-resume
        // metadata if the write destination turns out not to exist yet —
        // see the re-derivation block near the end of this method.
        let pre_scan_meta = target.meta.clone();
        let mut session_binding: Option<SessionRef> = None;
        let read = match options.resume {
            ResumeMode::Never => None,
            ResumeMode::LatestForAgent => target
                .locator
                .latest_for_agent(target.resume_agent.as_deref().unwrap_or(&target.stem)),
            ResumeMode::Thread => options.thread_id.as_deref().and_then(|thread| {
                target
                    .locator
                    .root_for_thread_scoped(thread, target.meta.agent_id.as_deref())
            }),
            ResumeMode::Session => {
                let Some(session) = options.session.clone().or_else(|| target.session.clone())
                else {
                    return Ok(SessionResume {
                        loaded: false,
                        history: self.history.clone(),
                    });
                };
                // The head generation, not the session the host named: a
                // compaction may have sealed that one and opened a successor,
                // and the head is the conversation the model is continuing.
                let head = target.locator.head_generation(&session);
                let read = target.locator.read_session_transcript(&head);
                if read.is_some() {
                    session_binding = Some(head);
                } else if let Some(thread) = options.thread_id.as_deref() {
                    // Nothing under this identity yet. A conversation written
                    // before session identity existed is spread over one or
                    // more timestamped stems; fold them in once so the model
                    // regains the turns the newest-wins lookup had stranded.
                    // Adoption is best effort: it recovers history that would
                    // otherwise be stranded, but failing to recover it must not
                    // fail the turn the user is waiting on.
                    if let Err(error) = target.locator.adopt_legacy(&session, thread, &target.meta)
                    {
                        tracing::warn!(
                            "[session] legacy adoption failed session={} thread={thread}: {error}",
                            session.session_id()
                        );
                    }
                    session_binding = Some(session.clone());
                }
                match read {
                    Some(read) => Some(read),
                    None => session_binding
                        .as_ref()
                        .and_then(|bound| target.locator.read_session_transcript(bound)),
                }
            }
        };
        let Some(read) = read else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        let Some(transcript) = read
            .read_session()
            .map_err(|error| RuntimeError::Persistence(error.to_string()))?
        else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        if let Some(selected) = session_binding.as_ref() {
            let expected = selected.session_id();
            if transcript
                .meta
                .session_id
                .as_deref()
                .is_some_and(|actual| actual != expected)
            {
                return Err(RuntimeError::Persistence(
                    "exact session read returned a different transcript identity".into(),
                ));
            }
        }
        let codec = self
            .codec
            .as_ref()
            .ok_or(RuntimeError::MissingDependency("TranscriptCodec"))?;
        let mut decoded = codec.decode_history(&transcript)?;
        // A compacted head can start with a System summary immediately after
        // the original frozen prompt. Its role does not make it prefix
        // material. Determine how many *stored* rows to strip independently
        // of any current replacement prefix: a shorter replacement must not
        // leave an old instruction behind as conversational history. Once this
        // session has committed a turn, its prefix is already the persisted
        // boundary; on a cold resume, read the sealed first generation.
        let leading_len = decoded
            .iter()
            .take_while(|message| matches!(message, Message::System(_)))
            .count();
        let cached_boundary = self
            .transcript
            .as_ref()
            .filter(|bound| bound.path() == read.path())
            .and(self.persisted_prefix_len);
        let recorded_boundary = transcript.meta.prefix_message_count;
        let mut stored_len = recorded_boundary.or(cached_boundary).unwrap_or(leading_len);
        let compacted_head = session_binding
            .as_ref()
            .is_some_and(|session| session.generation > 0)
            || transcript.meta.parent_session_id.is_some();
        let mut recovered_boundary = None;
        if recorded_boundary.is_none() && cached_boundary.is_none() && compacted_head {
            // Without the sealed root there is no safe boundary in a head
            // containing a System summary. If a replacement prefix was
            // supplied, fail rather than replaying unverifiable old System
            // instructions beside it.
            stored_len = 0;
            let mut boundary_resolved = false;
            let bound_session = session_binding.as_ref().or(target.session.as_ref());
            if let Some(head) = bound_session
                .map(|session| target.locator.head_generation(session))
                .filter(|session| {
                    session.generation > 0
                        && transcript.meta.session_id.as_deref()
                            == Some(session.session_id().as_str())
                })
            {
                let root = head.first_generation();
                if let Some(read) = target.locator.read_session_transcript(&root) {
                    match read.read_session() {
                        Ok(Some(root_transcript)) => match codec.decode_history(&root_transcript) {
                            Ok(root_messages) => {
                                let root_len = root_transcript
                                    .meta
                                    .prefix_message_count
                                    .unwrap_or_else(|| {
                                        root_messages
                                            .iter()
                                            .take_while(|message| {
                                                matches!(message, Message::System(_))
                                            })
                                            .count()
                                    })
                                    .min(root_messages.len());
                                let root_prefix = &root_messages[..root_len];
                                if decoded.len() >= root_prefix.len()
                                    && root_prefix
                                        .iter()
                                        .zip(decoded.iter())
                                        .all(|(root, head)| root == head)
                                {
                                    stored_len = root_prefix.len();
                                    boundary_resolved = true;
                                    recovered_boundary = root_transcript
                                        .meta
                                        .prefix_message_count
                                        .map(|_| stored_len);
                                } else {
                                    tracing::warn!(
                                        session = %root.session_id(),
                                        "[session] sealed prefix differs from compacted head; leaving system rows unfrozen"
                                    );
                                }
                            }
                            Err(error) => tracing::warn!(
                                session = %root.session_id(),
                                %error,
                                "[session] could not decode sealed prefix; leaving head system rows unfrozen"
                            ),
                        },
                        Ok(None) => tracing::warn!(
                            session = %root.session_id(),
                            "[session] sealed prefix missing; leaving head system rows unfrozen"
                        ),
                        Err(error) => tracing::warn!(
                            session = %root.session_id(),
                            %error,
                            "[session] could not read sealed prefix; leaving head system rows unfrozen"
                        ),
                    }
                } else {
                    tracing::warn!(
                        session = %root.session_id(),
                        "[session] sealed prefix unavailable; leaving head system rows unfrozen"
                    );
                }
            } else {
                tracing::warn!(
                    scanned_session = ?transcript.meta.session_id,
                    "[session] scanned compacted head differs from bound session; leaving system rows unfrozen"
                );
            }
            if !boundary_resolved && !self.prefix.messages().is_empty() {
                return Err(RuntimeError::Persistence(
                    "cannot apply a replacement prompt without the scanned transcript's sealed prefix"
                        .into(),
                ));
            }
        }
        let authoritative_boundary = recorded_boundary.is_some()
            || cached_boundary.is_some()
            || recovered_boundary.is_some();
        let stored_len = if authoritative_boundary {
            // An explicit frozen prefix may include non-System few-shot rows.
            // Only the legacy inferred boundary is limited to leading System
            // messages; a recorded count is bounded by the transcript itself.
            stored_len.min(decoded.len())
        } else {
            stored_len.min(leading_len)
        };
        // A legacy transcript has no recorded boundary for
        // non-System few-shot messages. Leading System rows alone must not
        // turn that unknown extent into an authoritative cached boundary.
        self.persisted_prefix_len = authoritative_boundary.then_some(stored_len);
        if self.prefix.messages().is_empty() && stored_len != 0 {
            self.prefix = PrefixSnapshot::new(decoded[..stored_len].to_vec());
        }
        decoded.drain(..stored_len);
        let history = if authoritative_boundary {
            // Equal messages after a known boundary are real conversation.
            let mut history = self.prefix.messages().to_vec();
            history.extend(decoded);
            history
        } else {
            // Legacy few-shot rows can remain after the leading System rows
            // were stripped. Reconcile their overlap until a successful turn
            // persists the full configured prefix count.
            self.with_prefix(decoded)
        };
        self.history = history.clone();
        // Every turn already on disk counts as committed: the prefix those
        // turns were sent with is part of the conversation, in this process
        // or the one that wrote it.
        self.committed_turns = self.committed_turns.max(transcript.meta.turn_count);
        self.recorded_tools_json = transcript.tools.clone();
        self.recorded_tools = Self::decode_recorded_tools(transcript.tools.as_ref());
        tracing::debug!(
            "[session] resumed history={} committed_turns={} recorded_tools={}",
            history.len(),
            self.committed_turns,
            self.recorded_tools
                .as_ref()
                .map_or(0, |tools| tools.specs().len())
        );
        self.persisted = transcript.messages;
        // The discovered metadata, not the builder seed, is authoritative for
        // the subsequent append. This keeps resume-only host fields intact.
        if let Some(target) = self.target.as_mut() {
            target.meta = transcript.meta;
        }
        // A successful explicit resume always rebinds the write handle to the
        // selected transcript. Builder construction itself remains I/O-free.
        //
        // For a session resume the handle must address **the file that was just
        // read**, not the target's original stem. Binding elsewhere is what
        // used to re-materialise a resumed history into a fresh stem and
        // orphan the original, leaving two roots claiming one thread.
        if let (Some(target), Some(head)) = (self.target.as_mut(), session_binding) {
            target.rebind_session(head);
        } else if let Some(target) = self.target.as_mut()
            && let Some(session) = target.session.clone()
        {
            // `session_binding` above is set only on the `ResumeMode::Session`
            // path, so a session-bound target resumed through `Thread` or
            // `LatestForAgent` would otherwise reach the bind below still
            // naming generation 0 — even when an earlier compaction already
            // sealed it and opened a later head. That write would land in a
            // generation the design requires to stay sealed and byte-for-byte
            // unchanged. Resolving the head here, for every mode, is what
            // `persist`'s own equivalent guard (`self.transcript.is_none()`)
            // cannot substitute for: `self.transcript` is bound unconditionally
            // a few lines down, so by the time `persist` runs on this turn
            // that guard has already been satisfied.
            let head = target.locator.head_generation(&session);
            if head != session {
                target.rebind_session(head);
            }
        }
        let target = self.target.as_ref().expect("target checked above");
        let handle = match target.session.as_ref() {
            Some(session) => target
                .locator
                .open_session(session, target.meta.clone())
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
            None => target
                .locator
                .open_stem(&target.stem, target.meta.clone())
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
        };
        // For a session-bound target, `target.session`/`target.stem` always
        // name the same file (construction and `rebind_session` keep them in
        // lockstep) — the bind above is always that file, regardless of
        // resume mode. Under `ResumeMode::Session`, `read` was already that
        // same file, so `self.persisted` (set above from `transcript`,
        // i.e. from `read`) already matches what this turn will append to.
        // Under `Thread`/`LatestForAgent`, `read` can legitimately be a
        // *different* file — a newest-wins scan recovering history from
        // wherever it exists is exactly their contract — while the destination
        // this turn writes to is still the session's own, separately-tracked
        // file. Using the scan's raw rows as the append-diff baseline for a
        // write that lands elsewhere would corrupt whatever is already on
        // that other file. Re-derive the baseline from the file this turn
        // actually writes to; `self.history` (what the model sees) keeps
        // coming from the scanned `read`, which is the intended recovery
        // behavior for those modes.
        if target.session.is_some() && options.resume != ResumeMode::Session {
            // The scanned file's `_meta` (set a few lines up, from `read`)
            // is equally wrong as an append baseline when `read` was a
            // different file: without this, the destination's next `_meta`
            // record would carry over the scanned file's `agent_id`,
            // `created`, provider/model, token/cost totals and (unless the
            // head changed) session identifiers — none of which describe
            // the file actually being appended to.
            let destination = handle
                .read_session()
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
            match destination {
                Some(destination_transcript) => {
                    self.recorded_tools_json = destination_transcript.tools.clone();
                    self.recorded_tools =
                        Self::decode_recorded_tools(destination_transcript.tools.as_ref());
                    self.persisted_prefix_len = destination_transcript.meta.prefix_message_count;
                    self.persisted = destination_transcript.messages;
                    if let Some(target) = self.target.as_mut() {
                        target.meta = destination_transcript.meta;
                    }
                }
                None => {
                    // Nothing at the destination yet: fall back to this
                    // target's own pre-resume metadata rather than the
                    // scanned file's, then reapply the session binding so
                    // `session_id`/`parent_session_id` stay canonical for
                    // whatever session this target now names (`resume`'s
                    // own head-resolution above may have rebound it).
                    self.recorded_tools_json = None;
                    self.recorded_tools = None;
                    self.persisted_prefix_len = None;
                    self.persisted = Vec::new();
                    if let Some(target) = self.target.as_mut() {
                        target.meta = pre_scan_meta;
                        if let Some(session) = target.session.clone() {
                            target.meta.session_id = Some(session.session_id());
                            target.meta.parent_session_id = session.parent_session_id();
                        }
                    }
                }
            }
        }
        // Publish the handle only after its destination baseline has been
        // loaded successfully. Otherwise a failed read could leave this
        // session bound with rows from the scanned transcript and cause later
        // writes to fail baseline validation (or compare against the wrong file).
        self.transcript = Some(handle);
        Ok(SessionResume {
            loaded: true,
            history,
        })
    }

    /// Executes and commits one state transition.
    pub async fn turn(
        &mut self,
        mut request: SessionTurnRequest,
        mut options: TurnOptions<C>,
    ) -> Result<SessionTurnOutcome, RuntimeError> {
        let mut terminal_guard = TerminalGuard::new(self.hooks.clone());
        let result = self
            .turn_inner(&mut request, &mut options, &mut terminal_guard)
            .await;
        if !terminal_guard.is_committed() {
            let terminal = match &result {
                Ok(outcome) => SessionTerminal::Completed(outcome.clone()),
                Err(RuntimeError::Cancelled) => SessionTerminal::Cancelled,
                Err(error) => SessionTerminal::Failed(error.to_string()),
            };
            terminal_guard.set(terminal);
        }
        // Terminal observation cannot revoke a durable successful commit.
        let _ = terminal_guard.finish().await;
        result
    }

    async fn turn_inner(
        &mut self,
        request: &mut SessionTurnRequest,
        options: &mut TurnOptions<C>,
        terminal_guard: &mut TerminalGuard<C>,
    ) -> Result<SessionTurnOutcome, RuntimeError> {
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let cancellation = options.cancellation.clone();
        let resume_preparation = cancelable(
            &cancellation,
            self.hooks
                .before_resume(request, options, self.state_view(false)),
        )
        .await?;
        self.apply_resume_preparation(resume_preparation)?;
        // Resume preparation may install a lazy target or change the selected
        // session, so choose the shared locks only after applying it. The lock
        // still covers the complete resume read and subsequent write.
        let _turn_lock =
            cancelable(&cancellation, async { Ok(self.lock_turn(options).await) }).await?;
        let resumed = if options.resume == ResumeMode::Never {
            false
        } else {
            self.resume(options).await?.loaded
        };
        // `resume` is synchronous after its read, so this explicit boundary
        // makes cancellation between loading and before-turn preparation
        // observable without handing work to the driver.
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let preparation = cancelable(
            &cancellation,
            self.hooks
                .before_turn(request, options, self.state_view(resumed)),
        )
        .await?;
        let previous_history = self.history.clone();
        let (tools, prepared_prefix) = self.apply_preparation(preparation)?;
        // Preparation stays local until persistence succeeds. Errors, cancellation,
        // and dropping this future cannot publish an uncommitted prefix.
        let (prefix, prepared_history) = match prepared_prefix {
            Some(prefix) => self.prepare_prefix(prefix)?,
            None => (self.prefix.clone(), self.history.clone()),
        };
        let exact_tools = tools.is_exact();
        let tools = if exact_tools {
            tools
        } else {
            self.retain_recorded(tools)?
        };
        // What this turn records as the session's tools: the set actually
        // sent, unless the host marked the turn's set as one-off.
        let record_tools = (!exact_tools).then(|| tools.clone());

        let mut input = prepared_history;
        if input.last() != Some(&request.input) {
            input.push(request.input.clone());
        }
        let codec_options = options.transcript_options();
        let request_id = options.request_id.clone();
        let thread_id = options.thread_id.clone();
        let stream = options.stream;
        let cancellation = options.cancellation.clone();
        // `RunContext` is consumed exactly once. The host context captured in
        // `codec_options` is the one after preparation and before handoff.
        let run_context = std::mem::replace(
            &mut options.run_context,
            tinyagents_harness::context::RunContext::new(
                tinyagents_harness::context::RunConfig::new("consumed-session-context"),
                codec_options.context.clone(),
            ),
        )
        .with_cancellation(cancellation.clone());
        let run_context = if prefix
            .messages()
            .iter()
            .all(|message| matches!(message, Message::System(_)))
        {
            run_context.with_frozen_system_prefix_len(prefix.messages().len())
        } else {
            // A mixed-role prefix is still restored by its recorded count,
            // but the harness's System-tier cache layout cannot represent its
            // non-System rows. Keep conservative request construction there.
            run_context
        };
        let driver_result = tokio::select! {
            _ = cancellation.cancelled() => return Err(RuntimeError::Cancelled),
            result = self.driver.execute(DriverRequest { history: input, tools, run_context, stream }) => result,
        };
        let mut outcome = match driver_result {
            Ok(outcome) => outcome,
            Err(failure) => {
                if let Some(partial) = failure.partial {
                    if cancellation.is_cancelled() {
                        return Err(RuntimeError::Cancelled);
                    }
                    let partial_history = Self::with_prefix_snapshot(&prefix, partial.history);
                    let raw = self.encode(&previous_history, &partial_history, &codec_options)?;
                    let turn_usage = self.turn_usage(&codec_options)?;
                    let receipt = self.persist(
                        &raw,
                        (request_id.as_deref(), thread_id.as_deref()),
                        partial.partial.as_ref(),
                        turn_usage.as_ref(),
                        record_tools.as_ref(),
                        &prefix,
                    )?;
                    self.remember_sent_tools(record_tools.as_ref());
                    self.prefix = prefix;
                    self.history = partial_history;
                    self.persisted = raw;
                    if receipt.is_some() {
                        self.committed_turns += 1;
                    }
                }
                terminal_guard.set_outcome(failure.outcome);
                return Err(failure.error);
            }
        };
        // Kept for `finalize_commit`: only a *committed* turn reports the
        // driver's success classification; a later failure uses its own.
        terminal_guard.driver_success_outcome = outcome.outcome.take();
        let candidate = Self::with_prefix_snapshot(&prefix, outcome.history);
        let committed = SessionTurnOutcome {
            history: candidate.clone(),
            output: outcome.output,
            interrupted: outcome.interrupted,
        };
        cancelable(
            &cancellation,
            self.hooks.before_commit(&committed, &codec_options),
        )
        .await?;
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let raw = self.encode(&previous_history, &candidate, &codec_options)?;
        let turn_usage = self.turn_usage(&codec_options)?;
        let transcript = self.persist(
            &raw,
            (request_id.as_deref(), thread_id.as_deref()),
            None,
            turn_usage.as_ref(),
            record_tools.as_ref(),
            &prefix,
        )?;
        self.remember_sent_tools(record_tools.as_ref());
        self.prefix = prefix;
        self.history = committed.history.clone();
        self.persisted = raw;
        self.committed_turns += 1;
        // Post-commit hooks may deliver another background message into this
        // session. Do not hold the non-reentrant turn lock while they run.
        drop(_turn_lock);
        // The receipt is constructed only after append and state replacement.
        // Its hook and the completed terminal are owned by one task: errors or
        // cancellation cannot relabel the successful durable transition, and
        // dropping the caller future cannot drop finalization mid-flight.
        let receipt = CommitReceipt {
            outcome: committed.clone(),
            options: codec_options,
            transcript,
        };
        let finalization = terminal_guard.finalize_commit(receipt);
        // This await deliberately does not observe cancellation. If this turn
        // future is dropped, dropping `JoinHandle` detaches rather than aborts
        // the owned finalization task.
        let _ = finalization.await;
        Ok(committed)
    }

    /// Takes the session's turn lock when the target is session-bound and its
    /// locator names a destination; otherwise there is nothing to share it
    /// with and the turn runs unlocked.
    async fn lock_turn(&self, options: &TurnOptions<C>) -> Vec<SessionTurnGuard> {
        let Some(target) = self.target.as_ref() else {
            return Vec::new();
        };
        let Some(target_session) = target.session.as_ref() else {
            let Some(session) = (options.resume == ResumeMode::Session)
                .then_some(options.session.as_ref())
                .flatten()
            else {
                return Vec::new();
            };
            return lock_session_turn(target.locator.as_ref(), session)
                .await
                .into_iter()
                .collect();
        };

        // An explicit session can be absent. In that case resume leaves the
        // target bound to `target_session`, which is where persist will write.
        // Hold both locks across the read/run/write span. Sort first so two
        // turns that name each other's session cannot deadlock while acquiring
        // their fallback and selected locks in opposite order.
        let mut sessions = vec![target_session];
        if options.resume == ResumeMode::Session
            && let Some(selected) = options.session.as_ref()
            && selected != target_session
        {
            sessions.push(selected);
        }
        sessions.sort_by_key(|session| session_stem(&session.first_generation()));
        sessions.dedup_by_key(|session| session_stem(&session.first_generation()));
        let mut guards = Vec::with_capacity(sessions.len());
        for session in sessions {
            if let Some(guard) = lock_session_turn(target.locator.as_ref(), session).await {
                guards.push(guard);
            }
        }
        guards
    }

    fn apply_preparation(
        &mut self,
        preparation: TurnPreparation,
    ) -> Result<(ToolSnapshot, Option<PrefixSnapshot>), RuntimeError> {
        // A returned snapshot never updates `default_tools`: it applies only
        // to the `DriverRequest` being built by this call.
        Ok((
            preparation
                .tools
                .unwrap_or_else(|| self.default_tools.clone()),
            preparation.prefix,
        ))
    }

    /// Merges back recorded declarations the host did not re-supply, when
    /// retention is on. See [`crate::SessionBuilder::retain_recorded_tools`].
    fn retain_recorded(&self, tools: ToolSnapshot) -> Result<ToolSnapshot, RuntimeError> {
        let Some(recorded) = self
            .recorded_tools
            .as_ref()
            .filter(|_| self.retain_recorded_tools)
        else {
            return Ok(tools);
        };
        let (merged, retained) = tools.with_retained(recorded)?;
        if retained != 0 {
            tracing::info!(
                "[session] retained {retained} recorded tool declaration(s) the host did not re-supply (sending {})",
                merged.specs().len()
            );
        }
        Ok(merged)
    }

    fn decode_recorded_tools(value: Option<&serde_json::Value>) -> Option<ToolSnapshot> {
        value.and_then(|value| match ToolSnapshot::from_json(value) {
            Ok(tools) => Some(tools),
            Err(error) => {
                tracing::warn!("[session] ignoring unreadable recorded tools: {error}");
                None
            }
        })
    }

    fn apply_resume_preparation(
        &mut self,
        preparation: ResumePreparation,
    ) -> Result<(), RuntimeError> {
        if let Some(target) = preparation.transcript {
            if self.transcript.is_some() || self.committed_turns != 0 {
                if !self
                    .target
                    .as_ref()
                    .is_some_and(|bound| bound.same_binding(&target))
                {
                    return Err(RuntimeError::InvalidSessionState(
                        "cannot change a transcript target after it is bound or committed".into(),
                    ));
                }
            } else {
                self.target = Some(target);
            }
        }
        if self.target.is_some() && self.codec.is_none() {
            return Err(RuntimeError::MissingDependency("TranscriptCodec"));
        }
        Ok(())
    }

    fn prepare_prefix(
        &self,
        prefix: PrefixSnapshot,
    ) -> Result<(PrefixSnapshot, Vec<Message>), RuntimeError> {
        let refresh = prefix.allows_refresh();
        let prefix = prefix.frozen();
        if prefix == self.prefix {
            return Ok((prefix, self.history.clone()));
        }
        if self.committed_turns != 0 && !refresh {
            return Err(RuntimeError::InvalidSessionState(
                "cannot change a session prefix after a committed turn".into(),
            ));
        }
        let conversation = self
            .history
            .strip_prefix(self.prefix.messages())
            .unwrap_or(&self.history);
        // These rows are conversation data after stripping the known prefix,
        // even when they equal a suffix of the replacement prefix.
        let mut history = prefix.messages().to_vec();
        history.extend_from_slice(conversation);
        Ok((prefix, history))
    }

    fn state_view(&self, resumed: bool) -> SessionStateView<'_> {
        SessionStateView {
            history: &self.history,
            raw_history: &self.persisted,
            prefix: &self.prefix,
            transcript_target: self.target.as_ref(),
            committed_turns: self.committed_turns,
            resumed,
        }
    }

    fn encode(
        &self,
        previous: &[Message],
        next: &[Message],
        options: &TranscriptTurnOptions<C>,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        match &self.codec {
            Some(codec) => codec.reconcile(&self.persisted, previous, next, options),
            None => Ok(Vec::new()),
        }
    }

    fn turn_usage(
        &self,
        options: &TranscriptTurnOptions<C>,
    ) -> Result<Option<TurnUsage>, RuntimeError> {
        match &self.codec {
            Some(codec) => codec.turn_usage(options),
            None => Ok(None),
        }
    }

    fn persist(
        &mut self,
        raw: &[TranscriptMessage],
        identifiers: (Option<&str>, Option<&str>),
        partial: Option<&TranscriptPartial>,
        turn_usage: Option<&TurnUsage>,
        tools: Option<&ToolSnapshot>,
        prefix: &PrefixSnapshot,
    ) -> Result<Option<TranscriptCommitReceipt>, RuntimeError> {
        // A committed prefix refresh is a replacement even when its rows
        // happen to begin with every old raw row. Never reclassify conversation
        // rows as prefix in the old generation, including persisted partials.
        let prefix_changed = self.committed_turns != 0 && prefix != &self.prefix;
        let prefix_len = prefix.messages().len();
        let (request_id, thread_id) = identifiers;
        let Some(target) = self.target.as_mut() else {
            return Ok(None);
        };
        if self.transcript.is_none() {
            // A turn can reach the first bind through a resume mode other
            // than `ResumeMode::Session` (e.g. `Never`, `LatestForAgent`,
            // `Thread`) on a session-bound target — `resume` only rebinds to
            // the head generation on its own `Session` path. Without this,
            // such a turn binds generation 0 even when a later `.g{n}`
            // exists: it appends into a generation the design requires to
            // stay sealed, and the next compaction's `begin_generation` then
            // fails outright because that later generation already exists.
            if let Some(session) = target.session.clone() {
                let head = target.locator.head_generation(&session);
                if head != session {
                    target.rebind_session(head);
                }
            }
            let handle = match target.session.as_ref() {
                Some(session) => target
                    .locator
                    .open_session(session, target.meta.clone())
                    .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
                None => target
                    .locator
                    .open_stem(&target.stem, target.meta.clone())
                    .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
            };
            // Binding can happen without resume (for example ResumeMode::Never).
            // Keep the durable comparison baseline current without changing
            // the visible in-memory history or invoking resume hooks.
            let persisted = handle
                .messages()
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
            self.transcript = Some(handle);
            self.persisted = persisted;
        }

        let previous_len = self.persisted.len();
        let next_len = raw.len();
        let common_len = previous_len.min(next_len);
        // Compare in normalized form: a legacy-string row from the host and the
        // typed row the transcript lifted from it are the same row.
        let extends = !prefix_changed
            && next_len >= previous_len
            && raw[..common_len]
                .iter()
                .zip(&self.persisted[..common_len])
                .all(|(next, previous)| {
                    next.same_row_as(previous)
                        && next.clone().normalized() == previous.clone().normalized()
                });

        // A turn that no longer extends what is persisted is a compaction. For
        // a session-bound target that seals the current generation and opens
        // the next one rather than appending a replacement record: rewriting
        // the logical set in place would make the replaced turns unreadable
        // forever, and they are the conversation's own history.
        //
        // The successor generation and handle are kept in locals, not written
        // onto `target`/`self.transcript`, until the append into them below
        // actually succeeds. Committing them first — as this used to — left
        // `target` pointing at `.g{n+1}` even when the append failed to
        // create it: the next turn's `begin_generation` would then find no
        // file at `.g{n+1}`, mint `.g{n+2}` instead, and `head_generation`
        // would keep resolving the old sealed generation as the head,
        // orphaning both the failed generation and the one after it.
        let mut prev: &[TranscriptMessage] = &self.persisted;
        let empty: [TranscriptMessage; 0] = [];
        let mut pending_generation: Option<(SessionRef, Arc<dyn TranscriptHistory>)> = None;
        let mut meta = target.meta.clone();
        if !extends && let Some(session) = target.session.clone() {
            let (successor, handle) = target
                .locator
                .begin_generation_from_baseline(&session, target.meta.clone(), &self.persisted)
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
            // The successor starts empty, so the retained set is written
            // through the ordinary turn path below and keeps its usage,
            // request ids and display partial.
            meta.turn_count = 0;
            meta.session_id = Some(successor.session_id());
            meta.parent_session_id = successor.parent_session_id();
            pending_generation = Some((successor, handle));
            prev = &empty;
        }

        let transcript: &dyn TranscriptHistory = match pending_generation.as_ref() {
            Some((_, handle)) => handle.as_ref(),
            None => self.transcript.as_deref().expect("bound above"),
        };
        meta.turn_count += 1;
        meta.prefix_message_count = Some(prefix_len);
        meta.updated = chrono::Utc::now().to_rfc3339();
        // Record every ordinary turn's declarations. Comparing against this
        // session's cached snapshot is unsafe when another live Session has
        // appended to the same transcript since our last turn; this append is
        // performed under the history's path lock.
        let tools_json = tools.map(ToolSnapshot::to_json);
        let tools_record = if pending_generation.is_some() {
            // Exact-tool turns deliberately do not replace the durable tool
            // list. A successor generation is a fresh file, though, so it
            // must carry that list forward or a later resume would lose it.
            tools_json.as_ref().or(self.recorded_tools_json.as_ref())
        } else {
            tools_json.as_ref()
        };
        meta.thread_id = thread_id.map(str::to_owned).or(meta.thread_id);
        transcript
            .append_turn_with_partial(
                TranscriptTurn {
                    prev,
                    next: raw,
                    meta: &meta,
                    turn_usage,
                    request_id,
                    tools: tools_record,
                },
                partial,
            )
            .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
        // Captured before `pending_generation`/`self.transcript` are moved
        // from below — `transcript` borrows out of whichever of the two held
        // the just-appended handle.
        let path = transcript.path().to_path_buf();
        // Only now that the append into the successor generation has
        // actually succeeded does the target move onto it.
        if let Some((successor, handle)) = pending_generation {
            target.rebind_session(successor);
            self.transcript = Some(handle);
        }
        target.meta = meta;
        self.persisted_prefix_len = Some(prefix_len);
        let delta = if extends {
            TranscriptDelta::Append {
                previous_len,
                appended: previous_len..next_len,
            }
        } else {
            TranscriptDelta::Replace {
                previous_len,
                next_len,
            }
        };
        Ok(Some(TranscriptCommitReceipt { path, delta }))
    }

    /// Records the declarations a successfully completed ordinary turn sent.
    /// This is deliberately outside `persist`: sessions without a transcript
    /// target still need retention to work between their in-memory turns.
    fn remember_sent_tools(&mut self, tools: Option<&ToolSnapshot>) {
        match tools {
            Some(tools) => {
                self.recorded_tools = Some(tools.clone());
                self.recorded_tools_json = Some(tools.to_json());
            }
            // An exact-tools turn is deliberately one-off. Do not let a
            // snapshot sent before it leak back into a later retained turn.
            None => {
                self.recorded_tools = None;
                self.recorded_tools_json = None;
            }
        }
    }

    fn with_prefix(&self, history: Vec<Message>) -> Vec<Message> {
        Self::with_prefix_snapshot(&self.prefix, history)
    }

    fn with_prefix_snapshot(snapshot: &PrefixSnapshot, history: Vec<Message>) -> Vec<Message> {
        let prefix = snapshot.messages();
        let overlap = (0..=prefix.len().min(history.len()))
            .rev()
            .find(|&len| prefix[prefix.len() - len..] == history[..len])
            .unwrap_or_default();
        let mut reconciled = prefix[..prefix.len() - overlap].to_vec();
        reconciled.extend(history);
        reconciled
    }
}

/// Ensures a terminal hook is scheduled once even if a caller drops a turn
/// future while it is awaiting preparation, driving, persistence, or hooks.
struct TerminalGuard<C: Clone + Send + Sync + 'static> {
    hooks: Arc<dyn SessionHooks<C>>,
    terminal: Option<SessionTerminal>,
    /// The driver's own typed classification of a failure, when it supplied one.
    outcome: Option<TerminalOutcome>,
    /// The driver's typed classification of a run it returned successfully;
    /// used only once the turn commits.
    driver_success_outcome: Option<TerminalOutcome>,
    /// `true` until a real terminal is set: the drop-time default means the
    /// caller abandoned the turn, which is a cancellation.
    abandoned: bool,
    committed: bool,
}

impl<C: Clone + Send + Sync + 'static> TerminalGuard<C> {
    fn new(hooks: Arc<dyn SessionHooks<C>>) -> Self {
        Self {
            hooks,
            terminal: Some(SessionTerminal::Failed("session turn dropped".into())),
            outcome: None,
            driver_success_outcome: None,
            abandoned: true,
            committed: false,
        }
    }

    fn set(&mut self, terminal: SessionTerminal) {
        self.terminal = Some(terminal);
        self.abandoned = false;
    }

    fn set_outcome(&mut self, outcome: Option<TerminalOutcome>) {
        self.outcome = outcome;
    }

    /// Takes the pending terminal with its typed outcome: the driver's if it
    /// gave one, a cancellation if the turn was abandoned, else derived.
    fn take_pending(&mut self) -> Option<(SessionTerminal, TerminalOutcome)> {
        let terminal = self.terminal.take()?;
        let outcome = self.outcome.take().unwrap_or_else(|| {
            if self.abandoned {
                TerminalOutcome::new(TerminalReason::Cancelled, "session turn dropped")
            } else {
                terminal.outcome()
            }
        });
        Some((terminal, outcome))
    }

    fn finalize_commit(&mut self, receipt: CommitReceipt<C>) -> tokio::task::JoinHandle<()> {
        let terminal = SessionTerminal::Completed(receipt.outcome.clone());
        let outcome = self
            .driver_success_outcome
            .take()
            .unwrap_or_else(|| terminal.outcome());
        // Removing the guard's terminal transfers exactly-once ownership to
        // the finalizer. `finish` and `Drop` then become no-ops for this turn.
        self.terminal = None;
        self.committed = true;
        let hooks = self.hooks.clone();
        tokio::spawn(async move {
            let _ = hooks.after_commit(receipt).await;
            let _ = hooks.on_terminal_outcome(outcome).await;
            let _ = hooks.on_terminal(terminal).await;
        })
    }

    fn is_committed(&self) -> bool {
        self.committed
    }

    async fn finish(mut self) -> Result<(), RuntimeError> {
        let Some((terminal, outcome)) = self.take_pending() else {
            return Ok(());
        };
        // The terminal is already removed from the guard, so `Drop` can no
        // longer deliver it. Run the hooks on their own task: if the caller's
        // future is dropped while a hook is pending, the remaining hooks still
        // run exactly once.
        let hooks = self.hooks.clone();
        let task = tokio::spawn(async move {
            let _ = hooks.on_terminal_outcome(outcome).await;
            hooks.on_terminal(terminal).await
        });
        match task.await {
            Ok(result) => result,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(_) => Ok(()),
        }
    }
}

impl<C: Clone + Send + Sync + 'static> Drop for TerminalGuard<C> {
    fn drop(&mut self) {
        let Some((terminal, outcome)) = self.take_pending() else {
            return;
        };
        let hooks = self.hooks.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = hooks.on_terminal_outcome(outcome).await;
                let _ = hooks.on_terminal(terminal).await;
            });
        }
    }
}

async fn cancelable<T>(
    cancellation: &CancellationToken,
    future: impl Future<Output = Result<T, RuntimeError>>,
) -> Result<T, RuntimeError> {
    if cancellation.is_cancelled() {
        return Err(RuntimeError::Cancelled);
    }
    tokio::select! {
        _ = cancellation.cancelled() => Err(RuntimeError::Cancelled),
        result = future => result,
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod session_tests;
