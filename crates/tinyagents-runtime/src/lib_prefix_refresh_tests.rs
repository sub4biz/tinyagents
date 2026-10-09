use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
struct RefreshCodec {
    fail_encode: AtomicBool,
    fail_usage: AtomicBool,
}
impl TranscriptCodec for RefreshCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        Ok(transcript
            .messages
            .iter()
            .map(|row| match row.role.as_str() {
                "system" => Message::system(&row.content),
                "assistant" => Message::assistant(&row.content),
                _ => Message::user(&row.content),
            })
            .collect())
    }
    fn reconcile(
        &self,
        _: &[TranscriptMessage],
        _: &[Message],
        next: &[Message],
        _: &TranscriptTurnOptions,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        if self.fail_encode.load(Ordering::SeqCst) {
            return Err(RuntimeError::Driver("encode".into()));
        }
        Ok(next
            .iter()
            .map(|message| {
                TranscriptMessage::new(
                    match message {
                        Message::System(_) => "system",
                        Message::Assistant(_) => "assistant",
                        _ => "user",
                    },
                    message.text(),
                )
            })
            .collect())
    }
    fn turn_usage(&self, _: &TranscriptTurnOptions) -> Result<Option<TurnUsage>, RuntimeError> {
        if self.fail_usage.load(Ordering::SeqCst) {
            return Err(RuntimeError::Driver("usage".into()));
        }
        Ok(None)
    }
}
struct RefreshHook {
    prefix: Mutex<Option<PrefixSnapshot>>,
    reject: AtomicBool,
    wait: AtomicBool,
    entered: tokio::sync::Notify,
}
impl Default for RefreshHook {
    fn default() -> Self {
        Self {
            prefix: Mutex::new(None),
            reject: AtomicBool::new(false),
            wait: AtomicBool::new(false),
            entered: tokio::sync::Notify::new(),
        }
    }
}
#[async_trait]
impl SessionHooks for RefreshHook {
    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        Ok(TurnPreparation {
            prefix: self.prefix.lock().unwrap().clone(),
            ..Default::default()
        })
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        if self.reject.load(Ordering::SeqCst) {
            return Err(RuntimeError::Hook("reject".into()));
        }
        if self.wait.load(Ordering::SeqCst) {
            self.entered.notify_one();
            std::future::pending::<()>().await;
        }
        Ok(())
    }
}
fn old() -> Vec<Message> {
    vec![Message::system("old"), Message::user("first question")]
}
fn new_prefix() -> PrefixSnapshot {
    PrefixSnapshot::new(vec![Message::system("new")]).refreshing()
}

/// Replay equal model rows losslessly, including when refresh duplicates them.
struct RetainedCodec;
impl TranscriptCodec for RetainedCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        RefreshCodec::default().decode_history(transcript)
    }
    fn reconcile(
        &self,
        prior: &[TranscriptMessage],
        previous: &[Message],
        next: &[Message],
        options: &TranscriptTurnOptions,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        let mut rows = RefreshCodec::default().reconcile(prior, previous, next, options)?;
        for (row, message) in rows.iter_mut().zip(next) {
            if let Some(index) = previous.iter().position(|old| old == message)
                && let Some(old) = prior.get(index)
            {
                *row = old.clone();
            }
        }
        Ok(rows)
    }
}

#[tokio::test]
async fn uncommitted_refresh_errors_preserve_prefix_history_and_accept_old_frozen_prefix() {
    for failure in [
        "driver",
        "hook",
        "codec",
        "usage",
        "persistence",
        "partial codec",
        "partial usage",
        "partial persistence",
    ] {
        let hooks = Arc::new(RefreshHook::default());
        let codec = Arc::new(RefreshCodec::default());
        let refreshed = vec![
            Message::system("new"),
            Message::user("first question"),
            Message::user("second"),
        ];
        let result = if failure == "driver" {
            Err(DriverFailure {
                outcome: None,
                error: RuntimeError::Driver("failed".into()),
                partial: None,
            })
        } else if failure.starts_with("partial") {
            Err(DriverFailure {
                outcome: None,
                error: RuntimeError::Driver("partial".into()),
                partial: Some(outcome(refreshed)),
            })
        } else {
            Ok(outcome(refreshed))
        };
        let driver = Arc::new(Driver::new(vec![
            Ok(outcome(old())),
            result,
            Ok(outcome(old())),
        ]));
        let (locator, history) = locator(None);
        let mut session = SessionBuilder::new(driver)
            .prefix(PrefixSnapshot::new(vec![Message::system("old")]))
            .hooks(hooks.clone())
            .codec(codec.clone())
            .transcript(locator, "refresh", meta())
            .build()
            .unwrap();
        session
            .turn(
                SessionTurnRequest::new(Message::user("first question")),
                TurnOptions::default(),
            )
            .await
            .unwrap();
        let durable = history.state.lock().unwrap().clone().unwrap();
        *hooks.prefix.lock().unwrap() = Some(new_prefix());
        hooks.reject.store(failure == "hook", Ordering::SeqCst);
        codec
            .fail_encode
            .store(failure.ends_with("codec"), Ordering::SeqCst);
        codec
            .fail_usage
            .store(failure.ends_with("usage"), Ordering::SeqCst);
        *history.fail.lock().unwrap() = failure.ends_with("persistence");
        assert!(
            session
                .turn(
                    SessionTurnRequest::new(Message::user("second")),
                    TurnOptions::default()
                )
                .await
                .is_err()
        );
        assert_eq!(
            session.prefix_snapshot().messages(),
            &[Message::system("old")],
            "{failure}"
        );
        assert_eq!(session.history(), old(), "{failure}");
        assert_eq!(
            history.state.lock().unwrap().as_ref().unwrap().messages,
            durable.messages
        );
        *hooks.prefix.lock().unwrap() = Some(PrefixSnapshot::new(vec![Message::system("old")]));
        hooks.reject.store(false, Ordering::SeqCst);
        codec.fail_encode.store(false, Ordering::SeqCst);
        codec.fail_usage.store(false, Ordering::SeqCst);
        *history.fail.lock().unwrap() = false;
        session
            .turn(
                SessionTurnRequest::new(Message::user("retry")),
                TurnOptions::default(),
            )
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn cancelled_or_dropped_refresh_before_commit_keeps_committed_prefix() {
    for drop_future in [false, true] {
        let hooks = Arc::new(RefreshHook::default());
        let driver = Arc::new(Driver::new(vec![
            Ok(outcome(old())),
            Ok(outcome(vec![
                Message::system("new"),
                Message::user("second"),
            ])),
        ]));
        let (locator, history) = locator(None);
        let mut session = SessionBuilder::new(driver)
            .prefix(PrefixSnapshot::new(vec![Message::system("old")]))
            .hooks(hooks.clone())
            .codec(Arc::new(RefreshCodec::default()))
            .transcript(locator, "refresh", meta())
            .build()
            .unwrap();
        session
            .turn(
                SessionTurnRequest::new(Message::user("first question")),
                TurnOptions::default(),
            )
            .await
            .unwrap();
        let durable = history.state.lock().unwrap().clone().unwrap();
        *hooks.prefix.lock().unwrap() = Some(new_prefix());
        hooks.wait.store(true, Ordering::SeqCst);
        let options = TurnOptions::default();
        let cancellation = options.cancellation.clone();
        {
            let turn = session.turn(SessionTurnRequest::new(Message::user("second")), options);
            tokio::pin!(turn);
            tokio::select! { _ = hooks.entered.notified() => {}, _ = &mut turn => panic!("turn ended before hook") }
            if !drop_future {
                cancellation.cancel();
                assert!(matches!(turn.await, Err(RuntimeError::Cancelled)));
            }
        }
        assert_eq!(
            session.prefix_snapshot().messages(),
            &[Message::system("old")]
        );
        assert_eq!(session.history(), old());
        assert_eq!(
            history.state.lock().unwrap().as_ref().unwrap().messages,
            durable.messages
        );
        assert_eq!(
            history
                .state
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .meta
                .prefix_message_count,
            Some(1)
        );
    }
}

#[tokio::test]
async fn successfully_persisted_partial_refresh_keeps_new_prefix() {
    let hooks = Arc::new(RefreshHook::default());
    let partial_history = vec![
        Message::system("new"),
        Message::user("first question"),
        Message::user("partial"),
    ];
    let driver = Arc::new(Driver::new(vec![
        Ok(outcome(old())),
        Err(DriverFailure {
            outcome: None,
            error: RuntimeError::Driver("partial".into()),
            partial: Some(outcome(partial_history.clone())),
        }),
    ]));
    let (locator, history) = locator(None);
    let mut session = SessionBuilder::new(driver)
        .prefix(PrefixSnapshot::new(vec![Message::system("old")]))
        .hooks(hooks.clone())
        .codec(Arc::new(RefreshCodec::default()))
        .transcript(locator, "refresh", meta())
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("first question")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    *hooks.prefix.lock().unwrap() = Some(new_prefix());
    assert!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("second")),
                TurnOptions::default()
            )
            .await
            .is_err()
    );
    assert_eq!(
        session.prefix_snapshot().messages(),
        &[Message::system("new")]
    );
    assert_eq!(session.history(), partial_history);
    assert_eq!(
        history.state.lock().unwrap().as_ref().unwrap().messages[0].content,
        "new"
    );
}

#[tokio::test]
async fn legacy_mixed_role_prefix_resumes_repeatedly_then_records_boundary_on_commit() {
    legacy_prefix_contract(false, false).await;
}

#[tokio::test]
async fn mixed_role_refresh_preserves_overlapping_conversation_on_request_and_cold_resume() {
    let directory = tempfile::tempdir().unwrap();
    let locator = Arc::new(FileTranscriptLocator::new(directory.path().to_path_buf()));
    let identity = SessionRef::scoped("refresh", "agent-id");
    let hooks = Arc::new(RefreshHook::default());
    let prefix = vec![Message::system("new"), Message::user("first question")];
    let mut expected = prefix.clone();
    expected.extend(old()[1..].iter().cloned());
    expected.push(Message::user("second"));
    let driver = Arc::new(Driver::new(vec![
        Ok(outcome(old())),
        Ok(outcome(expected.clone())),
    ]));
    let codec = Arc::new(RefreshCodec::default());
    let mut session = SessionBuilder::new(driver.clone())
        .prefix(PrefixSnapshot::new(vec![Message::system("old")]))
        .hooks(hooks.clone())
        .codec(codec.clone())
        .session(locator.clone(), identity.clone(), meta())
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("first question")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    *hooks.prefix.lock().unwrap() = Some(PrefixSnapshot::new(prefix.clone()).refreshing());
    session
        .turn(
            SessionTurnRequest::new(Message::user("second")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(driver.requests.lock().unwrap()[1].history, expected);
    let mut resumed = SessionBuilder::new(Arc::new(Driver::new(vec![])))
        .codec(codec)
        .session(locator, identity, meta())
        .build()
        .unwrap();
    resumed
        .resume(&session_turn_options(ResumeMode::Session, "refresh"))
        .await
        .unwrap();
    assert_eq!(resumed.prefix_snapshot().messages(), prefix);
    assert_eq!(resumed.history(), expected);
}

#[tokio::test]
async fn prefix_extension_containing_prior_rows_forces_successor_for_normal_and_partial_turns() {
    for partial in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let locator = Arc::new(FileTranscriptLocator::new(directory.path()));
        let identity = SessionRef::scoped("extension", "agent-id");
        let mut initial_meta = meta();
        initial_meta.turn_count = 1;
        initial_meta.prefix_message_count = Some(1);
        let sealed = locator.open_session(&identity, initial_meta).unwrap();
        sealed
            .append(TranscriptMessage::new("system", "base"))
            .unwrap();
        let mut example = TranscriptMessage::new("user", "example");
        example.extra_metadata = Some(serde_json::json!({"opaque":"conversation provenance"}));
        example.request_id = Some("original-turn".into());
        sealed.append(example).unwrap();
        let sealed_before = std::fs::read(sealed.path()).unwrap();
        let original_row = sealed.read_session().unwrap().unwrap().messages[1].clone();
        let prefix = vec![Message::system("base"), Message::user("example")];
        let expected = vec![
            Message::system("base"),
            Message::user("example"),
            Message::user("example"),
            Message::user("next"),
        ];
        let result = if partial {
            Err(DriverFailure {
                outcome: None,
                error: RuntimeError::Driver("partial".into()),
                partial: Some(outcome(expected.clone())),
            })
        } else {
            Ok(outcome(expected.clone()))
        };
        let driver = Arc::new(Driver::new(vec![result]));
        let hooks = Arc::new(RefreshHook::default());
        *hooks.prefix.lock().unwrap() = Some(PrefixSnapshot::new(prefix.clone()).refreshing());
        let mut session = SessionBuilder::new(driver.clone())
            .hooks(hooks)
            .codec(Arc::new(RetainedCodec))
            .session(locator.clone(), identity.clone(), meta())
            .build()
            .unwrap();
        let options = session_turn_options(ResumeMode::Session, "extension");
        let result = session
            .turn(
                SessionTurnRequest::new(Message::user("next")),
                session_turn_options(ResumeMode::Session, "extension"),
            )
            .await;
        assert_eq!(result.is_err(), partial);
        assert_eq!(driver.requests.lock().unwrap()[0].history, expected);
        let head = locator.head_generation(&identity);
        assert_eq!(
            head.generation, 1,
            "changed committed prefix must seal generation zero"
        );
        assert_eq!(std::fs::read(sealed.path()).unwrap(), sealed_before);
        let persisted = locator
            .read_session_transcript(&head)
            .unwrap()
            .read_session()
            .unwrap()
            .unwrap();
        assert_eq!(persisted.meta.prefix_message_count, Some(2));
        assert_eq!(
            persisted.messages[2].extra_metadata,
            original_row.extra_metadata
        );
        assert_eq!(persisted.messages[2].request_id, original_row.request_id);
        drop(session);
        let mut reopened = SessionBuilder::new(Arc::new(Driver::new(vec![])))
            .codec(Arc::new(RetainedCodec))
            .session(
                Arc::new(FileTranscriptLocator::new(directory.path())),
                identity,
                meta(),
            )
            .build()
            .unwrap();
        assert_eq!(reopened.resume(&options).await.unwrap().history, expected);
        assert_eq!(reopened.prefix_snapshot().messages(), prefix);
    }
}

#[tokio::test]
async fn compacted_legacy_mixed_role_prefix_resumes_then_records_exact_boundary() {
    legacy_prefix_contract(true, false).await;
    legacy_prefix_contract(true, true).await;
}

async fn legacy_prefix_contract(compacted: bool, recorded_root: bool) {
    let directory = tempfile::tempdir().unwrap();
    let locator = Arc::new(FileTranscriptLocator::new(directory.path().to_path_buf()));
    let identity = SessionRef::scoped("legacy-prefix", "agent-id");
    let mut legacy_meta = meta();
    legacy_meta.turn_count = 1;
    let mut root_meta = legacy_meta.clone();
    root_meta.prefix_message_count = recorded_root.then_some(3);
    let root = locator.open_session(&identity, root_meta).unwrap();
    let history = if compacted {
        for (role, content) in [
            ("system", "policy"),
            ("user", "example"),
            ("assistant", "example answer"),
        ] {
            root.append(TranscriptMessage::new(role, content)).unwrap();
        }
        locator.begin_generation(&identity, legacy_meta).unwrap().1
    } else {
        root
    };
    for (role, content) in [
        ("system", "policy"),
        ("user", "example"),
        ("assistant", "example answer"),
        ("user", "real turn"),
        ("user", "another real turn"),
    ] {
        history
            .append(TranscriptMessage::new(role, content))
            .unwrap();
        if recorded_root && content == "example answer" {
            // An authoritative root count must preserve a real conversation
            // row equal to the configured prefix's suffix.
            history
                .append(TranscriptMessage::new("assistant", content))
                .unwrap();
        }
    }
    let prefix = vec![
        Message::system("policy"),
        Message::user("example"),
        Message::assistant("example answer"),
    ];
    let mut expected = prefix.clone();
    if recorded_root {
        expected.push(Message::assistant("example answer"));
    }
    expected.extend([
        Message::user("real turn"),
        Message::user("another real turn"),
    ]);
    let mut committed = expected.clone();
    committed.push(Message::user("next turn"));
    let driver = Arc::new(Driver::new(vec![Ok(outcome(committed.clone()))]));
    let codec = Arc::new(RefreshCodec::default());
    let mut session = SessionBuilder::new(driver.clone())
        .prefix(PrefixSnapshot::new(prefix.clone()))
        .codec(codec.clone())
        .session(locator.clone(), identity.clone(), meta())
        .build()
        .unwrap();
    let options = session_turn_options(ResumeMode::Session, "legacy-prefix");
    for _ in 0..2 {
        assert_eq!(session.resume(&options).await.unwrap().history, expected);
        assert_eq!(session.history(), expected);
        assert_eq!(
            history
                .read_session()
                .unwrap()
                .unwrap()
                .meta
                .prefix_message_count,
            None
        );
    }
    session
        .turn(
            SessionTurnRequest::new(Message::user("next turn")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(driver.requests.lock().unwrap()[0].history, committed);
    assert_eq!(
        locator
            .read_session_transcript(&locator.head_generation(&identity))
            .unwrap()
            .read_session()
            .unwrap()
            .unwrap()
            .meta
            .prefix_message_count,
        Some(3)
    );
    drop(session);
    let mut reopened = SessionBuilder::new(Arc::new(Driver::new(vec![])))
        .codec(codec)
        .session(
            Arc::new(FileTranscriptLocator::new(directory.path().to_path_buf())),
            identity,
            meta(),
        )
        .build()
        .unwrap();
    assert_eq!(reopened.resume(&options).await.unwrap().history, committed);
    assert_eq!(reopened.prefix_snapshot().messages(), prefix);
}
