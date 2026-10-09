use super::*;
use crate::testkit::conformance::transcript_history_conformance;
use tinystoragedrivers_core::{MemoryStorage, Scope, StorageBackend, Version};

pub(in crate::port::drivers) fn meta(thread_id: &str) -> TranscriptMeta {
    let mut meta = discovered_seed("planner");
    meta.agent_id = Some("planner".to_string());
    meta.dispatcher = "native".to_string();
    meta.created = "2026-01-01T00:00:00Z".to_string();
    meta.updated = "2026-01-01T00:00:00Z".to_string();
    meta.thread_id = Some(thread_id.to_string());
    meta
}

fn message(role: &str, content: &str) -> TranscriptMessage {
    serde_json::from_value(json!({ "role": role, "content": content })).unwrap()
}

fn docs() -> Arc<dyn DocumentStore> {
    let storage = MemoryStorage::new();
    Arc::clone(storage.for_scope(&Scope::local()).unwrap().documents())
}

fn locator(docs: &Arc<dyn DocumentStore>) -> DriverTranscriptLocator {
    DriverTranscriptLocator::new(Arc::clone(docs), Blocking::new().unwrap(), "memory://test")
}

fn turn<'a>(
    prev: &'a [TranscriptMessage],
    next: &'a [TranscriptMessage],
    meta: &'a TranscriptMeta,
) -> TranscriptTurn<'a> {
    TranscriptTurn {
        prev,
        next,
        meta,
        turn_usage: None,
        request_id: Some("req"),
        tools: None,
    }
}

/// The raw log entries of `stem`, in order.
fn entries(docs: &Arc<dyn DocumentStore>, stem: &str) -> Vec<Value> {
    let docs = Arc::clone(docs);
    let stem = stem.to_string();
    Blocking::new()
        .unwrap()
        .run(async move {
            docs.query_all(
                ENTRIES,
                &Query::filter(Filter::eq("stem", stem)).sort(Sort::asc("seq")),
            )
            .await
            .unwrap()
            .into_iter()
            .map(|found| found.doc)
            .collect()
        })
        .unwrap()
}

fn on_bridge<T: Send + 'static>(
    future: impl Future<Output = Result<T, StorageError>> + Send + 'static,
) -> T {
    Blocking::new().unwrap().run(future).unwrap().unwrap()
}

#[test]
fn a_handle_meets_the_transcript_history_contract() {
    let docs = docs();
    let history = locator(&docs).open_stem("contract", meta("t")).unwrap();
    transcript_history_conformance(history.as_ref());
}

#[test]
fn an_ordinary_turn_stores_only_its_new_rows() {
    let docs = docs();
    let history = locator(&docs).open_stem("s", meta("t")).unwrap();
    let first = vec![message("user", "a"), message("assistant", "b")];
    history.append_turn(turn(&[], &first, &meta("t"))).unwrap();
    let mut second = first.clone();
    second.extend([message("user", "c"), message("assistant", "d")]);
    history
        .append_turn(turn(&first, &second, &meta("t")))
        .unwrap();
    let compacted = vec![message("user", "summary")];
    history
        .append_turn(turn(&second, &compacted, &meta("t")))
        .unwrap();

    let log = entries(&docs, "s");
    assert_eq!(log.len(), 3);
    assert_eq!(log[0]["set"].as_array().unwrap().len(), 2);
    assert_eq!(log[1]["extend"].as_array().unwrap().len(), 2);
    assert!(log[1].get("set").is_none());
    assert_eq!(log[2]["set"].as_array().unwrap().len(), 1);
    assert_eq!(log[2]["seq"], json!(2));
    let contents: Vec<String> = history
        .messages()
        .unwrap()
        .into_iter()
        .map(|row| row.content)
        .collect();
    assert_eq!(contents, ["summary"]);
}

#[test]
fn two_handles_on_one_stem_never_lose_each_others_writes() {
    let docs = docs();
    let locator = locator(&docs);
    let one = locator.open_stem("s", meta("t")).unwrap();
    let two = locator.open_stem("s", meta("t")).unwrap();
    one.append(message("user", "from one")).unwrap();
    two.append(message("user", "from two")).unwrap();
    one.append(message("user", "one again")).unwrap();
    let contents: Vec<String> = two
        .messages()
        .unwrap()
        .into_iter()
        .map(|row| row.content)
        .collect();
    assert_eq!(contents, ["from one", "from two", "one again"]);
}

#[test]
fn concurrent_writers_serialize_through_the_log() {
    let docs = docs();
    let locator = locator(&docs);
    let threads: Vec<_> = (0..4)
        .map(|writer| {
            let handle = locator.open_stem("busy", meta("t")).unwrap();
            std::thread::spawn(move || {
                for i in 0..5 {
                    handle
                        .append(message("user", &format!("{writer}-{i}")))
                        .unwrap();
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    let reader = locator.open_stem("busy", meta("t")).unwrap();
    assert_eq!(reader.messages().unwrap().len(), 20);
    let seqs: Vec<u64> = entries(&docs, "busy")
        .iter()
        .map(|entry| entry["seq"].as_u64().unwrap())
        .collect();
    assert_eq!(seqs, (0..20).collect::<Vec<_>>(), "the log is dense");
}

#[test]
fn tools_and_meta_follow_the_latest_turn() {
    let docs = docs();
    let history = locator(&docs).open_stem("s", meta("t")).unwrap();
    let rows = vec![message("user", "a")];
    let tools = json!([{ "name": "search" }]);
    history
        .append_turn(TranscriptTurn {
            tools: Some(&tools),
            ..turn(&[], &rows, &meta("t"))
        })
        .unwrap();
    let mut later = meta("t");
    later.turn_count = 2;
    history.append_turn(turn(&rows, &rows, &later)).unwrap();
    let read = history.read_session().unwrap().unwrap();
    assert_eq!(
        read.tools,
        Some(tools),
        "a turn without tools keeps the last"
    );
    assert_eq!(read.meta.turn_count, 2);
}

#[test]
fn partials_stay_out_of_the_replay_and_a_clear_drops_them() {
    let docs = docs();
    let locator = locator(&docs);
    let handle = locator.handle("s", meta("t"));
    assert!(
        !handle
            .record_partial(&TranscriptPartial::new(""), None)
            .unwrap()
    );
    let rows = vec![message("user", "a")];
    let mut partial = TranscriptPartial::new("half");
    partial.reasoning_content = Some("thinking".into());
    partial.iteration = Some(2);
    handle
        .append_turn_with_partial(turn(&[], &rows, &meta("t")), Some(&partial))
        .unwrap();
    assert!(
        handle
            .record_partial(&TranscriptPartial::new("more"), Some("r2"))
            .unwrap()
    );
    let partials = handle.partials().unwrap();
    assert_eq!(partials.len(), 2);
    assert_eq!(partials[0], (partial, Some("req".to_string())));
    assert_eq!(partials[1].1.as_deref(), Some("r2"));
    assert_eq!(handle.messages().unwrap().len(), 1);
    handle.clear().unwrap();
    assert!(handle.partials().unwrap().is_empty());
    assert!(handle.messages().unwrap().is_empty());
    assert!(
        handle.read_session().unwrap().is_some(),
        "a cleared transcript still exists"
    );
}

#[test]
fn a_partial_alone_makes_the_transcript_exist() {
    let docs = docs();
    let handle = locator(&docs).handle("s", meta("t"));
    assert!(
        handle
            .record_partial(&TranscriptPartial::new("half"), None)
            .unwrap()
    );
    let read = handle.read_session().unwrap().unwrap();
    assert!(read.messages.is_empty());
    assert_eq!(
        read.meta.thread_id.as_deref(),
        Some("t"),
        "the seed is the meta"
    );
}

#[test]
fn a_sealed_generation_refuses_writes() {
    let docs = docs();
    let locator = locator(&docs);
    let session = SessionRef::scoped("t", "planner");
    let history = locator.open_session(&session, meta("t")).unwrap();
    history.append(message("user", "a")).unwrap();
    let (successor, _next) = locator.begin_generation(&session, meta("t")).unwrap();
    assert_eq!(successor.generation, 1);
    for error in [
        history.append(message("user", "late")).unwrap_err(),
        history.replace(&[]).unwrap_err(),
        history.clear().unwrap_err(),
        history.append_turn(turn(&[], &[], &meta("t"))).unwrap_err(),
    ] {
        assert!(error.to_string().contains("sealed"), "{error}");
    }
    let handle = locator.handle(&session_stem(&session), meta("t"));
    assert!(
        !handle
            .record_partial(&TranscriptPartial::new("x"), None)
            .unwrap()
    );
    handle.seal(None).unwrap();
    assert_eq!(
        entries(&docs, &session_stem(&session))
            .iter()
            .filter(|entry| entry["seal"] == json!(true))
            .count(),
        1,
        "sealing twice writes one seal"
    );
}

#[test]
fn a_generation_is_opened_once() {
    let docs = docs();
    let locator = locator(&docs);
    let session = SessionRef::scoped("t", "planner");
    locator
        .open_session(&session, meta("t"))
        .unwrap()
        .append(message("user", "a"))
        .unwrap();
    let (successor, next) = locator.begin_generation(&session, meta("t")).unwrap();
    let error = locator.begin_generation(&session, meta("t")).err().unwrap();
    assert!(error.to_string().contains("reserved"), "{error}");
    next.append(message("user", "summary")).unwrap();
    let error = locator.begin_generation(&session, meta("t")).err().unwrap();
    assert!(error.to_string().contains("already exists"), "{error}");
    assert_eq!(locator.head_generation(&session), successor);
    let read = next.read_session().unwrap().unwrap();
    assert_eq!(read.meta.session_id, Some(successor.session_id()));
    assert_eq!(read.meta.parent_session_id, successor.parent_session_id());
}

#[test]
fn a_stale_reservation_is_taken_over() {
    let docs = docs();
    let locator = locator(&docs);
    let session = SessionRef::scoped("t", "planner");
    locator
        .open_session(&session, meta("t"))
        .unwrap()
        .append(message("user", "a"))
        .unwrap();
    locator.begin_generation(&session, meta("t")).unwrap();
    // The process that reserved generation 1 stopped before writing it.
    let id = doc_key(&[&session_stem(&session.next_generation())]);
    let aged = Arc::clone(&docs);
    on_bridge(async move {
        let found = aged.get(INDEX, &id).await?.unwrap();
        let mut doc = found.doc;
        doc["reserved_at"] = json!(0);
        aged.put(INDEX, &id, doc, Precondition::None).await
    });
    let (successor, next) = locator.begin_generation(&session, meta("t")).unwrap();
    next.append(message("user", "summary")).unwrap();
    assert_eq!(locator.head_generation(&session), successor);
}

#[test]
fn the_generation_limit_holds() {
    let docs = docs();
    let mut session = SessionRef::scoped("t", "planner");
    session.generation = MAX_GENERATIONS;
    let error = locator(&docs)
        .begin_generation(&session, meta("t"))
        .err()
        .unwrap();
    assert!(error.to_string().contains("limit"), "{error}");
}

#[test]
fn lookups_find_the_newest_written_root() {
    let docs = docs();
    let locator = locator(&docs);
    assert!(locator.root_for_thread("  ").is_none());
    assert!(
        !locator
            .append_interrupted_partial(" ", None, &TranscriptPartial::new("x"), None)
            .unwrap()
    );
    assert!(
        !locator
            .append_interrupted_partial("t", None, &TranscriptPartial::new(""), None)
            .unwrap()
    );

    // An opened but unwritten stem is not found.
    let _unwritten = locator.open_stem("never", meta("t")).unwrap();
    assert!(locator.root_for_thread("t").is_none());

    locator
        .open_stem("old", meta("t"))
        .unwrap()
        .append(message("user", "old"))
        .unwrap();
    locator
        .open_stem("new", meta("t"))
        .unwrap()
        .append(message("user", "new"))
        .unwrap();
    locator
        .open_stem("new__child", meta("t"))
        .unwrap()
        .append(message("user", "child"))
        .unwrap();
    let newest = locator.root_for_thread("t").unwrap();
    assert_eq!(newest.path(), Path::new("memory://test/new"));
    assert_eq!(
        locator.latest_for_agent("planner").unwrap().path(),
        Path::new("memory://test/new"),
        "sub-agent transcripts are never roots"
    );
    assert!(locator.root_for_thread_scoped("t", Some("other")).is_none());
    assert!(locator.latest_for_agent("nobody").is_none());

    assert!(
        locator
            .append_interrupted_partial("t", Some("planner"), &TranscriptPartial::new("p"), None)
            .unwrap()
    );
    assert_eq!(
        locator.handle("new", meta("t")).partials().unwrap().len(),
        1
    );
}

#[test]
fn a_thread_change_moves_the_transcript_in_the_index() {
    let docs = docs();
    let locator = locator(&docs);
    let history = locator.open_stem("s", meta("first")).unwrap();
    let rows = vec![message("user", "a")];
    history
        .append_turn(turn(&[], &rows, &meta("first")))
        .unwrap();
    history
        .append_turn(turn(&rows, &rows, &meta("second")))
        .unwrap();
    assert!(locator.root_for_thread("first").is_none());
    assert!(locator.root_for_thread("second").is_some());
}

#[test]
fn ids_are_unambiguous_and_bounded() {
    assert_ne!(doc_key(&["a/b", "c"]), doc_key(&["a", "b/c"]));
    let long = "x".repeat(1_000);
    let hashed = doc_key(&[&long]);
    assert!(hashed.starts_with("h:") && hashed.len() == 66, "{hashed}");
    assert!(entry_id(&long, 9).len() < tinystoragedrivers_core::MAX_ID_LEN);
    assert_eq!(entry_id("s", 7), "1:s#0000000007");
    assert!(is_subagent("parent__child"));
    assert!(!is_subagent("__child"));
    assert!(!is_subagent("plain"));
}

#[test]
fn a_long_stem_still_round_trips() {
    let docs = docs();
    let stem = "s".repeat(600);
    let history = locator(&docs).open_stem(&stem, meta("t")).unwrap();
    history.append(message("user", "a")).unwrap();
    assert_eq!(history.messages().unwrap().len(), 1);
}

#[test]
fn an_unreadable_entry_is_an_error() {
    let docs = docs();
    let locator = locator(&docs);
    let history = locator.open_stem("s", meta("t")).unwrap();
    history.append(message("user", "a")).unwrap();
    let raw = Arc::clone(&docs);
    on_bridge(async move {
        raw.put(
            ENTRIES,
            &entry_id("s", 1),
            json!({ "stem": "s", "seq": 1, "set": "not rows" }),
            Precondition::None,
        )
        .await
    });
    assert!(history.messages().is_err());
}

#[test]
fn the_index_keeps_its_creation_time() {
    let docs = docs();
    let history = locator(&docs).open_stem("s", meta("t")).unwrap();
    let rows = vec![message("user", "a")];
    history.append_turn(turn(&[], &rows, &meta("t"))).unwrap();
    let read = Arc::clone(&docs);
    let first = on_bridge(async move { read.get(INDEX, &doc_key(&["s"])).await });
    history
        .append_turn(turn(&rows, &rows, &meta("t2")))
        .unwrap();
    let read = Arc::clone(&docs);
    let second = on_bridge(async move { read.get(INDEX, &doc_key(&["s"])).await });
    let (first, second) = (first.unwrap(), second.unwrap());
    assert_eq!(first.doc["created_at"], second.doc["created_at"]);
    assert_eq!(second.doc["thread_id"], json!("t2"));
    assert!(second.version > first.version || first.version == Version(1));
}

#[test]
fn handles_describe_themselves() {
    let docs = docs();
    let locator = locator(&docs);
    assert!(format!("{locator:?}").contains("memory://test"));
    assert!(format!("{:?}", locator.handle("s", meta("t"))).contains("memory://test/s"));
    assert_eq!(
        locator.destination_key().as_deref(),
        Some("memory://test/transcripts")
    );
}

#[test]
fn a_stale_turn_is_refused_instead_of_replacing_newer_rows() {
    let docs = docs();
    let locator = locator(&docs);
    let one = locator.open_stem("s", meta("t")).unwrap();
    let two = locator.open_stem("s", meta("t")).unwrap();
    let a = vec![message("user", "a")];
    one.append_turn(turn(&[], &a, &meta("t"))).unwrap();
    let mut ab = a.clone();
    ab.push(message("assistant", "b"));
    one.append_turn(turn(&a, &ab, &meta("t"))).unwrap();
    // `two` still believes the transcript is `a`.
    let mut ac = a.clone();
    ac.push(message("assistant", "c"));
    let error = two.append_turn(turn(&a, &ac, &meta("t"))).unwrap_err();
    assert!(error.to_string().contains("stale"), "{error}");
    let contents: Vec<String> = two
        .messages()
        .unwrap()
        .into_iter()
        .map(|row| row.content)
        .collect();
    assert_eq!(contents, ["a", "b"], "the newer rows survive");
}

#[test]
fn a_turn_keeps_its_usage_and_request_id() {
    let docs = docs();
    let history = locator(&docs).open_stem("s", meta("t")).unwrap();
    let rows = vec![message("user", "q"), message("assistant", "a")];
    let usage: crate::transcript::TurnUsage = serde_json::from_value(json!({
        "provider": "p",
        "model": "m",
        "usage": { "input": 7, "output": 3, "cached_input": 0, "cost_usd": 0.01 },
    }))
    .unwrap();
    history
        .append_turn(TranscriptTurn {
            turn_usage: Some(&usage),
            request_id: Some("req-1"),
            ..turn(&[], &rows, &meta("t"))
        })
        .unwrap();
    let read = history.messages().unwrap();
    assert!(
        read.iter()
            .all(|row| row.request_id.as_deref() == Some("req-1"))
    );
    assert!(
        read[1].turn_usage.is_some(),
        "the assistant row carries the usage"
    );
    assert!(read[0].turn_usage.is_none());

    // The next turn, built from the rows as read back, extends them.
    let mut next = read.clone();
    next.push(message("user", "again"));
    history
        .append_turn(TranscriptTurn {
            request_id: Some("req-2"),
            ..turn(&read, &next, &meta("t"))
        })
        .unwrap();
    let log = entries(&docs, "s");
    assert_eq!(log[1]["extend"].as_array().unwrap().len(), 1);
    let read = history.messages().unwrap();
    assert_eq!(
        read[0].request_id.as_deref(),
        Some("req-1"),
        "old rows keep theirs"
    );
    assert_eq!(read[2].request_id.as_deref(), Some("req-2"));
}

#[test]
fn a_stale_baseline_fails_the_seal_and_frees_the_successor() {
    let docs = docs();
    let locator = locator(&docs);
    let session = SessionRef::scoped("t", "planner");
    let history = locator.open_session(&session, meta("t")).unwrap();
    let a = vec![message("user", "a")];
    history.append_turn(turn(&[], &a, &meta("t"))).unwrap();
    let mut ab = a.clone();
    ab.push(message("assistant", "b"));
    history.append_turn(turn(&a, &ab, &meta("t"))).unwrap();

    let error = locator
        .begin_generation_from_baseline(&session, meta("t"), &a)
        .err()
        .unwrap();
    assert!(error.to_string().contains("stale"), "{error}");
    let mut abc = ab.clone();
    abc.push(message("user", "c"));
    history
        .append_turn(turn(&ab, &abc, &meta("t")))
        .expect("the predecessor was not sealed");

    let (successor, _) = locator
        .begin_generation_from_baseline(&session, meta("t"), &abc)
        .expect("the released reservation is free again");
    assert_eq!(successor.generation, 1);
}

#[test]
fn a_stale_handle_never_rewinds_the_index() {
    let docs = docs();
    let locator = locator(&docs);
    let rows = vec![message("user", "a")];
    let old = locator.handle("s", meta("old"));
    old.append_turn(turn(&[], &rows, &meta("old"))).unwrap();
    let new = locator.open_stem("s", meta("new")).unwrap();
    new.append_turn(turn(&rows, &rows, &meta("new"))).unwrap();

    // `old` re-indexes from a replay that predates `new`'s entry.
    let inner = Arc::clone(&old.inner);
    on_bridge(async move {
        let mut replay = inner.replay.lock().await;
        replay.indexed = None;
        replay.next_seq = 1;
        inner.index(&mut replay).await
    });
    assert!(locator.root_for_thread("new").is_some());
    assert!(locator.root_for_thread("old").is_none());
}

#[test]
fn a_read_repairs_an_index_a_write_could_not_refresh() {
    let docs = docs();
    let locator = locator(&docs);
    let history = locator.handle("s", meta("t"));
    history.append(message("user", "a")).unwrap();
    // The index refresh after that write was lost.
    let raw = Arc::clone(&docs);
    on_bridge(async move {
        raw.delete(INDEX, &doc_key(&["s"]), Precondition::None)
            .await
    });
    assert!(locator.root_for_thread("t").is_none());
    locator
        .handle("s", meta("t"))
        .read_session()
        .unwrap()
        .unwrap();
    assert!(
        locator.root_for_thread("t").is_some(),
        "the read re-indexed it"
    );
}
