use super::*;
use tinystoragedrivers_core::{MemoryStorage, Scope, StorageBackend};

fn store() -> (DriverTurnStates, Arc<dyn DocumentStore>) {
    let storage = MemoryStorage::new();
    let docs = Arc::clone(storage.for_scope(&Scope::local()).unwrap().documents());
    (
        DriverTurnStates::new(Arc::clone(&docs), Blocking::new().unwrap()),
        docs,
    )
}

fn started(thread: &str, request: &str, minute: u32) -> TurnState {
    TurnState::started(thread, request, 8, format!("2026-01-01T00:{minute:02}:00Z"))
}

fn completed(thread: &str, request: &str, minute: u32) -> TurnState {
    let mut turn = started(thread, request, minute);
    turn.lifecycle = TurnLifecycle::Completed;
    turn.updated_at = turn.started_at.clone();
    turn
}

#[test]
fn completed_turns_are_kept_up_to_the_retention_limit() {
    let (turns, _) = store();
    for minute in 0..(COMPLETED_RETENTION as u32 + 3) {
        turns
            .put(&completed("t", &format!("r{minute:02}"), minute))
            .unwrap();
    }
    turns.put(&started("t", "live", 59)).unwrap();
    let kept = turns.list_thread("t").unwrap();
    assert_eq!(kept.len(), COMPLETED_RETENTION + 1);
    assert!(
        turns.get_turn("t", "r00").unwrap().is_none(),
        "the oldest completed turns go first"
    );
    assert!(
        turns.get_turn("t", "live").unwrap().is_some(),
        "live turns stay"
    );
}

#[test]
fn settling_prunes_too() {
    let (turns, _) = store();
    for minute in 0..COMPLETED_RETENTION as u32 {
        turns
            .put(&completed("t", &format!("r{minute:02}"), minute))
            .unwrap();
    }
    turns.put(&started("t", "last", 58)).unwrap();
    assert!(
        turns
            .settle_turn(
                "t",
                "last",
                TurnLifecycle::Completed,
                "2026-01-01T01:00:00Z"
            )
            .unwrap()
    );
    assert_eq!(turns.list_thread("t").unwrap().len(), COMPLETED_RETENTION);
}

#[test]
fn put_unless_completed_inserts_updates_and_then_holds() {
    let (turns, _) = store();
    let mut turn = started("t", "r", 0);
    assert!(
        turns.put_unless_completed(&turn).unwrap(),
        "a new turn is written"
    );
    turn.lifecycle = TurnLifecycle::Completed;
    assert!(
        turns.put_unless_completed(&turn).unwrap(),
        "a live turn is updated"
    );
    assert!(
        !turns.put_unless_completed(&started("t", "r", 1)).unwrap(),
        "a completed turn is never overwritten conditionally"
    );
    assert_eq!(
        turns.get_turn("t", "r").unwrap().unwrap().lifecycle,
        TurnLifecycle::Completed
    );
}

#[test]
fn settling_needs_a_live_turn() {
    let (turns, _) = store();
    let now = "2026-01-01T01:00:00Z";
    assert!(
        !turns
            .settle_turn("t", "missing", TurnLifecycle::Interrupted, now)
            .unwrap()
    );
    turns.put(&completed("t", "done", 0)).unwrap();
    assert!(
        !turns
            .settle_turn("t", "done", TurnLifecycle::Interrupted, now)
            .unwrap()
    );
    let mut live = started("t", "live", 1);
    live.phase = Some(crate::turn_state::TurnPhase::Thinking);
    turns.put(&live).unwrap();
    assert!(
        turns
            .settle_turn("t", "live", TurnLifecycle::Interrupted, now)
            .unwrap()
    );
    let settled = turns.get_turn("t", "live").unwrap().unwrap();
    assert_eq!(settled.lifecycle, TurnLifecycle::Interrupted);
    assert_eq!(settled.phase, None);
    assert_eq!(settled.updated_at, now);
}

#[test]
fn deletes_report_whether_anything_went() {
    let (turns, _) = store();
    assert!(!turns.delete("t").unwrap());
    assert!(!turns.delete_turn("t", "r").unwrap());
    turns.put(&started("t", "r1", 0)).unwrap();
    turns.put(&started("t", "r2", 1)).unwrap();
    turns.put(&started("u", "r1", 2)).unwrap();
    assert!(turns.delete("t").unwrap());
    assert!(turns.list_thread("t").unwrap().is_empty());
    assert_eq!(turns.list().unwrap().len(), 1);
    assert_eq!(turns.clear_all().unwrap(), 1);
    assert_eq!(turns.clear_all().unwrap(), 0);
}

#[test]
fn list_answers_each_threads_newest_turn() {
    let (turns, _) = store();
    turns.put(&started("t", "old", 0)).unwrap();
    turns.put(&started("t", "new", 5)).unwrap();
    turns.put(&started("u", "only", 1)).unwrap();
    let mut latest: Vec<(String, String)> = turns
        .list()
        .unwrap()
        .into_iter()
        .map(|turn| (turn.thread_id, turn.request_id))
        .collect();
    latest.sort();
    assert_eq!(
        latest,
        [
            ("t".to_string(), "new".to_string()),
            ("u".to_string(), "only".to_string())
        ]
    );
}

#[test]
fn ids_with_separators_do_not_collide() {
    let (turns, _) = store();
    turns.put(&started("a/b", "c", 0)).unwrap();
    turns.put(&started("a", "b/c", 1)).unwrap();
    assert_eq!(turns.list().unwrap().len(), 2);
}

#[test]
fn an_unreadable_snapshot_is_an_error() {
    let (turns, docs) = store();
    turns.put(&started("t", "r", 0)).unwrap();
    let bridge = Blocking::new().unwrap();
    let raw = Arc::clone(&docs);
    bridge
        .run(async move {
            raw.put(
                COLLECTION,
                &id("t", "bodiless"),
                json!({"thread_id": "t"}),
                Precondition::None,
            )
            .await
        })
        .unwrap()
        .unwrap();
    let error = turns.list_thread("t").unwrap_err();
    assert!(error.contains("no body"), "{error}");
    let raw = Arc::clone(&docs);
    bridge
        .run(async move {
            raw.put(
                COLLECTION,
                &id("t", "bodiless"),
                json!({"thread_id": "t", "turn": "nope"}),
                Precondition::None,
            )
            .await
        })
        .unwrap()
        .unwrap();
    let error = turns.get_turn("t", "bodiless").unwrap_err();
    assert!(error.contains("unreadable"), "{error}");
}

#[test]
fn contention_is_reported_after_the_last_attempt() {
    let error = contended("t", "r");
    assert_eq!(error.kind(), ErrorKind::Conflict);
    assert!(is_race(&error));
    assert!(!is_race(&StorageError::unavailable("down")));
}

#[test]
fn concurrent_settles_and_sweeps_never_lose_a_turn() {
    let (turns, _) = store();
    for i in 0..8 {
        turns.put(&started("t", &format!("r{i}"), i)).unwrap();
    }
    let handles: Vec<_> = (0..4)
        .map(|worker| {
            let turns = turns.clone();
            std::thread::spawn(move || {
                if worker == 0 {
                    turns.mark_all_interrupted("2026-01-01T02:00:00Z").unwrap();
                } else {
                    for i in 0..8 {
                        turns
                            .settle_turn(
                                "t",
                                &format!("r{i}"),
                                TurnLifecycle::Completed,
                                "2026-01-01T02:00:00Z",
                            )
                            .unwrap();
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let all = turns.list_thread("t").unwrap();
    assert_eq!(all.len(), 8);
    assert!(all.iter().all(is_terminal), "every turn ended up terminal");
}
