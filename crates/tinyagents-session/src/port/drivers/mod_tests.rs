use super::*;
use crate::testkit::conformance::{session_store_conformance, session_store_isolation_conformance};
use crate::turn_state::{TurnLifecycle, TurnState};
use tinystoragedrivers_core::MemoryStorage;
use tinystoragedrivers_sqlite::SqliteStorage;

fn memory() -> DriverSessionStores {
    DriverSessionStores::new(Arc::new(MemoryStorage::new())).unwrap()
}

fn sqlite(dir: &tempfile::TempDir) -> DriverSessionStores {
    let storage = SqliteStorage::open(dir.path().join("sessions.db")).unwrap();
    DriverSessionStores::new(Arc::new(storage)).unwrap()
}

#[tokio::test]
async fn memory_backed_stores_meet_the_session_store_contract() {
    session_store_conformance(&memory()).await;
}

#[tokio::test]
async fn memory_backed_stores_keep_agents_apart() {
    session_store_isolation_conformance(&memory()).await;
}

#[tokio::test]
async fn sqlite_backed_stores_meet_the_session_store_contract() {
    let dir = tempfile::tempdir().unwrap();
    session_store_conformance(&sqlite(&dir)).await;
}

#[tokio::test]
async fn sqlite_backed_stores_keep_agents_apart() {
    let dir = tempfile::tempdir().unwrap();
    session_store_isolation_conformance(&sqlite(&dir)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn stores_work_from_a_multi_threaded_runtime() {
    session_store_conformance(&memory()).await;
}

#[test]
fn stores_work_outside_any_runtime() {
    let stores = memory().for_agent("plain");
    let turn = TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z");
    stores.turn_states.put(&turn).unwrap();
    assert_eq!(stores.turn_states.get("t").unwrap(), Some(turn));
}

#[test]
fn a_valid_agent_id_is_its_own_scope() {
    assert_eq!(
        DriverSessionStores::scope_for("agent-7").as_str(),
        "agent-7"
    );
}

#[test]
fn any_other_agent_id_maps_to_a_stable_hashed_scope() {
    let spaced = DriverSessionStores::scope_for("has a space");
    assert!(spaced.as_str().starts_with("sha256:"));
    assert_eq!(spaced, DriverSessionStores::scope_for("has a space"));
    assert_ne!(spaced, DriverSessionStores::scope_for("has  a space"));
    assert!(
        DriverSessionStores::scope_for("")
            .as_str()
            .starts_with("sha256:")
    );
    let long = "a".repeat(300);
    assert!(
        DriverSessionStores::scope_for(&long)
            .as_str()
            .starts_with("sha256:")
    );
}

#[test]
fn a_raw_id_can_never_name_a_hashed_scope() {
    let hashed = DriverSessionStores::scope_for("has a space");
    let impostor = DriverSessionStores::scope_for(hashed.as_str());
    assert_ne!(
        impostor, hashed,
        "ids in the reserved prefix are hashed too"
    );
    assert!(impostor.as_str().starts_with("sha256:"));
}

#[test]
fn two_backends_never_share_a_destination() {
    let one = memory().for_agent("a");
    let two = memory().for_agent("a");
    assert_ne!(
        one.transcripts.destination_key(),
        two.transcripts.destination_key()
    );
}

#[test]
fn an_agent_gets_the_same_stores_every_time() {
    let provider = memory();
    let first = provider.for_agent("a");
    let again = provider.for_agent("a");
    assert!(Arc::ptr_eq(&first.turn_states, &again.turn_states));
    assert_eq!(
        first.transcripts.destination_key(),
        again.transcripts.destination_key()
    );
    assert_ne!(
        first.transcripts.destination_key(),
        provider.for_agent("b").transcripts.destination_key()
    );
    let debug = format!("{provider:?}");
    assert!(debug.contains("agents: 2"), "{debug}");
    assert!(debug.contains("memory"), "{debug}");
    assert!(
        provider
            .destination_key()
            .is_some_and(|key| key.starts_with("memory://"))
    );
}

#[test]
fn data_survives_a_new_provider_on_the_same_backend() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let turn = TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z");
    DriverSessionStores::new(Arc::clone(&backend))
        .unwrap()
        .for_agent("a")
        .turn_states
        .put(&turn)
        .unwrap();
    let reopened = DriverSessionStores::new(backend).unwrap().for_agent("a");
    assert_eq!(reopened.turn_states.get("t").unwrap(), Some(turn));
}

#[test]
fn recover_marks_open_agents_turns_interrupted() {
    let provider = memory();
    let stores = provider.for_agent("a");
    stores
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap();
    provider.recover().unwrap();
    assert_eq!(
        stores.turn_states.get("t").unwrap().unwrap().lifecycle,
        TurnLifecycle::Interrupted
    );
}

#[test]
fn recover_on_open_interrupts_turns_an_earlier_process_left() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let earlier = DriverSessionStores::new(Arc::clone(&backend)).unwrap();
    earlier
        .for_agent("a")
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap();

    let plain = DriverSessionStores::new(Arc::clone(&backend)).unwrap();
    assert_eq!(
        plain
            .for_agent("a")
            .turn_states
            .get("t")
            .unwrap()
            .unwrap()
            .lifecycle,
        TurnLifecycle::Started,
        "without the option nothing is touched"
    );

    let recovering = DriverSessionStores::new(backend)
        .unwrap()
        .recover_on_open(true);
    assert!(format!("{recovering:?}").contains("recover_on_open: true"));
    let stores = recovering.for_agent("a");
    assert_eq!(
        stores.turn_states.get("t").unwrap().unwrap().lifecycle,
        TurnLifecycle::Interrupted
    );

    // Only the first open recovers: a turn started afterwards stays live.
    stores
        .turn_states
        .put(&TurnState::started("t2", "r", 8, "2026-01-01T00:01:00Z"))
        .unwrap();
    recovering.agents.lock().unwrap().clear();
    assert_eq!(
        recovering
            .for_agent("a")
            .turn_states
            .get("t2")
            .unwrap()
            .unwrap()
            .lifecycle,
        TurnLifecycle::Started
    );
}

/// A backend that cannot bind any scope.
#[derive(Debug)]
struct Unbindable;

impl StorageBackend for Unbindable {
    fn driver(&self) -> &'static str {
        "unbindable"
    }

    fn capabilities(&self) -> tinystoragedrivers_core::Capabilities {
        tinystoragedrivers_core::Capabilities::default()
    }

    fn for_scope(&self, _scope: &Scope) -> Result<ScopedStorage, StorageError> {
        Err(StorageError::unavailable("database is down"))
    }

    fn database(&self, _name: &str) -> Result<Arc<dyn StorageBackend>, StorageError> {
        Err(StorageError::unavailable("database is down"))
    }
}

#[tokio::test]
async fn an_unbindable_scope_fails_closed() {
    let provider = DriverSessionStores::new(Arc::new(Unbindable)).unwrap();
    assert!(provider.try_for_agent("a").is_err());
    let stores = provider.for_agent("a");

    let error = stores
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap_err();
    assert!(error.contains("database is down"), "{error}");
    assert!(stores.turn_states.list().is_err());
    assert!(stores.kv.get("ns", "k").await.is_err());
    assert!(
        stores
            .journal
            .append("s", serde_json::json!(1))
            .await
            .is_err()
    );
    assert!(stores.journal.len("s").await.is_err());

    let session = crate::transcript::SessionRef::scoped("t", "a");
    assert!(!stores.transcripts.session_exists(&session));
    assert!(stores.transcripts.root_for_thread("t").is_none());
    assert!(stores.transcripts.latest_for_agent("a").is_none());
    let handle = stores
        .transcripts
        .open_stem("stem", transcripts::tests::meta("t"))
        .unwrap();
    assert!(handle.read_session().is_err());
    assert!(handle.messages().is_err());
    assert!(
        stores
            .transcripts
            .begin_generation(&session, transcripts::tests::meta("t"))
            .is_err()
    );

    assert!(
        provider.agents.lock().unwrap().is_empty(),
        "refusals are not cached"
    );
    provider.recover().unwrap();
}
