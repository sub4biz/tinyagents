use super::*;
use crate::store::conformance::run_store_conformance;
use tinystoragedrivers_core::{MemoryStorage, Scope, StorageBackend};

fn scoped(storage: &MemoryStorage, scope: &str) -> tinystoragedrivers_core::ScopedStorage {
    storage.for_scope(&Scope::new(scope).unwrap()).unwrap()
}

#[tokio::test]
async fn driver_store_passes_the_store_conformance_suite() {
    let storage = MemoryStorage::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    run_store_conformance(&store).await;
}

#[tokio::test]
async fn namespaces_and_keys_may_hold_any_characters() {
    let storage = MemoryStorage::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    store.put("a/b", "c", json!(1)).await.unwrap();
    store.put("a", "b/c", json!(2)).await.unwrap();
    store.put("odd ns:é", "k y", json!(3)).await.unwrap();
    assert_eq!(store.get("a/b", "c").await.unwrap(), Some(json!(1)));
    assert_eq!(store.get("a", "b/c").await.unwrap(), Some(json!(2)));
    assert_eq!(store.get("odd ns:é", "k y").await.unwrap(), Some(json!(3)));
    assert_eq!(store.list("a").await.unwrap(), vec!["b/c".to_owned()]);
}

#[tokio::test]
async fn stores_in_different_scopes_and_collections_are_isolated() {
    let storage = MemoryStorage::new();
    let alice = DriverStore::new(Arc::clone(scoped(&storage, "alice").documents()));
    let bob = DriverStore::new(Arc::clone(scoped(&storage, "bob").documents()));
    let other = DriverStore::with_collection(
        Arc::clone(scoped(&storage, "alice").documents()),
        "other_store",
    );
    alice.put("ns", "k", json!("a")).await.unwrap();
    assert_eq!(bob.get("ns", "k").await.unwrap(), None);
    assert_eq!(other.get("ns", "k").await.unwrap(), None);
    assert!(bob.list("ns").await.unwrap().is_empty());
}

#[tokio::test]
async fn a_list_follows_every_page() {
    let storage = MemoryStorage::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    for i in 0..1_005 {
        store
            .put("big", &format!("k{i:04}"), json!(i))
            .await
            .unwrap();
    }
    assert_eq!(store.list("big").await.unwrap().len(), 1_005);
}

#[tokio::test]
async fn an_unstorable_key_is_a_validation_error() {
    let storage = MemoryStorage::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    let long = "k".repeat(600);
    let error = store.put("ns", &long, json!(1)).await.unwrap_err();
    assert!(matches!(error, TinyAgentsError::Validation(_)), "{error:?}");
    let error = DriverStore::with_collection(
        Arc::clone(scoped(&storage, "local").documents()),
        "bad name",
    )
    .get("ns", "k")
    .await
    .unwrap_err();
    assert!(matches!(error, TinyAgentsError::Validation(_)), "{error:?}");
}

#[test]
fn backend_failures_map_to_storage_errors() {
    let error = map_error(StorageError::unavailable("busy"));
    assert!(matches!(error, TinyAgentsError::Storage(ref m) if m.contains("busy")));
}

#[tokio::test]
async fn driver_append_store_keeps_dense_offsets() {
    let storage = MemoryStorage::new();
    let journal = DriverAppendStore::new(Arc::clone(scoped(&storage, "local").streams()));
    assert_eq!(journal.len("run-1").await.unwrap(), 0);
    assert!(journal.read_from("run-1", 0).await.unwrap().is_empty());
    for i in 0..3 {
        assert_eq!(journal.append("run-1", json!({ "i": i })).await.unwrap(), i);
    }
    assert_eq!(journal.len("run-1").await.unwrap(), 3);
    assert_eq!(
        journal.read_from("run-1", 1).await.unwrap(),
        vec![(1, json!({ "i": 1 })), (2, json!({ "i": 2 }))]
    );
    assert_eq!(
        journal.read_window("run-1", 0, 2).await.unwrap(),
        vec![(0, json!({ "i": 0 })), (1, json!({ "i": 1 }))]
    );
    assert!(journal.read_from("run-1", 9).await.unwrap().is_empty());
}

#[tokio::test]
async fn read_from_follows_long_streams_and_prefixes_isolate() {
    let storage = MemoryStorage::new();
    let streams = Arc::clone(scoped(&storage, "local").streams());
    let journal = DriverAppendStore::with_prefix(Arc::clone(&streams), "journal/");
    let other = DriverAppendStore::with_prefix(Arc::clone(&streams), "other/");
    for i in 0..(READ_PAGE as u64 + 5) {
        journal.append("s", json!(i)).await.unwrap();
    }
    let all = journal.read_from("s", 0).await.unwrap();
    assert_eq!(all.len(), READ_PAGE + 5);
    assert_eq!(all.last().unwrap().0, READ_PAGE as u64 + 4);
    assert_eq!(other.len("s").await.unwrap(), 0);
    let error = journal.append("", json!(1)).await;
    assert!(error.is_ok(), "a prefix makes the empty stream name valid");
    let ab = DriverAppendStore::with_prefix(Arc::clone(&streams), "ab");
    let a = DriverAppendStore::with_prefix(Arc::clone(&streams), "a");
    ab.append("c", json!("ab/c")).await.unwrap();
    assert_eq!(a.len("bc").await.unwrap(), 0, "prefixes never overlap");
    let bare = DriverAppendStore::new(Arc::clone(scoped(&storage, "local").streams()));
    bare.append("", json!("bare")).await.unwrap();
    assert_eq!(bare.len("").await.unwrap(), 1, "a bare name is encoded too");
    // A bare stream spelled like a prefixed one stays its own stream.
    bare.append("1:ab", json!("bare")).await.unwrap();
    assert_eq!(a.len("b").await.unwrap(), 0);
    assert_eq!(a.len("").await.unwrap(), 0);
}
