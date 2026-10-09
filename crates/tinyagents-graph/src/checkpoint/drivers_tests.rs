use std::time::Duration;

use super::*;
use crate::testkit::conformance::{
    checkpointer_concurrent_contract, checkpointer_contract, checkpointer_lineage_contract,
    checkpointer_writes_contract,
};
use tinystoragedrivers_core::{MemoryStorage, Scope, StorageBackend};

fn docs(storage: &MemoryStorage, scope: &str) -> Arc<dyn DocumentStore> {
    Arc::clone(
        storage
            .for_scope(&Scope::new(scope).unwrap())
            .unwrap()
            .documents(),
    )
}

fn sample(thread: &str, id: &str, _parent: Option<&str>, step: i32) -> Checkpoint<i32> {
    Checkpoint::new(step, Vec::new())
        .with_thread_id(thread.to_string())
        .with_checkpoint_id(id.to_string())
}

fn checkpointer() -> DriverCheckpointer<i32> {
    DriverCheckpointer::new(docs(&MemoryStorage::new(), "local"))
}

#[tokio::test]
async fn passes_the_checkpointer_contract() {
    checkpointer_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_writes_contract() {
    checkpointer_writes_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_lineage_contract() {
    checkpointer_lineage_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_concurrent_contract() {
    checkpointer_concurrent_contract(Arc::new(checkpointer())).await;
}

#[tokio::test]
async fn scopes_and_prefixes_keep_threads_apart() {
    let storage = MemoryStorage::new();
    let alice = DriverCheckpointer::<i32>::new(docs(&storage, "alice"));
    let bob = DriverCheckpointer::<i32>::new(docs(&storage, "bob"));
    let other = DriverCheckpointer::<i32>::with_prefix(docs(&storage, "alice"), "other");
    let checkpoint = sample("t", "c1", None, 1);
    alice.put(checkpoint).await.unwrap();
    assert!(bob.get("t", None).await.unwrap().is_none());
    assert!(other.get("t", None).await.unwrap().is_none());
    assert_eq!(alice.list_threads().await.unwrap(), vec!["t".to_owned()]);
    assert!(bob.list_threads().await.unwrap().is_empty());
    assert!(format!("{alice:?}").contains("graph_checkpoints"));
    let _ = alice.clone();
}

#[tokio::test]
async fn leases_follow_the_claim_protocol() {
    let cp = checkpointer();
    let minute = Duration::from_secs(60);
    assert!(cp.try_claim("t", "a", minute).await.unwrap());
    assert!(
        !cp.try_claim("t", "b", minute).await.unwrap(),
        "live lease is refused"
    );
    assert!(
        cp.try_claim("t", "a", minute).await.unwrap(),
        "same owner re-claims"
    );
    assert!(cp.renew("t", "a", minute).await.unwrap());
    assert!(
        !cp.renew("t", "b", minute).await.unwrap(),
        "only the owner renews"
    );
    assert!(!cp.renew("missing", "a", minute).await.unwrap());

    cp.release("t", "b").await.unwrap();
    assert!(
        !cp.try_claim("t", "b", minute).await.unwrap(),
        "foreign release is a no-op"
    );
    cp.release("t", "a").await.unwrap();
    cp.release("t", "a").await.unwrap();
    assert!(
        cp.try_claim("t", "b", minute).await.unwrap(),
        "released lease is free"
    );

    assert!(cp.try_claim("z", "dead", Duration::ZERO).await.unwrap());
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(
        !cp.renew("z", "dead", minute).await.unwrap(),
        "expired lease cannot renew"
    );
    assert!(
        cp.try_claim("z", "new", minute).await.unwrap(),
        "expired lease is reclaimable"
    );
}

#[test]
fn long_keys_are_hashed_and_tuples_never_collide() {
    assert_ne!(key(&["a/b", "c"]), key(&["a", "b/c"]));
    let long = "t".repeat(500);
    let hashed = key(&[&long]);
    assert!(hashed.starts_with("h:") && hashed.len() == 66, "{hashed}");
    assert_ne!(hashed, key(&[&"u".repeat(500)]));
    assert_ne!(
        key(&[&"t".repeat(500)]),
        key(&[&"t".repeat(499), "t"]),
        "the hash covers the length-prefixed tuple"
    );
}

#[test]
fn namespaces_encode_injectively() {
    let ns = |parts: &[&str]| parts.iter().map(|p| (*p).to_string()).collect::<Vec<_>>();
    assert_ne!(
        namespace_key(&ns(&["a", "b"])),
        namespace_key(&ns(&["a\u{1f}b"]))
    );
    assert_ne!(
        namespace_key(&ns(&["a", "b"])),
        namespace_key(&ns(&["a/b"]))
    );
    assert_ne!(namespace_key(&ns(&["ab"])), namespace_key(&ns(&["a", "b"])));
    assert_eq!(namespace_key(&[]), "");
}

#[tokio::test]
async fn scoped_reads_stay_in_their_namespace_when_ids_repeat() {
    let saver = checkpointer();
    let child = vec!["sub".to_string()];
    saver.put(sample("t", "same", None, 1)).await.unwrap();
    saver
        .put(sample("t", "same", None, 2).with_namespace(child.clone()))
        .await
        .unwrap();
    let root = saver
        .get_scoped("t", Some("same"), &[])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        root.state, 1,
        "the root never loads the subgraph's checkpoint"
    );
    let nested = saver
        .get_scoped("t", Some("same"), &child)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(nested.state, 2);
    assert_eq!(
        saver
            .get_scoped("t", None, &[])
            .await
            .unwrap()
            .unwrap()
            .state,
        1
    );
    assert!(
        saver
            .get_scoped("t", None, &["other".to_string()])
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn pending_writes_never_merge_across_lookalike_namespaces() {
    let saver = checkpointer();
    let config = |namespace: Vec<String>| CheckpointConfig {
        thread_id: "t".to_string(),
        checkpoint_id: Some("c".to_string()),
        namespace,
    };
    let split = config(vec!["a".to_string(), "b".to_string()]);
    let joined = config(vec!["a\u{1f}b".to_string()]);
    saver
        .put_writes(
            &split,
            &[PendingWrite::data("n", "task", 0, "out", json!("split"))],
        )
        .await
        .unwrap();
    assert!(saver.get_writes(&joined).await.unwrap().is_empty());
    assert_eq!(saver.get_writes(&split).await.unwrap().len(), 1);
}

#[test]
fn driver_failures_are_checkpoint_errors() {
    let error = map_error(StorageError::unavailable("busy"));
    assert!(matches!(error, TinyAgentsError::Checkpoint(ref m) if m.contains("busy")));
}

#[tokio::test]
async fn a_corrupt_record_is_a_checkpoint_error() {
    let storage = MemoryStorage::new();
    let docs = docs(&storage, "local");
    let cp = DriverCheckpointer::<i32>::new(Arc::clone(&docs));
    cp.put(sample("t", "c1", None, 1)).await.unwrap();
    let stored = cp.thread_docs("t").await.unwrap().remove(0);
    docs.put(
        "graph_checkpoints",
        &stored.id,
        json!({"thread": "t", "seq": 0, "checkpoint_id": "c1", "record": "nope"}),
        Precondition::None,
    )
    .await
    .unwrap();
    let error = cp.get("t", None).await.unwrap_err();
    assert!(matches!(error, TinyAgentsError::Checkpoint(_)), "{error:?}");
}
