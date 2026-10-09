//! Tests for [`FileStatusStore`], run-id minting and the env redaction seed.

use std::time::{Duration, UNIX_EPOCH};

use super::*;
use crate::events::{AgentEvent, EventSink};
use crate::ids::{ComponentId, HarnessPhase, ThreadId};
use crate::observability::{
    AgentObservation, FanOutSink, HarnessEventJournal, JournalSink, RedactingSink,
    StoreEventJournal,
};
use crate::store::{JsonlAppendStore, Store};
use std::sync::Arc;

fn tmp_root(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ta-file-status-{tag}-{}", uuid::Uuid::new_v4()))
}

#[test]
fn mint_run_id_is_slash_free_and_unique() {
    let a = mint_run_id();
    let b = mint_run_id();
    assert_ne!(a.as_str(), b.as_str());
    let hex = a.as_str().strip_prefix("run.").expect("run. prefix");
    assert_eq!(hex.len(), 32);
    assert!(hex.bytes().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn secrets_from_vars_masks_only_long_secretish_values() {
    let secrets = secrets_from_vars([
        ("OPENAI_API_KEY".to_string(), "sk-0123456789".to_string()),
        ("my_bearer".to_string(), "  abcdefgh  ".to_string()),
        ("SHORT_TOKEN".to_string(), "short".to_string()),
        ("HOME".to_string(), "/home/someone-long-path".to_string()),
    ]);
    assert_eq!(
        secrets,
        vec!["sk-0123456789".to_string(), "  abcdefgh  ".to_string()]
    );
}

/// On-disk format contract: one pretty JSON file per run at
/// `<kv>/run_status/<run_id>.json`, decoded back through the store.
#[tokio::test]
async fn on_disk_record_matches_literal_fixture() {
    let root = tmp_root("fixture");
    let store = FileStatusStore::new(FileStore::new(&root));
    let mut status = HarnessRunStatus::new(
        RunId::new("run.fixture"),
        ComponentId::new("mock-model".to_string()),
    )
    .with_thread(ThreadId::new("thread-1"));
    status.started_at = UNIX_EPOCH + Duration::from_secs(100);
    status.updated_at = UNIX_EPOCH + Duration::from_secs(100);
    store.put_status(status).await.unwrap();

    let raw = std::fs::read(root.join("run_status").join("run.fixture.json")).unwrap();
    let on_disk: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let expected: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(on_disk, expected);

    // And a record in that exact shape reads back through the store.
    let back = store.get_status("run.fixture").await.unwrap().unwrap();
    assert_eq!(back.thread_id.as_ref().unwrap().as_str(), "thread-1");
    assert_eq!(back.status, ExecutionStatus::Pending);
    let _ = std::fs::remove_dir_all(&root);
}

const FIXTURE: &str = r#"{"active_tool_calls":[],"component":"mock-model","cost":{"cache_cost":0.0,"input_cost":0.0,"output_cost":0.0,"reasoning_cost":0.0,"total_cost":0.0},"current_phase":"idle","metadata":null,"model_calls":0,"root_run_id":"run.fixture","run_id":"run.fixture","started_at":100,"status":"pending","thread_id":"thread-1","tool_calls":0,"updated_at":100,"usage":{"calls":0,"usage":{"cache_creation_tokens":0,"cache_read_tokens":0,"input_tokens":0,"output_tokens":0,"reasoning_tokens":0,"total_tokens":0}}}"#;

#[tokio::test]
async fn status_store_round_trips_and_answers_queries() {
    let root = tmp_root("queries");
    let store = FileStatusStore::new(FileStore::new(&root));
    let run_id = mint_run_id();
    let mut status =
        HarnessRunStatus::new(run_id.clone(), ComponentId::new("mock-model".to_string()))
            .with_thread(ThreadId::new("thread-42"));
    status.mark_running(HarnessPhase::Model);
    store.put_status(status.clone()).await.unwrap();

    let active = store.list_active().await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].status, ExecutionStatus::Running);
    assert_eq!(store.list_by_thread("thread-42").await.unwrap().len(), 1);
    assert!(store.list_by_thread("nope").await.unwrap().is_empty());

    status.mark_completed();
    store.put_status(status).await.unwrap();
    let by_root = store.list_by_root(run_id.as_str()).await.unwrap();
    assert_eq!(by_root.len(), 1);
    assert_eq!(by_root[0].status, ExecutionStatus::Completed);
    assert!(store.list_active().await.unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn undecodable_record_is_skipped() {
    let root = tmp_root("corrupt");
    let store = FileStatusStore::new(FileStore::new(&root));
    std::fs::create_dir_all(root.join("run_status")).unwrap();
    std::fs::write(root.join("run_status").join("bad.json"), "{\"nope\":1}").unwrap();
    assert!(store.list_active().await.unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn malformed_json_record_does_not_hide_valid_runs() {
    let root = tmp_root("malformed");
    let store = FileStatusStore::new(FileStore::new(&root));
    let mut status = HarnessRunStatus::new(mint_run_id(), ComponentId::new("m".to_string()));
    status.mark_running(HarnessPhase::Model);
    store.put_status(status).await.unwrap();
    std::fs::write(root.join("run_status").join("broken.json"), "{not json").unwrap();
    assert_eq!(store.list_active().await.unwrap().len(), 1);
    let _ = std::fs::remove_dir_all(&root);
}

async fn read_all(root: &std::path::Path, run_id: &str, from: u64) -> Vec<AgentObservation> {
    StoreEventJournal::new(JsonlAppendStore::new(root))
        .read_from(run_id, from)
        .await
        .unwrap()
}

/// Journal persists, masks secrets and replays with stable ids.
#[tokio::test]
async fn journal_persists_masks_and_replays_run() {
    let root = tmp_root("journal");
    let run_id = mint_run_id();
    let journal: Arc<dyn HarnessEventJournal> =
        Arc::new(StoreEventJournal::new(JsonlAppendStore::new(&root)));
    let sink = EventSink::with_stream_id(run_id.as_str());
    let journal_sink = Arc::new(JournalSink::new(journal, run_id.clone()));
    let redacting = RedactingSink::new(journal_sink.clone(), vec!["sk-super-secret".into()]);
    sink.subscribe(Arc::new(FanOutSink::new().with(Arc::new(redacting))));

    sink.emit(AgentEvent::ModelStarted {
        call_id: "c1".into(),
        model: "sk-super-secret leaked here".to_string(),
    });
    sink.emit(AgentEvent::ToolStarted {
        parent_call_id: None,
        call_id: "c1".into(),
        tool_name: "echo".to_string(),
        input: None,
    });
    journal_sink.flush();

    let replayed = read_all(&root, run_id.as_str(), 0).await;
    assert_eq!(replayed.len(), 2);
    for (offset, obs) in replayed.iter().enumerate() {
        assert_eq!(obs.offset, offset as u64);
        assert_eq!(
            obs.event_id.as_str(),
            format!("{}-evt-{offset}", run_id.as_str())
        );
    }
    if let AgentEvent::ModelStarted { model, .. } = &replayed[0].event {
        assert!(!model.contains("sk-super-secret"));
        assert!(model.contains("[REDACTED]"));
    } else {
        panic!("expected ModelStarted first");
    }
    let tail = read_all(&root, run_id.as_str(), 1).await;
    assert_eq!(tail.len(), 1);
    assert!(matches!(tail[0].event, AgentEvent::ToolStarted { .. }));
    let _ = std::fs::remove_dir_all(&root);
}

/// Multi-byte UTF-8 straddling the 4096-byte lookback window (#5599).
#[tokio::test]
async fn journal_sink_handles_multibyte_utf8_spanning_window_boundary() {
    const WINDOW: usize = 4096;
    let mut torn = Vec::new();
    for pad in 0..20usize {
        let root = tmp_root(&format!("utf8-{pad}"));
        let run_id = mint_run_id();
        let journal: Arc<dyn HarnessEventJournal> =
            Arc::new(StoreEventJournal::new(JsonlAppendStore::new(&root)));
        let sink = EventSink::with_stream_id(run_id.as_str());
        let journal_sink = Arc::new(JournalSink::new(journal, run_id.clone()));
        sink.subscribe(Arc::new(FanOutSink::new().with(journal_sink.clone())));
        for i in 0..10 {
            sink.emit(AgentEvent::ModelStarted {
                call_id: format!("call-{i}").into(),
                model: "🦀".repeat(50),
            });
        }
        sink.emit(AgentEvent::ModelStarted {
            call_id: "call-10".into(),
            model: "x".repeat(pad),
        });
        journal_sink.flush();

        let raw = std::fs::read(root.join(format!("{}.jsonl", run_id.as_str())))
            .expect("read stream file");
        if raw.len() > WINDOW {
            let start = raw.len() - WINDOW;
            if !String::from_utf8_lossy(&raw).is_char_boundary(start) {
                torn.push(pad);
            }
        }
        sink.emit(AgentEvent::ToolStarted {
            parent_call_id: None,
            call_id: "call-11".into(),
            tool_name: "test_tool".to_string(),
            input: None,
        });
        journal_sink.flush();

        let replayed = read_all(&root, run_id.as_str(), 0).await;
        assert_eq!(replayed.len(), 12, "pad {pad}");
        for (i, obs) in replayed.iter().enumerate() {
            assert_eq!(obs.offset, i as u64, "pad {pad}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
    assert!(
        !torn.is_empty(),
        "no pad induced a torn multi-byte boundary"
    );
}

#[tokio::test]
async fn a_status_store_over_any_store_reads_back_what_it_wrote() {
    // A host's own store, injected as a trait object rather than a FileStore.
    let kv: Arc<dyn Store> = Arc::new(crate::store::InMemoryStore::new());
    let store = FileStatusStore::over(kv.clone());
    let run_id = RunId::new("run.host-store");
    let mut status = HarnessRunStatus::new(run_id.clone(), ComponentId::new("model"))
        .with_thread(ThreadId::new("thread-1"));
    status.mark_running(HarnessPhase::Model);
    store.put_status(status).await.unwrap();
    let read = store
        .get_status(run_id.as_str())
        .await
        .unwrap()
        .expect("written");
    assert_eq!(read.run_id.as_str(), "run.host-store");
    // The record lives in the injected store, keyed as on disk.
    assert_eq!(kv.list(STATUS_NS).await.unwrap().len(), 1);
}

#[tokio::test]
async fn unsafe_run_ids_read_and_migrate_legacy_raw_keys() {
    let kv: Arc<dyn Store> = Arc::new(crate::store::InMemoryStore::new());
    let store = FileStatusStore::over(kv.clone());
    let run_id = RunId::new("legacy/run");
    let status = HarnessRunStatus::new(run_id.clone(), ComponentId::new("model"));
    kv.put(
        STATUS_NS,
        run_id.as_str(),
        serde_json::to_value(&status).unwrap(),
    )
    .await
    .unwrap();

    assert_eq!(
        store
            .get_status(run_id.as_str())
            .await
            .unwrap()
            .map(|status| status.run_id),
        Some(run_id.clone())
    );
    store.put_status(status).await.unwrap();
    assert!(kv.get(STATUS_NS, run_id.as_str()).await.unwrap().is_none());
    assert!(
        kv.get(STATUS_NS, &status_key(run_id.as_str()))
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn a_journal_over_a_shared_append_store_handle_is_a_journal() {
    let shared: Arc<dyn crate::store::AppendStore> =
        Arc::new(crate::store::InMemoryAppendStore::new());
    let journal = StoreEventJournal::new(shared.clone());
    assert!(journal.read_from("run.x", 0).await.unwrap().is_empty());
    shared.append("s", serde_json::json!(1)).await.unwrap();
    assert_eq!(shared.read_window("s", 0, 5).await.unwrap().len(), 1);
    assert_eq!(shared.len("s").await.unwrap(), 1);
}
