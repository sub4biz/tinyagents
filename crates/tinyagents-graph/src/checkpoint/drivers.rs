//! A [`Checkpointer`] over `tinystoragedrivers` document ports.
//!
//! [`DriverCheckpointer`] puts graph checkpoints, their pending writes and the
//! per-thread execution lease on whatever backend the host opened (SQLite on a
//! desktop, MongoDB in the cloud, memory in tests). The document handle is
//! already bound to a tenant scope, so two tenants' threads never meet even
//! when they share a thread id.
//!
//! # Layout
//!
//! Three collections, named from a prefix (default `graph`):
//!
//! - `<prefix>_checkpoints`: one document per stored checkpoint,
//!   `{thread, namespace, seq, checkpoint_id, record}` (`namespace` is the
//!   encoded subgraph namespace, so scoped reads filter on it). `seq` is a per-thread insertion
//!   counter, so listing is a query sorted on it and duplicate checkpoint ids
//!   resolve to the latest write, exactly like the append-only backends.
//! - `<prefix>_threads`: one counter document per thread, advanced with a
//!   compare-and-swap so concurrent writers never share a `seq`.
//! - `<prefix>_writes`: one document per `(thread, namespace, checkpoint)`
//!   holding the merged pending writes (see [`merge_writes`]).
//! - `<prefix>_leases`: one document per leased thread, claimed and renewed
//!   with compare-and-swap.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinystoragedrivers_core::{
    CollectionSpec, DocumentStore, DocumentStoreExt, ErrorKind, Filter, IndexSpec, Precondition,
    Query, Sort, StorageError, Versioned,
};
use tokio::sync::OnceCell;

use super::{
    Checkpoint, CheckpointConfig, CheckpointId, CheckpointMetadata, Checkpointer, PendingWrite,
    decode_json_err, merge_writes, require_checkpoint_id,
};

/// How many times a compare-and-swap loop retries before giving up.
const CAS_ATTEMPTS: usize = 64;

/// Map a storage driver failure onto the graph's checkpoint error.
fn map_error(error: StorageError) -> TinyAgentsError {
    TinyAgentsError::Checkpoint(format!("storage driver: {error}"))
}

/// Longest id stored as is; longer ones are hashed.
const MAX_KEY_LEN: usize = 400;

/// A document id for `parts`: length-prefixed so no two tuples collide, and
/// replaced by its SHA-256 when it would exceed the driver's id limit.
fn key(parts: &[&str]) -> String {
    let joined: String = parts
        .iter()
        .map(|part| format!("{}:{part}", part.len()))
        .collect::<Vec<_>>()
        .join("/");
    if joined.len() <= MAX_KEY_LEN {
        joined
    } else {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(joined.as_bytes());
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        format!("h:{hex}")
    }
}

/// `namespace` as one string, injectively: each component is
/// length-prefixed, so `["a", "b"]` and `["a/b"]` never meet.
fn namespace_key(namespace: &[String]) -> String {
    namespace
        .iter()
        .map(|part| format!("{}:{part};", part.len()))
        .collect()
}

/// A [`Checkpointer`] that stores everything in a driver [`DocumentStore`].
pub struct DriverCheckpointer<State> {
    docs: Arc<dyn DocumentStore>,
    checkpoints: String,
    threads: String,
    writes: String,
    leases: String,
    declared: Arc<OnceCell<()>>,
    _state: PhantomData<fn() -> State>,
}

impl<State> Clone for DriverCheckpointer<State> {
    fn clone(&self) -> Self {
        Self {
            docs: Arc::clone(&self.docs),
            checkpoints: self.checkpoints.clone(),
            threads: self.threads.clone(),
            writes: self.writes.clone(),
            leases: self.leases.clone(),
            declared: Arc::clone(&self.declared),
            _state: PhantomData,
        }
    }
}

impl<State> std::fmt::Debug for DriverCheckpointer<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverCheckpointer")
            .field("checkpoints", &self.checkpoints)
            .finish_non_exhaustive()
    }
}

impl<State> DriverCheckpointer<State> {
    /// Store checkpoints in the `graph_*` collections of `docs`.
    pub fn new(docs: Arc<dyn DocumentStore>) -> Self {
        Self::with_prefix(docs, "graph")
    }

    /// Store checkpoints in the `<prefix>_*` collections of `docs`, so
    /// several independent checkpointers can share one backend.
    pub fn with_prefix(docs: Arc<dyn DocumentStore>, prefix: &str) -> Self {
        Self {
            docs,
            checkpoints: format!("{prefix}_checkpoints"),
            threads: format!("{prefix}_threads"),
            writes: format!("{prefix}_writes"),
            leases: format!("{prefix}_leases"),
            declared: Arc::new(OnceCell::new()),
            _state: PhantomData,
        }
    }

    async fn declared(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                let specs = [
                    CollectionSpec::new(&self.checkpoints)
                        .index(IndexSpec::new("by_thread_seq", ["thread", "seq"]))
                        .index(IndexSpec::new("by_thread_id", ["thread", "checkpoint_id"]))
                        .index(IndexSpec::new(
                            "by_thread_namespace",
                            ["thread", "namespace", "seq"],
                        )),
                    CollectionSpec::new(&self.threads),
                    CollectionSpec::new(&self.writes)
                        .index(IndexSpec::new("by_thread", ["thread"])),
                    CollectionSpec::new(&self.leases),
                ];
                for spec in &specs {
                    self.docs.ensure_collection(spec).await.map_err(map_error)?;
                }
                Ok::<(), TinyAgentsError>(())
            })
            .await
            .map(|_| ())
    }

    /// Reserve the next insertion sequence number for `thread`.
    async fn next_seq(&self, thread: &str) -> Result<u64> {
        let id = key(&[thread]);
        for _ in 0..CAS_ATTEMPTS {
            let current = self.docs.get(&self.threads, &id).await.map_err(map_error)?;
            let (seq, precondition) = match &current {
                Some(found) => (
                    found.doc.get("next").and_then(Value::as_u64).unwrap_or(0),
                    found.unchanged(),
                ),
                None => (0, Precondition::Absent),
            };
            let doc = json!({ "thread": thread, "next": seq + 1 });
            match self.docs.put(&self.threads, &id, doc, precondition).await {
                Ok(_) => return Ok(seq),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(map_error(error)),
            }
        }
        Err(TinyAgentsError::Checkpoint(format!(
            "could not reserve a checkpoint sequence for thread `{thread}`"
        )))
    }

    /// Every stored checkpoint document of `thread`, in insertion order.
    async fn thread_docs(&self, thread: &str) -> Result<Vec<Versioned<Value>>> {
        self.declared().await?;
        let query = Query::filter(Filter::eq("thread", thread)).sort(Sort::asc("seq"));
        self.docs
            .query_all(&self.checkpoints, &query)
            .await
            .map_err(map_error)
    }

    fn writes_id(thread: &str, namespace: &[String], checkpoint_id: &str) -> String {
        key(&[thread, &namespace_key(namespace), checkpoint_id])
    }

    async fn drop_writes(&self, filter: Filter) -> Result<()> {
        self.docs
            .delete_where(&self.writes, &filter)
            .await
            .map(|_| ())
            .map_err(map_error)
    }
}

impl<State> DriverCheckpointer<State>
where
    State: DeserializeOwned,
{
    fn decode(stored: Versioned<Value>) -> Result<Checkpoint<State>> {
        let record = stored.doc.get("record").cloned().unwrap_or(Value::Null);
        // Through the shared classifier, so a `State` that no longer decodes
        // is tagged `[schema]` exactly as the file and SQLite backends tag it
        // (durable delegations prune such a checkpoint and start fresh).
        let mut checkpoint: Checkpoint<State> = serde_json::from_value(record)
            .map_err(|error| decode_json_err("storage driver checkpointer", "record", error))?;
        checkpoint.normalize();
        Ok(checkpoint)
    }
}

#[async_trait]
impl<State> Checkpointer<State> for DriverCheckpointer<State>
where
    State: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId> {
        self.declared().await?;
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());
        let seq = self.next_seq(&checkpoint.thread_id).await?;
        let doc = json!({
            "thread": checkpoint.thread_id,
            "namespace": namespace_key(&checkpoint.namespace),
            "seq": seq,
            "checkpoint_id": checkpoint.checkpoint_id,
            "record": serde_json::to_value(&checkpoint)?,
        });
        let doc_id = key(&[&checkpoint.thread_id, &format!("{seq:020}")]);
        self.docs
            .put(&self.checkpoints, &doc_id, doc, Precondition::Absent)
            .await
            .map_err(map_error)?;
        Ok(id)
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint<State>>> {
        self.declared().await?;
        let mut filter = Filter::eq("thread", thread_id);
        if let Some(id) = checkpoint_id {
            filter = filter.and(Filter::eq("checkpoint_id", id));
        }
        let query = Query::filter(filter).sort(Sort::desc("seq")).limit(1);
        let page = self
            .docs
            .query(&self.checkpoints, &query)
            .await
            .map_err(map_error)?;
        page.items.into_iter().next().map(Self::decode).transpose()
    }

    /// One indexed query on the stored namespace, so a parent run and a
    /// subgraph sharing a thread never load each other's checkpoints, even
    /// when they reuse a checkpoint id.
    async fn get_scoped(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        namespace: &[String],
    ) -> Result<Option<Checkpoint<State>>> {
        self.declared().await?;
        let mut filter =
            Filter::eq("thread", thread_id).and(Filter::eq("namespace", namespace_key(namespace)));
        if let Some(id) = checkpoint_id {
            filter = filter.and(Filter::eq("checkpoint_id", id));
        }
        let query = Query::filter(filter).sort(Sort::desc("seq")).limit(1);
        let page = self
            .docs
            .query(&self.checkpoints, &query)
            .await
            .map_err(map_error)?;
        page.items.into_iter().next().map(Self::decode).transpose()
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        Ok(self
            .get_thread(thread_id)
            .await?
            .iter()
            .map(Checkpoint::to_metadata)
            .collect())
    }

    async fn get_thread(&self, thread_id: &str) -> Result<Vec<Checkpoint<State>>> {
        self.thread_docs(thread_id)
            .await?
            .into_iter()
            .map(Self::decode)
            .collect()
    }

    async fn list_threads(&self) -> Result<Vec<String>> {
        self.declared().await?;
        let counters = self
            .docs
            .query_all(&self.threads, &Query::all())
            .await
            .map_err(map_error)?;
        let mut threads = Vec::new();
        for counter in counters {
            let Some(thread) = counter.doc.get("thread").and_then(Value::as_str) else {
                continue;
            };
            let live = self
                .docs
                .count(&self.checkpoints, &Filter::eq("thread", thread))
                .await
                .map_err(map_error)?;
            if live > 0 {
                threads.push(thread.to_owned());
            }
        }
        Ok(threads)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.declared().await?;
        self.docs
            .delete_where(&self.checkpoints, &Filter::eq("thread", thread_id))
            .await
            .map_err(map_error)?;
        // Keep the sequence counter: a later thread of the same name continues
        // after it, so its records never interleave with stale cursors.
        self.drop_writes(Filter::eq("thread", thread_id)).await
    }

    async fn delete_checkpoints(&self, thread_id: &str, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        self.declared().await?;
        let filter = Filter::eq("thread", thread_id).and(Filter::one_of(
            "checkpoint_id",
            ids.iter().map(String::as_str),
        ));
        let removed = self
            .docs
            .delete_where(&self.checkpoints, &filter)
            .await
            .map_err(map_error)?;
        self.drop_writes(Filter::eq("thread", thread_id).and(Filter::one_of(
            "checkpoint_id",
            ids.iter().map(String::as_str),
        )))
        .await?;
        Ok(usize::try_from(removed).unwrap_or(usize::MAX))
    }

    async fn put_writes(&self, config: &CheckpointConfig, writes: &[PendingWrite]) -> Result<()> {
        let checkpoint_id = require_checkpoint_id(config)?;
        if writes.is_empty() {
            return Ok(());
        }
        self.declared().await?;
        let id = Self::writes_id(&config.thread_id, &config.namespace, &checkpoint_id);
        for _ in 0..CAS_ATTEMPTS {
            let current = self.docs.get(&self.writes, &id).await.map_err(map_error)?;
            let (mut stored, precondition): (Vec<PendingWrite>, Precondition) = match &current {
                Some(found) => (
                    serde_json::from_value(
                        found.doc.get("writes").cloned().unwrap_or(Value::Null),
                    )?,
                    found.unchanged(),
                ),
                None => (Vec::new(), Precondition::Absent),
            };
            merge_writes(&mut stored, writes);
            let doc = json!({
                "thread": config.thread_id,
                "namespace": config.namespace,
                "checkpoint_id": checkpoint_id,
                "writes": serde_json::to_value(&stored)?,
            });
            match self.docs.put(&self.writes, &id, doc, precondition).await {
                Ok(_) => return Ok(()),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(map_error(error)),
            }
        }
        Err(TinyAgentsError::Checkpoint(format!(
            "could not record pending writes for thread `{}`",
            config.thread_id
        )))
    }

    async fn get_writes(&self, config: &CheckpointConfig) -> Result<Vec<PendingWrite>> {
        let Some(checkpoint_id) = self.resolve_write_target(config).await? else {
            return Ok(Vec::new());
        };
        self.declared().await?;
        let id = Self::writes_id(&config.thread_id, &config.namespace, &checkpoint_id);
        let Some(found) = self.docs.get(&self.writes, &id).await.map_err(map_error)? else {
            return Ok(Vec::new());
        };
        Ok(serde_json::from_value(
            found.doc.get("writes").cloned().unwrap_or(Value::Null),
        )?)
    }

    async fn try_claim(&self, thread: &str, owner: &str, ttl: Duration) -> Result<bool> {
        self.declared().await?;
        let id = key(&[thread]);
        for _ in 0..CAS_ATTEMPTS {
            let now = tinyagents_harness::ids::now_ms();
            let current = self.docs.get(&self.leases, &id).await.map_err(map_error)?;
            let precondition = match &current {
                None => Precondition::Absent,
                Some(found) => {
                    let held_by = found.doc.get("owner").and_then(Value::as_str);
                    let expires = found.doc.get("expires_at_ms").and_then(Value::as_u64);
                    let live = expires.is_some_and(|at| at > now);
                    if live && held_by != Some(owner) {
                        return Ok(false);
                    }
                    found.unchanged()
                }
            };
            let expires_at_ms =
                now.saturating_add(u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX));
            let doc = json!({ "owner": owner, "expires_at_ms": expires_at_ms });
            match self.docs.put(&self.leases, &id, doc, precondition).await {
                Ok(_) => return Ok(true),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(map_error(error)),
            }
        }
        Ok(false)
    }

    async fn renew(&self, thread: &str, owner: &str, ttl: Duration) -> Result<bool> {
        self.declared().await?;
        let id = key(&[thread]);
        let now = tinyagents_harness::ids::now_ms();
        let Some(found) = self.docs.get(&self.leases, &id).await.map_err(map_error)? else {
            return Ok(false);
        };
        let held_by = found.doc.get("owner").and_then(Value::as_str);
        let live = found
            .doc
            .get("expires_at_ms")
            .and_then(Value::as_u64)
            .is_some_and(|at| at > now);
        if held_by != Some(owner) || !live {
            return Ok(false);
        }
        let expires_at_ms = now.saturating_add(u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX));
        let doc = json!({ "owner": owner, "expires_at_ms": expires_at_ms });
        match self
            .docs
            .put(&self.leases, &id, doc, found.unchanged())
            .await
        {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == ErrorKind::Conflict => Ok(false),
            Err(error) => Err(map_error(error)),
        }
    }

    async fn release(&self, thread: &str, owner: &str) -> Result<()> {
        self.declared().await?;
        let id = key(&[thread]);
        let Some(found) = self.docs.get(&self.leases, &id).await.map_err(map_error)? else {
            return Ok(());
        };
        if found.doc.get("owner").and_then(Value::as_str) != Some(owner) {
            return Ok(());
        }
        match self.docs.delete(&self.leases, &id, found.unchanged()).await {
            Ok(_) => Ok(()),
            // Someone else reclaimed it in between; it is no longer ours.
            Err(error) if error.kind() == ErrorKind::Conflict => Ok(()),
            Err(error) => Err(map_error(error)),
        }
    }
}

#[cfg(test)]
#[path = "drivers_tests.rs"]
mod tests;
