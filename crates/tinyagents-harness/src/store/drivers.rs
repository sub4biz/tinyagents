//! [`Store`] and [`AppendStore`] over `tinystoragedrivers` ports.
//!
//! A host that has opened a storage backend (SQLite on a desktop, MongoDB in
//! the cloud, memory in tests) hands the harness a scoped
//! [`DocumentStore`] / [`StreamStore`] and wraps it here; the harness keeps
//! talking to its own narrow traits and never learns which backend it runs
//! on. The handles are already bound to a tenant scope, so nothing here takes
//! one.
//!
//! # Layout
//!
//! - [`DriverStore`] keeps every harness namespace in one document collection
//!   (default [`DriverStore::DEFAULT_COLLECTION`]). Each entry is a document
//!   `{ "ns": <namespace>, "key": <key>, "value": <value> }`, so a namespace
//!   or key may hold any characters and `list` is an indexed query on `ns`.
//! - [`DriverAppendStore`] maps each harness stream to the driver stream
//!   `<len>:<prefix><stream>`, the prefix empty when there is none. The
//!   length is always written, so no two `(prefix, stream)` pairs, prefixed
//!   or not, ever address one stream. Offsets are the
//!   driver's dense offsets, which already match the [`AppendStore`]
//!   contract.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinystoragedrivers_core::{
    CollectionSpec, DocumentStore, DocumentStoreExt, ErrorKind, Filter, IndexSpec, Precondition,
    Query, StorageError, StreamStore,
};
use tokio::sync::OnceCell;

use super::{AppendStore, Store};
use crate::error::{Result, TinyAgentsError};

/// Map a storage driver failure onto the harness error type. Invalid input
/// (a name the backend cannot store) is a validation error; everything else
/// is a storage error carrying the driver's message.
pub(crate) fn map_error(error: StorageError) -> TinyAgentsError {
    match error.kind() {
        ErrorKind::InvalidInput => TinyAgentsError::Validation(error.to_string()),
        _ => TinyAgentsError::Storage(error.to_string()),
    }
}

/// A harness [`Store`] over a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverStore {
    docs: Arc<dyn DocumentStore>,
    collection: String,
    declared: Arc<OnceCell<()>>,
}

impl DriverStore {
    /// The collection used by [`DriverStore::new`].
    pub const DEFAULT_COLLECTION: &'static str = "harness_store";

    /// Store harness namespaces in [`Self::DEFAULT_COLLECTION`].
    pub fn new(docs: Arc<dyn DocumentStore>) -> Self {
        Self::with_collection(docs, Self::DEFAULT_COLLECTION)
    }

    /// Store harness namespaces in `collection`, so several independent
    /// harness stores can share one backend.
    pub fn with_collection(docs: Arc<dyn DocumentStore>, collection: impl Into<String>) -> Self {
        Self {
            docs,
            collection: collection.into(),
            declared: Arc::new(OnceCell::new()),
        }
    }

    /// The document id of `key` in `namespace`. The namespace is
    /// length-prefixed so `("a/b", "c")` and `("a", "b/c")` never collide.
    fn id(namespace: &str, key: &str) -> String {
        format!("{}:{namespace}/{key}", namespace.len())
    }

    /// Declare the collection (with its `ns` index) once per handle.
    async fn declared(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                let spec =
                    CollectionSpec::new(&self.collection).index(IndexSpec::new("by_ns", ["ns"]));
                self.docs.ensure_collection(&spec).await.map_err(map_error)
            })
            .await
            .map(|_| ())
    }
}

#[async_trait]
impl Store for DriverStore {
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<Value>> {
        self.declared().await?;
        let found = self
            .docs
            .get(&self.collection, &Self::id(namespace, key))
            .await
            .map_err(map_error)?;
        Ok(found.and_then(|mut stored| stored.doc.get_mut("value").map(Value::take)))
    }

    async fn put(&self, namespace: &str, key: &str, value: Value) -> Result<()> {
        self.declared().await?;
        let doc = json!({ "ns": namespace, "key": key, "value": value });
        self.docs
            .put(
                &self.collection,
                &Self::id(namespace, key),
                doc,
                Precondition::None,
            )
            .await
            .map(|_| ())
            .map_err(map_error)
    }

    async fn delete(&self, namespace: &str, key: &str) -> Result<()> {
        self.declared().await?;
        self.docs
            .delete(
                &self.collection,
                &Self::id(namespace, key),
                Precondition::None,
            )
            .await
            .map(|_| ())
            .map_err(map_error)
    }

    async fn list(&self, namespace: &str) -> Result<Vec<String>> {
        self.declared().await?;
        let query = Query::filter(Filter::eq("ns", namespace));
        let found = self
            .docs
            .query_all(&self.collection, &query)
            .await
            .map_err(map_error)?;
        Ok(found
            .into_iter()
            .filter_map(|stored| {
                stored
                    .doc
                    .get("key")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect())
    }
}

/// A harness [`AppendStore`] over a driver [`StreamStore`].
#[derive(Debug, Clone)]
pub struct DriverAppendStore {
    streams: Arc<dyn StreamStore>,
    prefix: String,
}

impl DriverAppendStore {
    /// Map each harness stream `s` to the driver stream `0:s`.
    pub fn new(streams: Arc<dyn StreamStore>) -> Self {
        Self::with_prefix(streams, "")
    }

    /// Map each harness stream to `<len>:<prefix><stream>`, so several
    /// independent append stores can share one backend. The prefix length is
    /// part of the name, so `("a", "bc")` and `("ab", "c")` stay apart.
    pub fn with_prefix(streams: Arc<dyn StreamStore>, prefix: impl Into<String>) -> Self {
        Self {
            streams,
            prefix: prefix.into(),
        }
    }

    fn name(&self, stream: &str) -> String {
        format!("{}:{}{stream}", self.prefix.len(), self.prefix)
    }
}

/// Entries read per driver call when following a stream to its end.
const READ_PAGE: usize = 1_000;

#[async_trait]
impl AppendStore for DriverAppendStore {
    async fn append(&self, stream: &str, value: Value) -> Result<u64> {
        self.streams
            .append(&self.name(stream), value)
            .await
            .map_err(map_error)
    }

    async fn read_from(&self, stream: &str, offset: u64) -> Result<Vec<(u64, Value)>> {
        let name = self.name(stream);
        let mut out = Vec::new();
        let mut next = offset;
        loop {
            let page = self
                .streams
                .read_window(&name, next, READ_PAGE)
                .await
                .map_err(map_error)?;
            let full = page.len() == READ_PAGE;
            for entry in page {
                next = entry.offset + 1;
                out.push((entry.offset, entry.value));
            }
            if !full {
                return Ok(out);
            }
        }
    }

    async fn read_window(
        &self,
        stream: &str,
        offset: u64,
        limit: usize,
    ) -> Result<Vec<(u64, Value)>> {
        let page = self
            .streams
            .read_window(&self.name(stream), offset, limit)
            .await
            .map_err(map_error)?;
        Ok(page
            .into_iter()
            .map(|entry| (entry.offset, entry.value))
            .collect())
    }

    async fn len(&self, stream: &str) -> Result<u64> {
        self.streams
            .len(&self.name(stream))
            .await
            .map_err(map_error)
    }
}

#[cfg(test)]
#[path = "drivers_tests.rs"]
mod tests;
