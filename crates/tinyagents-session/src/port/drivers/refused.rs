//! Stores that refuse every call: what an agent gets when the backend could
//! not bind its scope, so the failure surfaces on use instead of the agent's
//! data landing somewhere it does not belong.

use std::sync::Arc;

use serde_json::Value;
use tinystoragedrivers_core::{
    Blocking, Capabilities, CollectionSpec, DocumentStore, ErrorKind, Filter, Page, Precondition,
    Query, Result, Sort, StorageError, StreamEntry, StreamStore, Version, Versioned, async_trait,
};

use super::super::AgentStores;
use super::{DriverAppendStore, DriverStore, DriverTranscriptLocator, DriverTurnStates};

/// A document and stream store whose every call fails with one error.
#[derive(Debug)]
pub(super) struct Refused {
    kind: ErrorKind,
    message: String,
}

impl Refused {
    fn error(&self) -> StorageError {
        StorageError::new(self.kind, self.message.clone())
    }
}

/// Stores over [`Refused`], reporting `error` on every call.
pub(super) fn stores(error: &StorageError, bridge: &Blocking) -> AgentStores {
    let refused = Arc::new(Refused {
        kind: error.kind(),
        message: format!("session stores unavailable: {}", error.message()),
    });
    let docs: Arc<dyn DocumentStore> = refused.clone();
    let streams: Arc<dyn StreamStore> = refused;
    AgentStores {
        transcripts: Arc::new(DriverTranscriptLocator::new(
            Arc::clone(&docs),
            bridge.clone(),
            "refused",
        )),
        turn_states: Arc::new(DriverTurnStates::new(Arc::clone(&docs), bridge.clone())),
        kv: Arc::new(DriverStore::new(docs)),
        journal: Arc::new(DriverAppendStore::new(streams)),
    }
}

#[async_trait]
impl DocumentStore for Refused {
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    async fn ensure_collection(&self, _spec: &CollectionSpec) -> Result<()> {
        Err(self.error())
    }

    async fn get(&self, _collection: &str, _id: &str) -> Result<Option<Versioned<Value>>> {
        Err(self.error())
    }

    async fn put(
        &self,
        _collection: &str,
        _id: &str,
        _doc: Value,
        _precondition: Precondition,
    ) -> Result<Version> {
        Err(self.error())
    }

    async fn delete(&self, _collection: &str, _id: &str, _pre: Precondition) -> Result<bool> {
        Err(self.error())
    }

    async fn query(&self, _collection: &str, _query: &Query) -> Result<Page<Versioned<Value>>> {
        Err(self.error())
    }

    async fn count(&self, _collection: &str, _filter: &Filter) -> Result<u64> {
        Err(self.error())
    }

    async fn delete_where(&self, _collection: &str, _filter: &Filter) -> Result<u64> {
        Err(self.error())
    }

    async fn claim(
        &self,
        _collection: &str,
        _filter: &Filter,
        _sort: &[Sort],
        _patch: &Value,
    ) -> Result<Option<Versioned<Value>>> {
        Err(self.error())
    }

    async fn drop_collection(&self, _collection: &str) -> Result<()> {
        Err(self.error())
    }
}

#[async_trait]
impl StreamStore for Refused {
    async fn append(&self, _stream: &str, _value: Value) -> Result<u64> {
        Err(self.error())
    }

    async fn append_batch(&self, _stream: &str, _values: Vec<Value>) -> Result<u64> {
        Err(self.error())
    }

    async fn read_window(
        &self,
        _stream: &str,
        _from: u64,
        _limit: usize,
    ) -> Result<Vec<StreamEntry>> {
        Err(self.error())
    }

    async fn len(&self, _stream: &str) -> Result<u64> {
        Err(self.error())
    }

    async fn truncate_before(&self, _stream: &str, _offset: u64) -> Result<u64> {
        Err(self.error())
    }

    async fn delete_stream(&self, _stream: &str) -> Result<bool> {
        Err(self.error())
    }

    async fn streams(&self, _prefix: &str) -> Result<Vec<String>> {
        Err(self.error())
    }
}

#[cfg(test)]
#[path = "refused_tests.rs"]
mod tests;
