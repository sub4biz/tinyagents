//! [`TurnStates`] over a driver [`DocumentStore`].
//!
//! One document per turn in [`COLLECTION`], keyed by `(thread, request)`:
//! `{ "thread_id", "request_id", "lifecycle", "turn": <TurnState> }`, with
//! the thread indexed so a thread's turns are one query. Every conditional
//! change (a write that must not clobber a completed turn, a settle, an
//! interruption sweep) is a compare-and-swap on the document's version, so
//! two processes sharing a database cannot lose each other's updates.
//!
//! Semantics match [`InMemoryTurnStates`](crate::port::InMemoryTurnStates):
//! "latest" is the newest `started_at`, completed turns are kept up to
//! [`COMPLETED_RETENTION`] per thread, and a completed turn is never
//! overwritten by a conditional write.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use serde_json::{Value, json};
use tinystoragedrivers_core::{
    Blocking, CollectionSpec, DocumentStore, DocumentStoreExt, ErrorKind, Filter, IndexSpec,
    Precondition, Query, StorageError, Versioned,
};
use tokio::sync::OnceCell;

use super::super::TurnStates;
use super::super::memory::{
    COMPLETED_RETENTION, completed_newest_first, is_terminal, newest_first,
};
use super::transcripts::doc_key;
use crate::turn_state::{TurnLifecycle, TurnState};

/// The collection turn snapshots live in.
pub(super) const COLLECTION: &str = "session_turn_states";

/// Compare-and-swap attempts before a contended update gives up.
const CAS_ATTEMPTS: usize = 64;

/// [`TurnStates`] stored in a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverTurnStates {
    inner: Arc<Inner>,
    bridge: Blocking,
}

#[derive(Debug)]
struct Inner {
    docs: Arc<dyn DocumentStore>,
    declared: OnceCell<()>,
}

impl DriverTurnStates {
    /// Turn states in `docs`, with synchronous calls run on `bridge`.
    pub fn new(docs: Arc<dyn DocumentStore>, bridge: Blocking) -> Self {
        Self {
            inner: Arc::new(Inner {
                docs,
                declared: OnceCell::new(),
            }),
            bridge,
        }
    }

    /// Runs `op` against the store on the bridge.
    fn run<T, F, Fut>(&self, op: F) -> Result<T, String>
    where
        F: FnOnce(Arc<Inner>) -> Fut,
        Fut: Future<Output = Result<T, StorageError>> + Send + 'static,
        T: Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        let future = op(inner);
        match self.bridge.run(future) {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error)) | Err(error) => Err(error.to_string()),
        }
    }
}

fn id(thread_id: &str, request_id: &str) -> String {
    doc_key(&[thread_id, request_id])
}

fn encode(state: &TurnState) -> Result<Value, StorageError> {
    let turn = serde_json::to_value(state)
        .map_err(|error| StorageError::serialization(error.to_string()))?;
    let lifecycle = serde_json::to_value(state.lifecycle)
        .map_err(|error| StorageError::serialization(error.to_string()))?;
    Ok(json!({
        "thread_id": state.thread_id,
        "request_id": state.request_id,
        "lifecycle": lifecycle,
        "turn": turn,
    }))
}

fn decode(stored: Versioned<Value>) -> Result<Versioned<TurnState>, StorageError> {
    let Versioned { id, version, doc } = stored;
    let turn = doc
        .get("turn")
        .cloned()
        .ok_or_else(|| StorageError::serialization(format!("turn state {id} has no body")))?;
    let turn = serde_json::from_value(turn).map_err(|error| {
        StorageError::serialization(format!("turn state {id} is unreadable: {error}"))
    })?;
    Ok(Versioned {
        id,
        version,
        doc: turn,
    })
}

impl Inner {
    async fn declared(&self) -> Result<(), StorageError> {
        self.declared
            .get_or_try_init(|| async {
                let spec = CollectionSpec::new(COLLECTION)
                    .index(IndexSpec::new("by_thread", ["thread_id"]));
                self.docs.ensure_collection(&spec).await
            })
            .await
            .map(|_| ())
    }

    async fn get_turn(
        &self,
        thread_id: &str,
        request_id: &str,
    ) -> Result<Option<Versioned<TurnState>>, StorageError> {
        self.declared().await?;
        self.docs
            .get(COLLECTION, &id(thread_id, request_id))
            .await?
            .map(decode)
            .transpose()
    }

    async fn query(&self, filter: Filter) -> Result<Vec<Versioned<TurnState>>, StorageError> {
        self.declared().await?;
        self.docs
            .query_all(COLLECTION, &Query::filter(filter).limit(1_000))
            .await?
            .into_iter()
            .map(decode)
            .collect()
    }

    async fn thread(&self, thread_id: &str) -> Result<Vec<TurnState>, StorageError> {
        let mut turns: Vec<TurnState> = self
            .query(Filter::eq("thread_id", thread_id))
            .await?
            .into_iter()
            .map(|stored| stored.doc)
            .collect();
        turns.sort_by(newest_first);
        Ok(turns)
    }

    async fn write(&self, state: &TurnState, pre: Precondition) -> Result<(), StorageError> {
        self.declared().await?;
        self.docs
            .put(
                COLLECTION,
                &id(&state.thread_id, &state.request_id),
                encode(state)?,
                pre,
            )
            .await
            .map(|_| ())
    }

    /// Drops `thread_id`'s completed turns beyond the newest
    /// [`COMPLETED_RETENTION`].
    ///
    /// Each delete is conditional on the version the listing saw: a turn
    /// rewritten in the meantime is someone else's newer state and stays.
    async fn prune_completed(&self, thread_id: &str) -> Result<(), StorageError> {
        let mut completed: Vec<Versioned<TurnState>> = self
            .query(Filter::eq("thread_id", thread_id))
            .await?
            .into_iter()
            .filter(|stored| stored.doc.lifecycle == TurnLifecycle::Completed)
            .collect();
        completed.sort_by(|a, b| completed_newest_first(&a.doc, &b.doc));
        for stale in completed.iter().skip(COMPLETED_RETENTION) {
            match self
                .docs
                .delete(COLLECTION, &stale.id, stale.unchanged())
                .await
            {
                Ok(_) => {}
                Err(error) if is_race(&error) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Applies `change` to one turn under compare-and-swap; returns whether
    /// it changed anything. `change` answers `None` to leave the turn alone.
    async fn update(
        &self,
        thread_id: &str,
        request_id: &str,
        change: impl Fn(&TurnState) -> Option<TurnState>,
    ) -> Result<Option<TurnState>, StorageError> {
        for _ in 0..CAS_ATTEMPTS {
            let Some(stored) = self.get_turn(thread_id, request_id).await? else {
                return Ok(None);
            };
            let Some(next) = change(&stored.doc) else {
                return Ok(None);
            };
            match self.write(&next, stored.unchanged()).await {
                Ok(()) => return Ok(Some(next)),
                Err(error) if is_race(&error) => {}
                Err(error) => return Err(error),
            }
        }
        Err(contended(thread_id, request_id))
    }
}

/// Whether `error` means another writer got there first.
fn is_race(error: &StorageError) -> bool {
    error.kind() == ErrorKind::Conflict
}

fn contended(thread_id: &str, request_id: &str) -> StorageError {
    StorageError::conflict(format!(
        "turn {thread_id}/{request_id} kept changing under {CAS_ATTEMPTS} attempts"
    ))
}

/// `turn` marked interrupted as of `now`.
fn interrupted(turn: &TurnState, now: &str) -> Option<TurnState> {
    if is_terminal(turn) {
        return None;
    }
    let mut next = turn.clone();
    next.lifecycle = TurnLifecycle::Interrupted;
    next.updated_at = now.to_string();
    next.active_tool = None;
    next.active_subagent = None;
    Some(next)
}

impl TurnStates for DriverTurnStates {
    fn put(&self, state: &TurnState) -> Result<(), String> {
        let state = state.clone();
        self.run(|inner| async move {
            inner.write(&state, Precondition::None).await?;
            if state.lifecycle == TurnLifecycle::Completed {
                inner.prune_completed(&state.thread_id).await?;
            }
            Ok(())
        })
    }

    fn put_unless_completed(&self, state: &TurnState) -> Result<bool, String> {
        let state = state.clone();
        self.run(|inner| async move {
            for _ in 0..CAS_ATTEMPTS {
                let pre = match inner.get_turn(&state.thread_id, &state.request_id).await? {
                    Some(stored) if stored.doc.lifecycle == TurnLifecycle::Completed => {
                        return Ok(false);
                    }
                    Some(stored) => stored.unchanged(),
                    None => Precondition::Absent,
                };
                match inner.write(&state, pre).await {
                    Ok(()) => {
                        if state.lifecycle == TurnLifecycle::Completed {
                            inner.prune_completed(&state.thread_id).await?;
                        }
                        return Ok(true);
                    }
                    Err(error) if is_race(&error) => {}
                    Err(error) => return Err(error),
                }
            }
            Err(contended(&state.thread_id, &state.request_id))
        })
    }

    fn get(&self, thread_id: &str) -> Result<Option<TurnState>, String> {
        Ok(self.list_thread(thread_id)?.into_iter().next())
    }

    fn get_turn(&self, thread_id: &str, request_id: &str) -> Result<Option<TurnState>, String> {
        let (thread_id, request_id) = (thread_id.to_string(), request_id.to_string());
        self.run(|inner| async move {
            Ok(inner
                .get_turn(&thread_id, &request_id)
                .await?
                .map(|stored| stored.doc))
        })
    }

    fn delete(&self, thread_id: &str) -> Result<bool, String> {
        let thread_id = thread_id.to_string();
        self.run(|inner| async move {
            inner.declared().await?;
            let removed = inner
                .docs
                .delete_where(COLLECTION, &Filter::eq("thread_id", thread_id))
                .await?;
            Ok(removed > 0)
        })
    }

    fn delete_turn(&self, thread_id: &str, request_id: &str) -> Result<bool, String> {
        let key = id(thread_id, request_id);
        self.run(|inner| async move {
            inner.declared().await?;
            inner
                .docs
                .delete(COLLECTION, &key, Precondition::None)
                .await
        })
    }

    fn list(&self) -> Result<Vec<TurnState>, String> {
        self.run(|inner| async move {
            let mut latest: HashMap<String, TurnState> = HashMap::new();
            for turn in inner.query(Filter::All).await? {
                let turn = turn.doc;
                match latest.get(&turn.thread_id) {
                    Some(kept) if newest_first(&turn, kept) != Ordering::Less => {}
                    _ => {
                        latest.insert(turn.thread_id.clone(), turn);
                    }
                }
            }
            Ok(latest.into_values().collect())
        })
    }

    fn list_thread(&self, thread_id: &str) -> Result<Vec<TurnState>, String> {
        let thread_id = thread_id.to_string();
        self.run(|inner| async move { inner.thread(&thread_id).await })
    }

    fn clear_all(&self) -> Result<usize, String> {
        self.run(|inner| async move {
            inner.declared().await?;
            let removed = inner.docs.delete_where(COLLECTION, &Filter::All).await?;
            Ok(usize::try_from(removed).unwrap_or(usize::MAX))
        })
    }

    fn mark_all_interrupted(&self, now_rfc3339: &str) -> Result<usize, String> {
        let now = now_rfc3339.to_string();
        self.run(|inner| async move {
            let mut count = 0;
            for stored in inner.query(Filter::All).await? {
                let turn = stored.doc;
                if is_terminal(&turn) {
                    continue;
                }
                if inner
                    .update(&turn.thread_id, &turn.request_id, |current| {
                        interrupted(current, &now)
                    })
                    .await?
                    .is_some()
                {
                    count += 1;
                }
            }
            Ok(count)
        })
    }

    fn settle_turn(
        &self,
        thread_id: &str,
        request_id: &str,
        lifecycle: TurnLifecycle,
        now_rfc3339: &str,
    ) -> Result<bool, String> {
        let (thread_id, request_id) = (thread_id.to_string(), request_id.to_string());
        let now = now_rfc3339.to_string();
        self.run(|inner| async move {
            let settled = inner
                .update(&thread_id, &request_id, |current| {
                    if is_terminal(current) {
                        return None;
                    }
                    let mut next = current.clone();
                    next.lifecycle = lifecycle;
                    next.phase = None;
                    next.active_tool = None;
                    next.active_subagent = None;
                    next.updated_at.clone_from(&now);
                    Some(next)
                })
                .await?;
            if settled.is_some() && lifecycle == TurnLifecycle::Completed {
                inner.prune_completed(&thread_id).await?;
            }
            Ok(settled.is_some())
        })
    }
}

#[cfg(test)]
#[path = "turn_states_tests.rs"]
mod tests;
