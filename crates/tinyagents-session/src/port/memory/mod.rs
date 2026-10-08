//! Process-lifetime session stores, one isolated set per agent.
//!
//! [`InMemorySessionStores`] is a complete [`SessionStoreProvider`]: each
//! agent gets its own [`InMemoryTranscriptLocator`], [`InMemoryTurnStates`],
//! key-value store and journal, created on first use and kept until the
//! provider is dropped. Nothing touches a filesystem, so it serves tests and
//! hosts that deliberately keep no history — and it is the reference a host's
//! own provider is held to by the conformance suite.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use tinyagents_harness::store::{InMemoryAppendStore, InMemoryStore};

use super::{AgentStores, SessionStoreProvider, TurnStates};
use crate::testkit::InMemoryTranscriptHistory;
use crate::transcript::{
    SessionRef, TranscriptHistory, TranscriptLocator, TranscriptMeta, TranscriptPartial,
    TranscriptRead, session_stem,
};
use crate::turn_state::{TurnLifecycle, TurnState};

pub(super) const MAX_GENERATIONS: u32 = 4096;

/// Completed turns kept per thread, as the on-disk store keeps them.
pub(super) const COMPLETED_RETENTION: usize = 20;

/// A [`SessionStoreProvider`] keeping every agent's stores in memory, each
/// agent's apart from every other's.
#[derive(Default)]
pub struct InMemorySessionStores {
    agents: Mutex<HashMap<String, AgentStores>>,
}

impl std::fmt::Debug for InMemorySessionStores {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let agents = self.agents.lock().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("InMemorySessionStores")
            .field("agents", &agents.len())
            .finish()
    }
}

impl InMemorySessionStores {
    /// A provider with no agents yet.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SessionStoreProvider for InMemorySessionStores {
    fn for_agent(&self, agent_id: &str) -> AgentStores {
        self.agents
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(agent_id.to_string())
            .or_insert_with(|| {
                let transcripts = Arc::new(InMemoryTranscriptLocator::new(agent_id));
                AgentStores {
                    transcripts: transcripts.clone(),
                    turn_states: Arc::new(InMemoryTurnStates::default()),
                    kv: Arc::new(InMemoryStore::new()),
                    journal: Arc::new(InMemoryAppendStore::new()),
                }
            })
            .clone()
    }

    fn recover(&self) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let agents: Vec<AgentStores> = self
            .agents
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .cloned()
            .collect();
        for stores in agents {
            stores
                .turn_states
                .mark_all_interrupted(&now)
                .map_err(anyhow::Error::msg)?;
        }
        Ok(())
    }

    fn destination_key(&self) -> Option<String> {
        Some(format!("memory://{:p}", self))
    }
}

/// A [`TranscriptLocator`] over transcripts held in memory.
///
/// Transcripts are keyed by stem exactly as the file locator keys files, so
/// generations, sub-agent stems and thread lookups behave the same; "newest"
/// means most recently created.
pub struct InMemoryTranscriptLocator {
    label: String,
    stems: Mutex<Vec<(String, bool, Arc<InMemoryTranscriptHistory>)>>,
    reserved_generations: Mutex<HashSet<String>>,
    generation_gate: Arc<Mutex<()>>,
}

impl InMemoryTranscriptLocator {
    /// An empty locator. `label` names it in diagnostics only.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            stems: Mutex::new(Vec::new()),
            reserved_generations: Mutex::new(HashSet::new()),
            generation_gate: Arc::new(Mutex::new(())),
        }
    }

    /// Written root transcripts (no `__` sub-agent separator), newest first,
    /// with their metadata.
    fn written_roots(&self) -> Vec<(TranscriptMeta, Arc<InMemoryTranscriptHistory>)> {
        let stems = self.stems.lock().unwrap_or_else(PoisonError::into_inner);
        stems
            .iter()
            .rev()
            .filter(|(_, is_subagent, _)| !is_subagent)
            .filter_map(|(_, _, history)| {
                let session = history.read_session().ok().flatten()?;
                Some((session.meta, history.clone()))
            })
            .collect()
    }

    fn begin_generation_locked(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        let successor = session.next_generation();
        anyhow::ensure!(
            successor.generation <= MAX_GENERATIONS,
            "session generation limit reached"
        );
        let stem = session_stem(&successor);
        let mut reservations = self
            .reserved_generations
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let already_exists = self
            .stems
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .any(|(known, _, _)| known == &stem);
        anyhow::ensure!(
            !already_exists && reservations.insert(stem.clone()),
            "session generation already exists or is reserved"
        );
        let mut meta = seed;
        meta.session_id = Some(successor.session_id());
        meta.parent_session_id = successor.parent_session_id();
        let predecessor = self
            .stems
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|(known, _, _)| known == &session_stem(session))
            .map(|(_, _, history)| history.clone());
        if let Some(predecessor) = &predecessor {
            predecessor.seal();
        }
        match self.open_stem_locked(&stem, meta) {
            Ok(handle) => Ok((successor, handle)),
            Err(error) => {
                if let Some(predecessor) = &predecessor {
                    predecessor.unseal();
                }
                reservations.remove(&stem);
                Err(error)
            }
        }
    }
}

impl TranscriptLocator for InMemoryTranscriptLocator {
    fn begin_generation(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.begin_generation_locked(session, seed)
    }

    fn begin_generation_from_baseline(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
        baseline: &[crate::transcript::TranscriptMessage],
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        // Validate baseline before acquiring the generation gate lock; the gate protects
        // generation allocation, not transcript reads. Reading without the gate prevents
        // a deadlock where read_session_transcript's open_stem would re-acquire it.
        let stem = session_stem(session);
        if let Some(transcript) = self
            .stems
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|(known, _, _)| known == &stem)
            .map(|(_, _, history)| history.clone())
            && let Some(transcript) = transcript.read_session()?
        {
            anyhow::ensure!(
                crate::transcript::same_transcript_messages(&transcript.messages, baseline),
                "transcript baseline is stale; reload the session before creating a generation"
            );
        } else {
            anyhow::ensure!(
                baseline.is_empty(),
                "transcript baseline is stale; reload the session before creating a generation"
            );
        }
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.begin_generation_locked(session, seed)
    }

    fn destination_key(&self) -> Option<String> {
        Some(format!("memory://{}/{:p}", self.label, self))
    }

    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.written_roots()
            .into_iter()
            .find(|(meta, _)| {
                meta.agent_name == agent_name || meta.agent_id.as_deref() == Some(agent_name)
            })
            .map(|(_, history)| history as Arc<dyn TranscriptRead>)
    }

    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.root_for_thread_scoped(thread_id, None)
    }

    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return None;
        }
        self.written_roots()
            .into_iter()
            .find(|(meta, _)| {
                meta.thread_id.as_deref() == Some(thread_id)
                    && agent_id.is_none_or(|agent| meta.agent_id.as_deref() == Some(agent))
            })
            .map(|(_, history)| history as Arc<dyn TranscriptRead>)
    }

    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.open_stem_locked(stem, seed)
    }

    fn append_interrupted_partial(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
        partial: &TranscriptPartial,
        request_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        self.append_interrupted_partial_locked(thread_id, agent_id, partial, request_id)
    }
}

impl InMemoryTranscriptLocator {
    fn open_stem_locked(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        let mut stems = self.stems.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, _, history)) = stems.iter().find(|(known, _, _)| known == stem) {
            history.set_seed_if_unwritten(seed);
            return Ok(history.clone());
        }
        // Session stems reserve `__` for the parent/child separator. Check the
        // stem itself so bounded parent stems remain children even when the
        // parent prefix is not present in this locator's index.
        let is_subagent = stem
            .split_once("__")
            .is_some_and(|(parent, child)| !parent.is_empty() && !child.is_empty());
        let history = Arc::new(InMemoryTranscriptHistory::new_with_gate(
            format!("{}/{stem}", self.label),
            seed,
            self.generation_gate.clone(),
        ));
        stems.push((stem.to_string(), is_subagent, history.clone()));
        Ok(history)
    }

    fn append_interrupted_partial_locked(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
        partial: &TranscriptPartial,
        request_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if partial.content.is_empty() {
            return Ok(false);
        }
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return Ok(false);
        }
        let roots = self.written_roots();
        let Some((_meta, history)) = roots.into_iter().find(|(meta, _)| {
            meta.thread_id.as_deref() == Some(thread_id)
                && agent_id.is_none_or(|agent| meta.agent_id.as_deref() == Some(agent))
        }) else {
            return Ok(false);
        };
        Ok(history.record_partial_under_gate(partial.clone(), request_id.map(str::to_string)))
    }
}

/// [`TurnStates`] held in memory, with the on-disk store's semantics:
/// "latest" is the newest `started_at`, completed turns are kept up to
/// [`COMPLETED_RETENTION`] per thread, and a completed turn is never
/// overwritten by a conditional write.
#[derive(Debug, Default)]
pub struct InMemoryTurnStates {
    turns: Mutex<HashMap<(String, String), TurnState>>,
}

impl InMemoryTurnStates {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<(String, String), TurnState>> {
        self.turns.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Newest first: by `started_at`, then `updated_at`.
pub(super) fn newest_first(a: &TurnState, b: &TurnState) -> Ordering {
    compare_rfc3339(&b.started_at, &a.started_at)
        .then_with(|| compare_rfc3339(&b.updated_at, &a.updated_at))
}

/// Completed-turn retention follows completion time, as the durable store
/// does. Listing the latest live turn still follows its start time.
pub(super) fn completed_newest_first(a: &TurnState, b: &TurnState) -> Ordering {
    compare_rfc3339(&b.updated_at, &a.updated_at)
        .then_with(|| compare_rfc3339(&b.started_at, &a.started_at))
}

fn compare_rfc3339(left: &str, right: &str) -> Ordering {
    match (
        chrono::DateTime::parse_from_rfc3339(left),
        chrono::DateTime::parse_from_rfc3339(right),
    ) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

fn key(thread_id: &str, request_id: &str) -> (String, String) {
    (thread_id.to_string(), request_id.to_string())
}

pub(super) fn is_terminal(state: &TurnState) -> bool {
    matches!(
        state.lifecycle,
        TurnLifecycle::Interrupted | TurnLifecycle::Completed
    )
}

/// Drops `thread_id`'s completed turns beyond the newest
/// [`COMPLETED_RETENTION`].
fn prune_completed(turns: &mut HashMap<(String, String), TurnState>, thread_id: &str) {
    let mut completed: Vec<TurnState> = turns
        .values()
        .filter(|turn| turn.thread_id == thread_id && turn.lifecycle == TurnLifecycle::Completed)
        .cloned()
        .collect();
    completed.sort_by(completed_newest_first);
    for stale in completed.iter().skip(COMPLETED_RETENTION) {
        turns.remove(&key(&stale.thread_id, &stale.request_id));
    }
}

impl TurnStates for InMemoryTurnStates {
    fn put(&self, state: &TurnState) -> Result<(), String> {
        let mut turns = self.lock();
        turns.insert(key(&state.thread_id, &state.request_id), state.clone());
        if state.lifecycle == TurnLifecycle::Completed {
            prune_completed(&mut turns, &state.thread_id);
        }
        Ok(())
    }

    fn put_unless_completed(&self, state: &TurnState) -> Result<bool, String> {
        let mut turns = self.lock();
        let slot = key(&state.thread_id, &state.request_id);
        if turns
            .get(&slot)
            .is_some_and(|stored| stored.lifecycle == TurnLifecycle::Completed)
        {
            return Ok(false);
        }
        turns.insert(slot, state.clone());
        if state.lifecycle == TurnLifecycle::Completed {
            prune_completed(&mut turns, &state.thread_id);
        }
        Ok(true)
    }

    fn get(&self, thread_id: &str) -> Result<Option<TurnState>, String> {
        Ok(self.list_thread(thread_id)?.into_iter().next())
    }

    fn get_turn(&self, thread_id: &str, request_id: &str) -> Result<Option<TurnState>, String> {
        Ok(self.lock().get(&key(thread_id, request_id)).cloned())
    }

    fn delete(&self, thread_id: &str) -> Result<bool, String> {
        let mut turns = self.lock();
        let before = turns.len();
        turns.retain(|(thread, _), _| thread != thread_id);
        Ok(turns.len() != before)
    }

    fn delete_turn(&self, thread_id: &str, request_id: &str) -> Result<bool, String> {
        Ok(self.lock().remove(&key(thread_id, request_id)).is_some())
    }

    fn list(&self) -> Result<Vec<TurnState>, String> {
        let mut latest: HashMap<String, TurnState> = HashMap::new();
        for turn in self.lock().values() {
            match latest.get(&turn.thread_id) {
                Some(kept) if newest_first(turn, kept) != Ordering::Less => {}
                _ => {
                    latest.insert(turn.thread_id.clone(), turn.clone());
                }
            }
        }
        Ok(latest.into_values().collect())
    }

    fn list_thread(&self, thread_id: &str) -> Result<Vec<TurnState>, String> {
        let mut turns: Vec<TurnState> = self
            .lock()
            .values()
            .filter(|turn| turn.thread_id == thread_id)
            .cloned()
            .collect();
        turns.sort_by(newest_first);
        Ok(turns)
    }

    fn clear_all(&self) -> Result<usize, String> {
        let mut turns = self.lock();
        let removed = turns.len();
        turns.clear();
        Ok(removed)
    }

    fn mark_all_interrupted(&self, now_rfc3339: &str) -> Result<usize, String> {
        let mut count = 0;
        for turn in self.lock().values_mut().filter(|turn| !is_terminal(turn)) {
            turn.lifecycle = TurnLifecycle::Interrupted;
            turn.updated_at = now_rfc3339.to_string();
            turn.active_tool = None;
            turn.active_subagent = None;
            count += 1;
        }
        Ok(count)
    }

    fn settle_turn(
        &self,
        thread_id: &str,
        request_id: &str,
        lifecycle: TurnLifecycle,
        now_rfc3339: &str,
    ) -> Result<bool, String> {
        let mut turns = self.lock();
        let Some(turn) = turns.get_mut(&key(thread_id, request_id)) else {
            return Ok(false);
        };
        if is_terminal(turn) {
            return Ok(false);
        }
        turn.lifecycle = lifecycle;
        turn.phase = None;
        turn.active_tool = None;
        turn.active_subagent = None;
        turn.updated_at = now_rfc3339.to_string();
        if lifecycle == TurnLifecycle::Completed {
            prune_completed(&mut turns, thread_id);
        }
        Ok(true)
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
