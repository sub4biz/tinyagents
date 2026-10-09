//! The completion router.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::run_queue::{QueueLane, RunQueueHandle};

use super::format::{CompletionFormatter, NeutralCompletionFormatter};
use super::store::CompletionStore;
use super::types::{CompletionRecord, CompletionState, NotifyMode};
use crate::recovery::{RecoveryChild, build_restart_recovery_note};

const LOG_PREFIX: &str = "[completion-router]";

/// Delivery attempts before a record gives up, unless overridden.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 5;

/// What [`CompletionRouter::record`] did with a completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordOutcome {
    /// Stored. `lane` is set when it was pushed onto a live parent's queue; the
    /// record is then pending and leased until the host calls
    /// [`CompletionRouter::mark_delivered`].
    Recorded {
        /// The lane the completion was pushed onto, if any.
        lane: Option<QueueLane>,
    },
    /// A record for this task id already exists; nothing changed.
    Duplicate,
    /// Dropped: the task was tombstoned, or its parent was cancelled.
    Suppressed,
}

/// What [`CompletionRouter::tombstone`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TombstoneOutcome {
    /// A pending completion was withdrawn.
    Suppressed,
    /// Nothing was recorded yet; a completion arriving later will be dropped.
    Reserved,
    /// The record was already delivered, given up, or tombstoned.
    AlreadyFinal,
}

#[derive(Default)]
struct RouterState {
    /// Task ids handed out by a claim and not yet resolved. Process-local on
    /// purpose: after a restart nobody holds a lease, so everything still
    /// pending is claimable again.
    leased: HashSet<String>,
    /// Live parents and their steering queues.
    parents: HashMap<String, RunQueueHandle>,
    /// Parents whose completions are dropped (deleted or stopped).
    cancelled_parents: HashSet<String>,
}

/// Durable, deduplicating hand-off of finished children to their parents.
///
/// See the `completions` module docs for the division of labour with the host. One
/// router serves one process: its read-modify-write sequences are serialised
/// by an in-process lock, so two processes must not share a store.
pub struct CompletionRouter {
    store: Arc<dyn CompletionStore>,
    formatter: Arc<dyn CompletionFormatter>,
    max_attempts: u32,
    state: Mutex<RouterState>,
}

impl CompletionRouter {
    /// A router over `store` with the neutral formatter and
    /// [`DEFAULT_MAX_ATTEMPTS`].
    pub fn new(store: Arc<dyn CompletionStore>) -> Self {
        Self {
            store,
            formatter: Arc::new(NeutralCompletionFormatter),
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            state: Mutex::new(RouterState::default()),
        }
    }

    /// Replaces the formatter (the host's own wording).
    pub fn with_formatter(mut self, formatter: Arc<dyn CompletionFormatter>) -> Self {
        self.formatter = formatter;
        self
    }

    /// Sets how many failed attempts a record survives. At least one.
    pub fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }

    /// The formatter in use.
    pub fn formatter(&self) -> &Arc<dyn CompletionFormatter> {
        &self.formatter
    }

    /// The attempt limit.
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    fn lock(&self) -> Result<MutexGuard<'_, RouterState>> {
        self.state
            .lock()
            .map_err(|_| TinyAgentsError::Graph("completion router lock poisoned".into()))
    }

    /// Marks `parent_key` live: [`NotifyMode::Followup`] and
    /// [`NotifyMode::Collect`] completions for it are pushed onto `queue`.
    pub fn attach_parent(&self, parent_key: impl Into<String>, queue: RunQueueHandle) {
        let parent_key = parent_key.into();
        tracing::debug!(parent_key = %parent_key, "{LOG_PREFIX} parent attached");
        if let Ok(mut state) = self.lock() {
            state.parents.insert(parent_key, queue);
        }
    }

    /// Marks `parent_key` no longer live. Later completions stay pending.
    ///
    /// The parent's queue is presumed gone, so leases on its in-flight pushes
    /// are released and those records can be claimed again. Acknowledge what the
    /// parent already received with [`Self::mark_delivered`] first.
    pub fn detach_parent(&self, parent_key: &str) {
        tracing::debug!(parent_key = %parent_key, "{LOG_PREFIX} parent detached");
        if let Ok(mut state) = self.lock() {
            state.parents.remove(parent_key);
            self.release_parent_locked(&mut state, parent_key);
        }
    }

    fn release_parent_locked(&self, state: &mut RouterState, parent_key: &str) {
        for record in self.store.list(Some(parent_key)) {
            state.leased.remove(&record.task_id);
        }
    }

    /// Releases the lease on `task_ids` without counting a failure, so they can
    /// be claimed again. Use it when a claim or push was abandoned (a dropped
    /// delivery future, a cleared queue). Attempts already counted stay counted.
    pub fn release<S: AsRef<str>>(&self, task_ids: &[S]) {
        if let Ok(mut state) = self.lock() {
            for id in task_ids {
                state.leased.remove(id.as_ref());
            }
        }
    }

    /// Records pushed onto `parent_key`'s live queue (or claimed) and not yet
    /// acknowledged, oldest first.
    pub fn in_flight_for(&self, parent_key: &str) -> Vec<CompletionRecord> {
        let Ok(state) = self.lock() else {
            return Vec::new();
        };
        let mut records: Vec<_> = self
            .store
            .list(Some(parent_key))
            .into_iter()
            .filter(|r| r.state == CompletionState::Pending && state.leased.contains(&r.task_id))
            .collect();
        sort_oldest_first(&mut records);
        records
    }

    /// Records a finished child, idempotent per task id.
    ///
    /// A new record is stored `Pending`. If its [`NotifyMode`] is `Followup` or
    /// `Collect` and the parent is attached, it is also pushed onto that lane,
    /// counted as attempt one and leased. The queue is in memory and has no
    /// acknowledgement, so the record settles only when the host calls
    /// [`Self::mark_delivered`] after the parent has the message; if the queue
    /// is lost first ([`Self::detach_parent`], [`Self::release`], or a restart)
    /// the record is claimable again. Delivery is at-least-once.
    pub async fn record(&self, mut record: CompletionRecord) -> Result<RecordOutcome> {
        let push = {
            let state = self.lock()?;
            if state.cancelled_parents.contains(&record.parent_key) {
                tracing::debug!(
                    task_id = %record.task_id,
                    parent_key = %record.parent_key,
                    "{LOG_PREFIX} dropped: parent cancelled"
                );
                return Ok(RecordOutcome::Suppressed);
            }
            match self.store.get(&record.task_id) {
                Some(existing) if existing.state == CompletionState::Tombstoned => {
                    tracing::debug!(
                        task_id = %record.task_id,
                        "{LOG_PREFIX} dropped: tombstoned"
                    );
                    return Ok(RecordOutcome::Suppressed);
                }
                Some(existing) => {
                    tracing::debug!(
                        task_id = %record.task_id,
                        state = existing.state.as_str(),
                        "{LOG_PREFIX} duplicate completion ignored"
                    );
                    return Ok(RecordOutcome::Duplicate);
                }
                None => {}
            }
            record.state = CompletionState::Pending;
            record.attempts = 0;
            record.updated_at = SystemTime::now();
            self.store.put(&record)?;
            tracing::debug!(
                task_id = %record.task_id,
                parent_key = %record.parent_key,
                status = record.status.as_str(),
                notify_mode = record.notify_mode.as_str(),
                "{LOG_PREFIX} recorded"
            );
            let lane = match record.notify_mode {
                NotifyMode::Followup => Some(QueueLane::Followup),
                NotifyMode::Collect => Some(QueueLane::Collect),
                NotifyMode::HoldForNextTurn | NotifyMode::Off => None,
            };
            // A live push is a delivery attempt, not a delivery: the record
            // stays pending and leased until the host acknowledges it with
            // `mark_delivered`, so a crash, a cleared queue or a detached parent
            // never loses it (it is redelivered through `claim_pending`).
            match lane.and_then(|lane| {
                state
                    .parents
                    .get(&record.parent_key)
                    .map(|queue| (lane, queue.clone()))
            }) {
                Some((lane, queue)) => {
                    record.attempts = 1;
                    record.updated_at = SystemTime::now();
                    self.store.put(&record)?;
                    state.leased.insert(record.task_id.clone());
                    Some((lane, queue))
                }
                None => None,
            }
        };
        let Some((lane, queue)) = push else {
            return Ok(RecordOutcome::Recorded { lane: None });
        };
        let message = self.formatter.to_message(std::slice::from_ref(&record));
        queue.push(lane, message).await;
        tracing::debug!(
            task_id = %record.task_id,
            parent_key = %record.parent_key,
            lane = lane.as_str(),
            "{LOG_PREFIX} pushed to live parent"
        );
        Ok(RecordOutcome::Recorded { lane: Some(lane) })
    }

    /// Withdraws a child's completion because the parent collected it itself
    /// (waited on it, or read its result). A completion that has not arrived
    /// yet is dropped when it does.
    pub fn tombstone(&self, task_id: &str) -> Result<TombstoneOutcome> {
        let mut state = self.lock()?;
        let outcome = match self.store.get(task_id) {
            Some(mut existing) if existing.state == CompletionState::Pending => {
                existing.state = CompletionState::Tombstoned;
                existing.updated_at = SystemTime::now();
                self.store.put(&existing)?;
                state.leased.remove(task_id);
                TombstoneOutcome::Suppressed
            }
            Some(_) => TombstoneOutcome::AlreadyFinal,
            None => {
                self.store.put(&CompletionRecord::tombstone_stub(task_id))?;
                TombstoneOutcome::Reserved
            }
        };
        tracing::debug!(task_id = %task_id, outcome = ?outcome, "{LOG_PREFIX} tombstone");
        Ok(outcome)
    }

    /// Drops every pending completion for `parent_key` and everything that
    /// finishes for it later, until [`Self::resume_parent`]. For a deleted or
    /// stopped parent. Returns how many pending records were withdrawn.
    pub fn cancel_parent(&self, parent_key: &str) -> Result<usize> {
        let mut state = self.lock()?;
        state.cancelled_parents.insert(parent_key.to_owned());
        state.parents.remove(parent_key);
        let mut withdrawn = 0;
        for mut record in self.store.list(Some(parent_key)) {
            if record.state == CompletionState::Pending {
                record.state = CompletionState::Tombstoned;
                record.updated_at = SystemTime::now();
                self.store.put(&record)?;
                state.leased.remove(&record.task_id);
                withdrawn += 1;
            }
        }
        tracing::debug!(parent_key = %parent_key, withdrawn, "{LOG_PREFIX} parent cancelled");
        Ok(withdrawn)
    }

    /// Lets completions for `parent_key` through again after
    /// [`Self::cancel_parent`] (a stopped thread that the user reopened).
    pub fn resume_parent(&self, parent_key: &str) {
        if let Ok(mut state) = self.lock() {
            state.cancelled_parents.remove(parent_key);
        }
    }

    /// Every undelivered completion for `parent_key`, oldest first, in every
    /// [`NotifyMode`] and whether or not a claim currently holds it. Read-only:
    /// this is what surfaces after a restart.
    pub fn pending_for(&self, parent_key: &str) -> Vec<CompletionRecord> {
        let mut pending: Vec<_> = self
            .store
            .list(Some(parent_key))
            .into_iter()
            .filter(|r| r.state == CompletionState::Pending)
            .collect();
        sort_oldest_first(&mut pending);
        pending
    }

    /// Claims up to `max` `Followup`/`Collect` completions for the idle-time
    /// batch, oldest first, counting an attempt on each. A claimed record is
    /// leased until [`Self::mark_delivered`] or [`Self::mark_failed`].
    pub fn claim_pending(&self, parent_key: &str, max: usize) -> Result<Vec<CompletionRecord>> {
        self.claim(parent_key, max, |mode| {
            matches!(mode, NotifyMode::Followup | NotifyMode::Collect)
        })
    }

    /// Claims the `HoldForNextTurn` completions for the turn that is starting.
    pub fn begin_turn(&self, parent_key: &str) -> Result<Vec<CompletionRecord>> {
        self.claim(parent_key, usize::MAX, |mode| {
            mode == NotifyMode::HoldForNextTurn
        })
    }

    /// Claims up to `max` `Off` completions the parent asked to pull.
    pub fn pull(&self, parent_key: &str, max: usize) -> Result<Vec<CompletionRecord>> {
        self.claim(parent_key, max, |mode| mode == NotifyMode::Off)
    }

    fn claim(
        &self,
        parent_key: &str,
        max: usize,
        eligible: impl Fn(NotifyMode) -> bool,
    ) -> Result<Vec<CompletionRecord>> {
        let mut state = self.lock()?;
        let mut candidates: Vec<_> = self
            .store
            .list(Some(parent_key))
            .into_iter()
            .filter(|r| {
                r.state == CompletionState::Pending
                    && eligible(r.notify_mode)
                    && !state.leased.contains(&r.task_id)
            })
            .collect();
        sort_oldest_first(&mut candidates);
        candidates.truncate(max);
        let mut claimed = Vec::with_capacity(candidates.len());
        for mut record in candidates {
            record.attempts += 1;
            record.updated_at = SystemTime::now();
            self.store.put(&record)?;
            state.leased.insert(record.task_id.clone());
            claimed.push(record);
        }
        if !claimed.is_empty() {
            tracing::debug!(
                parent_key = %parent_key,
                claimed = claimed.len(),
                "{LOG_PREFIX} claimed"
            );
        }
        Ok(claimed)
    }

    /// Settles claimed (or pulled) records as delivered. Returns how many
    /// changed; records that are not pending are left alone.
    pub fn mark_delivered<S: AsRef<str>>(&self, task_ids: &[S]) -> Result<usize> {
        let mut state = self.lock()?;
        let mut changed = 0;
        for id in task_ids {
            let id = id.as_ref();
            state.leased.remove(id);
            if let Some(mut record) = self.store.get(id)
                && record.state == CompletionState::Pending
            {
                record.state = CompletionState::Delivered;
                record.updated_at = SystemTime::now();
                self.store.put(&record)?;
                changed += 1;
            }
        }
        tracing::debug!(changed, "{LOG_PREFIX} delivered");
        Ok(changed)
    }

    /// Reports a failed delivery. The lease is released so the record can be
    /// claimed again, unless its attempts have reached [`Self::max_attempts`]:
    /// then it becomes [`CompletionState::GaveUp`] and is returned so the host
    /// can apply its own give-up policy (the router never delivers it again).
    pub fn mark_failed<S: AsRef<str>>(&self, task_ids: &[S]) -> Result<Vec<CompletionRecord>> {
        let mut state = self.lock()?;
        let mut gave_up = Vec::new();
        for id in task_ids {
            let id = id.as_ref();
            state.leased.remove(id);
            let Some(mut record) = self.store.get(id) else {
                continue;
            };
            if record.state != CompletionState::Pending || record.attempts < self.max_attempts {
                continue;
            }
            record.state = CompletionState::GaveUp;
            record.updated_at = SystemTime::now();
            self.store.put(&record)?;
            tracing::warn!(
                task_id = %record.task_id,
                parent_key = %record.parent_key,
                attempts = record.attempts,
                "{LOG_PREFIX} gave up"
            );
            gave_up.push(record);
        }
        Ok(gave_up)
    }

    /// Drops settled records older than `retain` from the store (and, for the
    /// JSONL store, the log). See [`CompletionStore::compact`].
    pub fn compact(&self, retain: Duration) -> Result<usize> {
        let dropped = self.store.compact(retain)?;
        tracing::debug!(dropped, "{LOG_PREFIX} compacted");
        Ok(dropped)
    }

    /// The restart recovery note for `parent_key`: the interrupted `children`
    /// (see [`recovery_children`](crate::recovery_children)) plus every
    /// completion that finished but was never delivered, framed by the
    /// formatter. Empty when there is nothing to say. Read-only; the host
    /// settles the completions with [`Self::mark_delivered`] once it has
    /// injected the note. `Off` (pull-only) records and ones a claim currently
    /// holds are left out. Nothing is relaunched.
    pub fn restart_recovery_note(&self, parent_key: &str, children: &[RecoveryChild]) -> String {
        let interrupted = build_restart_recovery_note(children);
        let leased = self.lock().map(|s| s.leased.clone()).unwrap_or_default();
        let pending: Vec<_> = self
            .pending_for(parent_key)
            .into_iter()
            .filter(|r| r.notify_mode != NotifyMode::Off && !leased.contains(&r.task_id))
            .collect();
        if pending.is_empty() {
            return interrupted;
        }
        let finished = self.formatter.format_batch(&pending);
        if interrupted.is_empty() {
            finished
        } else {
            format!("{finished}\n\n{interrupted}")
        }
    }
}

fn sort_oldest_first(records: &mut [CompletionRecord]) {
    records.sort_by(|a, b| {
        a.finished_at
            .cmp(&b.finished_at)
            .then_with(|| a.task_id.cmp(&b.task_id))
    });
}
