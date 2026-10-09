//! Sub-agent trails: project each delegated run's sibling transcript and
//! place it next to the tool call that spawned it.
//!
//! Correlation is by evidence, strongest first:
//!
//! 0. **Explicit link** — a sub-agent spawn result records the child's
//!    identity (`subagent_run_id` and `job_id` in the queued JSON), and the
//!    child's `_meta` repeats it as its `task_id` or in its thread id
//!    (`…-subagent-{run_id}`). A parent tool call whose result names the
//!    child is its spawner, with no guessing ([`find_explicit_spawning_call`]).
//!    The run ledger's `parentCallId` is the same kind of exact evidence for
//!    hosts that keep one ([`find_exact_spawning_call`]).
//!
//! Transcripts written before that link existed carry neither, so they fall
//! back to timing and targets:
//!
//! 1. **Turn** — the child's spawn time (the leading unix seconds of its stem
//!    suffix) against the parent turns' commit timestamps
//!    ([`anchor_request_id`]).
//! 2. **Call** — within that turn, the first unclaimed tool call that targets
//!    the child's agent (`delegate_{agent}`, an `agent_id` argument, …), else
//!    the first unclaimed delegation-shaped call.
//!
//! An uncorrelated child lands at the end of its turn (or of the list when
//! there are no turns) instead of after every root item, which is where all
//! sub-agents used to go.

use std::path::{Path, PathBuf};

use crate::transcript::{self, DisplayRecord};

use super::project::{native_tool_round, project_records};
use super::types::{DisplayItem, ToolCallStatus, TranscriptSubagentStatus};

const LOG_PREFIX: &str = "[threads][transcript][subagents]";

/// Max sub-agent nesting depth the projection descends; bounded so a worker
/// that itself delegates still surfaces, without unbounded fan-out.
const MAX_SUBAGENT_DEPTH: usize = 3;

/// Legacy text prefix a delegation runner put on a result it gave up on.
/// Still read so old transcripts keep their status; new results carry the
/// typed `"status": "incomplete"` instead (see [`is_incomplete_result`]).
const INCOMPLETE_MARKER: &str = "[SUBAGENT_INCOMPLETE]";

/// Prefix of an async spawn's acknowledgement — success of the *spawn*, not
/// of the run, so it says nothing about the child's terminal state.
const ASYNC_ACCEPTED_PREFIX: &str = "Accepted async sub-agent";

/// Argument keys a spawn/delegate tool uses to name its target agent.
const TARGET_ARG_KEYS: &[&str] = &["agent_id", "agent", "subagent", "subagent_type", "target"];

/// A projected child run awaiting placement.
struct ChildRun {
    /// Unix seconds the child was spawned at, from its stem.
    spawn_unix: Option<i64>,
    agent_id: Option<String>,
    /// Spawn task id, when the transcript recorded one — the key the run
    /// ledger's `AgentRunUpsert.id` uses, so it's also the key for the exact
    /// `parentCallId` correlation in [`find_exact_spawning_call`].
    task_id: Option<String>,
    /// Identifiers a parent spawn result may name this child by (task id and
    /// the run id embedded in the child thread id) for the explicit-link pass.
    link_ids: Vec<String>,
    item: DisplayItem,
    /// The child's own terminal evidence, before the spawning call is known.
    own_state: OwnState,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OwnState {
    Completed,
    Interrupted,
    Unknown,
}

/// Place every direct child of the root (`__`-once stems) into `items`.
/// `segments` are the root turns' `(request_id, commit unix)` pairs.
pub(super) fn attach(
    items: &mut Vec<DisplayItem>,
    sub_paths: &[PathBuf],
    segments: &[(String, i64)],
    root_thread_id: Option<&str>,
    workspace_dir: Option<&Path>,
) {
    let children = build_children(sub_paths, None, root_thread_id, 0, workspace_dir);
    place(items, children, segments, workspace_dir);
}

/// Project the direct children of `parent_stem` (or of the roots, when
/// `None`), recursing into their own children.
fn build_children(
    sub_paths: &[PathBuf],
    parent_stem: Option<&str>,
    parent_thread_id: Option<&str>,
    depth: usize,
    workspace_dir: Option<&Path>,
) -> Vec<ChildRun> {
    if depth >= MAX_SUBAGENT_DEPTH {
        return Vec::new();
    }
    let mut children = Vec::new();
    for path in sub_paths {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // A compacted child transcript is a generation chain. The newest
        // generation contains the retained prefix plus its subsequent rows;
        // project that head once instead of rendering every generation as a
        // separate child card.
        if let Some((base, generation)) = child_generation(stem)
            && sub_paths
                .iter()
                .filter_map(|candidate| candidate.file_stem().and_then(|name| name.to_str()))
                .filter_map(|candidate| child_generation(candidate))
                .any(|(candidate_base, candidate_generation)| {
                    candidate_base == base && candidate_generation > generation
                })
        {
            continue;
        }
        let suffix = match parent_stem {
            Some(parent) => match stem.strip_prefix(parent).and_then(|r| r.strip_prefix("__")) {
                Some(rest) if !rest.contains("__") => rest,
                _ => continue,
            },
            // A root's direct child has exactly one `__` separator.
            None => match stem.split_once("__") {
                Some((_, rest)) if !rest.contains("__") => rest,
                _ => continue,
            },
        };
        let stem_parent_thread_id = parent_stem
            .is_none()
            .then(|| stem.split_once("__").map(|(parent, _)| parent))
            .flatten();
        if let Some(child) = build_child(
            path,
            stem,
            suffix,
            parent_thread_id.or(stem_parent_thread_id),
            sub_paths,
            depth,
            workspace_dir,
        ) {
            children.push(child);
        }
    }
    children.sort_by_key(|child| child.spawn_unix);
    children
}

fn child_generation(stem: &str) -> Option<(&str, u32)> {
    let (base, suffix) = stem.rsplit_once(".g")?;
    Some((base, suffix.parse().ok()?))
}

fn build_child(
    path: &Path,
    stem: &str,
    suffix: &str,
    parent_thread_id: Option<&str>,
    sub_paths: &[PathBuf],
    depth: usize,
    workspace_dir: Option<&Path>,
) -> Option<ChildRun> {
    let display = match transcript::read_transcript_display(path) {
        Ok(display) => display,
        Err(err) => {
            tracing::warn!(
                "{LOG_PREFIX} failed to read sub-agent transcript {}: {err}",
                path.display()
            );
            return None;
        }
    };
    let own_state = own_state(&display.records);
    let mut items = project_records(&display.records);
    let grandchildren = build_children(
        sub_paths,
        Some(stem),
        display.meta.thread_id.as_deref(),
        depth + 1,
        workspace_dir,
    );
    place(
        &mut items,
        grandchildren,
        &turn_segments(&display.records),
        workspace_dir,
    );

    let task_id = display.meta.task_id.clone().filter(|id| !id.is_empty());
    let agent_id = display
        .meta
        .agent_id
        .clone()
        .or_else(|| Some(display.meta.agent_name.clone()))
        .filter(|id| !id.is_empty());
    let link_ids = link_ids(
        task_id.as_deref(),
        display.meta.thread_id.as_deref(),
        parent_thread_id,
    );
    let id = task_id.clone().unwrap_or_else(|| suffix.to_string());
    let spawn_unix = child_spawn_unix(suffix);
    // The spawn timestamp encoded in the sub-agent's own file stem (used
    // above to anchor it to a parent turn) doubles as this item's `ts` —
    // sub-agent transcripts carry no back-link to a delegating request, so
    // there is no per-message `ts` to inherit the way the root projector
    // pulls one from `DisplayMessage.ts`.
    let ts = spawn_unix
        .and_then(|unix| chrono::DateTime::from_timestamp(unix, 0).map(|dt| dt.to_rfc3339()));
    Some(ChildRun {
        spawn_unix,
        agent_id: agent_id.clone(),
        task_id: task_id.clone(),
        link_ids,
        item: DisplayItem::Subagent {
            id,
            agent_id,
            task_id,
            call_id: None,
            status: TranscriptSubagentStatus::Running,
            request_id: None,
            ts,
            items,
        },
        own_state,
    })
}

/// Marker the harness puts in a child's derived thread id:
/// `{parent_thread}-subagent-{run_id}`. A grandchild's parent thread already
/// holds the marker, so the run id is the part after the *last* one.
const CHILD_THREAD_MARKER: &str = "-subagent-";

/// The ids a parent spawn result could use to name this child.
fn link_ids(
    task_id: Option<&str>,
    thread_id: Option<&str>,
    parent_thread_id: Option<&str>,
) -> Vec<String> {
    let mut ids: Vec<String> = task_id.map(str::to_owned).into_iter().collect();
    let parent_run_id = parent_thread_id.and_then(|parent| {
        thread_id?
            .strip_prefix(parent)?
            .strip_prefix(CHILD_THREAD_MARKER)
    });
    if let Some(run_id) = parent_run_id
        && !run_id.is_empty()
    {
        ids.push(run_id.to_owned());
    }
    if let Some(run_id) =
        thread_id.and_then(|thread| thread.rsplit_once(CHILD_THREAD_MARKER).map(|(_, id)| id))
        && !run_id.is_empty()
        && !ids.iter().any(|id| id == run_id)
    {
        ids.push(run_id.to_owned());
    }
    ids
}

/// Explicit correlation: the unclaimed [`DisplayItem::ToolCall`] whose result
/// is a spawn payload naming one of the child's `ids` as its
/// `subagent_run_id` or `job_id`.
///
/// Only spawn payloads count: queued/inline results that carry both `job_id`
/// and `subagent_run_id`. `subagent_jobs` query/cancel snapshots (keyed `id`)
/// and `subagent_message` acknowledgements do not, so a later status check
/// cannot steal the match. When the payload records the spawning
/// `parent_tool_call_id` it must equal the item's own `call_id`.
///
/// Ids are unique per run, so the whole item list is searched rather than a
/// turn range. `None` when no result carries an id (a transcript written
/// before the link existed), so callers fall through to the older evidence.
fn find_explicit_spawning_call(
    items: &[DisplayItem],
    claimed: &[bool],
    ids: &[String],
) -> Option<usize> {
    if ids.is_empty() {
        return None;
    }
    (0..items.len()).find(|&index| {
        if claimed[index] {
            return false;
        }
        let DisplayItem::ToolCall {
            call_id,
            result: Some(result),
            ..
        } = &items[index]
        else {
            return false;
        };
        // Cheap precheck: skip JSON parsing for results that cannot be spawn payloads.
        if !result.contains("subagent_run_id") {
            return false;
        }
        let Ok(serde_json::Value::Object(payload)) = serde_json::from_str(result) else {
            return false;
        };
        if !(payload.contains_key("job_id") && payload.contains_key("subagent_run_id")) {
            return false;
        }
        if let Some(recorded) = payload.get("parent_tool_call_id")
            && recorded.as_str() != Some(call_id)
        {
            return false;
        }
        ["subagent_run_id", "job_id"].iter().any(|key| {
            payload
                .get(*key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| ids.iter().any(|id| id == value))
        })
    })
}

/// Exact correlation: the run ledger's `AgentRunUpsert.metadata.parentCallId`
/// for this task (stamped by `progress_bridge`'s `SubagentSpawned` handling),
/// resolved to the unclaimed [`DisplayItem::ToolCall`] with that `call_id`.
///
/// Preferred over [`find_spawning_call`]'s timestamp/target-argument
/// heuristic whenever it resolves — the ledger has the actual call id, no
/// guessing required. `None` on any miss (no workspace, no task id, no
/// ledger row, no matching/unclaimed call), so callers fall back to the
/// heuristic unconditionally.
fn find_exact_spawning_call(
    items: &[DisplayItem],
    claimed: &[bool],
    start: usize,
    end: usize,
    task_id: Option<&str>,
    workspace_dir: Option<&Path>,
) -> Option<usize> {
    let workspace_dir = workspace_dir?;
    let task_id = task_id?;
    let run = crate::run_ledger::get_agent_run(workspace_dir, task_id)
        .ok()
        .flatten()?;
    let parent_call_id = run.metadata.get("parentCallId")?.as_str()?;
    let matching = |index: usize| matches!(&items[index], DisplayItem::ToolCall { call_id, .. } if call_id == parent_call_id);
    let end = end.min(items.len());
    let start = start.min(end);
    // Providers may reuse call ids across turns, so prefer the anchored turn.
    if let Some(index) = (start..end).find(|&index| !claimed[index] && matching(index)) {
        return Some(index);
    }
    // Only look outside the anchored turn when that turn has no call with
    // this id at all (an imprecise spawn anchor); a claimed in-range match
    // means the id repeats, and another turn's call would be the wrong one.
    if (start..end).any(matching) {
        return None;
    }
    (0..items.len()).find(|&index| !claimed[index] && matching(index))
}

/// What the child's own transcript says about how it ended.
fn own_state(records: &[DisplayRecord]) -> OwnState {
    let last = records.iter().rev().find_map(|record| match record {
        DisplayRecord::Message(msg) if msg.message.role != "system" => Some(msg),
        _ => None,
    });
    match last {
        Some(msg) if msg.interrupted => OwnState::Interrupted,
        Some(msg)
            if msg.message.role == "assistant"
                && msg
                    .turn_usage
                    .as_ref()
                    .is_none_or(|usage| usage.tool_calls.is_empty())
                && native_tool_round(&msg.message).is_none_or(|(_, calls)| calls.is_empty()) =>
        {
            OwnState::Completed
        }
        _ => OwnState::Unknown,
    }
}

/// Insert `children` into `items`, each after its correlated spawning call
/// (claimed at most once), else at the end of its anchored turn.
fn place(
    items: &mut Vec<DisplayItem>,
    children: Vec<ChildRun>,
    segments: &[(String, i64)],
    workspace_dir: Option<&Path>,
) {
    if children.is_empty() {
        return;
    }
    let mut claimed = vec![false; items.len()];
    // (insert position, order) — applied back-to-front afterwards.
    let mut inserts: Vec<(usize, usize, DisplayItem)> = Vec::new();
    for (order, mut child) in children.into_iter().enumerate() {
        let request_id = anchor_request_id(child.spawn_unix, segments);
        let (start, end) = turn_range(items, request_id.as_deref());
        let pick = find_explicit_spawning_call(items, &claimed, &child.link_ids)
            .inspect(|index| {
                tracing::debug!(
                    "{LOG_PREFIX} explicit link child_ids={:?} call_index={index}",
                    child.link_ids
                );
            })
            .or_else(|| {
                find_exact_spawning_call(
                    items,
                    &claimed,
                    start,
                    end,
                    child.task_id.as_deref(),
                    workspace_dir,
                )
            })
            .or_else(|| find_spawning_call(items, &claimed, start, end, child.agent_id.as_deref()));
        let (position, call) = match pick {
            Some(index) => {
                claimed[index] = true;
                (index + 1, Some(index))
            }
            None => (end, None),
        };
        let (call_id, call_status, call_result) = match call.and_then(|i| items.get(i)) {
            Some(DisplayItem::ToolCall {
                call_id,
                status,
                result,
                ..
            }) => (Some(call_id.clone()), Some(*status), result.clone()),
            _ => (None, None, None),
        };
        let status = derive_status(child.own_state, call_status, call_result.as_deref());
        if let DisplayItem::Subagent {
            id,
            call_id: call_slot,
            status: status_slot,
            request_id: request_slot,
            ..
        } = &mut child.item
        {
            tracing::debug!(
                "{LOG_PREFIX} subagent id={id} agent={:?} request_id={request_id:?} call_id={call_id:?} status={status:?}",
                child.agent_id
            );
            *call_slot = call_id;
            *status_slot = status;
            *request_slot = request_id;
        }
        inserts.push((position, order, child.item));
    }
    // Back-to-front keeps earlier positions valid; for one position, the
    // later child is inserted first so spawn order is preserved.
    inserts.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    for (position, _, item) in inserts {
        items.insert(position.min(items.len()), item);
    }
}

/// Terminal state from the spawning call's outcome and the child's own
/// transcript. A failed or incomplete delegation wins; then the child's own
/// ending; then a settled synchronous call.
fn derive_status(
    own: OwnState,
    call_status: Option<ToolCallStatus>,
    call_result: Option<&str>,
) -> TranscriptSubagentStatus {
    let result = call_result.map(str::trim_start).unwrap_or_default();
    if is_incomplete_result(result) {
        return TranscriptSubagentStatus::Incomplete;
    }
    if call_status == Some(ToolCallStatus::Error) {
        return TranscriptSubagentStatus::Failed;
    }
    match own {
        OwnState::Interrupted => TranscriptSubagentStatus::Interrupted,
        OwnState::Completed => TranscriptSubagentStatus::Completed,
        OwnState::Unknown
            if call_status == Some(ToolCallStatus::Success)
                && !result.starts_with(ASYNC_ACCEPTED_PREFIX) =>
        {
            TranscriptSubagentStatus::Completed
        }
        OwnState::Unknown => TranscriptSubagentStatus::Running,
    }
}

/// Whether a spawn result reports the run incomplete: the typed
/// `{"status": "incomplete"}` of the harness's own job payload (which always
/// names its `job_id` or `subagent_run_id`), or the legacy text marker.
/// Foreign JSON that merely has a `status` key is not trusted.
fn is_incomplete_result(result: &str) -> bool {
    if result.starts_with(INCOMPLETE_MARKER) {
        return true;
    }
    // Cheap precheck: most results are prose and never reach the JSON parser.
    if !result.starts_with('{') || !result.contains("incomplete") {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(result)
        .ok()
        .is_some_and(|value| {
            value.get("status").and_then(|s| s.as_str()) == Some("incomplete")
                && (value.get("job_id").is_some() || value.get("subagent_run_id").is_some())
        })
}

/// `[start, end)` of `request_id`'s items (after its boundary, up to the next
/// one); the whole list when the turn is unknown.
fn turn_range(items: &[DisplayItem], request_id: Option<&str>) -> (usize, usize) {
    let Some(request_id) = request_id else {
        return (0, items.len());
    };
    let Some(boundary) = items.iter().position(
        |item| matches!(item, DisplayItem::TurnBoundary { request_id: rid } if rid == request_id),
    ) else {
        return (0, items.len());
    };
    let end = items[boundary + 1..]
        .iter()
        .position(|item| matches!(item, DisplayItem::TurnBoundary { .. }))
        .map_or(items.len(), |offset| boundary + 1 + offset);
    (boundary + 1, end)
}

fn find_spawning_call(
    items: &[DisplayItem],
    claimed: &[bool],
    start: usize,
    end: usize,
    agent_id: Option<&str>,
) -> Option<usize> {
    let candidates = || {
        (start..end).filter_map(|index| match &items[index] {
            DisplayItem::ToolCall { name, args, .. } if !claimed[index] => {
                Some((index, name.as_str(), args.as_ref()))
            }
            _ => None,
        })
    };
    if let Some(agent_id) = agent_id
        && let Some((index, ..)) =
            candidates().find(|(_, name, args)| call_targets_agent(name, *args, agent_id))
    {
        return Some(index);
    }
    candidates()
        .find(|(_, name, _)| is_delegation_tool(name))
        .map(|(index, ..)| index)
}

/// Whether a tool call names `agent_id` as its delegation target.
fn call_targets_agent(name: &str, args: Option<&serde_json::Value>, agent_id: &str) -> bool {
    let agent = agent_id.to_ascii_lowercase();
    let name = name.to_ascii_lowercase();
    if name == format!("delegate_{agent}") || name == format!("delegate_to_{agent}") {
        return true;
    }
    let named_in_args = args
        .and_then(serde_json::Value::as_object)
        .is_some_and(|args| {
            TARGET_ARG_KEYS.iter().any(|key| {
                args.get(*key)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|value| value.eq_ignore_ascii_case(&agent))
            })
        });
    if named_in_args {
        return true;
    }
    // Alias tools such as `plan` → `planner`. The length floor keeps a
    // short generic tool name from matching an agent by accident.
    let stripped = name
        .strip_prefix("delegate_to_")
        .or_else(|| name.strip_prefix("delegate_"))
        .unwrap_or(&name);
    stripped.len() >= 5 && agent.starts_with(stripped)
}

fn is_delegation_tool(name: &str) -> bool {
    name.starts_with("delegate") || name.starts_with("spawn_")
}

/// The turns' `(request_id, commit unix)` pairs, in file order: the last
/// parseable timestamp of each `request_id` run.
///
/// Every stamped row of a turn carries the turn's *commit* time (the writer
/// stamps it when the turn is appended), so this is when the turn ended, not
/// when it began.
pub(super) fn turn_segments(records: &[DisplayRecord]) -> Vec<(String, i64)> {
    let mut segments: Vec<(String, i64)> = Vec::new();
    for record in records {
        let DisplayRecord::Message(msg) = record else {
            continue;
        };
        let (Some(rid), Some(ts)) = (msg.request_id.as_deref(), msg.ts.as_deref()) else {
            continue;
        };
        let Some(unix) = parse_rfc3339_unix(ts) else {
            continue;
        };
        match segments.last_mut() {
            Some((last, end)) if last == rid => *end = (*end).max(unix),
            _ => segments.push((rid.to_string(), unix)),
        }
    }
    segments
}

/// Extract a sub-agent's spawn unix timestamp (seconds) from its stem suffix
/// (`{unix}_{nanos}_{agent}…`). `None` for non-numeric legacy stems.
fn child_spawn_unix(stem_suffix: &str) -> Option<i64> {
    stem_suffix
        .split('_')
        .next()
        .and_then(|s| s.parse::<i64>().ok())
}

fn parse_rfc3339_unix(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|dt| dt.timestamp())
}

/// Anchor a sub-agent to the turn that was running at `child_unix`: the first
/// turn whose commit time is at or after the spawn.
///
/// Fallbacks: no segments → `None` (unanchored); unknown spawn time, or a
/// spawn after every recorded commit (a turn still in flight) → the newest
/// turn.
fn anchor_request_id(child_unix: Option<i64>, segments: &[(String, i64)]) -> Option<String> {
    let last = segments.last()?;
    let Some(child_unix) = child_unix else {
        return Some(last.0.clone());
    };
    segments
        .iter()
        .find(|(_, end)| *end >= child_unix)
        .or(Some(last))
        .map(|(rid, _)| rid.clone())
}
