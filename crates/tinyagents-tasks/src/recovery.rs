//! Parent-facing restart-recovery note for interrupted child tasks.
//!
//! After a restart, [`reconcile_orphaned_tasks`](crate::reconcile_orphaned_tasks)
//! settles every task whose executor died. That keeps the store honest but tells
//! the *parent agent* nothing: its children vanished mid-flight, possibly after
//! a tool call already changed the outside world. This module turns the sweep's
//! report into a prompt-ready note listing what was interrupted and how to
//! reconcile it.
//!
//! It is pure: it reads a [`ReconcileReport`], touches no store, and **never
//! relaunches anything**. Interrupted work is re-run only if the parent decides
//! to, after reconciling against saved results and verifying uncertain side
//! effects — see [`RESTART_RECOVERY_INSTRUCTION`].
//!
//! Wiring the note into a host's next parent turn is the host's job.

use serde::Serialize;

use crate::reconcile::{ReconcileOutcome, ReconcileReport, task_status_label};
use crate::types::{OrchestrationTaskKind, OrchestrationTaskRecord};

/// Most children rendered in one note; the remainder is summarised as `+N more`.
pub const MAX_RECOVERY_CHILDREN: usize = 32;

/// Longest label or interrupted reason (in `char`s) kept in a roster row.
pub const MAX_RECOVERY_LABEL_CHARS: usize = 256;

/// Instruction appended to every non-empty note.
pub const RESTART_RECOVERY_INSTRUCTION: &str = "Interrupted child tasks are not automatically relaunched. \
Reconcile every listed task against its saved results, current status, and the user's requested outcome. \
Prefer using a saved result when it already satisfies the task; otherwise continue it, assign replacement \
work, or finish it yourself. Before continuing, confirm the previous execution has stopped and verify \
uncertain side effects (files written, messages sent, commands run) rather than blindly re-running them. \
Do not duplicate work that is still running. A restart interruption alone is not a blocker: finish the \
original task or report the specific remaining blocker that needs user input.";

/// One interrupted child as shown to the parent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryChild {
    /// The task's id.
    pub task_id: String,
    /// Stable kind label (`sub_agent`, `graph`, `tool`, `external_process`).
    pub kind: String,
    /// Human label: the task's `label` metadata, else the kind's target
    /// (agent name, graph id, tool name). Truncated to
    /// [`MAX_RECOVERY_LABEL_CHARS`].
    pub label: String,
    /// Status the task held when the executor died (`running`, `awaiting`, ...).
    pub last_status: String,
    /// Why it was interrupted, as supplied by the host's reconcile reason.
    /// Truncated to [`MAX_RECOVERY_LABEL_CHARS`].
    pub interrupted_reason: String,
}

fn truncate_chars(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

fn target_of(kind: &OrchestrationTaskKind) -> &str {
    match kind {
        OrchestrationTaskKind::Graph { graph_id } => graph_id.as_str(),
        OrchestrationTaskKind::SubAgent { agent } => agent,
        OrchestrationTaskKind::Tool { tool } => tool,
        OrchestrationTaskKind::ExternalProcess { label } => label,
    }
}

/// Selects the children a parent should reconcile from a sweep `report`,
/// ordered by creation time then task id.
///
/// Tasks settled as cancelled are excluded: a recorded cancellation is intent,
/// not an interruption. Tasks the sweep failed to transition are included,
/// since they are unfinished and unverified. A failed task reports the reason the
/// sweep actually persisted (`ReconciledTask::recorded_reason`), so the note
/// matches the store even when the host's closure is not repeatable; `reason`
/// is evaluated only for tasks the sweep did not record one for.
pub fn recovery_children(
    report: &ReconcileReport,
    reason: &dyn Fn(&OrchestrationTaskRecord) -> String,
) -> Vec<RecoveryChild> {
    let mut tasks: Vec<_> = report
        .tasks
        .iter()
        .filter(|task| !matches!(task.outcome, ReconcileOutcome::Cancelled))
        .collect();
    tasks.sort_by(|a, b| {
        a.record
            .created_at
            .cmp(&b.record.created_at)
            .then_with(|| a.task_id.as_str().cmp(b.task_id.as_str()))
    });
    let children: Vec<RecoveryChild> = tasks
        .into_iter()
        .map(|task| {
            let spec = &task.record.spec;
            let label = spec
                .metadata
                .get("label")
                .map(|label| label.trim())
                .filter(|label| !label.is_empty())
                .unwrap_or_else(|| target_of(&spec.kind));
            RecoveryChild {
                task_id: task.task_id.as_str().to_owned(),
                kind: spec.kind.as_str().to_owned(),
                label: truncate_chars(label, MAX_RECOVERY_LABEL_CHARS),
                last_status: task_status_label(task.prior_status).to_owned(),
                interrupted_reason: truncate_chars(
                    &task
                        .recorded_reason
                        .clone()
                        .unwrap_or_else(|| reason(&task.record)),
                    MAX_RECOVERY_LABEL_CHARS,
                ),
            }
        })
        .collect();
    tracing::debug!(
        interrupted = children.len(),
        "[orchestration] selected restart-recovery children"
    );
    children
}

/// Renders the recovery note for `children`, or `""` when there are none.
///
/// The roster is capped at [`MAX_RECOVERY_CHILDREN`] rows with a `+N more`
/// line for the rest. Rows are JSON inside a `<child_task_facts>` block, and
/// every `<` is escaped as `\u003c` so a hostile label cannot close the block
/// and masquerade as instructions.
pub fn build_restart_recovery_note(children: &[RecoveryChild]) -> String {
    if children.is_empty() {
        return String::new();
    }
    let shown = &children[..children.len().min(MAX_RECOVERY_CHILDREN)];
    let roster = serde_json::to_string_pretty(shown)
        .expect("recovery roster serializes")
        .replace('<', "\\u003c");
    let mut note = format!(
        "Interrupted child tasks to reconcile:\n<child_task_facts>\n{roster}\n</child_task_facts>\n"
    );
    let hidden = children.len() - shown.len();
    if hidden > 0 {
        note.push_str(&format!(
            "+{hidden} more interrupted tasks are not shown. Inspect them before completing recovery.\n"
        ));
    }
    note.push_str(RESTART_RECOVERY_INSTRUCTION);
    note
}

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod tests;
