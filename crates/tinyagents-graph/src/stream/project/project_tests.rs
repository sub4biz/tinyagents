use tinyagents_harness::events::AgentEvent;
use tinyagents_harness::ids::{CallId, NodeId, RunId};
use tinyinference_llm::message::MessageDelta;

use super::*;

fn envelope(event: GraphEvent) -> GraphEventEnvelope {
    GraphEventEnvelope {
        run_id: RunId::from("run-1".to_string()),
        task_id: None,
        ns: Vec::new(),
        seq: 0,
        event,
    }
}

#[test]
fn project_graph_event_routes_task_events_to_the_tasks_mode() {
    let event = GraphEvent::NodeStarted {
        node: NodeId::from("n".to_string()),
        step: 1,
    };
    assert!(project_graph_event(&event, &[StreamMode::Tasks]));
    assert!(!project_graph_event(&event, &[StreamMode::Checkpoints]));
    // Debug always sees everything, including narrow-mode events.
    assert!(project_graph_event(&event, &[StreamMode::Debug]));
}

#[test]
fn project_graph_event_lifecycle_events_are_debug_only() {
    let event = GraphEvent::RunStarted {
        run_id: RunId::from("run-1".to_string()),
    };
    assert!(!project_graph_event(&event, &[StreamMode::Tasks]));
    assert!(!project_graph_event(&event, &[StreamMode::Checkpoints]));
    assert!(project_graph_event(&event, &[StreamMode::Debug]));
}

#[test]
fn project_graph_event_routes_checkpoint_events_to_the_checkpoints_mode() {
    let event = GraphEvent::CheckpointSaved {
        checkpoint_id: "ckpt-1".to_string().into(),
        step: Some(2),
    };
    assert!(project_graph_event(&event, &[StreamMode::Checkpoints]));
    assert!(!project_graph_event(&event, &[StreamMode::Updates]));
}

#[test]
fn stream_projection_folds_model_deltas_into_messages_in_order() {
    let mut projection = StreamProjection::new();
    projection.fold_agent_event(&AgentEvent::ModelDelta {
        run_id: RunId::from("run-1".to_string()),
        call_id: CallId::from("call-1".to_string()),
        delta: MessageDelta::text("hel"),
    });
    projection.fold_agent_event(&AgentEvent::ModelDelta {
        run_id: RunId::from("run-1".to_string()),
        call_id: CallId::from("call-1".to_string()),
        delta: MessageDelta::text("lo"),
    });

    assert_eq!(projection.messages.len(), 2);
    assert_eq!(projection.messages[0].cursor, 1);
    assert_eq!(projection.messages[1].cursor, 2);
    assert_eq!(projection.messages[0].value.delta.text, "hel");
    assert_eq!(projection.messages[1].value.delta.text, "lo");
    assert_eq!(projection.cursor(), 2);
}

#[test]
fn stream_projection_folds_tool_lifecycle_as_two_entries() {
    let mut projection = StreamProjection::new();
    projection.fold_agent_event(&AgentEvent::ToolStarted {
        parent_call_id: None,
        call_id: CallId::from("call-1".to_string()),
        tool_name: "search".into(),
        input: None,
    });
    projection.fold_agent_event(&AgentEvent::ToolCompleted {
        parent_call_id: None,
        call_id: CallId::from("call-1".to_string()),
        tool_name: "search".into(),
        started_at_ms: None,
        input: None,
        output: None,
        duration_ms: None,
        output_bytes: None,
        error: None,
        metadata: None,
    });

    assert_eq!(projection.tool_calls.len(), 2);
    assert_eq!(projection.tool_calls[0].value.phase, ToolCallPhase::Started);
    assert_eq!(
        projection.tool_calls[1].value.phase,
        ToolCallPhase::Completed
    );
    assert_eq!(projection.tool_calls[0].value.call_id.as_str(), "call-1");
}

#[test]
fn stream_projection_folds_failed_tool_completion_as_failed_phase() {
    let mut projection = StreamProjection::new();
    projection.fold_agent_event(&AgentEvent::ToolCompleted {
        parent_call_id: None,
        call_id: CallId::from("call-1".to_string()),
        tool_name: "search".into(),
        started_at_ms: None,
        input: None,
        output: None,
        duration_ms: None,
        output_bytes: None,
        error: Some("boom".into()),
        metadata: None,
    });
    assert_eq!(
        projection.tool_calls[0].value.phase,
        ToolCallPhase::Failed {
            error: "boom".into()
        }
    );
}

#[test]
fn stream_projection_folds_subgraph_events_as_subagents() {
    let mut projection = StreamProjection::new();
    projection.fold_graph_event(&envelope(GraphEvent::SubgraphStarted {
        node: NodeId::from("researcher".to_string()),
        namespace: vec!["researcher".into()],
    }));
    projection.fold_graph_event(&envelope(GraphEvent::SubgraphCompleted {
        node: NodeId::from("researcher".to_string()),
        namespace: vec!["researcher".into()],
    }));

    assert_eq!(projection.subagents.len(), 2);
    assert_eq!(projection.subagents[0].value.name, "researcher");
    assert_eq!(projection.subagents[0].value.phase, SubagentPhase::Started);
    assert_eq!(
        projection.subagents[1].value.phase,
        SubagentPhase::Completed
    );
}

#[test]
fn stream_projection_since_replays_only_items_after_the_given_cursor() {
    let mut projection = StreamProjection::new();
    // Cursors start at one, while zero is the empty-projection sentinel. A
    // snapshot can therefore be passed directly to `since` without dropping
    // the first event subsequently folded.
    projection.fold_agent_event(&AgentEvent::ToolStarted {
        parent_call_id: None,
        call_id: CallId::from("call-1".to_string()),
        tool_name: "search".into(),
        input: None,
    });
    let cursor_after_first = projection.cursor();
    projection.fold_agent_event(&AgentEvent::ModelDelta {
        run_id: RunId::from("run-1".to_string()),
        call_id: CallId::from("call-2".to_string()),
        delta: MessageDelta::text("hi"),
    });
    projection.fold_graph_event(&envelope(GraphEvent::SubgraphStarted {
        node: NodeId::from("n".to_string()),
        namespace: vec!["n".into()],
    }));

    let replay = projection.since(0);
    assert_eq!(replay.len(), 3, "every item after the empty cursor");

    let replay = projection.since(cursor_after_first);
    assert_eq!(replay.len(), 2, "everything after the first item");
    assert!(matches!(replay[1], ProjectedSince::Subagent(_)));

    // Nothing new since the last item's own cursor.
    assert!(projection.since(projection.cursor()).is_empty());
}
