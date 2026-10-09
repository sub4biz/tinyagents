//! TRUE end-to-end, fully OFFLINE: the flagship registry showcase.
//!
//! An *orchestrator* agent designs which specialized sub-agents to run by
//! emitting a structured selection of **names**, those names are resolved out of
//! a [`CapabilityRegistry`] at runtime, the resolved sub-agents are run in
//! parallel, and their outputs are composed into a final answer.
//!
//! Everything is deterministic: the orchestrator and every sub-agent are backed
//! by [`ScriptedModel`]s, so the test asserts on *structure* — which capabilities
//! the registry resolved and ran (testkit [`Trajectory`]) and that the unchosen
//! capability was never resolved or invoked — never on model prose.
//!
//! Topology:
//!
//! ```text
//!   registry: { researcher, coder, summarizer }   (3 named SubAgentTools)
//!         │
//!   orchestrator (structured output) ── selects ──> ["researcher", "summarizer"]
//!         │
//!   resolve each name from the registry, run in parallel, compose
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use serde_json::{Value, json};

use tinyagents_graph::*;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::events::{AgentEvent, EventSink};
use tinyagents_harness::ids::{CallId, RunId};
use tinyagents_harness::middleware::AgentRun;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::testkit::{EventRecorder, ScriptedModel, Trajectory};
use tinyagents_harness::tool::ToolDispatch;
use tinyagents_orchestration::subagent::{
    ChildDataPolicy, SubAgent, SubAgentJobRegistry, SubAgentTool,
};
use tinyagents_registry::*;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ResponseFormat;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Wraps a [`ScriptedModel`] as a named [`SubAgentTool`] whose child run always
/// answers with `answer`.
fn specialist(
    name: &str,
    description: &str,
    model: Arc<ScriptedModel>,
    jobs: SubAgentJobRegistry,
) -> SubAgentTool<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("model", model)
        .set_default_model("model");
    let subagent = SubAgent::new(name, description, Arc::new(harness))
        .with_system_prompt(format!("You are the {name}."));
    SubAgentTool::new(
        Arc::new(subagent),
        ChildDataPolicy::new(|parent: &()| *parent),
    )
    .with_job_registry(jobs)
}

/// Reads the `{ "agents": [..] }` selection out of an [`AgentRun`], preferring
/// extracted structured output and falling back to parsing the raw text.
fn parse_selection(run: &AgentRun) -> Vec<String> {
    let value: Value = run
        .structured
        .clone()
        .or_else(|| run.text().and_then(|t| serde_json::from_str(&t).ok()))
        .unwrap_or(Value::Null);
    value
        .get("agents")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn orchestrator_resolves_and_runs_only_the_chosen_subagents() -> Result<()> {
    let task = "research and summarize Rust trait objects";

    // Keep concrete handles so we can assert *which* sub-agent models were
    // actually invoked (and which were not).
    let researcher_model = Arc::new(ScriptedModel::replies(vec!["RESEARCH_NOTES_ON_TRAITS"]));
    let coder_model = Arc::new(ScriptedModel::replies(vec!["CODE_SNIPPET_DYN_TRAIT"]));
    let summarizer_model = Arc::new(ScriptedModel::replies(vec!["PLAIN_SUMMARY_DONE"]));

    // 1. Register three named specialist sub-agents in the capability registry.
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    let mut dispatches: HashMap<String, Arc<SubAgentTool<()>>> = HashMap::new();
    let jobs = SubAgentJobRegistry::new();
    for (name, description, model) in [
        (
            "researcher",
            "Gathers factual background.",
            researcher_model.clone(),
        ),
        ("coder", "Writes code snippets.", coder_model.clone()),
        (
            "summarizer",
            "Condenses material.",
            summarizer_model.clone(),
        ),
    ] {
        let dispatch = Arc::new(specialist(name, description, model, jobs.clone()));
        registry.register_tool(dispatch.tool())?;
        dispatches.insert(name.to_owned(), dispatch);
    }

    let available = registry.names(ComponentKind::Tool);
    assert_eq!(available, vec!["coder", "researcher", "summarizer"]);

    // 2. Orchestrator designs the plan: a scripted structured selection of TWO
    //    of the three registered names (deliberately skipping `coder`).
    let mut orchestrator: AgentHarness<()> = AgentHarness::new();
    orchestrator
        .register_model(
            "model",
            Arc::new(ScriptedModel::replies(vec![
                r#"{"agents":["researcher","summarizer"]}"#,
            ])),
        )
        .set_default_model("model")
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::json_schema(
                "agent_selection",
                json!({
                    "type": "object",
                    "properties": {
                        "agents": { "type": "array", "items": { "type": "string" } }
                    },
                    "required": ["agents"]
                }),
            )),
            ..RunPolicy::default()
        });

    let plan = orchestrator
        .invoke_default(&(), vec![Message::user(task)])
        .await?;

    // The orchestrator's selection is extracted as structured output.
    assert_eq!(
        plan.structured,
        Some(json!({ "agents": ["researcher", "summarizer"] }))
    );

    let chosen = parse_selection(&plan);
    assert_eq!(chosen, vec!["researcher", "summarizer"]);
    assert!(
        !chosen.iter().any(|n| n == "coder"),
        "the orchestrator did not choose `coder`"
    );

    // 3. Bind at runtime: resolve each chosen name from the registry and run the
    //    resolved sub-agents in parallel, bracketing each dispatch with tool
    //    events on a shared sink so the trajectory is observable.
    let recorder = EventRecorder::new();
    let sink: EventSink = recorder.sink();
    sink.emit(AgentEvent::RunStarted {
        run_id: RunId::new("orchestrator"),
        thread_id: None,
    });

    let dispatches = chosen.iter().enumerate().map(|(i, name)| {
        let name = name.clone();
        let dispatch = dispatches
            .get(&name)
            .cloned()
            .expect("a chosen name must resolve in the registry");
        let sink = sink.clone();
        let call_id = format!("dispatch-{i}");
        async move {
            sink.emit(AgentEvent::ToolStarted {
                parent_call_id: None,
                call_id: CallId::new(call_id.clone()),
                tool_name: name.clone(),
                input: None,
            });
            let parent = RunContext::new(RunConfig::new(format!("dispatch-{i}")), ());
            let result = dispatch
                .invoke_in_parent_context(
                    &(),
                    json!({ "input": task }),
                    tinytools::ToolCallOptions::default(),
                    &parent,
                )
                .await?;
            sink.emit(AgentEvent::ToolCompleted {
                parent_call_id: None,
                call_id: CallId::new(call_id),
                tool_name: name.clone(),
                started_at_ms: None,
                input: None,
                output: None,
                duration_ms: None,
                output_bytes: None,
                error: None,
                metadata: None,
            });
            let job_id = serde_json::from_str::<Value>(&result.output())?
                .get("job_id")
                .and_then(Value::as_str)
                .expect("spawn result contains a job id")
                .to_owned();
            Ok::<(String, String), TinyAgentsError>((name, job_id))
        }
    });
    let spawned: Vec<(String, String)> = join_all(dispatches)
        .await
        .into_iter()
        .collect::<Result<_>>()?;

    let outputs = tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            let completed = spawned
                .iter()
                .filter_map(|(name, job_id)| {
                    let job = jobs.get(job_id)?;
                    job.status
                        .is_terminal()
                        .then(|| (name.clone(), job.output.unwrap_or_default()))
                })
                .collect::<Vec<_>>();
            if completed.len() == spawned.len() {
                return completed;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("chosen subagent jobs reach terminal states");

    sink.emit(AgentEvent::RunCompleted {
        run_id: RunId::new("orchestrator"),
        outcome: None,
    });

    // 4. Compose the resolved sub-agents' outputs into one final answer.
    let composed = outputs
        .iter()
        .map(|(name, text)| format!("[{name}] {text}"))
        .collect::<Vec<_>>()
        .join("\n");

    // ── Assertions ───────────────────────────────────────────────────────────

    // Registry resolution: all three are registered, but only the chosen two
    // were ever resolved into a runnable handle.
    assert!(registry.has(ComponentKind::Tool, "coder"));
    assert!(registry.tool("researcher").is_some());
    assert!(registry.tool("summarizer").is_some());

    // Trajectory: ONLY the chosen sub-agents ran; the unchosen one never did.
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_tool_called("researcher");
    traj.assert_tool_called("summarizer");
    assert_eq!(traj.tool_call_count("researcher"), 1);
    assert_eq!(traj.tool_call_count("summarizer"), 1);
    assert!(
        !traj.tool_was_called("coder"),
        "the unchosen `coder` capability must never run"
    );
    traj.assert_completed();
    traj.assert_order(&["run.started", "run.completed"])
        .expect("dispatch is bracketed by run start/completion");

    // Model-level proof: only the chosen sub-agents' models were invoked.
    assert_eq!(
        researcher_model.requests().len(),
        1,
        "researcher sub-agent ran exactly once"
    );
    assert_eq!(
        summarizer_model.requests().len(),
        1,
        "summarizer sub-agent ran exactly once"
    );
    assert_eq!(
        coder_model.requests().len(),
        0,
        "the unchosen coder sub-agent was never invoked"
    );

    // The composition weaves in BOTH chosen sub-agents' outputs.
    assert!(composed.contains("RESEARCH_NOTES_ON_TRAITS"));
    assert!(composed.contains("PLAIN_SUMMARY_DONE"));
    assert!(
        !composed.contains("CODE_SNIPPET_DYN_TRAIT"),
        "the unchosen sub-agent's output must not appear in the composition"
    );

    Ok(())
}
