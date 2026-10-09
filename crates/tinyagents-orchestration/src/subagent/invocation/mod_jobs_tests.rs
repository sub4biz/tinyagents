//! Job-control contracts for [`SubAgentTool`]: independent per-job
//! cancellation, parent-cancel cascade, panic safety.

use super::*;
use std::sync::Arc;

use serde_json::json;

use super::test::{BlockedModel, spawned_job_id, wait_for_terminal};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::ids::new_call_id;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::tool::ToolDispatch;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};

pub(super) struct PanickingModel;

#[async_trait::async_trait]
impl ChatModel<()> for PanickingModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        panic!("child model exploded");
    }
}

fn blocked_tool() -> (
    Arc<SubAgentTool<(), ()>>,
    Arc<tokio::sync::Semaphore>,
    Arc<tokio::sync::Semaphore>,
) {
    let started = Arc::new(tokio::sync::Semaphore::new(0));
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let mut harness = AgentHarness::new();
    harness.register_model(
        "worker",
        Arc::new(BlockedModel {
            started: started.clone(),
            release: release.clone(),
        }),
    );
    let tool = Arc::new(SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    ));
    (tool, started, release)
}

async fn spawn(tool: &SubAgentTool<(), ()>, parent: &RunContext<()>) -> String {
    let result = tool
        .invoke_in_parent_context(
            &(),
            json!({"input": "work"}),
            tinytools::ToolCallOptions::default(),
            parent,
        )
        .await
        .expect("spawn succeeds");
    spawned_job_id(&result)
}

async fn cancel_via_tool(
    jobs: &SubAgentJobsTool,
    parent: &RunContext<()>,
    job_id: &str,
) -> anyhow::Result<tinytools::ToolResult> {
    ToolDispatch::<(), ()>::execute(
        jobs,
        &(),
        new_call_id(),
        json!({"action": "cancel", "job_id": job_id}),
        tinytools::ToolCallOptions::default(),
        parent,
    )
    .await
}

#[tokio::test]
async fn cancelling_one_job_leaves_its_sibling_running() {
    let (tool, started, release) = blocked_tool();
    let jobs = tool.job_registry().clone();
    let jobs_tool = SubAgentJobsTool::new(jobs.clone());
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let owner = parent.instance_id();

    let first = spawn(&tool, &parent).await;
    let second = spawn(&tool, &parent).await;
    let _first = started.acquire().await.unwrap();
    let _second = started.acquire().await.unwrap();

    let result = cancel_via_tool(&jobs_tool, &parent, &first)
        .await
        .expect("owner may cancel");
    assert!(result.output().contains("cancelled"));
    let cancelled = wait_for_terminal(&jobs, &first, owner).await;
    assert_eq!(cancelled.status, SubAgentJobStatus::Cancelled);

    assert_eq!(
        jobs.get_owned(&second, owner).unwrap().status,
        SubAgentJobStatus::Running,
        "sibling keeps running"
    );
    assert!(
        !parent.cancellation.is_cancelled(),
        "child cancel must not cancel the parent"
    );

    release.add_permits(1);
    let done = wait_for_terminal(&jobs, &second, owner).await;
    assert_eq!(done.status, SubAgentJobStatus::Completed);
}

#[tokio::test]
async fn cancel_is_owner_checked_and_rejects_terminal_jobs() {
    let (tool, started, release) = blocked_tool();
    let jobs = tool.job_registry().clone();
    let jobs_tool = SubAgentJobsTool::new(jobs.clone());
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let stranger = RunContext::new(RunConfig::new("stranger"), ());

    let job = spawn(&tool, &parent).await;
    let _permit = started.acquire().await.unwrap();

    assert!(
        cancel_via_tool(&jobs_tool, &stranger, &job).await.is_err(),
        "a non-owner cannot cancel"
    );
    assert_eq!(
        jobs.get_owned(&job, parent.instance_id()).unwrap().status,
        SubAgentJobStatus::Running
    );

    release.add_permits(1);
    wait_for_terminal(&jobs, &job, parent.instance_id()).await;
    assert!(
        cancel_via_tool(&jobs_tool, &parent, &job).await.is_err(),
        "terminal jobs cannot be cancelled again"
    );
}

#[tokio::test]
async fn parent_cancellation_cascades_to_running_jobs() {
    let (tool, started, _release) = blocked_tool();
    let jobs = tool.job_registry().clone();
    let parent = RunContext::new(RunConfig::new("parent"), ());

    let job = spawn(&tool, &parent).await;
    let _permit = started.acquire().await.unwrap();
    parent.cancellation.cancel();

    let job = wait_for_terminal(&jobs, &job, parent.instance_id()).await;
    assert_eq!(job.status, SubAgentJobStatus::Cancelled);
}

#[tokio::test]
async fn panicking_child_marks_the_job_failed() {
    let mut harness = AgentHarness::new();
    harness.register_model("worker", Arc::new(PanickingModel));
    let tool = SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    );
    let jobs = tool.job_registry().clone();
    let parent = RunContext::new(RunConfig::new("parent"), ());

    let job_id = spawn(&tool, &parent).await;
    let job = wait_for_terminal(&jobs, &job_id, parent.instance_id()).await;
    assert_eq!(job.status, SubAgentJobStatus::Failed);
    assert!(
        job.error.as_deref().is_some_and(|e| e.contains("panicked")),
        "error explains the panic: {:?}",
        job.error
    );
}

async fn message_via_tool(
    tool: &SubAgentMessageTool,
    parent: &RunContext<()>,
    args: serde_json::Value,
) -> serde_json::Value {
    let result = ToolDispatch::<(), ()>::execute(
        tool,
        &(),
        new_call_id(),
        args,
        tinytools::ToolCallOptions::default(),
        parent,
    )
    .await
    .expect("message accepted");
    serde_json::from_str(&result.output()).expect("JSON result")
}

#[tokio::test]
async fn message_with_a_repeated_request_id_is_queued_once() {
    let jobs = SubAgentJobRegistry::new();
    let tool = SubAgentMessageTool::new(jobs.clone());
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let (job_id, steering) = jobs.create("worker", parent.instance_id());
    let args = json!({"job_id": job_id.as_str(), "message": "look here", "request_id": "r1"});

    let first = message_via_tool(&tool, &parent, args.clone()).await;
    assert_eq!(first["status"], "message_queued");
    assert!(first.get("duplicate").is_none());
    assert_eq!(steering.pending(), 1);

    let again = message_via_tool(&tool, &parent, args).await;
    assert_eq!(again["duplicate"], true);
    assert_eq!(steering.pending(), 1, "duplicate is not enqueued again");

    // A new id and an id-less message are both delivered.
    message_via_tool(
        &tool,
        &parent,
        json!({"job_id": job_id.as_str(), "message": "m", "request_id": "r2"}),
    )
    .await;
    message_via_tool(
        &tool,
        &parent,
        json!({"job_id": job_id.as_str(), "message": "m"}),
    )
    .await;
    assert_eq!(steering.pending(), 3);
}

#[tokio::test]
async fn message_request_ids_are_bounded_per_job() {
    let jobs = SubAgentJobRegistry::new();
    let (job_id, steering) = jobs.create("worker", 1);
    for index in 0..=tinyagents_harness::steering::RecentRequestIds::DEFAULT_CAPACITY {
        let duplicate = jobs
            .send_message_with_request_id(job_id.as_str(), 1, "m", Some(&format!("r{index}")))
            .unwrap();
        assert!(!duplicate);
    }
    // `r0` aged out of the 64-id window, so it is delivered again.
    assert!(
        !jobs
            .send_message_with_request_id(job_id.as_str(), 1, "m", Some("r0"))
            .unwrap()
    );
    assert_eq!(steering.pending(), 66);
}

#[tokio::test]
async fn terminal_jobs_never_return_to_running() {
    let jobs = SubAgentJobRegistry::new();
    let (job_id, _steering) = jobs.create("worker", 1);
    jobs.cancel_owned(job_id.as_str(), 1).expect("cancel");
    jobs.mark_result(&job_id, Err(TinyAgentsError::Cancelled));
    jobs.mark_running(&job_id);
    assert_eq!(
        jobs.get(job_id.as_str()).unwrap().status,
        SubAgentJobStatus::Cancelled
    );
}

#[tokio::test]
async fn settled_jobs_release_their_cancellation_token() {
    let jobs = SubAgentJobRegistry::new();
    let (job_id, _steering) = jobs.create("worker", 1);
    assert!(jobs.holds_live_cancellation(&job_id));
    jobs.cancel_owned(job_id.as_str(), 1).expect("cancel");
    assert!(!jobs.holds_live_cancellation(&job_id));
    assert!(!jobs.get(job_id.as_str()).unwrap().status.is_terminal());
    jobs.mark_result(&job_id, Err(TinyAgentsError::Cancelled));
    assert!(jobs.get(job_id.as_str()).unwrap().status.is_terminal());

    let (aborted, _steering) = jobs.create("worker", 1);
    jobs.mark_aborted(&aborted, true);
    assert!(!jobs.holds_live_cancellation(&aborted));
}

#[tokio::test]
async fn jobs_tool_treats_a_null_action_as_query() {
    let jobs = SubAgentJobRegistry::new();
    let tool = SubAgentJobsTool::new(jobs.clone());
    let parent = RunContext::new(RunConfig::new("parent"), ());
    jobs.create("worker", parent.instance_id());
    let result = ToolDispatch::<(), ()>::execute(
        &tool,
        &(),
        new_call_id(),
        json!({"action": null}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .expect("null action queries");
    let listed: serde_json::Value = serde_json::from_str(&result.output()).unwrap();
    assert_eq!(listed.as_array().map(Vec::len), Some(1));
}

#[tokio::test]
async fn oversized_message_request_ids_are_rejected() {
    let jobs = SubAgentJobRegistry::new();
    let tool = SubAgentMessageTool::new(jobs.clone());
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let (job_id, steering) = jobs.create("worker", parent.instance_id());
    let too_long = "x".repeat(129);
    let rejected = ToolDispatch::<(), ()>::execute(
        &tool,
        &(),
        new_call_id(),
        json!({"job_id": job_id.as_str(), "message": "m", "request_id": too_long}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await;
    assert!(rejected.is_err());
    assert_eq!(steering.pending(), 0);
    let at_limit = message_via_tool(
        &tool,
        &parent,
        json!({"job_id": job_id.as_str(), "message": "m", "request_id": "x".repeat(128)}),
    )
    .await;
    assert_eq!(at_limit["status"], "message_queued");
}

#[tokio::test]
async fn a_budget_overrun_that_raced_an_owner_cancel_settles_cancelled() {
    let jobs = SubAgentJobRegistry::new();
    let (job_id, _steering) = jobs.create("worker", 1);
    jobs.cancel_owned(job_id.as_str(), 1).expect("cancel");
    jobs.mark_budget_overrun(
        &job_id,
        crate::subagent::AppliedResult {
            text: "late".into(),
            ..Default::default()
        },
        "over budget".into(),
    );
    let job = jobs.get(job_id.as_str()).unwrap();
    assert_eq!(job.status, SubAgentJobStatus::Cancelled);
    assert!(job.incomplete_kind.is_none() && job.output.is_none());
}
