//! Tests for the host policy gate and approval resolver seam.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::middleware::{
    BoxToolFuture, MiddlewareStack, MiddlewareToolOutcome, ToolBaseCall, ToolMiddleware,
};
use tinyinference_llm::tool::ToolCall;
use tinytools::ToolResult;

/// Host context stand-in: the policy reads the channel from it.
struct Host {
    channel: &'static str,
}

fn ctx() -> RunContext<Host> {
    RunContext::new(RunConfig::new("gate"), Host { channel: "web" })
}

fn call(name: &str) -> ToolCall {
    ToolCall {
        id: "call-1".to_string(),
        name: name.to_string(),
        arguments: serde_json::json!({ "to": "a@b.c" }),
        invalid: None,
    }
}

struct Base {
    runs: Arc<Mutex<usize>>,
    fail: bool,
}

impl ToolBaseCall<(), Host> for Base {
    fn call<'a>(
        &'a self,
        _ctx: &'a RunContext<Host>,
        _state: &'a (),
        _call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            *self.runs.lock().unwrap() += 1;
            Ok(if self.fail {
                ToolResult::error("boom")
            } else {
                ToolResult::success("ran")
            })
        })
    }
}

/// Policy keyed on tool name; `channel_deny` also reads the host context.
struct NamePolicy;

#[async_trait]
impl ToolCallPolicy<Host> for NamePolicy {
    fn name(&self) -> &str {
        "name_policy"
    }

    async fn check(&self, ctx: &RunContext<Host>, call: &ToolCall) -> PolicyDecision {
        match call.name.as_str() {
            "rm" => PolicyDecision::deny("destructive"),
            "send" => PolicyDecision::require_approval("sends mail"),
            "web_only" if ctx.data.channel != "web" => PolicyDecision::deny("wrong channel"),
            _ => PolicyDecision::Allow,
        }
    }
}

#[derive(Default)]
struct Resolver {
    verdict: Mutex<Option<ApprovalResolution>>,
    recorded: Mutex<Vec<(String, bool)>>,
    resolved: Mutex<usize>,
}

#[async_trait]
impl ApprovalResolver<Host> for Resolver {
    async fn requires_approval(&self, _ctx: &RunContext<Host>, call: &ToolCall) -> bool {
        call.name == "send"
    }

    async fn resolve(&self, _ctx: &RunContext<Host>, _call: &ToolCall) -> ApprovalResolution {
        *self.resolved.lock().unwrap() += 1;
        self.verdict.lock().unwrap().clone().unwrap()
    }

    fn record(&self, ticket: &str, result: &ToolResult) {
        self.recorded
            .lock()
            .unwrap()
            .push((ticket.to_string(), result.is_error));
    }
}

async fn run(mw: Arc<dyn ToolMiddleware<(), Host>>, name: &str, fail: bool) -> (ToolResult, usize) {
    let runs = Arc::new(Mutex::new(0));
    let base = Base {
        runs: runs.clone(),
        fail,
    };
    let mut stack: MiddlewareStack<(), Host> = MiddlewareStack::new();
    stack.push_tool_middleware(mw);
    let c = ctx();
    let result = stack
        .run_wrapped_tool(&c, &(), call(name), &base)
        .await
        .unwrap()
        .into_result();
    let n = *runs.lock().unwrap();
    (result, n)
}

#[test]
fn decision_helpers_expose_the_blocking_reason() {
    assert_eq!(PolicyDecision::Allow.blocking_reason(), None);
    assert_eq!(
        PolicyDecision::require_approval("ask").blocking_reason(),
        Some("ask")
    );
    assert_eq!(PolicyDecision::deny("no").blocking_reason(), Some("no"));
    assert_eq!(
        PolicyDecision::deny("no"),
        PolicyDecision::Deny {
            reason: "no".to_string()
        }
    );
}

#[tokio::test]
async fn gate_check_passes_the_host_context_and_waives_only_approval() {
    let gate = ToolPolicyGate::new(Arc::new(NamePolicy));
    let mut c = ctx();
    assert_eq!(gate.policy_name(), "name_policy");
    assert_eq!(
        gate.check(&c, &call("read"), false).await,
        PolicyDecision::Allow
    );
    assert_eq!(
        gate.check(&c, &call("send"), false).await,
        PolicyDecision::require_approval("sends mail")
    );
    assert_eq!(
        gate.check(&c, &call("send"), true).await,
        PolicyDecision::Allow
    );
    // A deny is never waived.
    assert_eq!(
        gate.check(&c, &call("rm"), true).await,
        PolicyDecision::deny("destructive")
    );
    // The context parameter reaches the policy.
    assert_eq!(
        gate.check(&c, &call("web_only"), false).await,
        PolicyDecision::Allow
    );
    c.data.channel = "cron";
    assert_eq!(
        gate.check(&c, &call("web_only"), false).await,
        PolicyDecision::deny("wrong channel")
    );
}

#[tokio::test]
async fn policy_middleware_fails_closed_on_deny_and_require_approval() {
    let mw = || Arc::new(ToolPolicyGateMiddleware::new(Arc::new(NamePolicy)));
    let (r, runs) = run(mw(), "rm", false).await;
    assert!(r.is_error);
    assert_eq!(
        r.output(),
        "Tool 'rm' denied by policy 'name_policy': destructive"
    );
    assert_eq!(runs, 0);

    let (r, runs) = run(mw(), "send", false).await;
    assert!(r.is_error);
    assert_eq!(
        r.output(),
        "Tool 'send' requires approval by policy 'name_policy': sends mail"
    );
    assert_eq!(runs, 0);

    let (r, runs) = run(mw(), "read", false).await;
    assert!(!r.is_error);
    assert_eq!(runs, 1);
}

#[tokio::test]
async fn policy_middleware_uses_the_custom_denial_renderer() {
    let mw = Arc::new(
        ToolPolicyGateMiddleware::new(Arc::new(NamePolicy)).with_denial_renderer(Arc::new(
            |c, p, d| format!("{}|{p}|{}", c.name, d.blocking_reason().unwrap()),
        )),
    );
    let (r, _) = run(mw, "rm", false).await;
    assert_eq!(r.output(), "rm|name_policy|destructive");
}

#[tokio::test]
async fn require_approval_with_a_resolver_runs_once_approved_and_records() {
    let resolver = Arc::new(Resolver::default());
    *resolver.verdict.lock().unwrap() = Some(ApprovalResolution::Allow {
        ticket: Some("t-1".to_string()),
    });
    let mw = Arc::new(
        ToolPolicyGateMiddleware::new(Arc::new(NamePolicy))
            .with_approval_resolver(resolver.clone()),
    );
    let (r, runs) = run(mw, "send", true).await;
    assert!(r.is_error);
    assert_eq!(runs, 1);
    assert_eq!(
        *resolver.recorded.lock().unwrap(),
        vec![("t-1".to_string(), true)]
    );
}

#[tokio::test]
async fn require_approval_with_a_resolver_denial_never_runs_the_tool() {
    let resolver = Arc::new(Resolver::default());
    *resolver.verdict.lock().unwrap() = Some(ApprovalResolution::Deny {
        reason: "user said no".to_string(),
    });
    let mw = Arc::new(
        ToolPolicyGateMiddleware::new(Arc::new(NamePolicy))
            .with_approval_resolver(resolver.clone()),
    );
    let (r, runs) = run(mw, "send", false).await;
    assert_eq!(r.output(), "user said no");
    assert!(r.is_error);
    assert_eq!(runs, 0);
    assert!(resolver.recorded.lock().unwrap().is_empty());
}

#[tokio::test]
async fn approval_gate_only_asks_for_calls_the_resolver_flags() {
    let resolver = Arc::new(Resolver::default());
    *resolver.verdict.lock().unwrap() = Some(ApprovalResolution::Allow { ticket: None });
    let mw = || Arc::new(ApprovalGateMiddleware::new("approval", resolver.clone()));

    let (r, runs) = run(mw(), "read", false).await;
    assert!(!r.is_error);
    assert_eq!(runs, 1);
    assert_eq!(*resolver.resolved.lock().unwrap(), 0);

    // Approved without a ticket: runs, records nothing.
    let (r, runs) = run(mw(), "send", false).await;
    assert!(!r.is_error);
    assert_eq!(runs, 1);
    assert_eq!(*resolver.resolved.lock().unwrap(), 1);
    assert!(resolver.recorded.lock().unwrap().is_empty());
    assert_eq!(ToolMiddleware::<(), Host>::name(&*mw()), "approval");
}

#[tokio::test]
async fn approval_gate_records_success_and_failure_for_an_approved_ticket() {
    let resolver = Arc::new(Resolver::default());
    *resolver.verdict.lock().unwrap() = Some(ApprovalResolution::Allow {
        ticket: Some("t-9".to_string()),
    });
    let mw = Arc::new(ApprovalGateMiddleware::new("approval", resolver.clone()));
    run(mw.clone(), "send", false).await;
    run(mw, "send", true).await;
    assert_eq!(
        *resolver.recorded.lock().unwrap(),
        vec![("t-9".to_string(), false), ("t-9".to_string(), true)]
    );
}

#[tokio::test]
async fn evaluate_settles_each_decision_for_a_host_that_interleaves_the_gate() {
    let gate = ToolPolicyGate::new(Arc::new(NamePolicy));
    let resolver = Resolver::default();
    *resolver.verdict.lock().unwrap() = Some(ApprovalResolution::Allow {
        ticket: Some("t-9".to_string()),
    });

    // Allow never consults the resolver.
    assert_eq!(
        gate.evaluate(&ctx(), &call("read"), false, Some(&resolver))
            .await,
        GateVerdict::Proceed { ticket: None }
    );
    assert_eq!(*resolver.resolved.lock().unwrap(), 0);

    // Deny is terminal and carries the policy's own decision.
    assert_eq!(
        gate.evaluate(&ctx(), &call("rm"), false, Some(&resolver))
            .await,
        GateVerdict::Blocked(PolicyDecision::deny("destructive"))
    );

    // RequireApproval settles through the resolver...
    assert_eq!(
        gate.evaluate(&ctx(), &call("send"), false, Some(&resolver))
            .await,
        GateVerdict::Proceed {
            ticket: Some("t-9".to_string())
        }
    );
    assert_eq!(*resolver.resolved.lock().unwrap(), 1);

    // ...fails closed without one...
    assert_eq!(
        gate.evaluate(&ctx(), &call("send"), false, None).await,
        GateVerdict::Blocked(PolicyDecision::require_approval("sends mail"))
    );

    // ...is waivable without asking anyone...
    assert_eq!(
        gate.evaluate(&ctx(), &call("send"), true, Some(&resolver))
            .await,
        GateVerdict::Proceed { ticket: None }
    );
    assert_eq!(*resolver.resolved.lock().unwrap(), 1);

    // ...and a refusal carries the resolver's text.
    *resolver.verdict.lock().unwrap() = Some(ApprovalResolution::Deny {
        reason: "user said no".to_string(),
    });
    assert_eq!(
        gate.evaluate(&ctx(), &call("send"), false, Some(&resolver))
            .await,
        GateVerdict::Refused {
            reason: "user said no".to_string()
        }
    );
}

#[test]
fn record_approved_needs_a_ticket_and_a_result() {
    let resolver = Resolver::default();
    let outcome = MiddlewareToolOutcome::Result(ToolResult::error("boom"));
    record_approved(&resolver, None, &outcome);
    assert!(resolver.recorded.lock().unwrap().is_empty());
    record_approved(&resolver, Some("t-1"), &outcome);
    assert_eq!(
        *resolver.recorded.lock().unwrap(),
        vec![("t-1".to_string(), true)]
    );
}

// ── Concurrent batches: approval prompts are serialised, the rest is not ────

/// Resolver that records how many `resolve` calls overlap. Names starting
/// with `gated` need approval.
struct SlowResolver {
    active: std::sync::atomic::AtomicUsize,
    max_active: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl ApprovalResolver<Host> for SlowResolver {
    async fn requires_approval(&self, _ctx: &RunContext<Host>, call: &ToolCall) -> bool {
        call.name.starts_with("gated")
    }

    async fn resolve(&self, _ctx: &RunContext<Host>, _call: &ToolCall) -> ApprovalResolution {
        use std::sync::atomic::Ordering;
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        ApprovalResolution::Allow { ticket: None }
    }

    fn record(&self, _ticket: &str, _result: &ToolResult) {}
}

/// Policy that requires approval for the `gated*` tools.
struct GatedNamePolicy;

#[async_trait]
impl ToolCallPolicy<Host> for GatedNamePolicy {
    fn name(&self) -> &str {
        "gated_name_policy"
    }

    async fn check(&self, _ctx: &RunContext<Host>, call: &ToolCall) -> PolicyDecision {
        if call.name.starts_with("gated") {
            PolicyDecision::require_approval("needs approval")
        } else {
            PolicyDecision::Allow
        }
    }
}

/// Base that sleeps 100ms and records tool overlap.
struct SleepingBase {
    active: std::sync::atomic::AtomicUsize,
    max_active: std::sync::atomic::AtomicUsize,
}

impl ToolBaseCall<(), Host> for SleepingBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a RunContext<Host>,
        _state: &'a (),
        _call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            use std::sync::atomic::Ordering;
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolResult::success("ran"))
        })
    }
}

/// Runs two approval-requiring calls and one free call concurrently through
/// `middleware`, asserting prompts never overlap while the free call and the
/// post-approval tool runs still do.
async fn assert_approvals_serialise(
    middleware: Arc<dyn ToolMiddleware<(), Host>>,
    resolver: Arc<SlowResolver>,
) {
    use std::sync::atomic::Ordering;
    let mut stack: MiddlewareStack<(), Host> = MiddlewareStack::new();
    stack.push_tool_middleware(middleware);
    let base = SleepingBase {
        active: Default::default(),
        max_active: Default::default(),
    };
    let c = ctx();

    let started = tokio::time::Instant::now();
    let results = futures::future::join_all(["gated_one", "gated_two", "free"].map(|name| {
        let mut tool_call = call(name);
        tool_call.id = format!("call-{name}");
        stack.run_wrapped_tool(&c, &(), tool_call, &base)
    }))
    .await;
    let elapsed = started.elapsed();

    for result in results {
        assert_eq!(result.unwrap().into_result().output(), "ran");
    }
    assert_eq!(
        resolver.max_active.load(Ordering::SeqCst),
        1,
        "approval prompts must never overlap"
    );
    assert!(
        base.max_active.load(Ordering::SeqCst) >= 2,
        "the free call must run alongside the gated ones"
    );
    // Serial would be 2 * (50 + 100) + 100 = 400ms; serialising only the
    // prompt gives 50 + 50 + 100 = 200ms.
    assert!(
        elapsed < std::time::Duration::from_millis(300),
        "only the prompt is serialised, not the whole call; got {elapsed:?}"
    );
}

fn slow_resolver() -> Arc<SlowResolver> {
    Arc::new(SlowResolver {
        active: Default::default(),
        max_active: Default::default(),
    })
}

#[tokio::test(start_paused = true)]
async fn approval_gate_serialises_concurrent_prompts_only() {
    let resolver = slow_resolver();
    let middleware = Arc::new(ApprovalGateMiddleware::new("approval", resolver.clone()));
    assert_approvals_serialise(middleware, resolver).await;
}

#[tokio::test(start_paused = true)]
async fn policy_gate_serialises_concurrent_prompts_only() {
    let resolver = slow_resolver();
    let middleware = Arc::new(
        ToolPolicyGateMiddleware::new(Arc::new(GatedNamePolicy))
            .with_approval_resolver(resolver.clone()),
    );
    assert_approvals_serialise(middleware, resolver).await;
}
