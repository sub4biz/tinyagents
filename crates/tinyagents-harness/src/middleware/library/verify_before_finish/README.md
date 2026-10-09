# Verify before finish

`VerifyBeforeFinishMiddleware` can hold an eligible draft final answer for one
model check. It appends the check as a user turn, preserving the conversation's
prefix. The follow-up answer may call tools; the check fires at most once per
run. Hosts opt in and choose a tool-round threshold or a custom trigger.

`mod.rs` exposes the small public surface. `types.rs` defines activity, trigger,
and per-run state. `middleware.rs` contains configuration and lifecycle hooks;
`middleware_tests.rs` exercises those hooks and the run loop.

The check is skipped for tool-bearing, empty, truncated, or already continued
responses and when call or wall-clock budget is too small. The check asks the
agent loop for reasoning on the call it holds the answer for
(`RunContext::request_reasoning`, a no-op unless the loop's fallback has switched
reasoning off after dead calls): a result fitted on the wrong axis is caught
by asking what the request implied, which a model running without reasoning
(the loop's fallback after dead calls) does not do. Successful and
failed runs release activity through lifecycle hooks. Deferred runs put their
tool activity and check status in `DeferredToolRequests::resume_metadata`.
Hosts constructing `DeferredToolResults` manually must copy it with
`with_resume_metadata(&requests)`; `approve_all()` copies it automatically.
On resume the middleware restores that state even if compaction removed the
tool calls or the check message. Legacy results without metadata recover only
activity visible in the transcript. Interrupted runs are pruned once their
contexts have been dropped; active runs retain state beyond 1,024 entries.
