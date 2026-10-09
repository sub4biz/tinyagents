//! #6416: the redaction must reach the model, not only the log.
//!
//! The scrubber's behaviour is deliberately NOT relaxed here — a signed
//! magic-link token in an email is a credential, and letting it through would
//! re-open #4453 for the commonest credential shape in mail. What changes is
//! that a scrubbed result now says so, so the agent stops re-fetching content
//! that will be redacted identically every time.
//!
//! The first test is the one that matters: it proves the redaction still
//! happens. A notice that appeared while the scrubbing silently weakened would
//! be the worst outcome this change could have.

use super::*;
use crate::context::{RunConfig, RunContext};
use std::sync::Arc;

use crate::middleware::{BoxToolFuture, MiddlewareStack, ToolBaseCall};
use tinyinference_core::sanitize::scrub_credentials;
use tinyinference_llm::tool::ToolCall;
use tinytools::ToolResult;

/// A magic-link token — exactly the shape QA hit in #6416 — must still be
/// removed. This is the security half of the contract and comes first.
#[test]
fn a_link_token_in_email_content_is_still_redacted() {
    let email = "Click https://acme.example/verify?token=aB3dE5fG7hJ9kL1mN3pQ to sign in.";
    let scrubbed = scrub_credentials(email);

    assert!(
        !scrubbed.contains("aB3dE5fG7hJ9kL1mN3pQ"),
        "the token body must not survive scrubbing: {scrubbed}"
    );
    assert!(
        scrubbed.contains(REDACTION_PLACEHOLDER),
        "a redaction must leave its placeholder: {scrubbed}"
    );
}

/// The notice must not match the scrubber's own patterns, or a second pass
/// would redact the explanation and the model would be told even less than
/// before. Verified rather than reasoned about: `credential` is followed by
/// `]` and a space here, never by `["']?\s*[:=]`.
#[test]
fn the_notice_does_not_scrub_itself() {
    let notice = redaction_notice(2);
    assert_eq!(
        scrub_credentials(&notice),
        notice,
        "the notice must survive the scrubber unchanged, or it would redact its own explanation"
    );
}

/// Scrubbing already-scrubbed content must be a no-op, which is what makes the
/// middleware idempotent: the `scrubbed != content` guard is false on a second
/// pass, so no second notice is appended and the count is not restated.
#[test]
fn scrubbing_is_idempotent_so_a_second_pass_adds_no_second_notice() {
    let email = "Click https://acme.example/verify?token=aB3dE5fG7hJ9kL1mN3pQ to sign in.";
    let once = scrub_credentials(email);
    let twice = scrub_credentials(&once);
    assert_eq!(twice, once, "scrub must be stable on its own output");

    // The value the middleware would hand on: scrubbed body plus the notice.
    let annotated = format!("{once}\n\n{}", redaction_notice(1));
    assert_eq!(
        scrub_credentials(&annotated),
        annotated,
        "a result already carrying the notice must pass through untouched"
    );
}

/// The count is what the model relays to the user, so it must describe this
/// pass rather than every placeholder in the text.
#[test]
fn the_notice_counts_only_what_this_pass_removed() {
    let two = "api_key=aB3dE5fG7hJ9kL1mN3pQ and token=zX9yW8vU7tS6rQ5pO4nM";
    let scrubbed = scrub_credentials(two);
    let removed = scrubbed.matches(REDACTION_PLACEHOLDER).count()
        - two.matches(REDACTION_PLACEHOLDER).count();
    assert_eq!(removed, 2, "both values were redacted: {scrubbed}");

    assert!(redaction_notice(removed).contains("2 value(s)"));
    assert!(
        redaction_notice(removed).contains("do not retry"),
        "the notice must tell the model retrying cannot help — that loop is the defect"
    );
}

/// Ordinary email content with nothing credential-shaped must be untouched, so
/// the notice never appears on a result that lost nothing.
#[test]
fn unremarkable_content_is_left_alone_and_gets_no_notice() {
    let email = "Lunch at 12:30 tomorrow? The meeting room is booked until 2pm.";
    assert_eq!(
        scrub_credentials(email),
        email,
        "nothing here is credential-shaped; a notice would be a lie"
    );
}

/// The behaviour this change exists for: a scrubbed result **carries** the
/// notice. The tests above prove the notice text is well-formed and that the
/// redaction still happens; this proves the two are actually joined, which is
/// the part a caller sees.
#[test]
fn a_scrubbed_result_carries_the_notice_and_the_count() {
    let email = "Click https://acme.example/verify?token=aB3dE5fG7hJ9kL1mN3pQ to sign in.";
    let (annotated, redactions) =
        scrub_with_notice(email).expect("credential-shaped content must be scrubbed");

    assert_eq!(redactions, 1);
    assert!(
        !annotated.contains("aB3dE5fG7hJ9kL1mN3pQ"),
        "the token must not survive: {annotated}"
    );
    assert!(
        annotated.contains("[credential_scrub]"),
        "the model must be told the result was altered: {annotated}"
    );
    assert!(
        annotated.contains("do not retry"),
        "the model must be told retrying cannot help: {annotated}"
    );
}

/// And a result that lost nothing must be left exactly alone — no notice, and
/// no rewrite of the tool's own output.
#[test]
fn an_unscrubbed_result_is_not_annotated_at_all() {
    assert!(
        scrub_with_notice("Lunch at 12:30 tomorrow? Room booked until 2pm.").is_none(),
        "nothing was redacted, so the result must pass through untouched"
    );
}

struct FixedBase(&'static str);

impl ToolBaseCall<(), ()> for FixedBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a RunContext,
        _state: &'a (),
        _call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move { Ok(ToolResult::success(self.0)) })
    }
}

struct FieldsBase;

impl ToolBaseCall<(), ()> for FieldsBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a RunContext,
        _state: &'a (),
        _call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            Ok(ToolResult {
                markdown_formatted: Some("api_key=aB3dE5fG7hJ9kL1mN3pQ".into()),
                follow_up: vec![
                    tinytools::ToolContent::Text {
                        text: "token=zX9yW8vU7tS6rQ5pO4nM".into(),
                    },
                    tinytools::ToolContent::Image {
                        media_type: "image/png".into(),
                        data: tinytools::ImageData::Url(
                            "https://example.test/image?token=aB3dE5fG7hJ9kL1mN3pQ".into(),
                        ),
                    },
                ],
                ..ToolResult::default()
            })
        })
    }
}

#[tokio::test]
async fn scrubs_markdown_and_follow_up_fields() {
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_tool_middleware(Arc::new(CredentialScrubMiddleware::new()));
    let ctx: RunContext = RunContext::new(RunConfig::new("mw-test"), ());
    let result = stack
        .run_wrapped_tool(
            &ctx,
            &(),
            ToolCall::new("c1", "fetch", serde_json::json!({})),
            &FieldsBase,
        )
        .await
        .unwrap()
        .into_result();

    assert!(result.markdown_formatted.is_none());
    let follow_up = match &result.follow_up[0] {
        tinytools::ToolContent::Text { text } => text,
        _ => panic!("expected text follow-up"),
    };
    assert!(!follow_up.contains("zX9yW8vU7tS6rQ5pO4nM"));
    match &result.follow_up[1] {
        tinytools::ToolContent::Image {
            data: tinytools::ImageData::Url(url),
            ..
        } => assert!(!url.contains("aB3dE5fG7hJ9kL1mN3pQ")),
        _ => panic!("expected URL-backed image follow-up"),
    }
}

#[tokio::test]
async fn scrubs_a_bare_alphabetic_password_in_tool_result_fields() {
    let out = run(
        CredentialScrubMiddleware::new(),
        "fetch",
        r#"{"password":"correcthorse"}"#,
    )
    .await;
    assert!(!out.contains("correcthorse"), "{out}");
    assert!(out.contains(REDACTION_PLACEHOLDER), "{out}");
}

async fn run(mw: CredentialScrubMiddleware, tool: &str, body: &'static str) -> String {
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_tool_middleware(Arc::new(mw));
    let ctx: RunContext = RunContext::new(RunConfig::new("mw-test"), ());
    stack
        .run_wrapped_tool(
            &ctx,
            &(),
            ToolCall::new("c1", tool, serde_json::json!({})),
            &FixedBase(body),
        )
        .await
        .unwrap()
        .into_result()
        .output()
}

/// The model-visible result carries the notice, and nothing else changes.
#[tokio::test]
async fn the_middleware_annotates_a_scrubbed_result() {
    let out = run(
        CredentialScrubMiddleware::new(),
        "fetch",
        "Click https://acme.example/verify?token=aB3dE5fG7hJ9kL1mN3pQ to sign in.",
    )
    .await;
    assert!(!out.contains("aB3dE5fG7hJ9kL1mN3pQ"), "{out}");
    assert!(out.contains("[credential_scrub] 1 value(s)"), "{out}");
}

#[tokio::test]
async fn the_middleware_leaves_a_clean_result_exactly_alone() {
    let out = run(
        CredentialScrubMiddleware::new(),
        "fetch",
        "Lunch at 12:30 tomorrow?",
    )
    .await;
    assert_eq!(out, "Lunch at 12:30 tomorrow?");
}

/// A host scrubber is handed the tool name and its verdict is what lands.
#[tokio::test]
async fn a_host_scrubber_sees_the_tool_name() {
    let scrubber: ToolScrubber =
        Arc::new(|tool, content| (tool == "special").then(|| (format!("{content} [handled]"), 0)));
    let handled = run(
        CredentialScrubMiddleware::with_scrubber(scrubber.clone()),
        "special",
        "body",
    )
    .await;
    assert_eq!(handled, "body [handled]");
    let untouched = run(
        CredentialScrubMiddleware::with_scrubber(scrubber),
        "other",
        "body",
    )
    .await;
    assert_eq!(untouched, "body");
}
