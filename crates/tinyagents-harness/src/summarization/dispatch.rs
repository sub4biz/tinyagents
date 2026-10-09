//! Tracks whether a summarizer reached its provider.
//!
//! Summarizer calls bypass the run context's dispatch marker, so a compaction
//! that fails without usage (a transport error before any metered response)
//! would otherwise leave `TerminalOutcome::provider_started` false although a
//! provider call was made. The middleware scopes each summarization with
//! [`track_dispatch`]; a model-backed summarizer calls [`mark_dispatched`] at
//! the moment it hands the request to its model. A rejection before that point
//! (empty input, validation) never sets the flag.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

tokio::task_local! {
    static DISPATCHED: Arc<AtomicBool>;
}

/// Records that the summarizer in scope is about to dispatch a provider call.
/// A no-op outside a [`track_dispatch`] scope.
pub(crate) fn mark_dispatched() {
    let _ = DISPATCHED.try_with(|flag| {
        flag.store(true, Ordering::Relaxed);
    });
}

/// Runs `fut`, returning its output and whether it dispatched a provider call.
pub(crate) async fn track_dispatch<F: Future>(fut: F) -> (F::Output, bool) {
    let flag = Arc::new(AtomicBool::new(false));
    let out = DISPATCHED.scope(Arc::clone(&flag), fut).await;
    let dispatched = flag.load(Ordering::Relaxed);
    tracing::trace!(dispatched, "[tinyagents::summarize] dispatch tracked");
    (out, dispatched)
}
