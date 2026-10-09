# harness::retriever

Provider-neutral retrieval contracts used by agent context composition.

## Why this exists

The harness does not construct embedding providers, own credentials, or
decide memory/source scope — those are host concerns, typically backed
directly by `tinyinference_embeddings` or a host memory adapter. What the
harness needs is a narrow, stable seam: given a query, get back ranked
documents, and render them into a prompt section. This module is exactly that
seam and nothing more.

## Public surface

- [`Retriever`] — the provider- and storage-neutral async trait. A single
  method, `retrieve(RetrievalRequest) -> Result<Vec<RetrievedDocument>>`,
  returning at most `request.limit` documents in descending relevance order.
- [`RetrievalRequest`] — query text, result limit, opaque caller metadata, and
  a [`crate::CancellationToken`] for cooperative cancellation of expensive
  retrieval work.
- [`RetrievedDocument`] — one ranked result: id, model-facing content, score,
  and source metadata (for citations / host projection).
- [`compose_retrieval_context`] — the one entry point most callers need: drives
  a `Retriever` and renders the result as a [`crate::prompt::PromptSection`]
  ready to insert into an agent's composed prompt.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `compose_retrieval_context`, the sole free function tying the trait to prompt composition. |
| `types.rs` | `RetrievalRequest`, `RetrievedDocument`, and the `Retriever` trait. |
| `mod_tests.rs` | A recording fake `Retriever` exercising cancellation, ordering, and metadata round-tripping. |

## Operational constraints

- Authorization and source scope are the host's responsibility *before* a
  `RetrievalRequest` is constructed — the trait itself performs no filtering.
- `compose_retrieval_context` preserves whatever order the `Retriever`
  returns; it does not re-rank or deduplicate.
- Cancellation is cooperative: an implementation should check
  `RetrievalRequest::cancellation` and return
  [`crate::TinyAgentsError::Cancelled`] promptly rather than ignoring it.
