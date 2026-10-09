# harness::store::namespaced

A hierarchical, TTL-aware, batch-oriented long-term store — the richer
sibling of the flat [`crate::store::Store`] trait.

## Why this exists next to `Store`

The flat `Store` trait is get/put/delete/list over a flat `&str` namespace:
enough to key values by bucket, not enough for a real long-term memory layer.
There is no way to ask for everything under `users/alice`, no way to filter
by a field, no way to enumerate what namespaces exist, no expiry (so a store
only ever grows), and no way to issue several reads as one round trip.
[`NamespacedStore`] adds all of that. It does **not** replace `Store` — the
flat trait keeps working unchanged, and [`FlatNamespacedStore`] adapts any
`NamespacedStore` back to it, so existing callers are untouched.

## Public surface

- [`NamespacedStore`] — the trait. [`NamespacedStore::batch`] is the single
  **required** method; `get`, `put`, `put_with_ttl`, `delete`, `search`, and
  `list_namespaces` are default-provided convenience wrappers that each submit
  a one-operation batch, so a backend that implements `batch` well gets every
  other method correct by construction. `ttl_config()` and `sweep_expired()`
  round out the trait (both have no-op-ish defaults).
- [`InMemoryNamespacedStore`] — the bundled in-process implementation, with
  working TTL. `new()`/`with_ttl(TtlConfig)` construct it.
- [`FlatNamespacedStore<S>`] — wraps any `NamespacedStore` and implements the
  flat `Store` trait over it (the flat namespace becomes a single-segment
  `Namespace`).
- [`Namespace`] — an ordered tuple of path segments (`Namespace::new`,
  `segments`, `starts_with`, `ends_with`, `validate`); validated (non-empty,
  no segment containing `.`, first segment not the reserved `langgraph` root).
- [`Item`] — a stored value with its address and timestamps
  (`created_at_ms`/`updated_at_ms`/`expires_at_ms`); `is_expired(now_ms)`.
- [`TtlConfig`] — `default_ttl_minutes` (`None` = never expires) and
  `refresh_on_read` (sliding-window vs. hard retention).
- [`FilterOp`] — one comparison in a `SearchQuery` filter (`Eq`, `Ne`, `Gt`,
  `Gte`, `Lt`, `Lte`, `In`, `Exists`), mirroring LangGraph's operator set.
- [`SearchQuery`] / [`ListNamespacesQuery`] — namespace-prefixed search with
  filter/substring-query/pagination, and wildcard (`*`-segment)
  prefix/suffix/depth namespace listing.
- [`StoreOp`] / [`StoreResult`] — the batch request/response vocabulary
  (`Get`, `Put`, `Search`, `ListNamespaces`), with `StoreResult::into_item` /
  `into_items` / `into_namespaces` unwrap helpers.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `NamespacedStore` trait default methods, `InMemoryNamespacedStore` (the full `batch` implementation), `FlatNamespacedStore` adapter. |
| `types.rs` | `Namespace`, `TtlConfig`, `Item`, `FilterOp`, `SearchQuery`, `ListNamespacesQuery`, `StoreOp`, `StoreResult`. |
| `mod_tests.rs` | Namespace validation/matching, CRUD, subtree/filtered/paginated search, wildcard listing, TTL expiry/default, batch-result alignment, and the flat-`Store` adapter. |

## Operational constraints

- **`batch` results are positionally aligned with the request** — result `i`
  answers operation `i`. A backend that cannot guarantee this must not
  implement the trait; `NamespacedStore::get`/`put`/etc. rely on it via the
  private `one()` helper, which errors if a batch of one does not return
  exactly one result.
- Expiry is enforced **on read**, not only by `sweep_expired` — an expired
  item is never observable even if nothing has swept it yet. Sweeping exists
  purely to reclaim space.
- `SearchQuery::query` (free-text) is the seam for vector/semantic search;
  this crate has no embedding dependency, so the bundled backend does a plain
  case-insensitive substring scan. A backend with a real index should
  interpret the query properly and should not be assumed to preserve the
  in-memory backend's ordering.
- The first namespace segment `"langgraph"` is reserved (kept for
  compatibility with stores shared with that ecosystem) and rejected by
  `Namespace::validate`.
