# Multimodal Content Blocks

`tinyinference_llm::message::ContentBlock` carries non-text media alongside
text/JSON/thinking/provider-extension blocks, and adapters map it onto each
provider's actual wire format where one exists.

## Types

```rust
pub enum ContentBlock {
    // ...Text, Json, Image, Thinking, RedactedThinking, ProviderExtension...
    Audio(MediaRef),
    Video(MediaRef),
    Document(MediaRef),
}

pub enum MediaRef {
    Url { url: String, media_type: Option<String> },
    Base64 { data: String, media_type: String },
    Path { path: String, media_type: Option<String> },
}
```

The harness never fetches a `MediaRef::Url` or reads a `MediaRef::Path`
itself — resolving either into bytes is a host concern. An SSRF-guarded
downloader for that resolution was scoped out of this pass; a host that needs
one should build it at the boundary where it already owns network egress
policy, then hand the harness a `MediaRef::Base64` (or an `Image`/`ImageRef`
data URI, for images).

`tinyinference_llm::model::Modalities` gained `video_in`, `video_out`, and
`document_in` alongside the existing `image_in`/`image_out`/`audio_in`/
`audio_out`, so a `ModelProfile` can advertise which of these blocks a model
actually accepts.

## Adapter support

Support follows each provider's real wire format rather than a uniform
lowest-common-denominator:

- **OpenAI (Chat Completions).** `Audio(MediaRef::Base64 { data, media_type })`
  maps to an `input_audio` content part (`{"type":"input_audio","input_audio":
  {"data","format"}}`, `format` derived from the MIME subtype). A `Url`/`Path`
  audio reference, and any `Document`/`Video` block, has no Chat Completions
  wire representation and fails closed with `Error::Validation` — the same
  policy the adapter already applies to `ProviderExtension` — rather than
  silently dropping an attachment the caller expected to be sent.
- **Anthropic (Messages).** `Document` maps to a native `document` content
  block (`base64` or `url` source; a `Path` reference becomes a placeholder
  text block, since the harness does not read local files). `Audio` and
  `Video` have no Messages API wire form at all, so both render as a
  placeholder text block noting the omitted attachment
  (`[audio attachment omitted: <descriptor>]`) instead of being dropped.

See `vendor/tinyinference/crates/tinyinference-llm/src/providers/openai/mod_tests.rs`
and `.../providers/anthropic/mod_tests.rs` for the serialization fixtures, and
`.../message/mod_tests.rs` for `MediaRef`/`ContentBlock` round-trip and
accessor tests.

## Harness-side accounting

`tinyagents_harness::context::stats::ContextStatistics` gained a `media: usize`
counter (alongside the existing `images: usize`) that counts `Audio`/`Video`/
`Document` blocks across a transcript, and
`ContentBlock::estimated_char_weight` charges each of them the same flat
`IMAGE_CHAR_WEIGHT` an image gets, so a transcript dominated by non-text
attachments is not under-counted by token-budgeting/compaction gating.

## Generic attachment intake

For uploads that must retain their original media, use the optional
`multimodal` feature's `resolve_attachment` API. It returns decoded bytes plus
display name, MIME, and size without text extraction. `UnknownMimePolicy::Accept`
is the host's explicit choice to admit arbitrary formats; existing
`resolve_file` callers use `Reject` and keep their MIME allowlist. Only gzip
data URIs with `original_mime` are transport-decoded, exactly once. Hosts own
local path authorization, storage, and choice of native `ContentBlock` variants.

ZIP/TAR/TAR.GZ inspection uses `ArchiveFormat::detect` and `inspect_archive`.
Typed listings report names, kinds, declared sizes, and resource truncation;
inspection never extracts files or follows archive paths/links. Office ZIP
containers are detected from standard document parts and remain documents.
See [the module README](../../../crates/tinyagents-harness/src/multimodal/README.md)
for byte/name/entry budgets and partial-validation semantics.
