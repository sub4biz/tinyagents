# harness::artifacts

Filesystem offload for oversized worker results on long-horizon runs, plus the
fail-closed path resolution and host policy hooks it depends on.

## Why this exists

For minutes-to-hours runs, keeping every worker's full result inline in
context accumulates without bound, and a summary can never restore full
fidelity. The convention instead writes an oversized result to disk under the
agent's artifact root and hands the parent a short pointer (path + abstract)
instead of the payload. Context stays lean; the full artifact is recoverable
with an ordinary file read.

Two directories under the artifact root:

| Directory    | Holds                                                   |
| ------------ | -------------------------------------------------------- |
| `outputs/`   | Deliverables, handed between steps **by path**.          |
| `workspace/` | Scratch, not meant to be handed back.                     |

The convention has two halves, deliberately split: a host-owned **prompt**
half that tells workers to offload on purpose (not in this crate — it names
host tools), and the **harness** half here ([`offload_oversized_result`]),
which fires on every worker outcome so an oversized result is offloaded even
when the worker inlined it anyway.

Every failure mode in this module is soft: the caller falls back to its
original inline payload and whatever summarisation/truncation backstop it
already had.

## Public surface

- [`ArtifactOffload`] (`ops.rs`) — per-run writer holding the resolved
  artifact root, host policies, and the ids used to name artifacts and tag
  logs. `write` / `write_returning_stored` perform a redacted write;
  `resolve` validates a path without writing.
- [`offload_oversized_result`] (`ops.rs`) — the deterministic entry point:
  offloads `output` when it exceeds a threshold, returning the text the
  parent should receive plus the artifact when one was written.
- [`should_offload`], [`effective_offload_threshold`], [`build_abstract`],
  [`render_artifact_pointer`], [`extract_artifact_paths`],
  [`note_artifact_handoff`] (`ops.rs`) — the supporting free functions:
  threshold decision, cap-aware threshold, abstract truncation, pointer
  rendering, pointer parsing, and handoff logging.
- [`resolve_artifact_path`], [`relative_to_root`], [`sanitize_component`]
  (`paths.rs`) — fail-closed path resolution under the convention root and
  the helpers it builds on.
- [`policy`] module (`policy.rs`) — the two host-supplied gates:
  [`ArtifactPathPolicy`] (which paths are off limits) and
  [`ArtifactRedactor`] (what is scrubbed before bytes hit disk), plus the
  permissive [`OpenPathPolicy`] / [`NoRedaction`] defaults used in tests and
  by hosts with nothing to restrict.
- [`ArtifactKind`], [`OffloadedArtifact`], [`OffloadError`] and the
  convention constants ([`OUTPUTS_DIR`], [`SCRATCH_DIR`],
  [`DEFAULT_OFFLOAD_THRESHOLD_BYTES`], [`ABSTRACT_BUDGET_CHARS`],
  [`ARTIFACT_POINTER_PREFIX`]) (`types.rs`) — the inert value types, kept
  dependency-free (std + `thiserror` only) so a host can build against them
  without pulling in the rest of the engine.

## Files

| File         | Role                                                                 |
| ------------ | --------------------------------------------------------------------- |
| `mod.rs`     | Module overview, re-exports.                                          |
| `types.rs`   | `ArtifactKind`, `OffloadedArtifact`, `OffloadError`, constants.        |
| `paths.rs`   | Fail-closed path resolution (`resolve_artifact_path` and helpers).     |
| `policy.rs`  | `ArtifactPathPolicy`, `ArtifactRedactor` traits and their defaults.    |
| `tool_results.rs` | Per-tool-result persistence: `ToolResultArtifactStore` (`new`: under `<action_dir>/artifacts/tool-results/`, relative pointers; `detached`: under `<storage_dir>/tool-results/` outside the working tree, absolute pointers), `apply_per_result_persistence`, `spill_aggregate_tool_results`, paged artifact reads (`read_target`). Host supplies redactor, read/wrapper tool names, read limit, and for a detached store the read grant on its artifact namespace. |
| `tool_results_tests.rs` | Tests for the above, incl. a byte-stable envelope fixture. |
| `ops.rs`     | `ArtifactOffload` writer, `offload_oversized_result`, pointer/handoff plumbing. |
| `mod_tests.rs`, `contract_tests.rs`, `policy_tests.rs`, `request_truncation_tests.rs` | Unit tests: happy path, fallback path, fail-closed hardening. |

## Operational constraints

- `resolve_artifact_path` is lexical only — it cannot see symlinks, because
  the write target usually does not exist yet. `ArtifactOffload::write`
  re-validates the real, symlink-resolved parent directory after
  `create_dir_all` and before the write; that is the first moment a
  pre-existing symlink escape can be detected.
- A `None` redactor stores bytes verbatim. That is a legitimate choice for a
  host with nothing to scrub, but it is a security decision the host must make
  explicitly — this crate cannot supply credential/PII patterns for it.
- Any preview surfaced back into a model's context must be built from the
  **stored** (post-redaction) body, never from the caller's original input.
  `write_returning_stored` and `offload_oversized_result` exist specifically
  so callers have that stored value at hand.
- `ArtifactOffload::with_render_root` matters whenever a worker writes inside
  its own checkout: the parent receiving the pointer does not hold the
  worker's root, so paths must be rendered relative to the *parent's* root
  (or absolute, when the worker's root is not nested inside it).
- Every write and every handoff emits a structured `[artifact]` log line, so a
  run journal shows both ends of a pointer (see `mod.rs` for the exact
  messages).
