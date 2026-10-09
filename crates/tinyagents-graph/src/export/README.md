# graph::export

Graph export and visualization — the introspection surface that lets a
recursive harness read back the shape of any graph, including one a model
authored or assembled at runtime.

Graphs in this runtime can be built by hand (`GraphBuilder`) or assembled at
runtime. This module reduces both sources to one inspectable, behavior-free
description — `GraphTopology` — so a graph can be diffed, snapshotted in
tests, or drawn for a human reviewing what an agent just constructed. It
implements the spec's "graph serialization to JSON" and "Mermaid export"
future features (see `docs/modules/graph/visualization-testkit.md`).

## Where topology comes from

`GraphTopology` can be extracted from two sources, both yielding the same
shape so visualization and test snapshots share one truth:

- `crate::CompiledGraph::topology` — a validated, frozen graph.
- `crate::GraphBuilder::topology` — a graph still under construction (entry
  may be unresolved).

None of these expose runnable behavior (handler/router closures, reducers);
only structure is captured, which is what makes `GraphTopology` cheap to
clone, `serde`-serializable, and safe to compare across builds.

## Public surface

- `GraphTopology` — the top-level, `serde`-serializable structural
  description of a graph: nodes, edges, conditional routes, barrier edges,
  channel bindings, a policy summary, and a validation report. All
  collections are stored in stable sorted order so exports are
  deterministic regardless of `HashMap` iteration order.
- `NodeInfo` — one node's structural metadata (kind, routing flags, `goto`
  hints, metadata, derived `NodePolicySummary`).
- `EdgeInfo` / `ConditionalEdgeInfo` / `RouteInfo` — direct and
  router-resolved edges.
- `WaitingEdgeInfo` — a barrier/fan-in join: `target` activates only once
  every node in `predecessors` has completed.
- `ChannelInfo` — a state-channel-to-reducer binding (empty for compiled
  whole-state graphs).
- `GraphPolicySummary` / `NodePolicySummary` — derived, graph- and
  node-level execution policy summaries computed from the topology itself.
- `ValidationReport` — structural errors (dangling entry/edge/route/barrier
  targets) and non-fatal warnings (unreachable or dead-end nodes) computed
  over a topology.
- `to_json` / `from_json` — pretty JSON round-trip for `GraphTopology`.
- `to_mermaid` — deterministic [Mermaid](https://mermaid.js.org/) `flowchart`
  rendering of a topology.

## Files

| File | Role |
| --- | --- |
| `types.rs` | The `GraphTopology` shape and its constituent structs; no behavior, `serde`-only. |
| `mod.rs` | Topology extraction (`build_topology`, `node_parts`), structural `validate`ation, JSON round-trip, and Mermaid rendering. |
| `mod_tests.rs` | Unit tests covering topology extraction from built/compiled graphs, JSON round-tripping, Mermaid output, and validation errors/warnings. |

## Operational constraints

- A `GraphTopology` never carries runnable behavior — no handler closures,
  router closures, or reducers. Anything derived from a topology (JSON,
  Mermaid, diffs) is therefore safe to log, snapshot in tests, or show to an
  end user without leaking implementation closures.
- All topology collections are sorted before being handed back
  (`build_topology` normalizes ordering), so two extractions of the same
  logical graph produce byte-identical JSON and Mermaid output — this is
  what makes snapshot testing viable.
- `ValidationReport` is computed structurally from the topology itself, not
  re-derived from the live graph; a topology extracted from a `CompiledGraph`
  is typically clean (compilation already validated it), while a topology
  extracted from an in-progress `GraphBuilder` may surface real
  in-progress issues (e.g. an unresolved entry).
- `START`/`END` are virtual boundaries and are never treated as undeclared
  nodes by `validate`, even though they don't appear in `GraphTopology::nodes`.
