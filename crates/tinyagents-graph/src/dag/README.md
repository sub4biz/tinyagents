# graph::dag

Structural validation of a dependency DAG — duplicate ids, dangling edges, and
cycles — over a borrowed node view, independent of any one caller's node
shape or of the graph runtime.

Multiple graph surfaces (task boards, plan steps, and host-defined workflows)
keep re-deriving the same
question: given nodes that each name the nodes they must run after, is the
result a well-formed DAG? Rather than each caller reimplementing Kahn's
algorithm against its own struct, this module takes a borrowed [`DagNode`]
view that any caller can project its own type into for the duration of one
validation call. It touches no graph runtime state and is usable before
anything is compiled or executed.

## Public surface

- `DagNode<'a>` — a borrowed `id` plus the `depends_on` ids it names (edges
  read as `depends_on -> id`: a node runs after everything it names).
  `DagNode::new(id, depends_on)` builds one from an id and any iterator of
  dependency ids.
- `DagIssue` — a structural problem: `DuplicateNode { id }`,
  `UnknownDependency { node, depends_on }`, or `Cycle` (a self-edge counts as
  a cycle). Deliberately narrow: domain rules a host layers on top (e.g. "a
  phase must name at least one agent") are not this module's concern.
- `has_cycle(nodes) -> bool` — the cycle question alone, for a caller that
  already validated ids/edges its own way.
- `validate_dag(nodes) -> Vec<DagIssue>` — duplicates, dangling edges, and
  cycles in one pass, returning every issue found (not just the first) so a
  definition can be shown with all its problems at once. An empty result
  means the input is a well-formed DAG.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `DagNode`, `DagIssue`. |
| `mod.rs` | `has_cycle`, `validate_dag` (Kahn's algorithm, O(V + E)). |
| `mod_tests.rs` | Unit tests (acyclic acceptance, cycles, self-edges, dangling edges, the duplicate-id false-cycle guard). |

## Semantics worth knowing

- **Edges pointing at unknown ids are ignored by the cycle check** and
  reported separately as `DagIssue::UnknownDependency` — a dangling edge
  cannot close a loop, and letting it participate would turn one mistake
  into two errors.
- **Duplicate ids never trip a false cycle.** The cycle graph is built from
  exactly one declaration per id (first declaration wins); reachability is
  compared against the unique-node count, not the input length.
- **A self-edge is a cycle.** A host that wants to surface self-dependency as
  its own diagnostic must check for it before calling.
- A caller that wants to add dependency edges to a node id that already
  exists must merge those edges into that node's single `DagNode`
  declaration — a second `DagNode` with the same id has its edges silently
  dropped by the dedupe (first declaration wins), so a real cycle it would
  have introduced can go undetected. This is unreachable for a caller that
  only ever mints fresh ids, but is a live footgun for one that revisits an
  existing id.

## How it fits together

`graph::dag` has no dependency on the rest of `graph`; it is a leaf utility
re-exported at the crate root for graph modules and hosts that need to project
their own node types into `DagNode` views before compiling or scheduling them.
