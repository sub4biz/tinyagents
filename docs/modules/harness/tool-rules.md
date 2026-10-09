# Tool rules

Pattern rules decide which tools a run's model may **see** and **call**. The
vocabulary is [`tinytools::ToolRules`](../../../vendor/tinytools/docs/specs/tool-rules.md);
this page is how the agent loop applies it.

## Where rules come from

A run stacks up to three restrictions. Every one must admit a tool:

1. The hosted definition's exact `tools` allowlist (`AgentDefinition::tools`),
   with its fail-closed default (I-9). Unchanged.
2. `RunPolicy::tool_rules`: a `ToolRulePolicy { rules, context }`. Use this
   for harness-wide rules, and for the `RuleContext` (`channel`, `agent`,
   `origin`, …) that `when` conditions match.
3. `AgentDefinition::tool_rules` on a hosted run, added as another layer.

A layer can only narrow: two allowlists intersect, and a `deny` anywhere wins.

## One gate, every surface

`ToolGate` (crate-private, `tool/rules/`) answers every question the loop asks
about a tool:

| Site | Surface |
|---|---|
| Direct schemas on the request, and mid-run toolset changes | `catalog` |
| The deferred catalogue, the `tool_search` manifest and answers, replayed promotions | `search` |
| Admission of a model call; the unknown-tool "closest available" list | `call` / listing |
| Admission of a nested call | `call` |

A tool a rule removes from the catalogue therefore cannot be found through
`tool_search`, or called under a name the model guessed.

## Calls

- The rules run **before** `before_tool` middleware. An approval middleware
  never asks a human about a call the rules refuse.
- They see the raw provider arguments, which is what the host security gate
  also sees.
- A refused call is answered with a tool error naming the rule, its layer and
  its reason. It frees its tool-call budget slot, as a middleware refusal
  does.
- When a tool reports `Tool::indirect_target(args)` (a connector's execute
  tool, a skill runner), the target is checked as well.
- `require_approval` defers the call exactly as a declared
  `approval_required` would. `auto_approve` waives that declaration. In a
  nested call, both map to the existing nested approval refusal.

## Example

```rust
use tinyagents_harness::runtime::RunPolicy;
use tinyagents_harness::tool::ToolRulePolicy;
use tinytools::{RuleContext, ToolRules};

let rules: ToolRules = serde_json::from_value(serde_json::json!({
    "rules": [
        { "id": "no-mcp", "effect": "deny", "match": { "name": "mcp_*" },
          "except": { "family": "github" } },
        { "effect": "require_approval", "match": { "side_effects": ["payment"] } },
        { "effect": "deny", "match": { "name": "shell" }, "when": { "channel": "telegram" } },
    ],
}))?;
let policy = RunPolicy {
    tool_rules: ToolRulePolicy::new(rules)
        .with_context(RuleContext::new().with("channel", "telegram")),
    ..RunPolicy::default()
};
```

The default `ToolRulePolicy` holds no rules. A harness that never sets it
pays nothing per tool and behaves exactly as before.
