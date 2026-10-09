//! Host-owned agent definition vocabulary.
//!
//! This lower-level crate deliberately knows only what a runtime may ask about
//! an agent: its identity, description, declared model/tools/delegates, and a
//! read-only catalogue seam. Authorization, prompt construction, and execution
//! remain with the host and harness.

use std::collections::{HashMap, HashSet};
use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Result returned by a definition catalogue.
pub type Result<T> = std::result::Result<T, DefinitionRegistryError>;

/// A backing catalogue could not answer a definition query.
///
/// This is distinct from [`DefinitionRegistry::resolve`] returning `Ok(None)`,
/// which is the normal absence outcome for an agent omitted by a build or host
/// configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefinitionRegistryError {
    message: String,
}

impl DefinitionRegistryError {
    /// Creates a catalogue-failure error with a host-safe explanation.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for DefinitionRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for DefinitionRegistryError {}

/// What the runtime is permitted to know about an agent.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Host-assigned opaque identifier.
    pub id: String,
    /// Human-readable name for prompts and display.
    pub name: String,
    /// Concise capability summary for a delegating parent.
    pub description: String,
    /// Host-defined routing role, when the host classifies this agent.
    #[serde(default)]
    pub role: Option<String>,
    /// Preferred model identifier, if this agent pins one.
    #[serde(default)]
    pub model: Option<String>,
    /// Agent ids this agent declares as eligible delegates.
    #[serde(default)]
    pub subagents: Vec<String>,
    /// Canonical tool names this agent may use.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Pattern rules narrowing which tools this agent may see and call, on
    /// top of [`Self::tools`]. Evaluated by the harness on the catalogue,
    /// tool search and every call; see [`tinytools::ToolRules`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_rules: Option<tinytools::ToolRules>,
}

impl AgentDefinition {
    /// Creates a definition with the required host-owned fields.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            role: None,
            model: None,
            subagents: Vec::new(),
            tools: Vec::new(),
            tool_rules: None,
        }
    }

    /// Sets the preferred model identifier.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets declared delegate ids.
    #[must_use]
    pub fn with_subagents<I, S>(mut self, subagents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.subagents = subagents.into_iter().map(Into::into).collect();
        self
    }

    /// Sets permitted tool names.
    #[must_use]
    pub fn with_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tools = tools.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the pattern rules narrowing this agent's tools.
    #[must_use]
    pub fn with_tool_rules(mut self, rules: tinytools::ToolRules) -> Self {
        self.tool_rules = Some(rules);
        self
    }

    /// Sets the host-defined routing role.
    #[must_use]
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Returns deterministic diagnostics for malformed or ambiguous data.
    #[must_use]
    pub fn diagnostics(&self) -> Vec<AgentDefinitionDiagnostic> {
        let mut diagnostics = Vec::new();
        required_field(&mut diagnostics, "id", &self.id);
        required_field(&mut diagnostics, "name", &self.name);
        required_field(&mut diagnostics, "description", &self.description);
        duplicate_values(&mut diagnostics, "subagents", &self.subagents);
        duplicate_values(&mut diagnostics, "tools", &self.tools);
        for (field, values) in [("subagents", &self.subagents), ("tools", &self.tools)] {
            for value in values {
                if value.trim().is_empty() {
                    diagnostics.push(AgentDefinitionDiagnostic::empty_entry(field));
                }
            }
        }
        diagnostics
    }

    /// Whether every required field and declared list entry is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.diagnostics().is_empty()
    }
}

/// A deterministic, machine-readable definition validation finding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentDefinitionDiagnostic {
    /// Field containing the invalid value.
    pub field: String,
    /// Stable machine-oriented validation code.
    pub code: String,
    /// Explanation appropriate for a configuration diagnostic.
    pub message: String,
}

impl AgentDefinitionDiagnostic {
    fn required(field: &str) -> Self {
        Self {
            field: field.to_string(),
            code: "required".to_string(),
            message: format!("agent definition field `{field}` must not be blank"),
        }
    }

    fn duplicate(field: &str, value: &str) -> Self {
        Self {
            field: field.to_string(),
            code: "duplicate".to_string(),
            message: format!("agent definition field `{field}` repeats `{value}`"),
        }
    }

    fn empty_entry(field: &str) -> Self {
        Self {
            field: field.to_string(),
            code: "empty_entry".to_string(),
            message: format!("agent definition field `{field}` contains a blank entry"),
        }
    }
}

fn required_field(diagnostics: &mut Vec<AgentDefinitionDiagnostic>, field: &str, value: &str) {
    if value.trim().is_empty() {
        diagnostics.push(AgentDefinitionDiagnostic::required(field));
    }
}

fn duplicate_values(
    diagnostics: &mut Vec<AgentDefinitionDiagnostic>,
    field: &str,
    values: &[String],
) {
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(value.as_str()) {
            diagnostics.push(AgentDefinitionDiagnostic::duplicate(field, value));
        }
    }
}

/// Required, host-owned definition lookup capability.
///
/// `Ok(None)` is the normal absence outcome. Errors are reserved for a backing
/// catalogue that could not answer; callers must not turn feature-gated or
/// otherwise absent agent definitions into failed runs.
#[async_trait]
pub trait DefinitionRegistry: Send + Sync {
    /// Resolves an id or returns normal absence.
    async fn resolve(&self, id: &str) -> Result<Option<AgentDefinition>>;
    /// Lists definitions in stable catalogue order.
    async fn list(&self) -> Result<Vec<AgentDefinition>>;
    /// Returns host-authorized delegate ids, not merely the declaration.
    async fn delegates_for(&self, id: &str) -> Result<Vec<String>>;
}

/// A fixed, insertion-ordered definition catalogue.
#[derive(Clone, Debug, Default)]
pub struct InMemoryDefinitionRegistry {
    definitions: Vec<AgentDefinition>,
    index: HashMap<String, usize>,
}

impl InMemoryDefinitionRegistry {
    /// Retains the first definition for each id, preserving insertion order.
    #[must_use]
    pub fn new(definitions: Vec<AgentDefinition>) -> Self {
        let mut registry = Self::default();
        for definition in definitions {
            if registry.index.contains_key(&definition.id) {
                continue;
            }
            registry
                .index
                .insert(definition.id.clone(), registry.definitions.len());
            registry.definitions.push(definition);
        }
        registry
    }

    /// Returns diagnostics from every retained definition, in catalogue order.
    #[must_use]
    pub fn diagnostics(&self) -> Vec<AgentDefinitionDiagnostic> {
        self.definitions
            .iter()
            .flat_map(AgentDefinition::diagnostics)
            .collect()
    }
}

#[async_trait]
impl DefinitionRegistry for InMemoryDefinitionRegistry {
    async fn resolve(&self, id: &str) -> Result<Option<AgentDefinition>> {
        Ok(self
            .index
            .get(id)
            .and_then(|position| self.definitions.get(*position))
            .cloned())
    }

    async fn list(&self) -> Result<Vec<AgentDefinition>> {
        Ok(self.definitions.clone())
    }

    async fn delegates_for(&self, id: &str) -> Result<Vec<String>> {
        Ok(self
            .index
            .get(id)
            .and_then(|position| self.definitions.get(*position))
            .map(|definition| definition.subagents.clone())
            .unwrap_or_default())
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
