//! File-operation lists carried by compaction summaries.
//!
//! A summary that forgets which files the agent already read or changed sends
//! it back to re-read them. Compaction therefore appends `<read-files>` and
//! `<modified-files>` sections, derived from the tool calls being folded, and
//! carries the previous summary's lists forward so they accumulate across
//! compactions. Port of pi's `compaction/utils.ts` file-op tracking.
//!
//! Extraction is pluggable ([`FileOpExtractor`]). [`DefaultFileOpExtractor`]
//! reads the common path arguments (`path`, `file`, `file_path`, `paths`) and
//! classifies the call from its tool name, split into words (`apply_patch` is
//! `apply`, `patch`):
//!
//! * a **search or listing** tool (`search`, `grep`, `glob`, `find`, `list`,
//!   `ls`) contributes nothing: its path is a scope, not a file it read;
//! * a tool with a **mutating verb** (`write`, `edit`, `patch`, `create`,
//!   `delete`, `remove`, `append`, `replace`, `move`, `rename`, `save`,
//!   `touch`, `mkdir`) is a *modification* only when it is plainly about
//!   files: the verb is `write`/`edit`/`patch`, or the name also says `file`,
//!   `dir`, `folder`, `fs`, `path` or `notebook`, or the name is the bare verb.
//!   Any other mutating tool (`create_issue`, `github_create_pr`,
//!   `memory_save`) contributes nothing, even with a `path` argument;
//! * every other path-carrying call is a *read*.
//!
//! A host with differently named tools supplies its own extractor.
//!
//! ## Bounds and safety
//!
//! Each list shows its [`MAX_LISTED_FILES`] most recently touched files and a
//! `…and K more` line for the rest (the count carries across compactions).
//! Path strings are escaped injectively before they are stored (`&`, `<`, `>`,
//! control characters and the marker's `…`), so a hostile file name can neither
//! forge a section, add lines to one, nor pass for the `…and K more` line. The
//! omitted counts are approximate: collapsed paths keep no identity, so a path
//! touched again after being collapsed is counted in both places.

use tinyinference_llm::message::Message;
use tinyinference_llm::tool::ToolCall;

const READ_OPEN: &str = "<read-files>\n";
const READ_CLOSE: &str = "\n</read-files>";
const MODIFIED_OPEN: &str = "<modified-files>\n";
const MODIFIED_CLOSE: &str = "\n</modified-files>";

/// Files each list shows; older ones collapse into a `…and K more` line.
pub const MAX_LISTED_FILES: usize = 50;

/// Longest path kept; longer ones are cut (a path this long is not a path).
const MAX_PATH_CHARS: usize = 300;

/// Argument names [`DefaultFileOpExtractor`] treats as file paths.
const PATH_ARGS: [&str; 4] = ["path", "file", "file_path", "paths"];

/// Tool-name words that make a call a listing/search, which touches no file.
const SEARCH_WORDS: [&str; 6] = ["search", "grep", "glob", "find", "list", "ls"];

/// Tool-name word prefixes that mark a mutating call.
const MUTATING_VERBS: [&str; 13] = [
    "write", "edit", "patch", "create", "delete", "remove", "append", "replace", "move", "rename",
    "save", "touch", "mkdir",
];

/// Verbs that modify files whatever else the name says.
const FILE_VERBS: [&str; 3] = ["write", "edit", "patch"];

/// Tool-name word prefixes that say the tool works on files.
const FILE_WORDS: [&str; 5] = ["file", "dir", "folder", "path", "notebook"];

/// How [`DefaultFileOpExtractor`] reads a tool name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Touch {
    Read,
    Modify,
    Ignore,
}

fn classify_tool(name: &str) -> Touch {
    let lower = name.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if words.iter().any(|w| SEARCH_WORDS.contains(w)) {
        return Touch::Ignore;
    }
    let starts = |set: &[&str]| words.iter().any(|w| set.iter().any(|p| w.starts_with(p)));
    if !starts(&MUTATING_VERBS) {
        return Touch::Read;
    }
    if starts(&FILE_VERBS) || starts(&FILE_WORDS) || words.contains(&"fs") || words.len() == 1 {
        Touch::Modify
    } else {
        Touch::Ignore
    }
}

/// Neutralizes what would let a path break out of its list line or section.
///
/// The encoding is injective (distinct paths stay distinct, since the path's
/// rendering doubles as its identity): `&` becomes `&amp;`, `<` / `>` become
/// `&lt;` / `&gt;`, a control character becomes `&#N;`, and the ellipsis that
/// opens the `…and K more` marker becomes `&#8230;`, so no path can pass for
/// that marker.
fn sanitize_path(path: &str) -> String {
    let mut out = String::new();
    for c in path.chars().take(MAX_PATH_CHARS) {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '…' => out.push_str("&#8230;"),
            c if c.is_control() => out.push_str(&format!("&#{};", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Files touched by the tool calls of a stretch of conversation, each list in
/// order of recency (a file touched again moves to the end).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileOperations {
    read: Vec<String>,
    modified: Vec<String>,
    /// Older reads already collapsed into a `…and K more` line.
    read_omitted: usize,
    /// Older modifications already collapsed likewise.
    modified_omitted: usize,
}

/// Moves an already-encoded `path` to the end of `list`.
fn touch_encoded(list: &mut Vec<String>, path: &str) {
    if path.is_empty() {
        return;
    }
    list.retain(|existing| existing != path);
    list.push(path.to_string());
}

fn touch(list: &mut Vec<String>, path: &str) {
    touch_encoded(list, &sanitize_path(path));
}

impl FileOperations {
    /// Records a file that was read.
    pub fn add_read(&mut self, path: &str) {
        touch(&mut self.read, path);
    }

    /// Records a file that was created, written, edited or otherwise changed.
    pub fn add_modified(&mut self, path: &str) {
        touch(&mut self.modified, path);
    }

    /// Unions `other` into `self`; `other`'s files count as more recent.
    pub fn merge(&mut self, other: &FileOperations) {
        // `other`'s paths are already encoded; encoding them again would
        // change them on every merge.
        for path in &other.read {
            touch_encoded(&mut self.read, path);
        }
        for path in &other.modified {
            touch_encoded(&mut self.modified, path);
        }
        self.read_omitted = self.read_omitted.saturating_add(other.read_omitted);
        self.modified_omitted = self.modified_omitted.saturating_add(other.modified_omitted);
    }

    /// Files that were modified, oldest first.
    pub fn modified(&self) -> Vec<&str> {
        self.modified.iter().map(String::as_str).collect()
    }

    /// Files that were read and never modified, oldest first.
    pub fn read_only(&self) -> Vec<&str> {
        self.read
            .iter()
            .filter(|path| !self.modified.contains(*path))
            .map(String::as_str)
            .collect()
    }

    /// Whether nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.read.is_empty() && self.modified.is_empty()
    }
}

/// Reads the files one tool call touched into a [`FileOperations`].
pub trait FileOpExtractor: Send + Sync {
    /// Records the files `call` read or modified, if any.
    fn extract(&self, call: &ToolCall, ops: &mut FileOperations);
}

/// The default extractor; see the module docs.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultFileOpExtractor;

impl FileOpExtractor for DefaultFileOpExtractor {
    fn extract(&self, call: &ToolCall, ops: &mut FileOperations) {
        if call.invalid.is_some() {
            return;
        }
        let kind = classify_tool(&call.name);
        if kind == Touch::Ignore {
            return;
        }
        let mut record = |path: &str| match kind {
            Touch::Modify => ops.add_modified(path),
            _ => ops.add_read(path),
        };
        for arg in PATH_ARGS {
            match call.arguments.get(arg) {
                Some(serde_json::Value::String(path)) => record(path),
                Some(serde_json::Value::Array(paths)) => paths
                    .iter()
                    .filter_map(|p| p.as_str())
                    .for_each(&mut record),
                _ => {}
            }
        }
    }
}

/// Collects the file operations of every tool call in `messages`.
pub fn extract_file_operations(
    messages: &[Message],
    extractor: &dyn FileOpExtractor,
) -> FileOperations {
    let mut ops = FileOperations::default();
    let failed_calls: std::collections::HashSet<&str> = messages
        .iter()
        .filter_map(|message| match message {
            Message::Tool(tool)
                if tool
                    .artifact
                    .as_ref()
                    .and_then(|artifact| artifact.get("is_error"))
                    .and_then(serde_json::Value::as_bool)
                    == Some(true) =>
            {
                Some(tool.tool_call_id.as_str())
            }
            _ => None,
        })
        .collect();
    for message in messages {
        if let Message::Assistant(assistant) = message {
            for call in &assistant.tool_calls {
                if !failed_calls.contains(call.id.as_str()) {
                    extractor.extract(call, &mut ops);
                }
            }
        }
    }
    ops
}

/// One section body: the most recent [`MAX_LISTED_FILES`] paths, then the
/// `…and K more` line when any were left out.
fn render_list(paths: &[&str], already_omitted: usize) -> String {
    let shown = &paths[paths.len().saturating_sub(MAX_LISTED_FILES)..];
    let omitted = already_omitted.saturating_add(paths.len() - shown.len());
    let mut lines = shown.join("\n");
    if omitted > 0 {
        lines.push_str(&format!("\n{OMITTED_PREFIX}{omitted}{OMITTED_SUFFIX}"));
    }
    lines
}

const OMITTED_PREFIX: &str = "…and ";
const OMITTED_SUFFIX: &str = " more";

/// Appends `<read-files>` / `<modified-files>` sections to `summary`; returns
/// it unchanged when `ops` is empty. Each list is capped at
/// [`MAX_LISTED_FILES`] (most recent kept).
pub fn append_file_sections(summary: &str, ops: &FileOperations) -> String {
    let mut text = summary.trim_end().to_string();
    let read = ops.read_only();
    if !read.is_empty() {
        text.push_str(&format!(
            "\n\n{READ_OPEN}{}{READ_CLOSE}",
            render_list(&read, ops.read_omitted)
        ));
    }
    let modified = ops.modified();
    if !modified.is_empty() {
        text.push_str(&format!(
            "\n\n{MODIFIED_OPEN}{}{MODIFIED_CLOSE}",
            render_list(&modified, ops.modified_omitted)
        ));
    }
    text
}

/// Splits the file sections [`append_file_sections`] wrote off the end of
/// `text`, returning the remaining body and the operations they listed
/// (including the `…and K more` counts). Only the trailer is parsed — the
/// `<modified-files>` section last, the `<read-files>` section before it, each
/// set off by a blank line, with trailing whitespace after it tolerated — so
/// delimiters in the summary prose are left alone.
/// Text without such a trailer comes back byte-for-byte unchanged with an empty
/// set.
pub fn split_file_sections(text: &str) -> (String, FileOperations) {
    let mut ops = FileOperations::default();
    let mut body = text;
    for (open, close, is_modified) in [
        (MODIFIED_OPEN, MODIFIED_CLOSE, true),
        (READ_OPEN, READ_CLOSE, false),
    ] {
        let Some(head) = body.trim_end().strip_suffix(close) else {
            continue;
        };
        let Some(start) = head.rfind(open) else {
            continue;
        };
        let Some(prose) = head[..start].strip_suffix("\n\n") else {
            continue;
        };
        for line in head[start + open.len()..].lines() {
            let omitted = line
                .strip_prefix(OMITTED_PREFIX)
                .and_then(|rest| rest.strip_suffix(OMITTED_SUFFIX))
                .and_then(|count| count.parse::<usize>().ok());
            match (omitted, is_modified) {
                (Some(n), true) => ops.modified_omitted = ops.modified_omitted.saturating_add(n),
                (Some(n), false) => ops.read_omitted = ops.read_omitted.saturating_add(n),
                // The listed lines are already encoded.
                (None, true) => touch_encoded(&mut ops.modified, line),
                (None, false) => touch_encoded(&mut ops.read, line),
            }
        }
        body = prose;
    }
    (body.to_string(), ops)
}

#[cfg(test)]
#[path = "file_ops_tests.rs"]
mod tests;
