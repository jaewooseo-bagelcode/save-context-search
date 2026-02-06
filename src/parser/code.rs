//! Code parser using tree-sitter for extracting symbols from source files.
//!
//! Supports: Rust (.rs), TypeScript (.ts, .tsx), JavaScript (.js, .jsx), C# (.cs), Python (.py)

use std::path::Path;
use anyhow::{Context, Result};
use tree_sitter::{Parser, Query, QueryCursor, Language, Node};
use streaming_iterator::StreamingIterator;

use crate::{ChunkType, ChunkKind, RawChunk, Visibility};
use super::queries::{
    RUST_DEF_QUERY,
    TS_DEF_QUERY,
    JS_DEF_QUERY,
    CS_DEF_QUERY,
    PY_DEF_QUERY,
};

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq)]
enum SupportedLanguage {
    Rust,
    TypeScript,
    Tsx,
    JavaScript,
    CSharp,
    Python,
}

impl SupportedLanguage {
    fn get_def_query(&self) -> &'static str {
        match self {
            SupportedLanguage::Rust => RUST_DEF_QUERY,
            SupportedLanguage::TypeScript | SupportedLanguage::Tsx => TS_DEF_QUERY,
            SupportedLanguage::JavaScript => JS_DEF_QUERY,
            SupportedLanguage::CSharp => CS_DEF_QUERY,
            SupportedLanguage::Python => PY_DEF_QUERY,
        }
    }

    fn get_ts_language(&self) -> Language {
        match self {
            SupportedLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
            SupportedLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            SupportedLanguage::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            SupportedLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            SupportedLanguage::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            SupportedLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        }
    }
}

/// Code parser that uses tree-sitter to extract symbols from source files.
pub struct CodeParser {
    parser: Parser,
}

impl CodeParser {
    /// Create a new CodeParser.
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: Parser::new(),
        })
    }

    /// Parse a source file and extract symbols as RawChunks.
    pub fn parse(&mut self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let lang = match ext {
            "rs" => Some(SupportedLanguage::Rust),
            "ts" => Some(SupportedLanguage::TypeScript),
            "tsx" => Some(SupportedLanguage::Tsx),
            "js" | "jsx" => Some(SupportedLanguage::JavaScript),
            "cs" => Some(SupportedLanguage::CSharp),
            "py" => Some(SupportedLanguage::Python),
            _ => None,
        };

        match lang {
            Some(language) => self.parse_with_language(path, content, language),
            None => Ok(vec![]),
        }
    }

    fn parse_with_language(
        &mut self,
        path: &Path,
        content: &str,
        lang: SupportedLanguage,
    ) -> Result<Vec<RawChunk>> {
        let ts_lang = lang.get_ts_language();
        self.parser.set_language(&ts_lang)
            .context("Failed to set language")?;

        parse_with_queries(path, content, &mut self.parser, lang.get_def_query(), ts_lang, lang)
    }
}

/// Parse content using the given queries and extract symbols.
fn parse_with_queries(
    path: &Path,
    content: &str,
    parser: &mut Parser,
    def_query_str: &str,
    lang: Language,
    supported_lang: SupportedLanguage,
) -> Result<Vec<RawChunk>> {
    let tree = match parser.parse(content, None) {
        Some(t) => t,
        None => {
            eprintln!("Warning: Failed to parse {}", path.display());
            return Ok(vec![]);
        }
    };

    let root = tree.root_node();
    let bytes = content.as_bytes();

    let def_query = match Query::new(&lang, def_query_str) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("Warning: Failed to compile definition query for {}: {}", path.display(), e);
            return Ok(vec![]);
        }
    };

    let mut chunks: Vec<RawChunk> = Vec::new();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&def_query, root, bytes);

    while let Some(m) = matches.next() {
        let mut symbol_node: Option<Node> = None;
        let mut name_node: Option<Node> = None;
        let mut capture_name: Option<&str> = None;

        for capture in m.captures {
            let cap_name = def_query.capture_names()[capture.index as usize];

            if cap_name.ends_with(".name") || cap_name.ends_with(".type") {
                name_node = Some(capture.node);
            } else {
                symbol_node = Some(capture.node);
                capture_name = Some(cap_name);
            }
        }

        let (sym_node, name_n, cap_name) = match (symbol_node, name_node, capture_name) {
            (Some(s), Some(n), Some(c)) => (s, n, c),
            _ => continue,
        };

        let name = match name_n.utf8_text(bytes) {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };

        let kind = match cap_name {
            "function" => ChunkKind::Function,
            "method" => ChunkKind::Method,
            "class" => ChunkKind::Class,
            "struct" => ChunkKind::Struct,
            "interface" => ChunkKind::Interface,
            "enum" => ChunkKind::Enum,
            "const" => ChunkKind::Constant,
            "trait" => ChunkKind::Interface,
            "impl" => ChunkKind::Impl,
            "property" => ChunkKind::Field,
            "type" => ChunkKind::Interface,
            _ => continue,
        };

        let line_start = sym_node.start_position().row as u32 + 1;
        let line_end = sym_node.end_position().row as u32 + 1;

        let content_text = match sym_node.utf8_text(bytes) {
            Ok(c) => c.to_string(),
            Err(_) => continue,
        };

        let signature = extract_signature(&content_text);
        let context = extract_context(sym_node, bytes);

        // Post-process: function with parent context is actually a method
        let kind = if kind == ChunkKind::Function && context.is_some() {
            ChunkKind::Method
        } else {
            kind
        };

        // Parse doc comment and visibility based on language
        let (doc_summary, visibility) = match supported_lang {
            SupportedLanguage::Rust => {
                let doc = parse_rust_doc_comment(sym_node, bytes);
                let vis = parse_rust_visibility(sym_node);
                (doc, vis)
            }
            SupportedLanguage::TypeScript | SupportedLanguage::Tsx | SupportedLanguage::JavaScript => {
                let doc = parse_jsdoc_comment(sym_node, bytes);
                let vis = parse_ts_visibility(sym_node, bytes);
                (doc, vis)
            }
            SupportedLanguage::Python => {
                let doc = parse_python_docstring(sym_node, bytes);
                let vis = parse_python_visibility(&name);
                (doc, vis)
            }
            SupportedLanguage::CSharp => {
                let doc = parse_csharp_doc_comment(sym_node, bytes);
                let vis = parse_csharp_visibility(sym_node, bytes);
                (doc, vis)
            }
        };

        chunks.push(RawChunk {
            chunk_type: ChunkType::Code,
            name,
            kind,
            line_start,
            line_end,
            byte_start: sym_node.start_byte(),
            byte_end: sym_node.end_byte(),
            content: content_text,
            context,
            signature: Some(signature),
            doc_summary,
            visibility,
        });
    }

    Ok(chunks)
}

/// Extract the signature from a symbol's content.
fn extract_signature(content: &str) -> String {
    let mut signature = String::new();
    let mut paren_depth: i32 = 0;
    let mut angle_depth: i32 = 0;
    let mut line_count = 0;

    for line in content.lines() {
        let trimmed = line.trim();
        line_count += 1;

        for ch in trimmed.chars() {
            match ch {
                '(' => paren_depth += 1,
                ')' => paren_depth = (paren_depth - 1).max(0),
                '<' => angle_depth += 1,
                '>' => angle_depth = (angle_depth - 1).max(0),
                _ => {}
            }
        }

        if !signature.is_empty() {
            signature.push(' ');
        }
        signature.push_str(trimmed);

        if paren_depth == 0
            && angle_depth == 0
            && (trimmed.ends_with('{') || trimmed.ends_with(';'))
        {
            break;
        }

        if line_count >= 5 {
            break;
        }
    }

    signature
}

/// Extract context (parent class/struct name) for a node.
fn extract_context(node: Node, bytes: &[u8]) -> Option<String> {
    let mut current = node.parent();

    while let Some(parent) = current {
        let kind = parent.kind();

        if kind == "class_declaration"
            || kind == "class_definition"
            || kind == "struct_item"
            || kind == "trait_item"
            || kind == "interface_declaration"
        {
            if let Some(name_node) = parent.child_by_field_name("name") {
                if let Ok(name) = name_node.utf8_text(bytes) {
                    return Some(name.to_string());
                }
            }
        } else if kind == "impl_item" {
            if let Some(type_node) = parent.child_by_field_name("type") {
                let name = extract_type_name(type_node, bytes);
                if name.is_some() {
                    return name;
                }
            }
        }

        current = parent.parent();
    }

    None
}

/// Extract the type name from a type node.
fn extract_type_name(type_node: Node, bytes: &[u8]) -> Option<String> {
    match type_node.kind() {
        "type_identifier" | "identifier" => {
            type_node.utf8_text(bytes).ok().map(|s| s.to_string())
        }
        "generic_type" => {
            if let Some(inner) = type_node.child_by_field_name("type") {
                return extract_type_name(inner, bytes);
            }
            type_node.child(0).and_then(|c| extract_type_name(c, bytes))
        }
        "scoped_type_identifier" => {
            if let Some(name_node) = type_node.child_by_field_name("name") {
                return name_node.utf8_text(bytes).ok().map(|s| s.to_string());
            }
            None
        }
        _ => {
            type_node.utf8_text(bytes).ok().map(|s| s.to_string())
        }
    }
}

// ============================================================================
// Rust doc comment and visibility parsing
// ============================================================================

/// Parse Rust doc comments (///) from preceding siblings.
/// Returns the first paragraph as summary (stops at empty line or # section).
fn parse_rust_doc_comment(node: Node, bytes: &[u8]) -> Option<String> {
    let mut comments = Vec::new();
    let mut prev = node.prev_sibling();

    // Collect consecutive /// comments going backwards
    while let Some(sibling) = prev {
        if sibling.kind() == "line_comment" {
            if let Ok(text) = sibling.utf8_text(bytes) {
                if text.starts_with("///") {
                    // Remove /// prefix and trim
                    let content = text.trim_start_matches("///").trim();
                    comments.push(content.to_string());
                } else {
                    break; // Not a doc comment, stop
                }
            }
        } else if sibling.kind() == "attribute_item" || sibling.kind() == "inner_attribute_item" {
            // Skip attributes like #[derive(...)]
        } else {
            break;
        }
        prev = sibling.prev_sibling();
    }

    if comments.is_empty() {
        return None;
    }

    // Reverse to get correct order
    comments.reverse();

    // Take first paragraph (stop at empty line or # section)
    let summary: Vec<&str> = comments
        .iter()
        .take_while(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|s| s.as_str())
        .collect();

    if summary.is_empty() {
        None
    } else {
        Some(summary.join(" "))
    }
}

/// Parse Rust visibility (checks for `pub` keyword).
fn parse_rust_visibility(node: Node) -> Visibility {
    // Check if first child is visibility_modifier (pub, pub(crate), etc.)
    if let Some(child) = node.child(0) {
        if child.kind() == "visibility_modifier" {
            return Visibility::Public;
        }
    }

    // For impl items, check if the method has pub
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "visibility_modifier" {
                return Visibility::Public;
            }
        }
    }

    Visibility::Private
}

// ============================================================================
// TypeScript/JavaScript JSDoc and visibility parsing
// ============================================================================

/// Parse JSDoc comments (/** */) from preceding siblings.
/// Returns the description before @tags as summary.
fn parse_jsdoc_comment(node: Node, bytes: &[u8]) -> Option<String> {
    // Look for comment in previous sibling
    let prev = node.prev_sibling()?;

    // Handle case where the comment might be before export_statement wrapping
    let comment_node = if prev.kind() == "comment" {
        prev
    } else {
        // Try parent's prev sibling for export statements
        let parent = node.parent()?;
        if parent.kind() == "export_statement" {
            parent.prev_sibling().filter(|n| n.kind() == "comment")?
        } else {
            return None;
        }
    };

    let text = comment_node.utf8_text(bytes).ok()?;

    // Must be JSDoc style (/** ... */)
    if !text.starts_with("/**") {
        return None;
    }

    // Remove /** and */
    let content = text
        .trim_start_matches("/**")
        .trim_end_matches("*/")
        .trim();

    // Parse lines, remove * prefix
    let lines: Vec<&str> = content
        .lines()
        .map(|l| l.trim().trim_start_matches('*').trim())
        .filter(|l| !l.is_empty())
        .collect();

    // Take lines until we hit a @tag
    let summary_lines: Vec<&str> = lines
        .iter()
        .take_while(|line| !line.starts_with('@'))
        .copied()
        .collect();

    if summary_lines.is_empty() {
        None
    } else {
        Some(summary_lines.join(" "))
    }
}

/// Parse TypeScript/JavaScript visibility (checks for `export` keyword).
fn parse_ts_visibility(node: Node, bytes: &[u8]) -> Visibility {
    // Check if this is inside an export_statement
    if let Some(parent) = node.parent() {
        if parent.kind() == "export_statement" {
            return Visibility::Public;
        }
    }

    // Check if the node itself contains "export" (for lexical_declaration with export)
    if let Ok(text) = node.utf8_text(bytes) {
        if text.starts_with("export ") {
            return Visibility::Public;
        }
    }

    Visibility::Private
}

// ============================================================================
// Python docstring and visibility parsing
// ============================================================================

/// Parse Python docstring from function/class body.
/// Looks for the first expression_statement containing a string as the docstring.
/// Returns the first paragraph (stops at empty line) as summary.
fn parse_python_docstring(node: Node, bytes: &[u8]) -> Option<String> {
    // Find the body of the function/class
    let body = node.child_by_field_name("body")?;

    // Look for the first child that is an expression_statement containing a string
    for i in 0..body.child_count() {
        let child = body.child(i as u32)?;

        // Skip non-expression statements at the start (e.g., comments)
        if child.kind() != "expression_statement" {
            continue;
        }

        // Look for a string child (the docstring)
        for j in 0..child.child_count() {
            if let Some(string_node) = child.child(j as u32) {
                if string_node.kind() == "string" {
                    return extract_docstring_summary(string_node, bytes);
                }
            }
        }

        // First expression_statement wasn't a docstring, so there's no docstring
        break;
    }

    None
}

/// Extract summary from a Python string node (docstring).
/// Handles both """ and ''' styles.
fn extract_docstring_summary(string_node: Node, bytes: &[u8]) -> Option<String> {
    let text = string_node.utf8_text(bytes).ok()?;

    // Remove the triple quotes (""" or ''')
    let content = if text.starts_with("\"\"\"") && text.ends_with("\"\"\"") {
        text.trim_start_matches("\"\"\"").trim_end_matches("\"\"\"")
    } else if text.starts_with("'''") && text.ends_with("'''") {
        text.trim_start_matches("'''").trim_end_matches("'''")
    } else if text.starts_with("\"") && text.ends_with("\"") {
        // Regular single-line string
        text.trim_start_matches("\"").trim_end_matches("\"")
    } else if text.starts_with("'") && text.ends_with("'") {
        text.trim_start_matches("'").trim_end_matches("'")
    } else {
        return None;
    };

    let content = content.trim();

    if content.is_empty() {
        return None;
    }

    // Take first paragraph (stop at empty line)
    let summary_lines: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .take_while(|l| !l.is_empty())
        .collect();

    if summary_lines.is_empty() {
        None
    } else {
        Some(summary_lines.join(" "))
    }
}

/// Parse Python visibility based on naming conventions.
/// - Names starting with `_` (but not `__x__` dunder) are private
/// - Dunder methods like `__init__`, `__str__` are public
/// - Everything else is public
fn parse_python_visibility(name: &str) -> Visibility {
    // Dunder methods (__init__, __str__, etc.) are public
    if name.starts_with("__") && name.ends_with("__") {
        return Visibility::Public;
    }

    // Names starting with _ or __ are private (convention)
    if name.starts_with('_') {
        return Visibility::Private;
    }

    Visibility::Public
}

// ============================================================================
// C# XML doc comment and visibility parsing
// ============================================================================

/// Parse C# XML doc comments (///) from preceding siblings.
/// Extracts text from <summary>...</summary> tags.
fn parse_csharp_doc_comment(node: Node, bytes: &[u8]) -> Option<String> {
    let mut comments = Vec::new();
    let mut prev = node.prev_sibling();

    // Collect consecutive /// comments going backwards
    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            if let Ok(text) = sibling.utf8_text(bytes) {
                if text.starts_with("///") {
                    // Remove /// prefix and trim
                    let content = text.trim_start_matches("///").trim();
                    comments.push(content.to_string());
                } else {
                    break; // Not a doc comment, stop
                }
            }
        } else if sibling.kind() == "attribute_list" {
            // Skip attributes like [Attribute]
        } else {
            break;
        }
        prev = sibling.prev_sibling();
    }

    if comments.is_empty() {
        return None;
    }

    // Reverse to get correct order
    comments.reverse();

    // Join all lines into one string for XML parsing
    let full_text = comments.join(" ");

    // Extract content from <summary>...</summary> tags
    extract_xml_summary(&full_text)
}

/// Extract text content from <summary>...</summary> XML tags.
fn extract_xml_summary(text: &str) -> Option<String> {
    // Simple extraction: find <summary> and </summary>
    let start_tag = "<summary>";
    let end_tag = "</summary>";

    let start = text.find(start_tag)?;
    let end = text.find(end_tag)?;

    if start >= end {
        return None;
    }

    let content = &text[start + start_tag.len()..end];

    // Remove XML tags and clean up whitespace
    let cleaned = strip_xml_tags(content.trim());

    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Strip all XML tags from a string.
fn strip_xml_tags(text: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;

    for ch in text.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }

    // Normalize whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Parse C# visibility from modifier keywords.
/// - public, internal → Public
/// - private, protected, no modifier → Private
fn parse_csharp_visibility(node: Node, bytes: &[u8]) -> Visibility {
    // Iterate through children to find modifier
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "modifier" {
                if let Ok(text) = child.utf8_text(bytes) {
                    match text {
                        "public" | "internal" => return Visibility::Public,
                        "private" | "protected" => return Visibility::Private,
                        _ => {}
                    }
                }
            }
        }
    }

    // Default: no modifier means private in C#
    Visibility::Private
}

impl Default for CodeParser {
    fn default() -> Self {
        Self::new().expect("Failed to create CodeParser")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_new_parser() {
        let parser = CodeParser::new();
        assert!(parser.is_ok());
    }

    #[test]
    fn test_parse_rust() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.rs");
        let content = r#"
fn hello() {
    println!("Hello");
}

struct Player {
    name: String,
}

impl Player {
    fn new(name: String) -> Self {
        Self { name }
    }
}
"#;

        let chunks = parser.parse(&path, content).unwrap();
        assert!(chunks.len() >= 3);

        let hello = chunks.iter().find(|c| c.name == "hello");
        assert!(hello.is_some());
        assert_eq!(hello.unwrap().kind, ChunkKind::Function);

        let player = chunks.iter().find(|c| c.name == "Player" && c.kind == ChunkKind::Struct);
        assert!(player.is_some());
    }

    #[test]
    fn test_parse_python() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.py");
        let content = r#"
def calculate(x, y):
    return x + y

class Calculator:
    def add(self, a, b):
        return a + b
"#;

        let chunks = parser.parse(&path, content).unwrap();
        assert!(chunks.len() >= 2);

        let calc_fn = chunks.iter().find(|c| c.name == "calculate");
        assert!(calc_fn.is_some());
        assert_eq!(calc_fn.unwrap().kind, ChunkKind::Function);

        let add_method = chunks.iter().find(|c| c.name == "add");
        assert!(add_method.is_some());
        assert_eq!(add_method.unwrap().kind, ChunkKind::Method);
        assert_eq!(add_method.unwrap().context, Some("Calculator".to_string()));
    }

    #[test]
    fn test_parse_typescript() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.ts");
        let content = r#"
const MODELS = {
    SONNET: "claude-sonnet",
    OPUS: "claude-opus",
};

function hello() {
    console.log("Hello");
}

class Player {
    name: string;
    move() {
        return this.name;
    }
}

interface GameState {
    score: number;
}

export const API_KEY = "test";
"#;

        let chunks = parser.parse(&path, content).unwrap();
        println!("TS chunks: {:?}", chunks.iter().map(|c| (&c.name, &c.kind)).collect::<Vec<_>>());

        assert!(chunks.len() >= 4, "Expected at least 4 chunks, got {}", chunks.len());

        let hello = chunks.iter().find(|c| c.name == "hello");
        assert!(hello.is_some(), "Should find hello function");
        assert_eq!(hello.unwrap().kind, ChunkKind::Function);

        let player = chunks.iter().find(|c| c.name == "Player");
        assert!(player.is_some(), "Should find Player class");
        assert_eq!(player.unwrap().kind, ChunkKind::Class);

        let models = chunks.iter().find(|c| c.name == "MODELS");
        assert!(models.is_some(), "Should find MODELS constant");
        assert_eq!(models.unwrap().kind, ChunkKind::Constant);

        let api_key = chunks.iter().find(|c| c.name == "API_KEY");
        assert!(api_key.is_some(), "Should find API_KEY constant (export const)");
    }

    #[test]
    fn test_parse_tsx() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.tsx");
        let content = r#"
export function App() {
    return <div>Hello</div>;
}

function Helper() {
    return <span>Helper</span>;
}

export const Button = () => <button>Click</button>;
"#;

        let chunks = parser.parse(&path, content).unwrap();
        println!("Tsx chunks: {:?}", chunks.iter().map(|c| (&c.name, &c.kind)).collect::<Vec<_>>());

        let app = chunks.iter().find(|c| c.name == "App");
        assert!(app.is_some(), "Should find App function (export function)");
        assert_eq!(app.unwrap().kind, ChunkKind::Function);

        let helper = chunks.iter().find(|c| c.name == "Helper");
        assert!(helper.is_some(), "Should find Helper function");

        let button = chunks.iter().find(|c| c.name == "Button");
        assert!(button.is_some(), "Should find Button constant (arrow function)");
    }

    #[test]
    fn test_parse_csharp_struct() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.cs");
        let content = r#"
public struct MaskedInt : IEquatable<MaskedInt> {
    private int value;

    public MaskedInt(int v) {
        value = v;
    }
}

public class Player {
    public string Name { get; set; }
}
"#;

        let chunks = parser.parse(&path, content).unwrap();
        println!("C# chunks: {:?}", chunks.iter().map(|c| (&c.name, &c.kind)).collect::<Vec<_>>());

        let masked = chunks.iter().find(|c| c.name == "MaskedInt");
        assert!(masked.is_some(), "Should find MaskedInt struct");
        assert_eq!(masked.unwrap().kind, ChunkKind::Struct);

        let player = chunks.iter().find(|c| c.name == "Player");
        assert!(player.is_some(), "Should find Player class");
        assert_eq!(player.unwrap().kind, ChunkKind::Class);
    }

    #[test]
    fn test_signature_extraction() {
        let content = "fn hello(name: &str) -> String {\n    format!(\"Hello, {}\", name)\n}";
        let sig = extract_signature(content);
        assert_eq!(sig, "fn hello(name: &str) -> String {");
    }

    #[test]
    fn test_parse_unsupported_extension() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.xyz");
        let content = "some content";

        let chunks = parser.parse(&path, content).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_parse_rust_doc_comments() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.rs");
        let content = r#"
/// This is the main function.
/// It does something important.
///
/// # Arguments
/// * `x` - the input
pub fn main_func(x: i32) -> i32 {
    x + 1
}

/// A simple struct
pub struct Player {
    name: String,
}

fn private_func() {
    // no doc comment
}
"#;

        let chunks = parser.parse(&path, content).unwrap();

        // Check main_func has doc summary and is public
        let main_fn = chunks.iter().find(|c| c.name == "main_func");
        assert!(main_fn.is_some(), "Should find main_func");
        let main_fn = main_fn.unwrap();
        assert_eq!(main_fn.visibility, Visibility::Public, "main_func should be public");
        assert!(main_fn.doc_summary.is_some(), "main_func should have doc summary");
        assert_eq!(
            main_fn.doc_summary.as_ref().unwrap(),
            "This is the main function. It does something important."
        );

        // Check Player struct is public with doc
        let player = chunks.iter().find(|c| c.name == "Player");
        assert!(player.is_some(), "Should find Player");
        let player = player.unwrap();
        assert_eq!(player.visibility, Visibility::Public, "Player should be public");
        assert_eq!(player.doc_summary.as_ref().unwrap(), "A simple struct");

        // Check private_func is private with no doc
        let private_fn = chunks.iter().find(|c| c.name == "private_func");
        assert!(private_fn.is_some(), "Should find private_func");
        let private_fn = private_fn.unwrap();
        assert_eq!(private_fn.visibility, Visibility::Private, "private_func should be private");
        assert!(private_fn.doc_summary.is_none(), "private_func should have no doc");
    }

    #[test]
    fn test_parse_typescript_jsdoc() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.ts");
        let content = r#"
/**
 * Calculate the difficulty score.
 * This is a very important function.
 * @param level - The level setting
 * @returns The difficulty score
 */
export function calculateDifficulty(level: number): number {
    return level * 2;
}

/**
 * A player class
 */
export class Player {
    name: string;
}

function privateHelper() {
    // no doc, not exported
}

/**
 * Exported constant
 */
export const API_KEY = "test";
"#;

        let chunks = parser.parse(&path, content).unwrap();

        // Check calculateDifficulty has doc summary and is exported (public)
        let calc_fn = chunks.iter().find(|c| c.name == "calculateDifficulty");
        assert!(calc_fn.is_some(), "Should find calculateDifficulty");
        let calc_fn = calc_fn.unwrap();
        assert_eq!(calc_fn.visibility, Visibility::Public, "calculateDifficulty should be public");
        assert!(calc_fn.doc_summary.is_some(), "calculateDifficulty should have doc summary");
        assert_eq!(
            calc_fn.doc_summary.as_ref().unwrap(),
            "Calculate the difficulty score. This is a very important function."
        );

        // Check Player class is exported with doc
        let player = chunks.iter().find(|c| c.name == "Player");
        assert!(player.is_some(), "Should find Player");
        let player = player.unwrap();
        assert_eq!(player.visibility, Visibility::Public, "Player should be public");
        assert_eq!(player.doc_summary.as_ref().unwrap(), "A player class");

        // Check privateHelper is private with no doc
        let private_fn = chunks.iter().find(|c| c.name == "privateHelper");
        assert!(private_fn.is_some(), "Should find privateHelper");
        let private_fn = private_fn.unwrap();
        assert_eq!(private_fn.visibility, Visibility::Private, "privateHelper should be private");
        assert!(private_fn.doc_summary.is_none(), "privateHelper should have no doc");

        // Check API_KEY constant
        let api_key = chunks.iter().find(|c| c.name == "API_KEY");
        assert!(api_key.is_some(), "Should find API_KEY");
        let api_key = api_key.unwrap();
        assert_eq!(api_key.visibility, Visibility::Public, "API_KEY should be public");
    }

    #[test]
    fn test_parse_python_docstrings() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.py");
        let content = r#"
def calculate(x: int) -> int:
    """Calculate the value.

    This is additional description.

    Args:
        x: Input number
    """
    return x * 2

class Player:
    """A player class.

    This handles player logic.
    """

    def __init__(self, name: str):
        """Initialize the player."""
        self.name = name

    def _private_method(self):
        """This is private."""
        pass

    def __internal(self):
        """Double underscore private."""
        pass

def _helper():
    '''Single quotes docstring.'''
    pass
"#;

        let chunks = parser.parse(&path, content).unwrap();
        println!("Python chunks: {:?}", chunks.iter().map(|c| (&c.name, &c.kind, &c.visibility)).collect::<Vec<_>>());

        // Check calculate has doc summary and is public
        let calc_fn = chunks.iter().find(|c| c.name == "calculate");
        assert!(calc_fn.is_some(), "Should find calculate");
        let calc_fn = calc_fn.unwrap();
        assert_eq!(calc_fn.visibility, Visibility::Public, "calculate should be public");
        assert!(calc_fn.doc_summary.is_some(), "calculate should have doc summary");
        assert_eq!(
            calc_fn.doc_summary.as_ref().unwrap(),
            "Calculate the value."
        );

        // Check Player class is public with doc
        let player = chunks.iter().find(|c| c.name == "Player" && c.kind == ChunkKind::Class);
        assert!(player.is_some(), "Should find Player class");
        let player = player.unwrap();
        assert_eq!(player.visibility, Visibility::Public, "Player should be public");
        assert_eq!(player.doc_summary.as_ref().unwrap(), "A player class.");

        // Check __init__ is public (dunder method)
        let init_fn = chunks.iter().find(|c| c.name == "__init__");
        assert!(init_fn.is_some(), "Should find __init__");
        let init_fn = init_fn.unwrap();
        assert_eq!(init_fn.visibility, Visibility::Public, "__init__ should be public");
        assert_eq!(init_fn.doc_summary.as_ref().unwrap(), "Initialize the player.");

        // Check _private_method is private
        let private_method = chunks.iter().find(|c| c.name == "_private_method");
        assert!(private_method.is_some(), "Should find _private_method");
        let private_method = private_method.unwrap();
        assert_eq!(private_method.visibility, Visibility::Private, "_private_method should be private");
        assert_eq!(private_method.doc_summary.as_ref().unwrap(), "This is private.");

        // Check __internal is private (not dunder - no trailing __)
        let internal = chunks.iter().find(|c| c.name == "__internal");
        assert!(internal.is_some(), "Should find __internal");
        let internal = internal.unwrap();
        assert_eq!(internal.visibility, Visibility::Private, "__internal should be private");

        // Check _helper is private with single-quote docstring
        let helper = chunks.iter().find(|c| c.name == "_helper");
        assert!(helper.is_some(), "Should find _helper");
        let helper = helper.unwrap();
        assert_eq!(helper.visibility, Visibility::Private, "_helper should be private");
        assert_eq!(helper.doc_summary.as_ref().unwrap(), "Single quotes docstring.");
    }

    #[test]
    fn test_parse_csharp_xml_docs() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.cs");
        let content = r#"
/// <summary>A player class.</summary>
public class Player {
    /// <summary>
    /// The player's name.
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// Calculate the value.
    /// This is important.
    /// </summary>
    /// <param name="x">Input number</param>
    public int Calculate(int x) {
        return x * 2;
    }

    private void InternalMethod() {
        // no doc
    }
}

internal class InternalHelper {
    // internal is public
}

class DefaultClass {
    // no modifier, default is private
}
"#;

        let chunks = parser.parse(&path, content).unwrap();
        println!("C# chunks: {:?}", chunks.iter().map(|c| (&c.name, &c.kind, &c.visibility)).collect::<Vec<_>>());

        // Check Player class is public with doc
        let player = chunks.iter().find(|c| c.name == "Player" && c.kind == ChunkKind::Class);
        assert!(player.is_some(), "Should find Player class");
        let player = player.unwrap();
        assert_eq!(player.visibility, Visibility::Public, "Player should be public");
        assert_eq!(player.doc_summary.as_ref().unwrap(), "A player class.");

        // Check Calculate method has doc summary and is public
        let calc_fn = chunks.iter().find(|c| c.name == "Calculate");
        assert!(calc_fn.is_some(), "Should find Calculate");
        let calc_fn = calc_fn.unwrap();
        assert_eq!(calc_fn.visibility, Visibility::Public, "Calculate should be public");
        assert!(calc_fn.doc_summary.is_some(), "Calculate should have doc summary");
        assert_eq!(
            calc_fn.doc_summary.as_ref().unwrap(),
            "Calculate the value. This is important."
        );

        // Check InternalMethod is private
        let internal_method = chunks.iter().find(|c| c.name == "InternalMethod");
        assert!(internal_method.is_some(), "Should find InternalMethod");
        let internal_method = internal_method.unwrap();
        assert_eq!(internal_method.visibility, Visibility::Private, "InternalMethod should be private");
        assert!(internal_method.doc_summary.is_none(), "InternalMethod should have no doc");

        // Check InternalHelper is public (internal keyword)
        let internal_helper = chunks.iter().find(|c| c.name == "InternalHelper");
        assert!(internal_helper.is_some(), "Should find InternalHelper");
        let internal_helper = internal_helper.unwrap();
        assert_eq!(internal_helper.visibility, Visibility::Public, "InternalHelper should be public (internal)");

        // Check DefaultClass is private (no modifier)
        let default_class = chunks.iter().find(|c| c.name == "DefaultClass");
        assert!(default_class.is_some(), "Should find DefaultClass");
        let default_class = default_class.unwrap();
        assert_eq!(default_class.visibility, Visibility::Private, "DefaultClass should be private (no modifier)");
    }

}
