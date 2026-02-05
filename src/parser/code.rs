//! Code parser using tree-sitter for extracting symbols from source files.
//!
//! Supports: Rust (.rs), TypeScript (.ts, .tsx), JavaScript (.js, .jsx), C# (.cs), Python (.py)

use std::path::Path;
use anyhow::{Context, Result};
use tree_sitter::{Parser, Query, QueryCursor, Language, Node};
use streaming_iterator::StreamingIterator;

use crate::{ChunkType, ChunkKind, RawChunk};
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

        parse_with_queries(path, content, &mut self.parser, lang.get_def_query(), ts_lang)
    }
}

/// Parse content using the given queries and extract symbols.
fn parse_with_queries(
    path: &Path,
    content: &str,
    parser: &mut Parser,
    def_query_str: &str,
    lang: Language,
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

}
