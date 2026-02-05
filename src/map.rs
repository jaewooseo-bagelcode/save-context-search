//! Project Map generation for AI context injection.
//!
//! Generates a high-level overview of a codebase organized by language,
//! optimized for AI consumption (~2,000 tokens).

use std::collections::HashMap;
use std::path::Path;

use crate::{ChunkKind, ChunkType, Index, Visibility};

/// Detected source language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceLang {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    CSharp,
    Other,
}

impl SourceLang {
    /// Detect language from file extension.
    pub fn from_path(path: &str) -> Self {
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        match ext {
            "rs" => SourceLang::Rust,
            "ts" | "tsx" => SourceLang::TypeScript,
            "js" | "jsx" => SourceLang::JavaScript,
            "py" => SourceLang::Python,
            "cs" => SourceLang::CSharp,
            _ => SourceLang::Other,
        }
    }

    /// Get display name for the language.
    pub fn name(&self) -> &'static str {
        match self {
            SourceLang::Rust => "Rust",
            SourceLang::TypeScript => "TypeScript",
            SourceLang::JavaScript => "JavaScript",
            SourceLang::Python => "Python",
            SourceLang::CSharp => "C#",
            SourceLang::Other => "Other",
        }
    }
}

/// A module grouping of symbols from the same source file/directory.
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub symbols: Vec<MapSymbol>,
    pub lang: SourceLang,
}

/// A symbol in the project map.
#[derive(Debug, Clone)]
pub struct MapSymbol {
    pub name: String,
    pub kind: ChunkKind,
    pub signature: String,
    pub doc_summary: Option<String>,
}

/// Configuration for map generation.
#[derive(Debug, Clone)]
pub struct MapConfig {
    pub max_tokens: usize,
    pub include_undocumented: bool,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            include_undocumented: false,
        }
    }
}

/// Group chunks by language and module for map generation.
/// Only includes public code symbols.
/// Returns modules grouped by (language, module_name).
pub fn group_by_module(index: &Index, include_undocumented: bool) -> Vec<Module> {
    // Key: (SourceLang, module_name)
    let mut modules: HashMap<(SourceLang, String), Vec<MapSymbol>> = HashMap::new();

    for chunk in &index.chunks {
        // Filter: public only
        if chunk.visibility != Visibility::Public {
            continue;
        }

        // Filter: code only (no docs)
        if chunk.chunk_type != ChunkType::Code {
            continue;
        }

        // Filter: skip impl blocks (we want the methods, not the impl itself)
        if chunk.kind == ChunkKind::Impl {
            continue;
        }

        // Filter: if not including undocumented, require doc_summary
        if !include_undocumented && chunk.doc_summary_idx.is_none() {
            continue;
        }

        // Get string values
        let name = index.strings.get(chunk.name_idx).unwrap_or("").to_string();
        let file = index.strings.get(chunk.file_idx).unwrap_or("");
        let signature = chunk
            .signature_idx
            .and_then(|idx| index.strings.get(idx))
            .unwrap_or("")
            .to_string();
        let doc_summary = chunk
            .doc_summary_idx
            .and_then(|idx| index.strings.get(idx))
            .map(|s| s.to_string());

        // Detect language and extract module name
        let lang = SourceLang::from_path(file);
        let module_name = extract_module_name(file);

        let symbol = MapSymbol {
            name,
            kind: chunk.kind.clone(),
            signature,
            doc_summary,
        };

        modules.entry((lang, module_name)).or_default().push(symbol);
    }

    // Convert to sorted Vec
    let mut result: Vec<Module> = modules
        .into_iter()
        .map(|((lang, name), symbols)| Module { name, symbols, lang })
        .collect();

    // Sort: by language first (Rust before TS), then by module name
    result.sort_by(|a, b| {
        let lang_order = |l: &SourceLang| match l {
            SourceLang::Rust => 0,
            SourceLang::TypeScript => 1,
            SourceLang::JavaScript => 2,
            SourceLang::Python => 3,
            SourceLang::CSharp => 4,
            SourceLang::Other => 5,
        };
        lang_order(&a.lang)
            .cmp(&lang_order(&b.lang))
            .then_with(|| a.name.cmp(&b.name))
    });

    // Sort symbols within each module by kind then name
    for module in &mut result {
        module.symbols.sort_by(|a, b| {
            let kind_order = |k: &ChunkKind| match k {
                ChunkKind::Class | ChunkKind::Struct => 0,
                ChunkKind::Interface => 1,
                ChunkKind::Enum => 2,
                ChunkKind::Function => 3,
                ChunkKind::Method => 4,
                ChunkKind::Constant => 5,
                _ => 6,
            };
            kind_order(&a.kind)
                .cmp(&kind_order(&b.kind))
                .then_with(|| a.name.cmp(&b.name))
        });
    }

    result
}

/// Extract module name from file path.
/// Examples:
/// - src/parser/code.rs -> parser
/// - src/index/mod.rs -> index
/// - src/lib.rs -> lib
/// - src/main.rs -> main
fn extract_module_name(file_path: &str) -> String {
    let path = Path::new(file_path);

    // Get parent directory
    if let Some(parent) = path.parent() {
        if let Some(parent_name) = parent.file_name() {
            let parent_str = parent_name.to_string_lossy();
            // If parent is not "src", use parent name as module
            if parent_str != "src" {
                return parent_str.to_string();
            }
        }
    }

    // Fallback to file stem (e.g., lib.rs -> lib)
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

// ============================================================================
// Rust Formatter
// ============================================================================

/// Format modules as Rust code for AI context.
/// Output format:
/// ```text
/// //! Project Map: project_name
///
/// /// Module description
/// pub mod module_name {
///     /// Function doc
///     pub fn function_name(args) -> ReturnType;
/// }
/// ```
pub fn format_rust_map(project_name: &str, modules: &[Module], max_tokens: usize) -> String {
    let mut output = String::new();
    let mut tokens = 0;

    // Header
    let header = format!(
        "//! Project Map: {}\n//! Auto-generated by SCS\n\n",
        project_name
    );
    output.push_str(&header);
    tokens += estimate_tokens(&header);

    for module in modules {
        let mut module_str = String::new();

        // Module start
        module_str.push_str(&format!("/// {}\npub mod {} {{\n", module.name, module.name));

        for symbol in &module.symbols {
            // Doc comment
            if let Some(doc) = &symbol.doc_summary {
                module_str.push_str(&format!("    /// {}\n", doc));
            }

            // Signature based on kind
            match symbol.kind {
                ChunkKind::Function | ChunkKind::Method => {
                    let simplified = simplify_rust_signature(&symbol.signature);
                    module_str.push_str(&format!("    pub fn {};\n", simplified));
                }
                ChunkKind::Class | ChunkKind::Struct => {
                    module_str.push_str(&format!("    pub struct {};\n", symbol.name));
                }
                ChunkKind::Interface => {
                    module_str.push_str(&format!("    pub trait {};\n", symbol.name));
                }
                ChunkKind::Enum => {
                    module_str.push_str(&format!("    pub enum {};\n", symbol.name));
                }
                ChunkKind::Constant => {
                    module_str.push_str(&format!("    pub const {};\n", symbol.name));
                }
                _ => {}
            }
        }

        module_str.push_str("}\n\n");

        // Check token budget
        let cost = estimate_tokens(&module_str);
        if tokens + cost > max_tokens {
            output.push_str("// ... more modules\n");
            break;
        }

        output.push_str(&module_str);
        tokens += cost;
    }

    output
}

/// Simplify a Rust function signature for the map.
/// Removes: pub, async, fn keywords, preserves name(args) -> ReturnType
fn simplify_rust_signature(sig: &str) -> String {
    sig.trim()
        .trim_start_matches("pub ")
        .trim_start_matches("async ")
        .trim_start_matches("fn ")
        .trim_end_matches('{')
        .trim_end_matches(';')
        .trim()
        .to_string()
}

/// Estimate token count (rough: ~4 chars per token).
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

// ============================================================================
// TypeScript Formatter
// ============================================================================

/// Format modules as TypeScript types for AI context.
/// Output format:
/// ```text
/// /**
///  * Project Map: project_name
///  */
/// type ProjectMap = {
///   /** Module description */
///   moduleName: {
///     /** Function doc */
///     functionName: (args) => ReturnType;
///   };
/// };
/// ```
pub fn format_typescript_map(project_name: &str, modules: &[Module], max_tokens: usize) -> String {
    let mut output = String::new();
    let mut tokens = 0;

    // Header
    let header = format!(
        "/**\n * Project Map: {}\n * Auto-generated by SCS\n */\ntype ProjectMap = {{\n",
        project_name
    );
    output.push_str(&header);
    tokens += estimate_tokens(&header);

    for module in modules {
        let mut module_str = String::new();

        // Module start
        module_str.push_str(&format!("  /** {} */\n  {}: {{\n", module.name, module.name));

        for symbol in &module.symbols {
            // Doc comment
            if let Some(doc) = &symbol.doc_summary {
                module_str.push_str(&format!("    /** {} */\n", doc));
            }

            // Type based on kind
            match symbol.kind {
                ChunkKind::Function | ChunkKind::Method => {
                    let ts_type = convert_to_ts_type(&symbol.name, &symbol.signature);
                    module_str.push_str(&format!("    {};\n", ts_type));
                }
                ChunkKind::Class | ChunkKind::Struct => {
                    module_str.push_str(&format!("    {}: {{}};\n", symbol.name));
                }
                ChunkKind::Interface => {
                    module_str.push_str(&format!("    {}: {{}};\n", symbol.name));
                }
                ChunkKind::Enum => {
                    module_str.push_str(&format!("    {}: {{}};\n", symbol.name));
                }
                ChunkKind::Constant => {
                    module_str.push_str(&format!("    {}: unknown;\n", symbol.name));
                }
                _ => {}
            }
        }

        module_str.push_str("  };\n");

        // Check token budget
        let cost = estimate_tokens(&module_str);
        if tokens + cost > max_tokens {
            output.push_str("  // ... more modules\n");
            break;
        }

        output.push_str(&module_str);
        tokens += cost;
    }

    output.push_str("};\n");
    output
}

/// Convert a function signature to TypeScript arrow function type.
/// Example: "function calculateDifficulty(level: number): number"
///       -> "calculateDifficulty: (level: number) => number"
fn convert_to_ts_type(name: &str, signature: &str) -> String {
    // Try to extract params and return type from signature
    if let Some(params_start) = signature.find('(') {
        if let Some(params_end) = signature.rfind(')') {
            let params = &signature[params_start..=params_end];

            // Look for return type after ) or after :
            let after_params = &signature[params_end + 1..];
            let return_type = if let Some(colon_pos) = after_params.find(':') {
                after_params[colon_pos + 1..]
                    .trim()
                    .trim_end_matches('{')
                    .trim_end_matches(';')
                    .trim()
            } else {
                "void"
            };

            return format!("{}: {} => {}", name, params, return_type);
        }
    }

    // Fallback
    format!("{}: unknown", name)
}

// ============================================================================
// Mixed Language Formatter (Default)
// ============================================================================

/// Format modules grouped by language.
/// Each language section uses its native syntax.
/// This is the default format for mixed-language projects.
pub fn format_mixed_map(project_name: &str, modules: &[Module], max_tokens: usize) -> String {
    let mut output = String::new();
    let mut tokens = 0;

    // Header
    let header = format!(
        "// Project Map: {}\n// Auto-generated by SCS\n\n",
        project_name
    );
    output.push_str(&header);
    tokens += estimate_tokens(&header);

    // Group modules by language
    let mut current_lang: Option<SourceLang> = None;

    for module in modules {
        // Language section header
        if current_lang != Some(module.lang) {
            if current_lang.is_some() {
                output.push('\n');
            }
            let lang_header = format!("// ── {} ──\n\n", module.lang.name());
            output.push_str(&lang_header);
            tokens += estimate_tokens(&lang_header);
            current_lang = Some(module.lang);
        }

        // Format module based on its language
        let module_str = match module.lang {
            SourceLang::Rust => format_rust_module(module),
            SourceLang::TypeScript | SourceLang::JavaScript => format_ts_module(module),
            SourceLang::Python => format_python_module(module),
            SourceLang::CSharp => format_csharp_module(module),
            SourceLang::Other => format_rust_module(module), // fallback
        };

        // Check token budget
        let cost = estimate_tokens(&module_str);
        if tokens + cost > max_tokens {
            output.push_str("// ... more modules\n");
            break;
        }

        output.push_str(&module_str);
        tokens += cost;
    }

    output
}

/// Format a single module in Rust style.
fn format_rust_module(module: &Module) -> String {
    let mut s = String::new();
    s.push_str(&format!("/// {}\npub mod {} {{\n", module.name, module.name));

    for symbol in &module.symbols {
        if let Some(doc) = &symbol.doc_summary {
            s.push_str(&format!("    /// {}\n", doc));
        }
        match symbol.kind {
            ChunkKind::Function | ChunkKind::Method => {
                let simplified = simplify_rust_signature(&symbol.signature);
                s.push_str(&format!("    pub fn {};\n", simplified));
            }
            ChunkKind::Class | ChunkKind::Struct => {
                s.push_str(&format!("    pub struct {};\n", symbol.name));
            }
            ChunkKind::Interface => {
                s.push_str(&format!("    pub trait {};\n", symbol.name));
            }
            ChunkKind::Enum => {
                s.push_str(&format!("    pub enum {};\n", symbol.name));
            }
            ChunkKind::Constant => {
                s.push_str(&format!("    pub const {};\n", symbol.name));
            }
            _ => {}
        }
    }
    s.push_str("}\n\n");
    s
}

/// Format a single module in TypeScript style.
fn format_ts_module(module: &Module) -> String {
    let mut s = String::new();
    s.push_str(&format!("/** {} */\nnamespace {} {{\n", module.name, module.name));

    for symbol in &module.symbols {
        if let Some(doc) = &symbol.doc_summary {
            s.push_str(&format!("  /** {} */\n", doc));
        }
        match symbol.kind {
            ChunkKind::Function | ChunkKind::Method => {
                let ts_type = convert_to_ts_type(&symbol.name, &symbol.signature);
                s.push_str(&format!("  export {};\n", ts_type));
            }
            ChunkKind::Class => {
                s.push_str(&format!("  export class {} {{}}\n", symbol.name));
            }
            ChunkKind::Struct => {
                s.push_str(&format!("  export interface {} {{}}\n", symbol.name));
            }
            ChunkKind::Interface => {
                s.push_str(&format!("  export interface {} {{}}\n", symbol.name));
            }
            ChunkKind::Enum => {
                s.push_str(&format!("  export enum {} {{}}\n", symbol.name));
            }
            ChunkKind::Constant => {
                s.push_str(&format!("  export const {}: unknown;\n", symbol.name));
            }
            _ => {}
        }
    }
    s.push_str("}\n\n");
    s
}

/// Format a single module in Python style.
fn format_python_module(module: &Module) -> String {
    let mut s = String::new();
    s.push_str(&format!("# {}\nclass {}:\n", module.name, module.name));

    for symbol in &module.symbols {
        if let Some(doc) = &symbol.doc_summary {
            s.push_str(&format!("    \"\"\"{}\"\"\";\n", doc));
        }
        match symbol.kind {
            ChunkKind::Function | ChunkKind::Method => {
                s.push_str(&format!("    def {}(): ...\n", symbol.name));
            }
            ChunkKind::Class => {
                s.push_str(&format!("    class {}: ...\n", symbol.name));
            }
            _ => {}
        }
    }
    s.push_str("\n");
    s
}

/// Format a single module in C# style.
fn format_csharp_module(module: &Module) -> String {
    let mut s = String::new();
    s.push_str(&format!("/// {}\nnamespace {} {{\n", module.name, module.name));

    for symbol in &module.symbols {
        if let Some(doc) = &symbol.doc_summary {
            s.push_str(&format!("    /// <summary>{}</summary>\n", doc));
        }
        match symbol.kind {
            ChunkKind::Function | ChunkKind::Method => {
                s.push_str(&format!("    public void {}();\n", symbol.name));
            }
            ChunkKind::Class | ChunkKind::Struct => {
                s.push_str(&format!("    public class {} {{}}\n", symbol.name));
            }
            ChunkKind::Interface => {
                s.push_str(&format!("    public interface {} {{}}\n", symbol.name));
            }
            ChunkKind::Enum => {
                s.push_str(&format!("    public enum {} {{}}\n", symbol.name));
            }
            _ => {}
        }
    }
    s.push_str("}\n\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_module_name() {
        assert_eq!(extract_module_name("src/parser/code.rs"), "parser");
        assert_eq!(extract_module_name("src/index/mod.rs"), "index");
        assert_eq!(extract_module_name("src/lib.rs"), "lib");
        assert_eq!(extract_module_name("src/main.rs"), "main");
        assert_eq!(extract_module_name("src/search/semantic.rs"), "search");
    }

    #[test]
    fn test_extract_module_name_nested() {
        assert_eq!(
            extract_module_name("src/parser/queries/mod.rs"),
            "queries"
        );
        assert_eq!(extract_module_name("lib/utils/helpers.ts"), "utils");
    }

    #[test]
    fn test_simplify_rust_signature() {
        assert_eq!(
            simplify_rust_signature("pub fn hello(name: &str) -> String {"),
            "hello(name: &str) -> String"
        );
        assert_eq!(
            simplify_rust_signature("pub async fn fetch() -> Result<Data>"),
            "fetch() -> Result<Data>"
        );
        assert_eq!(
            simplify_rust_signature("fn private()"),
            "private()"
        );
    }

    #[test]
    fn test_format_rust_map() {
        let modules = vec![
            Module {
                name: "parser".to_string(),
                lang: SourceLang::Rust,
                symbols: vec![
                    MapSymbol {
                        name: "parse".to_string(),
                        kind: ChunkKind::Function,
                        signature: "pub fn parse(content: &str) -> Vec<Chunk>".to_string(),
                        doc_summary: Some("Parse source code".to_string()),
                    },
                ],
            },
        ];

        let output = format_rust_map("test-project", &modules, 2000);

        assert!(output.contains("Project Map: test-project"));
        assert!(output.contains("pub mod parser {"));
        assert!(output.contains("/// Parse source code"));
        assert!(output.contains("pub fn parse(content: &str) -> Vec<Chunk>;"));
    }

    #[test]
    fn test_format_rust_map_token_limit() {
        // Create many modules to exceed token limit
        let modules: Vec<Module> = (0..100)
            .map(|i| Module {
                name: format!("module_{}", i),
                lang: SourceLang::Rust,
                symbols: vec![MapSymbol {
                    name: format!("function_{}", i),
                    kind: ChunkKind::Function,
                    signature: format!("pub fn function_{}() -> Result<()>", i),
                    doc_summary: Some(format!("Does something {}", i)),
                }],
            })
            .collect();

        let output = format_rust_map("test", &modules, 500);

        assert!(output.contains("// ... more modules"));
    }

    #[test]
    fn test_convert_to_ts_type() {
        assert_eq!(
            convert_to_ts_type("calculate", "function calculate(x: number): number"),
            "calculate: (x: number) => number"
        );
        assert_eq!(
            convert_to_ts_type("greet", "greet(name: string): string {"),
            "greet: (name: string) => string"
        );
        assert_eq!(
            convert_to_ts_type("noReturn", "noReturn()"),
            "noReturn: () => void"
        );
    }

    #[test]
    fn test_format_typescript_map() {
        let modules = vec![
            Module {
                name: "utils".to_string(),
                lang: SourceLang::TypeScript,
                symbols: vec![
                    MapSymbol {
                        name: "calculate".to_string(),
                        kind: ChunkKind::Function,
                        signature: "function calculate(x: number): number".to_string(),
                        doc_summary: Some("Calculate a value".to_string()),
                    },
                ],
            },
        ];

        let output = format_typescript_map("my-app", &modules, 2000);

        assert!(output.contains("Project Map: my-app"));
        assert!(output.contains("type ProjectMap = {"));
        assert!(output.contains("utils: {"));
        assert!(output.contains("/** Calculate a value */"));
        assert!(output.contains("calculate: (x: number) => number;"));
        assert!(output.contains("};"));
    }

    #[test]
    fn test_format_typescript_map_token_limit() {
        let modules: Vec<Module> = (0..100)
            .map(|i| Module {
                name: format!("module_{}", i),
                lang: SourceLang::TypeScript,
                symbols: vec![MapSymbol {
                    name: format!("function_{}", i),
                    kind: ChunkKind::Function,
                    signature: format!("function function_{}(): void", i),
                    doc_summary: Some(format!("Does something {}", i)),
                }],
            })
            .collect();

        let output = format_typescript_map("test", &modules, 500);

        assert!(output.contains("// ... more modules"));
    }

    #[test]
    fn test_format_mixed_map() {
        let modules = vec![
            Module {
                name: "backend".to_string(),
                lang: SourceLang::Rust,
                symbols: vec![
                    MapSymbol {
                        name: "handle_request".to_string(),
                        kind: ChunkKind::Function,
                        signature: "pub fn handle_request() -> Response".to_string(),
                        doc_summary: Some("Handle HTTP request".to_string()),
                    },
                ],
            },
            Module {
                name: "frontend".to_string(),
                lang: SourceLang::TypeScript,
                symbols: vec![
                    MapSymbol {
                        name: "render".to_string(),
                        kind: ChunkKind::Function,
                        signature: "function render(): void".to_string(),
                        doc_summary: Some("Render UI".to_string()),
                    },
                ],
            },
        ];

        let output = format_mixed_map("tauri-app", &modules, 2000);

        // Check header
        assert!(output.contains("Project Map: tauri-app"));
        // Check language sections
        assert!(output.contains("// ── Rust ──"));
        assert!(output.contains("// ── TypeScript ──"));
        // Check Rust module
        assert!(output.contains("pub mod backend {"));
        assert!(output.contains("/// Handle HTTP request"));
        // Check TypeScript module
        assert!(output.contains("namespace frontend {"));
        assert!(output.contains("/** Render UI */"));
    }

    #[test]
    fn test_source_lang_from_path() {
        assert_eq!(SourceLang::from_path("src/lib.rs"), SourceLang::Rust);
        assert_eq!(SourceLang::from_path("src/app.ts"), SourceLang::TypeScript);
        assert_eq!(SourceLang::from_path("src/app.tsx"), SourceLang::TypeScript);
        assert_eq!(SourceLang::from_path("src/utils.js"), SourceLang::JavaScript);
        assert_eq!(SourceLang::from_path("src/main.py"), SourceLang::Python);
        assert_eq!(SourceLang::from_path("src/Program.cs"), SourceLang::CSharp);
        assert_eq!(SourceLang::from_path("README.md"), SourceLang::Other);
    }
}
