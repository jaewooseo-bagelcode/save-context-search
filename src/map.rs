//! Hierarchical Project Map generation for AI context injection.
//!
//! Generates a hierarchical overview of a codebase:
//!   L0: Directory    — src/parser/ (3 files, 15 symbols)
//!   L1: File         — src/parser/code.rs (8 symbols)
//!   L2: Top Symbol   — class PlayerController / fn main
//!   L3: Member       — method Move(), field Health
//!
//! Token budget drives depth: auto-selects maximum depth that fits.

use std::collections::BTreeMap;
use std::path::Path;

use crate::{ChunkKind, ChunkType, Index, Visibility};

// ============================================================================
// Data Model
// ============================================================================

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

/// Directory node — L0
#[derive(Debug, Clone)]
pub struct DirNode {
    pub path: String,
    pub file_count: usize,
    pub symbol_count: usize,
    pub files: Vec<FileNode>,
    pub summary: Option<String>,
}

/// File node — L1
#[derive(Debug, Clone)]
pub struct FileNode {
    pub name: String,
    pub lang: SourceLang,
    pub symbols: Vec<SymbolNode>,
    pub summary: Option<String>,
}

/// Symbol node — L2 (top-level) or L3 (member)
#[derive(Debug, Clone)]
pub struct SymbolNode {
    pub name: String,
    pub kind: ChunkKind,
    pub signature: Option<String>,
    pub summary: Option<String>,
    pub visibility: Visibility,
    pub members: Vec<SymbolNode>,
}

/// Configuration for map generation.
#[derive(Debug, Clone)]
pub struct MapConfig {
    pub max_tokens: usize,
    pub area: Option<String>,
}

/// Zoom level for --area drill-down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZoomLevel {
    /// Full project overview (no --area)
    Project,
    /// Directory zoom: shows files and symbols within an area directory
    Directory,
    /// File zoom: shows function list for a single file
    File,
}

/// Detect the zoom level based on the --area path.
pub fn detect_zoom_level(area: &str, dirs: &[DirNode]) -> ZoomLevel {
    let area_norm = area.trim_end_matches('/');

    // Check if area matches a file (has a file extension in the last component)
    let last_component = area_norm.rsplit('/').next().unwrap_or(area_norm);
    if last_component.contains('.') {
        // Verify it exists as a file in some directory
        let dir_part = extract_dir_path(area_norm);
        let file_part = extract_file_name(area_norm);
        for dir in dirs {
            if dir.path == dir_part
                && dir.files.iter().any(|f| f.name == file_part)
            {
                return ZoomLevel::File;
            }
        }
    }

    // Check if area matches any directory
    for dir in dirs {
        if dir.path == area_norm || dir.path.starts_with(&format!("{}/", area_norm)) {
            return ZoomLevel::Directory;
        }
    }

    ZoomLevel::Project // area not found
}

/// Generate a zoom hint line for the output footer.
pub fn zoom_hint(zoom: ZoomLevel, area: &str, dirs: &[DirNode]) -> String {
    let area_norm = area.trim_end_matches('/');

    match zoom {
        ZoomLevel::Project => {
            // Suggest top 3 largest directories
            let mut sorted: Vec<&DirNode> = dirs.iter().collect();
            sorted.sort_by(|a, b| b.symbol_count.cmp(&a.symbol_count));
            let suggestions: Vec<String> = sorted.iter()
                .take(3)
                .map(|d| format!("{} ({} symbols)", d.path, d.symbol_count))
                .collect();
            if suggestions.is_empty() {
                return String::new();
            }
            format!(
                "//\n// [zoom: project] Drill deeper: scs map --area <path>\n// Suggested: {}",
                suggestions.join(", ")
            )
        }
        ZoomLevel::Directory => {
            // Find subdirectories within the area, or files if no subdirs
            let area_prefix = format!("{}/", area_norm);

            // Collect direct child directories (one level deeper)
            let mut child_dirs: Vec<(&str, usize, usize)> = Vec::new();
            for dir in dirs {
                if let Some(rest) = dir.path.strip_prefix(&area_prefix) {
                    // Only direct children (no more slashes)
                    if !rest.contains('/') {
                        child_dirs.push((&dir.path, dir.file_count, dir.symbol_count));
                    }
                }
            }

            if !child_dirs.is_empty() {
                child_dirs.sort_by(|a, b| b.2.cmp(&a.2));
                let suggestions: Vec<String> = child_dirs.iter()
                    .take(3)
                    .map(|(p, f, s)| format!("{} ({} files, {} symbols)", p, f, s))
                    .collect();
                format!(
                    "//\n// [zoom: directory] Drill deeper: scs map --area <path>\n// Suggested: {}",
                    suggestions.join(", ")
                )
            } else {
                // No subdirs — suggest files in the area directory
                let mut area_files: Vec<(&str, &str, usize)> = Vec::new();
                for dir in dirs {
                    if dir.path == area_norm {
                        for file in &dir.files {
                            let sym_count = file.symbols.iter().map(|s| 1 + s.members.len()).sum::<usize>();
                            area_files.push((&dir.path, &file.name, sym_count));
                        }
                    }
                }
                if area_files.is_empty() {
                    return String::new();
                }
                area_files.sort_by(|a, b| b.2.cmp(&a.2));
                let suggestions: Vec<String> = area_files.iter()
                    .take(3)
                    .map(|(d, f, s)| format!("{}/{} ({} symbols)", d, f, s))
                    .collect();
                format!(
                    "//\n// [zoom: directory] Drill to file: scs map --area <file-path>\n// Suggested: {}",
                    suggestions.join(", ")
                )
            }
        }
        ZoomLevel::File => {
            "//\n// [zoom: file] Use 'scs lookup <symbol>' for full definition".to_string()
        }
    }
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            area: None,
        }
    }
}

/// Directory tree node for rendering large projects.
/// Unlike flat DirNode, this represents a proper tree with parent-child relationships.
#[derive(Debug, Clone)]
pub struct DirTree {
    pub name: String,
    pub total_files: usize,
    pub total_symbols: usize,
    pub children: Vec<DirTree>,
    pub summary: Option<String>,
}

// ============================================================================
// Build Hierarchy
// ============================================================================

/// Build hierarchical directory→file→symbol tree from the index.
/// All code symbols are included (public + private).
pub fn build_hierarchy(index: &Index, root_path: &str) -> Vec<DirNode> {
    // Group chunks by file_idx
    let mut file_chunks: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, chunk) in index.chunks.iter().enumerate() {
        if chunk.chunk_type != ChunkType::Code {
            continue;
        }
        // Skip impl blocks
        if chunk.kind == ChunkKind::Impl {
            continue;
        }
        file_chunks.entry(chunk.file_idx).or_default().push(i);
    }

    // Build file nodes grouped by directory
    // Key: directory relative path, Value: Vec<FileNode>
    let mut dir_files: BTreeMap<String, Vec<FileNode>> = BTreeMap::new();

    for (file_idx, chunk_idxs) in &file_chunks {
        let file_path = match index.strings.get(*file_idx) {
            Some(p) => p,
            None => continue,
        };

        let rel_path = make_relative(file_path, root_path);
        let dir_path = extract_dir_path(&rel_path);
        let file_name = extract_file_name(&rel_path);
        let lang = SourceLang::from_path(file_path);

        // Build symbol tree for this file
        // First pass: identify top-level symbols and parent contexts
        let mut top_symbols: Vec<SymbolNode> = Vec::new();
        // context string -> index in top_symbols
        let mut context_map: BTreeMap<String, usize> = BTreeMap::new();

        // First: add classes/structs/enums/interfaces as top-level containers
        for &ci in chunk_idxs {
            let chunk = &index.chunks[ci];
            if matches!(chunk.kind, ChunkKind::Class | ChunkKind::Struct | ChunkKind::Interface | ChunkKind::Enum) {
                let name = index.strings.get(chunk.name_idx).unwrap_or("").to_string();
                let signature = chunk.signature_idx.and_then(|i| index.strings.get(i)).map(|s| s.to_string());
                let summary = get_summary(index, chunk);

                let sym_idx = top_symbols.len();
                context_map.insert(name.clone(), sym_idx);
                top_symbols.push(SymbolNode {
                    name,
                    kind: chunk.kind.clone(),
                    signature,
                    summary,
                    visibility: chunk.visibility,
                    members: Vec::new(),
                });
            }
        }

        // Second: add functions/methods/constants/fields
        for &ci in chunk_idxs {
            let chunk = &index.chunks[ci];
            if matches!(chunk.kind, ChunkKind::Class | ChunkKind::Struct | ChunkKind::Interface | ChunkKind::Enum) {
                continue; // already handled
            }

            let name = index.strings.get(chunk.name_idx).unwrap_or("").to_string();
            let signature = chunk.signature_idx.and_then(|i| index.strings.get(i)).map(|s| s.to_string());
            let summary = get_summary(index, chunk);

            let sym = SymbolNode {
                name,
                kind: chunk.kind.clone(),
                signature,
                summary,
                visibility: chunk.visibility,
                members: Vec::new(),
            };

            // Check if this chunk has a parent context (class/struct)
            if let Some(context_idx) = chunk.context_idx {
                let context_name = index.strings.get(context_idx).unwrap_or("");
                if let Some(&parent_idx) = context_map.get(context_name) {
                    top_symbols[parent_idx].members.push(sym);
                    continue;
                }
            }

            // Top-level symbol
            top_symbols.push(sym);
        }

        // Sort: types first (class/struct/enum/interface), then functions, then others
        top_symbols.sort_by(|a, b| {
            symbol_sort_key(&a.kind).cmp(&symbol_sort_key(&b.kind))
                .then_with(|| a.name.cmp(&b.name))
        });
        for sym in &mut top_symbols {
            sym.members.sort_by(|a, b| {
                symbol_sort_key(&a.kind).cmp(&symbol_sort_key(&b.kind))
                    .then_with(|| a.name.cmp(&b.name))
            });
        }

        let file_node = FileNode {
            name: file_name,
            lang,
            symbols: top_symbols,
            summary: None,
        };

        dir_files.entry(dir_path).or_default().push(file_node);
    }

    // Build DirNode vec
    let mut dirs: Vec<DirNode> = dir_files
        .into_iter()
        .map(|(path, mut files)| {
            files.sort_by(|a, b| a.name.cmp(&b.name));
            let file_count = files.len();
            let symbol_count: usize = files.iter().map(count_symbols).sum();
            DirNode {
                path,
                file_count,
                symbol_count,
                files,
                summary: None,
            }
        })
        .collect();

    dirs.sort_by(|a, b| a.path.cmp(&b.path));
    dirs
}

fn get_summary(index: &Index, chunk: &crate::Chunk) -> Option<String> {
    chunk.doc_summary_idx
        .and_then(|idx| index.strings.get(idx))
        .map(|s| s.to_string())
}

fn count_symbols(file: &FileNode) -> usize {
    file.symbols.iter().map(|s| 1 + s.members.len()).sum()
}

fn symbol_sort_key(kind: &ChunkKind) -> u8 {
    match kind {
        ChunkKind::Class | ChunkKind::Struct => 0,
        ChunkKind::Interface => 1,
        ChunkKind::Enum => 2,
        ChunkKind::Function => 3,
        ChunkKind::Method => 4,
        ChunkKind::Constant => 5,
        ChunkKind::Field => 6,
        _ => 7,
    }
}

/// Make a file path relative to root.
fn make_relative(path: &str, root: &str) -> String {
    let root_prefix = if root.ends_with('/') {
        root.to_string()
    } else {
        format!("{}/", root)
    };
    if path.starts_with(&root_prefix) {
        path[root_prefix.len()..].to_string()
    } else {
        path.to_string()
    }
}

/// Extract directory path from a relative file path.
/// "src/parser/code.rs" → "src/parser"
/// "lib.rs" → "."
pub fn extract_dir_path(rel_path: &str) -> String {
    match rel_path.rfind('/') {
        Some(i) => rel_path[..i].to_string(),
        None => ".".to_string(),
    }
}

/// Extract file name from a relative path.
/// "src/parser/code.rs" → "code.rs"
fn extract_file_name(rel_path: &str) -> String {
    match rel_path.rfind('/') {
        Some(i) => rel_path[i + 1..].to_string(),
        None => rel_path.to_string(),
    }
}

// ============================================================================
// Auto Depth Selection
// ============================================================================

/// Count entries at each depth level.
#[cfg(test)]
fn count_at_depth(dirs: &[DirNode]) -> [usize; 4] {
    let mut counts = [0usize; 4];

    // L0: directories
    counts[0] = dirs.len();

    // L1: + files
    let total_files: usize = dirs.iter().map(|d| d.files.len()).sum();
    counts[1] = counts[0] + total_files;

    // L2: + top-level symbols
    let total_top_symbols: usize = dirs.iter()
        .flat_map(|d| &d.files)
        .map(|f| f.symbols.len())
        .sum();
    counts[2] = counts[1] + total_top_symbols;

    // L3: + members
    let total_members: usize = dirs.iter()
        .flat_map(|d| &d.files)
        .flat_map(|f| &f.symbols)
        .map(|s| s.members.len())
        .sum();
    counts[3] = counts[2] + total_members;

    counts
}

/// Estimate tokens per entry at different depths.
/// L0: ~25 chars per dir line ("  src/parser/       — 3 files, 15 symbols\n")
/// L1: ~30 chars per file line ("    code.rs         — 8 symbols\n")
/// L2: ~40 chars per symbol line ("      fn parse(content: &str) -> Vec<Chunk>\n")
/// L3: ~35 chars per member line ("        method Move()\n")
const CHARS_PER_TOKEN: usize = 4;
const HEADER_TOKENS: usize = 50;

fn estimate_tokens_for_depth(dirs: &[DirNode], depth: usize) -> usize {
    let mut chars = 0usize;

    for dir in dirs {
        // L0: directory line + optional summary (~50 chars)
        chars += 20 + dir.path.len() + 30;
        if dir.summary.is_some() {
            chars += 55; // " (summary text)\n"
        }

        if depth >= 1 {
            for file in &dir.files {
                // L1: file line + optional summary (~50 chars)
                chars += 24 + file.name.len() + 20;
                if file.summary.is_some() {
                    chars += 55; // " — summary text"
                }

                if depth >= 2 {
                    for sym in &file.symbols {
                        // L2: symbol line
                        chars += 6 + symbol_line_len(sym);

                        if depth >= 3 {
                            for member in &sym.members {
                                // L3: member line
                                chars += 8 + symbol_line_len(member);
                            }
                        }
                    }
                }
            }
        }
    }

    HEADER_TOKENS + chars.div_ceil(CHARS_PER_TOKEN)
}

fn symbol_line_len(sym: &SymbolNode) -> usize {
    let base = sym.name.len() + 10; // kind prefix + padding
    if let Some(ref sig) = sym.signature {
        base + sig.len().min(80) // cap signature display
    } else {
        base
    }
}

/// Check if the project is large enough to trigger tree mode rendering.
/// Tree mode doesn't display summaries, so summary generation can be skipped.
pub fn would_use_tree_mode(dirs: &[DirNode], max_tokens: usize) -> bool {
    estimate_tokens_for_depth(dirs, 0) > max_tokens
}

/// Select maximum depth that fits within the token budget.
fn auto_select_depth(dirs: &[DirNode], max_tokens: usize) -> usize {
    for depth in (0..=3).rev() {
        if estimate_tokens_for_depth(dirs, depth) <= max_tokens {
            return depth;
        }
    }
    0 // L0 always fits (fallback)
}

// ============================================================================
// Directory Tree (for large projects where flat L0 overflows)
// ============================================================================

/// Build a directory tree from a flat list of DirNodes.
/// Groups directories by their path components into a proper tree hierarchy.
pub fn build_dir_tree(dirs: &[DirNode]) -> Vec<DirTree> {
    // Group by first path component
    let mut groups: BTreeMap<String, (usize, usize, Vec<(String, usize, usize)>)> = BTreeMap::new();

    for dir in dirs {
        let (first, rest) = match dir.path.find('/') {
            Some(i) => (&dir.path[..i], Some(&dir.path[i + 1..])),
            None => (dir.path.as_str(), None),
        };

        let entry = groups
            .entry(first.to_string())
            .or_insert((0, 0, Vec::new()));

        match rest {
            None => {
                entry.0 += dir.file_count;
                entry.1 += dir.symbol_count;
            }
            Some(rest_path) => {
                entry
                    .2
                    .push((rest_path.to_string(), dir.file_count, dir.symbol_count));
            }
        }
    }

    groups
        .into_iter()
        .map(
            |(name, (direct_files, direct_symbols, child_entries))| {
                // Recurse on child entries
                let child_dirs: Vec<DirNode> = child_entries
                    .into_iter()
                    .map(|(path, fc, sc)| DirNode {
                        path,
                        file_count: fc,
                        symbol_count: sc,
                        files: Vec::new(),
                        summary: None,
                    })
                    .collect();

                let children = build_dir_tree(&child_dirs);
                let child_files: usize = children.iter().map(|c| c.total_files).sum();
                let child_symbols: usize = children.iter().map(|c| c.total_symbols).sum();

                DirTree {
                    name,
                    total_files: direct_files + child_files,
                    total_symbols: direct_symbols + child_symbols,
                    children,
                    summary: None,
                }
            },
        )
        .collect()
}

/// Collapse single-child chains: Assets/ → Blood Invasion/ → Scripts/
/// becomes Assets/Blood Invasion/Scripts/ (saves vertical space).
pub fn collapse_single_children(tree: &mut DirTree) {
    // Only collapse if parent has no direct files (all files are in children)
    while tree.children.len() == 1 {
        let child = &tree.children[0];
        // If child's total equals parent's total, parent has no direct files → safe to collapse
        if child.total_files < tree.total_files {
            break; // Parent has direct files, don't collapse
        }
        let child = tree.children.remove(0);
        tree.name = format!("{}/{}", tree.name, child.name);
        tree.children = child.children;
    }
    for child in &mut tree.children {
        collapse_single_children(child);
    }
}

/// Estimate tokens for tree rendering at a given max_depth.
fn estimate_tree_tokens(trees: &[DirTree], max_depth: usize, indent: usize) -> usize {
    let mut chars = 0usize;
    for tree in trees {
        // "// {indent}{name}/  — N files, M symbols (summary)\n"
        chars += 3 + indent * 2 + tree.name.len() + 35;
        if tree.summary.is_some() {
            chars += 55;
        }

        if !tree.children.is_empty() && indent < max_depth {
            chars += estimate_tree_tokens(&tree.children, max_depth, indent + 1);
        }
    }
    chars
}

/// Select maximum tree depth that fits within the token budget.
fn auto_select_tree_depth(trees: &[DirTree], max_tokens: usize) -> usize {
    let budget_chars = max_tokens * CHARS_PER_TOKEN;
    for depth in (0..=10).rev() {
        if estimate_tree_tokens(trees, depth, 0) <= budget_chars {
            return depth;
        }
    }
    0
}

/// Render directory tree at given depth with indentation.
fn render_trees(out: &mut String, trees: &[DirTree], max_depth: usize, indent: usize) {
    let prefix = format!("//{} ", "  ".repeat(indent));
    for tree in trees {
        let summary_suffix = match &tree.summary {
            Some(s) => format!(" ({})", s),
            None => String::new(),
        };
        out.push_str(&format!(
            "{}{}/  — {} files, {} symbols{}\n",
            prefix, tree.name, tree.total_files, tree.total_symbols, summary_suffix
        ));
        if !tree.children.is_empty() && indent < max_depth {
            render_trees(out, &tree.children, max_depth, indent + 1);
        }
    }
}

/// Render tree-based map for large projects.
pub fn render_tree_map(
    trees: &[DirTree],
    project_name: &str,
    primary_lang: SourceLang,
    total_symbols: usize,
    max_tokens: usize,
) -> String {
    let mut out = String::new();

    let header = format!(
        "// Project: {} ({}, {} symbols)\n//\n",
        project_name,
        primary_lang.name(),
        total_symbols,
    );
    let header_tokens = header.len().div_ceil(CHARS_PER_TOKEN);
    out.push_str(&header);

    let remaining = max_tokens.saturating_sub(header_tokens);
    let tree_depth = auto_select_tree_depth(trees, remaining);

    render_trees(&mut out, trees, tree_depth, 0);
    out
}

// ============================================================================
// Summary Attachment & Content Hashing
// ============================================================================

use crate::summarizer::SummaryCache;

/// Attach dir/file summaries from cache to DirNode/FileNode.
pub fn attach_summaries(dirs: &mut [DirNode], cache: &SummaryCache) {
    for dir in dirs.iter_mut() {
        let dir_key = format!("dir::{}", dir.path);
        let dir_hash = dir_content_hash(dir);
        if let Some(summary) = cache.get(&dir_key, &dir_hash, "none") {
            dir.summary = Some(summary.to_string());
        }
        for file in dir.files.iter_mut() {
            let file_key = format!("file::{}/{}", dir.path, file.name);
            let file_hash = file_content_hash(file);
            if let Some(summary) = cache.get(&file_key, &file_hash, "none") {
                file.summary = Some(summary.to_string());
            }
        }
    }
}

/// Content hash for a directory (based on file names + symbol names).
pub fn dir_content_hash(dir: &DirNode) -> String {
    use sha2::{Sha256, Digest};
    let mut parts: Vec<String> = dir.files.iter().map(|f| {
        let mut sym_names: Vec<&str> = f.symbols.iter().map(|s| s.name.as_str()).collect();
        sym_names.sort();
        format!("{}:{}", f.name, sym_names.join(","))
    }).collect();
    parts.sort();
    format!("{:x}", Sha256::digest(parts.join("|").as_bytes()))
}

/// Content hash for a file (based on symbol names).
pub fn file_content_hash(file: &FileNode) -> String {
    use sha2::{Sha256, Digest};
    let mut names: Vec<&str> = file.symbols.iter().map(|s| s.name.as_str()).collect();
    names.sort();
    format!("{:x}", Sha256::digest(names.join(",").as_bytes()))
}

// ============================================================================
// Tree Summary Helpers
// ============================================================================

/// Collect symbol names from DirNodes that fall under a tree node's path prefix.
/// Returns up to `limit` deduplicated names for summary generation.
pub fn collect_tree_symbol_names(tree_path: &str, dirs: &[DirNode], limit: usize) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for dir in dirs {
        if dir.path == tree_path || dir.path.starts_with(&format!("{}/", tree_path)) {
            for file in &dir.files {
                for sym in &file.symbols {
                    names.push(sym.name.clone());
                }
            }
        }
    }
    names.sort();
    names.dedup();
    names.truncate(limit);
    names
}

/// Content hash for a tree node (based on paths of DirNodes under it).
pub fn tree_content_hash(tree_path: &str, dirs: &[DirNode]) -> String {
    use sha2::{Sha256, Digest};
    let mut paths: Vec<&str> = dirs.iter()
        .filter(|d| d.path == tree_path || d.path.starts_with(&format!("{}/", tree_path)))
        .map(|d| d.path.as_str())
        .collect();
    paths.sort();
    // Include file counts for invalidation on file add/remove
    let parts: Vec<String> = paths.iter()
        .filter_map(|p| dirs.iter().find(|d| d.path == *p))
        .map(|d| format!("{}:{}", d.path, d.symbol_count))
        .collect();
    format!("{:x}", Sha256::digest(parts.join("|").as_bytes()))
}

/// Walk a DirTree and collect all nodes with their full paths (for summary generation).
pub fn collect_tree_nodes_with_paths(trees: &[DirTree], prefix: &str) -> Vec<(String, usize, usize)> {
    let mut result = Vec::new();
    for tree in trees {
        let full_path = if prefix.is_empty() {
            tree.name.clone()
        } else {
            format!("{}/{}", prefix, tree.name)
        };
        result.push((full_path.clone(), tree.total_files, tree.total_symbols));
        if !tree.children.is_empty() {
            result.extend(collect_tree_nodes_with_paths(&tree.children, &full_path));
        }
    }
    result
}

/// Attach summaries to DirTree nodes by matching full paths.
pub fn attach_tree_summaries(
    trees: &mut [DirTree],
    summaries: &std::collections::HashMap<String, String>,
    prefix: &str,
) {
    for tree in trees.iter_mut() {
        let full_path = if prefix.is_empty() {
            tree.name.clone()
        } else {
            format!("{}/{}", prefix, tree.name)
        };
        if let Some(summary) = summaries.get(&full_path) {
            tree.summary = Some(summary.clone());
        }
        if !tree.children.is_empty() {
            attach_tree_summaries(&mut tree.children, summaries, &full_path);
        }
    }
}

// ============================================================================
// Rendering
// ============================================================================

/// Generate the hierarchical project map string from pre-built dirs.
/// Use this when summaries have been attached externally.
pub fn generate_map_from_dirs(
    dirs: Vec<DirNode>,
    root_path: &str,
    config: &MapConfig,
) -> String {
    if dirs.is_empty() {
        return String::new();
    }

    let project_name = Path::new(root_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("project");

    let lang_counts = count_languages(&dirs);
    let primary_lang = lang_counts.first().map(|(l, _)| *l).unwrap_or(SourceLang::Other);
    let total_symbols: usize = dirs.iter().map(|d| d.symbol_count).sum();

    if let Some(ref area) = config.area {
        return render_with_area(&dirs, project_name, primary_lang, total_symbols, area, config.max_tokens);
    }

    let l0_tokens = estimate_tokens_for_depth(&dirs, 0);
    if l0_tokens <= config.max_tokens {
        let depth = auto_select_depth(&dirs, config.max_tokens);
        render_at_depth(&dirs, project_name, primary_lang, total_symbols, depth)
    } else {
        let mut trees = build_dir_tree(&dirs);
        for tree in &mut trees {
            collapse_single_children(tree);
        }
        render_tree_map(&trees, project_name, primary_lang, total_symbols, config.max_tokens)
    }
}

/// Generate the hierarchical project map string.
pub fn generate_map(index: &Index, root_path: &str, config: &MapConfig) -> String {
    let dirs = build_hierarchy(index, root_path);

    if dirs.is_empty() {
        return String::new();
    }

    // Get project name
    let project_name = Path::new(root_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("project");

    // Detect primary language
    let lang_counts = count_languages(&dirs);
    let primary_lang = lang_counts.first().map(|(l, _)| *l).unwrap_or(SourceLang::Other);
    let total_symbols: usize = dirs.iter().map(|d| d.symbol_count).sum();

    // Handle --area: show only the specified subtree at max depth
    if let Some(ref area) = config.area {
        return render_with_area(&dirs, project_name, primary_lang, total_symbols, area, config.max_tokens);
    }

    // Check if flat L0 fits in the budget
    let l0_tokens = estimate_tokens_for_depth(&dirs, 0);
    if l0_tokens <= config.max_tokens {
        // Small/medium project: use flat rendering with auto depth
        let depth = auto_select_depth(&dirs, config.max_tokens);
        render_at_depth(&dirs, project_name, primary_lang, total_symbols, depth)
    } else {
        // Large project: use tree rendering with auto-collapsing
        let mut trees = build_dir_tree(&dirs);
        for tree in &mut trees {
            collapse_single_children(tree);
        }
        render_tree_map(&trees, project_name, primary_lang, total_symbols, config.max_tokens)
    }
}

pub fn count_languages(dirs: &[DirNode]) -> Vec<(SourceLang, usize)> {
    let mut counts: BTreeMap<u8, (SourceLang, usize)> = BTreeMap::new();
    for dir in dirs {
        for file in &dir.files {
            let key = lang_sort_key(&file.lang);
            counts.entry(key).or_insert((file.lang, 0)).1 += count_symbols(file);
        }
    }
    let mut sorted: Vec<(SourceLang, usize)> = counts.into_values().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted
}

fn lang_sort_key(lang: &SourceLang) -> u8 {
    match lang {
        SourceLang::Rust => 0,
        SourceLang::TypeScript => 1,
        SourceLang::JavaScript => 2,
        SourceLang::Python => 3,
        SourceLang::CSharp => 4,
        SourceLang::Other => 5,
    }
}

/// Render map at the given depth for the full project.
fn render_at_depth(
    dirs: &[DirNode],
    project_name: &str,
    primary_lang: SourceLang,
    total_symbols: usize,
    depth: usize,
) -> String {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "// Project: {} ({}, {} symbols)\n//\n",
        project_name,
        primary_lang.name(),
        total_symbols,
    ));

    for dir in dirs {
        // L0: Directory (with optional summary)
        if let Some(ref summary) = dir.summary {
            out.push_str(&format!(
                "// {}/  — {} files, {} symbols ({})\n",
                dir.path, dir.file_count, dir.symbol_count, summary
            ));
        } else {
            out.push_str(&format!(
                "// {}/  — {} files, {} symbols\n",
                dir.path, dir.file_count, dir.symbol_count
            ));
        }

        if depth >= 1 {
            for file in &dir.files {
                let sym_count = count_symbols(file);
                if depth >= 2 {
                    // L1+L2: Show file with symbols (summary appended if present)
                    if let Some(ref summary) = file.summary {
                        out.push_str(&format!("//   {}  ({} symbols) — {}\n", file.name, sym_count, summary));
                    } else {
                        out.push_str(&format!("//   {}  ({} symbols)\n", file.name, sym_count));
                    }
                    render_symbols(&mut out, &file.symbols, depth, "//     ");
                } else {
                    // L1 only: Show file with summary (preferred) or symbol names preview
                    if let Some(ref summary) = file.summary {
                        out.push_str(&format!("//   {}  — {} — {}\n", file.name, sym_count, summary));
                    } else {
                        let names = symbol_names_preview(&file.symbols, 60);
                        if names.is_empty() {
                            out.push_str(&format!("//   {}  — {} symbols\n", file.name, sym_count));
                        } else {
                            out.push_str(&format!("//   {}  — {} ({})\n", file.name, sym_count, names));
                        }
                    }
                }
            }
        }
    }

    out
}

/// Render symbols (L2 and optionally L3).
fn render_symbols(out: &mut String, symbols: &[SymbolNode], depth: usize, prefix: &str) {
    for sym in symbols {
        let kind_str = kind_prefix(&sym.kind);
        let vis = if sym.visibility == Visibility::Public { "" } else { "(priv) " };

        if let Some(ref summary) = sym.summary {
            out.push_str(&format!("{}{}{}{} — {}\n", prefix, vis, kind_str, sym.name, summary));
        } else if let Some(ref sig) = sym.signature {
            let short_sig = shorten_signature(sig, 70);
            out.push_str(&format!("{}{}{}\n", prefix, vis, short_sig));
        } else {
            out.push_str(&format!("{}{}{}{}\n", prefix, vis, kind_str, sym.name));
        }

        // L3: members
        if depth >= 3 && !sym.members.is_empty() {
            let member_prefix = format!("{}  ", prefix);
            for member in &sym.members {
                let m_kind = kind_prefix(&member.kind);
                let m_vis = if member.visibility == Visibility::Public { "" } else { "(priv) " };
                if let Some(ref summary) = member.summary {
                    out.push_str(&format!("{}{}{}{} — {}\n", member_prefix, m_vis, m_kind, member.name, summary));
                } else {
                    out.push_str(&format!("{}{}{}{}\n", member_prefix, m_vis, m_kind, member.name));
                }
            }
        }
    }
}

/// Render a compact summary of other directories (outside the area).
/// Shows top 5 by symbol count to provide context without token waste.
fn render_other_dirs_summary(out: &mut String, dirs: &[&DirNode]) {
    if dirs.is_empty() {
        return;
    }

    let mut sorted: Vec<&DirNode> = dirs.to_vec();
    sorted.sort_by(|a, b| b.symbol_count.cmp(&a.symbol_count));

    let show_count = 5;
    let total = sorted.len();

    out.push_str("//\n// --- other directories (top by size) ---\n");
    for dir in sorted.iter().take(show_count) {
        if let Some(ref summary) = dir.summary {
            out.push_str(&format!(
                "// {}/  — {} files, {} symbols ({})\n",
                dir.path, dir.file_count, dir.symbol_count, summary
            ));
        } else {
            out.push_str(&format!(
                "// {}/  — {} files, {} symbols\n",
                dir.path, dir.file_count, dir.symbol_count
            ));
        }
    }
    if total > show_count {
        out.push_str(&format!("// ... and {} more directories\n", total - show_count));
    }
}

/// Render with --area: show area subtree at max depth, rest at L0.
fn render_with_area(
    dirs: &[DirNode],
    project_name: &str,
    primary_lang: SourceLang,
    total_symbols: usize,
    area: &str,
    max_tokens: usize,
) -> String {
    let mut out = String::new();
    let zoom = detect_zoom_level(area, dirs);

    // Header
    out.push_str(&format!(
        "// Project: {} ({}, {} symbols) [area: {}]\n//\n",
        project_name,
        primary_lang.name(),
        total_symbols,
        area,
    ));

    // Normalize area path (strip trailing slash)
    let area_norm = area.trim_end_matches('/');

    // Determine which dirs match the area
    let mut area_dirs: Vec<&DirNode> = Vec::new();
    let mut other_dirs: Vec<&DirNode> = Vec::new();

    for dir in dirs {
        if dir.path == area_norm || dir.path.starts_with(&format!("{}/", area_norm)) {
            area_dirs.push(dir);
        } else {
            other_dirs.push(dir);
        }
    }

    // Check if area matches a specific file within a directory
    let area_file: Option<(&DirNode, &str)> = if area_dirs.is_empty() {
        // Maybe it's a file path like "src/main.rs"
        let dir_part = extract_dir_path(area_norm);
        dirs.iter()
            .find(|d| d.path == dir_part)
            .map(|d| (d, &area_norm[dir_part.len() + 1..]))
    } else {
        None
    };

    if !area_dirs.is_empty() {
        // Area matches directories — auto-depth based on area entry count
        let area_entry_count: usize = area_dirs.iter()
            .map(|d| d.files.iter().map(|f| 1 + count_symbols(f)).sum::<usize>() + 1)
            .sum();

        let area_depth = if area_entry_count < 30 {
            3 // Few entries: show functions + members
        } else if area_entry_count <= 100 {
            2 // Medium: show functions
        } else {
            // Large area: auto-select within budget, at least depth 1
            auto_select_depth(
                &area_dirs.iter().map(|d| (*d).clone()).collect::<Vec<_>>(),
                max_tokens.saturating_sub(100),
            ).max(1)
        };

        for dir in &area_dirs {
            // L0: Directory (with summary)
            if let Some(ref summary) = dir.summary {
                out.push_str(&format!(
                    "// {}/  — {} files, {} symbols ({})\n",
                    dir.path, dir.file_count, dir.symbol_count, summary
                ));
            } else {
                out.push_str(&format!(
                    "// {}/  — {} files, {} symbols\n",
                    dir.path, dir.file_count, dir.symbol_count
                ));
            }

            if area_depth >= 1 {
                for file in &dir.files {
                    let sym_count = count_symbols(file);
                    if area_depth >= 2 {
                        // L1+L2: file with symbols
                        if let Some(ref summary) = file.summary {
                            out.push_str(&format!("//   {}  ({} symbols) — {}\n", file.name, sym_count, summary));
                        } else {
                            out.push_str(&format!("//   {}  ({} symbols)\n", file.name, sym_count));
                        }
                        render_symbols(&mut out, &file.symbols, area_depth, "//     ");
                    } else {
                        // L1 only: file with summary or symbol preview
                        if let Some(ref summary) = file.summary {
                            out.push_str(&format!("//   {}  — {} — {}\n", file.name, sym_count, summary));
                        } else {
                            let names = symbol_names_preview(&file.symbols, 60);
                            if names.is_empty() {
                                out.push_str(&format!("//   {}  — {} symbols\n", file.name, sym_count));
                            } else {
                                out.push_str(&format!("//   {}  — {} ({})\n", file.name, sym_count, names));
                            }
                        }
                    }
                }
            }
        }

        // Show context: top sibling directories (not full list)
        render_other_dirs_summary(&mut out, &other_dirs);
    } else if let Some((dir, file_name)) = area_file {
        // Area matches a specific file — show it at full depth with summary
        if let Some(file) = dir.files.iter().find(|f| f.name == file_name) {
            let sym_count = count_symbols(file);
            out.push_str(&format!("// {}/\n", dir.path));
            if let Some(ref summary) = file.summary {
                out.push_str(&format!("//   {}  ({} symbols) — {}\n", file.name, sym_count, summary));
            } else {
                out.push_str(&format!("//   {}  ({} symbols)\n", file.name, sym_count));
            }
            render_symbols(&mut out, &file.symbols, 3, "//     ");
        } else {
            out.push_str(&format!("// Area '{}' not found in index\n", area));
        }

        // Show context: sibling directories
        let siblings: Vec<&DirNode> = dirs.iter().filter(|d| !std::ptr::eq(*d, dir)).collect();
        render_other_dirs_summary(&mut out, &siblings);
    } else {
        out.push_str(&format!("// Area '{}' not found in index\n", area));
        // Show everything at L0 (limited)
        let all_refs: Vec<&DirNode> = dirs.iter().collect();
        render_other_dirs_summary(&mut out, &all_refs);
    }

    // Append zoom hint
    let hint = zoom_hint(zoom, area, dirs);
    if !hint.is_empty() {
        out.push_str(&format!("{}\n", hint));
    }

    out
}

/// Get a short preview of symbol names in a file.
fn symbol_names_preview(symbols: &[SymbolNode], max_len: usize) -> String {
    let mut parts = Vec::new();
    let mut len = 0;

    for sym in symbols {
        let addition = if parts.is_empty() {
            sym.name.len()
        } else {
            sym.name.len() + 2 // ", "
        };

        if len + addition > max_len {
            parts.push("...".to_string());
            break;
        }

        parts.push(sym.name.clone());
        len += addition;
    }

    parts.join(", ")
}

fn kind_prefix(kind: &ChunkKind) -> &'static str {
    match kind {
        ChunkKind::Class => "class ",
        ChunkKind::Struct => "struct ",
        ChunkKind::Interface => "trait ",
        ChunkKind::Enum => "enum ",
        ChunkKind::Function => "fn ",
        ChunkKind::Method => "fn ",
        ChunkKind::Constant => "const ",
        ChunkKind::Field => "field ",
        _ => "",
    }
}

/// Shorten a function signature for display.
fn shorten_signature(sig: &str, max_len: usize) -> String {
    let cleaned = sig.trim()
        .trim_start_matches("pub ")
        .trim_start_matches("async ")
        .trim_start_matches("export ")
        .trim_start_matches("function ")
        .trim_end_matches('{')
        .trim_end_matches(';')
        .trim();

    if cleaned.len() <= max_len {
        cleaned.to_string()
    } else {
        // Find a valid char boundary near max_len - 3
        let truncate_at = max_len - 3;
        let boundary = cleaned.floor_char_boundary(truncate_at);
        format!("{}...", &cleaned[..boundary])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, FileEntry, StringTable};

    /// Helper: build a minimal index for testing.
    fn make_test_index() -> Index {
        let mut strings = StringTable::new();
        let root_idx = strings.intern("/project");
        let model_idx = strings.intern("text-embedding-3-small");

        // Files
        let file1_idx = strings.intern("/project/src/parser/code.rs");
        let file2_idx = strings.intern("/project/src/lib.rs");
        let file3_idx = strings.intern("/project/src/parser/docs.rs");

        // Symbols in parser/code.rs
        let parse_name = strings.intern("parse");
        let parse_content = strings.intern("fn parse() { ... }");
        let parse_sig = strings.intern("pub fn parse(content: &str) -> Vec<Chunk>");
        let parse_doc = strings.intern("Parse source code into chunks");

        let extract_name = strings.intern("extract_symbols");
        let extract_content = strings.intern("fn extract_symbols() { ... }");

        // Class with members
        let class_name = strings.intern("CodeParser");
        let class_content = strings.intern("struct CodeParser { ... }");
        let method_name = strings.intern("new");
        let method_content = strings.intern("fn new() { ... }");
        let context_str = strings.intern("CodeParser");

        // Symbols in lib.rs
        let main_fn_name = strings.intern("load_or_create");
        let main_fn_content = strings.intern("fn load_or_create() { ... }");
        let main_fn_sig = strings.intern("pub fn load_or_create(root: &Path) -> Result<Self>");

        // Symbols in parser/docs.rs
        let docs_fn_name = strings.intern("parse_markdown");
        let docs_fn_content = strings.intern("fn parse_markdown() { ... }");

        let chunks = vec![
            // parser/code.rs: parse (public fn)
            Chunk {
                chunk_type: ChunkType::Code,
                name_idx: parse_name,
                kind: ChunkKind::Function,
                file_idx: file1_idx,
                line_start: 10,
                line_end: 50,
                content_idx: parse_content,
                context_idx: None,
                signature_idx: Some(parse_sig),
                doc_summary_idx: Some(parse_doc),
                visibility: Visibility::Public,
            },
            // parser/code.rs: extract_symbols (private fn)
            Chunk {
                chunk_type: ChunkType::Code,
                name_idx: extract_name,
                kind: ChunkKind::Function,
                file_idx: file1_idx,
                line_start: 55,
                line_end: 80,
                content_idx: extract_content,
                context_idx: None,
                signature_idx: None,
                doc_summary_idx: None,
                visibility: Visibility::Private,
            },
            // parser/code.rs: CodeParser (struct)
            Chunk {
                chunk_type: ChunkType::Code,
                name_idx: class_name,
                kind: ChunkKind::Struct,
                file_idx: file1_idx,
                line_start: 1,
                line_end: 8,
                content_idx: class_content,
                context_idx: None,
                signature_idx: None,
                doc_summary_idx: None,
                visibility: Visibility::Public,
            },
            // parser/code.rs: new (method of CodeParser)
            Chunk {
                chunk_type: ChunkType::Code,
                name_idx: method_name,
                kind: ChunkKind::Method,
                file_idx: file1_idx,
                line_start: 3,
                line_end: 7,
                content_idx: method_content,
                context_idx: Some(context_str),
                signature_idx: None,
                doc_summary_idx: None,
                visibility: Visibility::Public,
            },
            // lib.rs: load_or_create (public fn)
            Chunk {
                chunk_type: ChunkType::Code,
                name_idx: main_fn_name,
                kind: ChunkKind::Function,
                file_idx: file2_idx,
                line_start: 100,
                line_end: 130,
                content_idx: main_fn_content,
                context_idx: None,
                signature_idx: Some(main_fn_sig),
                doc_summary_idx: None,
                visibility: Visibility::Public,
            },
            // parser/docs.rs: parse_markdown
            Chunk {
                chunk_type: ChunkType::Code,
                name_idx: docs_fn_name,
                kind: ChunkKind::Function,
                file_idx: file3_idx,
                line_start: 1,
                line_end: 30,
                content_idx: docs_fn_content,
                context_idx: None,
                signature_idx: None,
                doc_summary_idx: None,
                visibility: Visibility::Public,
            },
        ];

        let files = vec![
            FileEntry {
                path_idx: file1_idx,
                mtime: 1000,
                hash_idx: strings.intern("hash1"),
                chunk_type: ChunkType::Code,
                chunk_range: (0, 4),
            },
            FileEntry {
                path_idx: file2_idx,
                mtime: 1000,
                hash_idx: strings.intern("hash2"),
                chunk_type: ChunkType::Code,
                chunk_range: (4, 5),
            },
            FileEntry {
                path_idx: file3_idx,
                mtime: 1000,
                hash_idx: strings.intern("hash3"),
                chunk_type: ChunkType::Code,
                chunk_range: (5, 6),
            },
        ];

        Index {
            version: 3,
            last_indexed: "2024-01-01T00:00:00Z".to_string(),
            root_idx,
            embedding_model_idx: model_idx,
            strings,
            chunks,
            files,
        }
    }

    #[test]
    fn test_build_hierarchy() {
        let index = make_test_index();
        let dirs = build_hierarchy(&index, "/project");

        assert_eq!(dirs.len(), 2); // "src" and "src/parser"

        // Find src/parser dir
        let parser_dir = dirs.iter().find(|d| d.path == "src/parser").unwrap();
        assert_eq!(parser_dir.file_count, 2); // code.rs, docs.rs
        assert!(parser_dir.symbol_count > 0);

        // Check code.rs file
        let code_file = parser_dir.files.iter().find(|f| f.name == "code.rs").unwrap();
        // Should have: CodeParser (struct with member new), parse, extract_symbols
        assert!(code_file.symbols.len() >= 3);

        // Check CodeParser has member "new"
        let code_parser = code_file.symbols.iter().find(|s| s.name == "CodeParser").unwrap();
        assert_eq!(code_parser.members.len(), 1);
        assert_eq!(code_parser.members[0].name, "new");
    }

    #[test]
    fn test_build_hierarchy_includes_private() {
        let index = make_test_index();
        let dirs = build_hierarchy(&index, "/project");

        let parser_dir = dirs.iter().find(|d| d.path == "src/parser").unwrap();
        let code_file = parser_dir.files.iter().find(|f| f.name == "code.rs").unwrap();

        // extract_symbols is private — should still be included
        let has_extract = code_file.symbols.iter().any(|s| s.name == "extract_symbols");
        assert!(has_extract, "Private symbols should be included");
    }

    #[test]
    fn test_auto_select_depth() {
        let index = make_test_index();
        let dirs = build_hierarchy(&index, "/project");

        // With a large budget, should go deep
        let depth = auto_select_depth(&dirs, 10000);
        assert!(depth >= 2);

        // With a tiny budget, should be L0
        let depth = auto_select_depth(&dirs, 30);
        assert_eq!(depth, 0);
    }

    #[test]
    fn test_generate_map_default() {
        let index = make_test_index();
        let config = MapConfig {
            max_tokens: 2000,
            area: None,
        };

        let output = generate_map(&index, "/project", &config);

        assert!(output.contains("Project:"));
        assert!(output.contains("src/parser"));
        assert!(output.contains("src"));
    }

    #[test]
    fn test_generate_map_with_area() {
        let index = make_test_index();
        let config = MapConfig {
            max_tokens: 2000,
            area: Some("src/parser".to_string()),
        };

        let output = generate_map(&index, "/project", &config);

        assert!(output.contains("[area: src/parser]"));
        assert!(output.contains("code.rs"));
        assert!(output.contains("parse"));
    }

    #[test]
    fn test_make_relative() {
        assert_eq!(make_relative("/project/src/lib.rs", "/project"), "src/lib.rs");
        assert_eq!(make_relative("/project/src/lib.rs", "/project/"), "src/lib.rs");
        assert_eq!(make_relative("src/lib.rs", "/other"), "src/lib.rs");
    }

    #[test]
    fn test_extract_dir_path() {
        assert_eq!(extract_dir_path("src/parser/code.rs"), "src/parser");
        assert_eq!(extract_dir_path("lib.rs"), ".");
        assert_eq!(extract_dir_path("src/lib.rs"), "src");
    }

    #[test]
    fn test_extract_file_name() {
        assert_eq!(extract_file_name("src/parser/code.rs"), "code.rs");
        assert_eq!(extract_file_name("lib.rs"), "lib.rs");
    }

    #[test]
    fn test_shorten_signature() {
        assert_eq!(
            shorten_signature("pub fn hello(name: &str) -> String {", 50),
            "fn hello(name: &str) -> String"
        );
        assert_eq!(
            shorten_signature("pub async fn fetch() -> Result<Data>", 50),
            "fn fetch() -> Result<Data>"
        );
    }

    #[test]
    fn test_symbol_names_preview() {
        let symbols = vec![
            SymbolNode { name: "foo".to_string(), kind: ChunkKind::Function, signature: None, summary: None, visibility: Visibility::Public, members: vec![] },
            SymbolNode { name: "bar".to_string(), kind: ChunkKind::Function, signature: None, summary: None, visibility: Visibility::Public, members: vec![] },
            SymbolNode { name: "baz".to_string(), kind: ChunkKind::Function, signature: None, summary: None, visibility: Visibility::Public, members: vec![] },
        ];

        let preview = symbol_names_preview(&symbols, 60);
        assert_eq!(preview, "foo, bar, baz");
    }

    #[test]
    fn test_symbol_names_preview_truncation() {
        let symbols: Vec<SymbolNode> = (0..20)
            .map(|i| SymbolNode {
                name: format!("very_long_function_name_{}", i),
                kind: ChunkKind::Function,
                signature: None,
                summary: None,
                visibility: Visibility::Public,
                members: vec![],
            })
            .collect();

        let preview = symbol_names_preview(&symbols, 60);
        assert!(preview.len() <= 65); // some tolerance
        assert!(preview.ends_with("..."));
    }

    #[test]
    fn test_count_at_depth() {
        let index = make_test_index();
        let dirs = build_hierarchy(&index, "/project");

        let counts = count_at_depth(&dirs);
        assert!(counts[0] > 0); // dirs
        assert!(counts[1] > counts[0]); // dirs + files
        assert!(counts[2] > counts[1]); // + symbols
        assert!(counts[3] >= counts[2]); // + members
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

    // ── Directory Tree tests ──

    fn make_large_dir_list() -> Vec<DirNode> {
        // Simulate a Unity-like project with many nested directories
        let mut dirs = Vec::new();
        let base = "Assets/Game/Scripts";
        let subdirs = ["Common", "GameControl", "UI", "InGame", "Debug"];
        let sub_subdirs = ["Actor", "Entity", "Interface", "Strategies", "Behaviours"];

        for sub in &subdirs {
            // Each subdir has files directly
            dirs.push(DirNode {
                path: format!("{}/{}", base, sub),
                file_count: 5,
                symbol_count: 50,
                files: Vec::new(),
                summary: None,
            });
            // And nested sub-subdirectories
            for subsub in &sub_subdirs {
                dirs.push(DirNode {
                    path: format!("{}/{}/{}", base, sub, subsub),
                    file_count: 10,
                    symbol_count: 100,
                    files: Vec::new(),
                    summary: None,
                });
            }
        }
        // Add some top-level dirs
        dirs.push(DirNode {
            path: "Packages".to_string(),
            file_count: 20,
            symbol_count: 200,
            files: Vec::new(),
            summary: None,
        });
        dirs.push(DirNode {
            path: ".ci".to_string(),
            file_count: 3,
            symbol_count: 10,
            files: Vec::new(),
            summary: None,
        });
        dirs
    }

    #[test]
    fn test_build_dir_tree() {
        let dirs = make_large_dir_list();
        let trees = build_dir_tree(&dirs);

        // Should have 3 top-level: .ci, Assets, Packages
        assert_eq!(trees.len(), 3);

        let assets = trees.iter().find(|t| t.name == "Assets").unwrap();
        assert!(assets.total_files > 0);
        assert!(assets.total_symbols > 0);
        // Assets has children
        assert!(!assets.children.is_empty());
    }

    #[test]
    fn test_collapse_single_children() {
        let dirs = make_large_dir_list();
        let mut trees = build_dir_tree(&dirs);
        for tree in &mut trees {
            collapse_single_children(tree);
        }

        // Assets/Game/Scripts should be collapsed into one node
        let assets = trees.iter().find(|t| t.name.starts_with("Assets")).unwrap();
        assert_eq!(assets.name, "Assets/Game/Scripts");
        // Should have 5 children (Common, GameControl, UI, InGame, Debug)
        assert_eq!(assets.children.len(), 5);
    }

    #[test]
    fn test_tree_rendering_fits_budget() {
        let dirs = make_large_dir_list();
        let mut trees = build_dir_tree(&dirs);
        for tree in &mut trees {
            collapse_single_children(tree);
        }

        let output = render_tree_map(&trees, "test-project", SourceLang::CSharp, 5000, 2000);

        // Should fit within budget
        let estimated_tokens = output.len().div_ceil(4);
        assert!(
            estimated_tokens <= 2200, // some tolerance
            "Output {} tokens should be <= 2200",
            estimated_tokens
        );

        // Should contain the project header
        assert!(output.contains("Project: test-project"));
        // Should show tree structure with indentation
        assert!(output.contains("Assets/Game/Scripts/"));
    }

    #[test]
    fn test_generate_map_uses_tree_for_large_projects() {
        // Build an index with many directories to trigger tree mode
        let mut strings = StringTable::new();
        let root_idx = strings.intern("/project");
        let model_idx = strings.intern("model");

        let mut chunks = Vec::new();
        let mut files = Vec::new();

        // Create 100+ directories with files
        for i in 0..120 {
            let dir = format!("dir{:03}", i / 4);
            let subdir = format!("{}/sub{}", dir, i % 4);
            let file_path = format!("/project/{}/file.cs", subdir);
            let file_idx = strings.intern(&file_path);
            let name_idx = strings.intern(&format!("Symbol{}", i));
            let content_idx = strings.intern("content");

            let chunk_start = chunks.len() as u32;
            chunks.push(Chunk {
                chunk_type: ChunkType::Code,
                name_idx,
                kind: ChunkKind::Function,
                file_idx,
                line_start: 1,
                line_end: 10,
                content_idx,
                context_idx: None,
                signature_idx: None,
                doc_summary_idx: None,
                visibility: Visibility::Public,
            });

            let hash_idx = strings.intern(&format!("hash{}", i));
            files.push(FileEntry {
                path_idx: file_idx,
                mtime: 1000,
                hash_idx,
                chunk_type: ChunkType::Code,
                chunk_range: (chunk_start, chunk_start + 1),
            });
        }

        let index = Index {
            version: 3,
            last_indexed: "2024-01-01T00:00:00Z".to_string(),
            root_idx,
            embedding_model_idx: model_idx,
            strings,
            chunks,
            files,
        };

        let config = MapConfig {
            max_tokens: 2000,
            area: None,
        };

        let output = generate_map(&index, "/project", &config);

        // Should use tree mode (indented children)
        assert!(output.contains("Project:"));
        // Should fit in budget
        let tokens = output.len().div_ceil(4);
        assert!(tokens <= 2200, "tokens={} should be <= 2200", tokens);
    }

    // ── Summary attachment tests ──

    #[test]
    fn test_attach_summaries() {
        let index = make_test_index();
        let mut dirs = build_hierarchy(&index, "/project");

        // Initially no summaries
        assert!(dirs.iter().all(|d| d.summary.is_none()));
        assert!(dirs.iter().flat_map(|d| &d.files).all(|f| f.summary.is_none()));

        // Build cache with some entries
        let mut cache = SummaryCache::new();
        let parser_dir = dirs.iter().find(|d| d.path == "src/parser").unwrap();
        let dir_hash = dir_content_hash(parser_dir);
        cache.insert("dir::src/parser".to_string(), dir_hash, "none".to_string(), "Code and doc parsing".to_string());

        let code_file = parser_dir.files.iter().find(|f| f.name == "code.rs").unwrap();
        let file_hash = file_content_hash(code_file);
        cache.insert("file::src/parser/code.rs".to_string(), file_hash, "none".to_string(), "Parses source code via tree-sitter".to_string());

        attach_summaries(&mut dirs, &cache);

        let parser_dir = dirs.iter().find(|d| d.path == "src/parser").unwrap();
        assert_eq!(parser_dir.summary.as_deref(), Some("Code and doc parsing"));

        let code_file = parser_dir.files.iter().find(|f| f.name == "code.rs").unwrap();
        assert_eq!(code_file.summary.as_deref(), Some("Parses source code via tree-sitter"));
    }

    #[test]
    fn test_dir_content_hash_deterministic() {
        let index = make_test_index();
        let dirs = build_hierarchy(&index, "/project");
        let parser_dir = dirs.iter().find(|d| d.path == "src/parser").unwrap();

        let hash1 = dir_content_hash(parser_dir);
        let hash2 = dir_content_hash(parser_dir);
        assert_eq!(hash1, hash2);
        assert!(!hash1.is_empty());
    }

    #[test]
    fn test_file_content_hash_deterministic() {
        let index = make_test_index();
        let dirs = build_hierarchy(&index, "/project");
        let parser_dir = dirs.iter().find(|d| d.path == "src/parser").unwrap();
        let code_file = parser_dir.files.iter().find(|f| f.name == "code.rs").unwrap();

        let hash1 = file_content_hash(code_file);
        let hash2 = file_content_hash(code_file);
        assert_eq!(hash1, hash2);
        assert!(!hash1.is_empty());
    }

    #[test]
    fn test_render_with_summaries() {
        let index = make_test_index();
        let mut dirs = build_hierarchy(&index, "/project");

        // Add summaries manually
        for dir in &mut dirs {
            if dir.path == "src/parser" {
                dir.summary = Some("Code and doc parsing".to_string());
                for file in &mut dir.files {
                    if file.name == "code.rs" {
                        file.summary = Some("Parses source code via tree-sitter".to_string());
                    }
                }
            }
        }

        let config = MapConfig { max_tokens: 2000, area: None };
        let output = generate_map_from_dirs(dirs, "/project", &config);

        // Directory summary should appear in parentheses
        assert!(output.contains("(Code and doc parsing)"), "Dir summary missing: {}", output);
        // File summary should appear
        assert!(output.contains("Parses source code via tree-sitter"), "File summary missing: {}", output);
    }

    #[test]
    fn test_generate_map_from_dirs_matches_generate_map() {
        let index = make_test_index();
        let config = MapConfig { max_tokens: 2000, area: None };

        // generate_map and generate_map_from_dirs should produce identical output
        // when no summaries are attached
        let output1 = generate_map(&index, "/project", &config);
        let dirs = build_hierarchy(&index, "/project");
        let output2 = generate_map_from_dirs(dirs, "/project", &config);

        assert_eq!(output1, output2);
    }
}
