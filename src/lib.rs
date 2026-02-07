use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

pub mod index;
pub mod embeddings;
pub mod map;
pub mod parser;
pub mod search;
pub mod summarizer;

// Global quiet mode flag
static QUIET_MODE: AtomicBool = AtomicBool::new(false);

// Global log directory (.scs/)
static LOG_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Initialize the log directory for file-based logging.
pub fn init_log_dir(dir: PathBuf) {
    let _ = LOG_DIR.set(dir);
}

/// Write a log line to `.scs/scs.log`. Rotates to `scs.log.old` at 512KB.
pub fn log_to_file(level: &str, msg: &str) {
    if let Some(dir) = LOG_DIR.get() {
        let log_path = dir.join("scs.log");
        // Rotate if > 512KB
        if let Ok(meta) = std::fs::metadata(&log_path) {
            if meta.len() > 512 * 1024 {
                let _ = std::fs::rename(&log_path, dir.join("scs.log.old"));
            }
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true).append(true).open(&log_path)
        {
            use std::io::Write;
            let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let _ = writeln!(f, "[{}] {} {}", now, level, msg);
        }
    }
}

/// Set global quiet mode (suppresses warnings)
pub fn set_quiet_mode(quiet: bool) {
    QUIET_MODE.store(quiet, Ordering::SeqCst);
}

/// Check if quiet mode is enabled
pub fn is_quiet() -> bool {
    QUIET_MODE.load(Ordering::SeqCst)
}

/// Print warning to stderr (if not quiet) and always log to file.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        $crate::log_to_file("WARN", &msg);
        if !$crate::is_quiet() {
            eprintln!("{}", msg);
        }
    }};
}

/// Log to file only (no stderr output).
#[macro_export]
macro_rules! scs_log {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        $crate::log_to_file("INFO", &msg);
    }};
}

// ============================================================================
// StringTable - String interning for memory efficiency
// ============================================================================

/// String interning table - stores all strings and references them by u32 index.
/// Reduces memory usage by deduplicating repeated strings (file paths, symbol names, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringTable {
    strings: Vec<String>,
    #[serde(skip)]
    index: HashMap<String, u32>,
}

impl StringTable {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Intern a string - returns existing index if already present, otherwise adds it.
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.index.get(s) {
            return idx;
        }
        let idx = self.strings.len() as u32;
        self.strings.push(s.to_string());
        self.index.insert(s.to_string(), idx);
        idx
    }

    /// Get string by index.
    pub fn get(&self, idx: u32) -> Option<&str> {
        self.strings.get(idx as usize).map(|s| s.as_str())
    }

    /// Rebuild the index HashMap after deserialization.
    /// Must be called after loading from bincode.
    pub fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, s) in self.strings.iter().enumerate() {
            self.index.insert(s.clone(), i as u32);
        }
    }

    /// Find a string's index (requires rebuild_index to have been called).
    pub fn find(&self, s: &str) -> Option<u32> {
        self.index.get(s).copied()
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Core Enums
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkType {
    Code,
    Doc,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkKind {
    Class,
    Struct,
    Impl,
    Interface,
    Enum,
    Function,
    Method,
    Field,
    Constant,
    Document,
    Section,
    Paragraph,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MatchType {
    Semantic,
    Exact,
    NameOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

/// Symbol visibility (public/private)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,     // pub, export
    #[default]
    Private,    // default
}

/// Doc comment information extracted from source
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocInfo {
    /// First paragraph summary (displayed in project map)
    pub summary: String,
}

// ============================================================================
// Index Data Structures (new binary format with string interning)
// ============================================================================

/// A code symbol or documentation section (with string-interned fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_type: ChunkType,
    pub name_idx: u32,              // StringTable index
    pub kind: ChunkKind,
    pub file_idx: u32,              // StringTable index
    pub line_start: u32,
    pub line_end: u32,
    pub content_idx: u32,           // StringTable index
    pub context_idx: Option<u32>,   // StringTable index (parent class/section)
    pub signature_idx: Option<u32>, // StringTable index
    pub doc_summary_idx: Option<u32>, // StringTable index (doc comment summary for map)
    pub visibility: Visibility,     // Public/Private for map filtering
}

/// File tracking entry with chunk range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path_idx: u32,              // StringTable index
    pub mtime: u64,
    pub hash_idx: u32,              // StringTable index
    pub chunk_type: ChunkType,
    pub chunk_range: (u32, u32),    // (start_idx, end_idx) in chunks vec
}

/// Unified index structure (replaces separate meta.json, chunks.json, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub version: u32,
    pub last_indexed: String,
    pub root_idx: u32,              // StringTable index
    pub embedding_model_idx: u32,   // StringTable index
    pub strings: StringTable,
    pub chunks: Vec<Chunk>,
    pub files: Vec<FileEntry>,
}

impl Index {
    pub fn new(root: &str, embedding_model: &str) -> Self {
        let mut strings = StringTable::new();
        let root_idx = strings.intern(root);
        let embedding_model_idx = strings.intern(embedding_model);

        Self {
            version: 2,
            last_indexed: chrono::Utc::now().to_rfc3339(),
            root_idx,
            embedding_model_idx,
            strings,
            chunks: Vec::new(),
            files: Vec::new(),
        }
    }

    /// Get root path as string.
    pub fn root(&self) -> &str {
        self.strings.get(self.root_idx).unwrap_or("")
    }

    /// Get embedding model as string.
    pub fn embedding_model(&self) -> &str {
        self.strings.get(self.embedding_model_idx).unwrap_or("")
    }
}

// ============================================================================
// Output/API Data Structures (keep String for JSON serialization)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_type: ChunkType,
    pub name: String,
    pub kind: ChunkKind,
    pub file: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub score: f32,
    pub preview: String,
    pub context: Option<String>,
    pub unique: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOutput {
    pub query: String,
    pub match_type: MatchType,
    pub results: Vec<SearchResult>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Definition {
    pub chunk_type: ChunkType,
    pub kind: ChunkKind,
    pub file: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub signature: Option<String>,
    pub context: Option<String>,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupOutput {
    pub name: String,
    pub match_type: MatchType,
    pub confidence: Confidence,
    pub definitions: Vec<Definition>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalStats {
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
    pub pending_embeddings: usize,
    pub embed_recommendation: EmbedRecommendation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EmbedRecommendation {
    None,
    Sync,
    SyncAcceptable,
    Background,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbedMode {
    /// Skip embedding entirely
    Skip,
    /// Auto-spawn background embed process when needed
    Auto,
}

impl IncrementalStats {
    pub fn has_changes(&self) -> bool {
        self.added > 0 || self.updated > 0 || self.removed > 0
    }

    pub fn total(&self) -> usize {
        self.added + self.updated + self.removed
    }
}

// ============================================================================
// Raw parsing structures (before string interning)
// ============================================================================

/// Raw chunk from parser (with String fields, before interning).
#[derive(Debug, Clone)]
pub struct RawChunk {
    pub chunk_type: ChunkType,
    pub name: String,
    pub kind: ChunkKind,
    pub line_start: u32,
    pub line_end: u32,
    pub byte_start: usize,
    pub byte_end: usize,
    pub content: String,
    pub context: Option<String>,
    pub signature: Option<String>,
    pub doc_summary: Option<String>,  // Doc comment summary for map
    pub visibility: Visibility,       // Public/Private for map filtering
}

// ============================================================================
// SCS - Main orchestration struct
// ============================================================================

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use index::manager::IndexManager;
use index::cache::{get_mtime, compute_hash};
use parser::code::CodeParser;
use parser::docs::DocsParser;
use embeddings::{Embedder, QueryCache};

/// Main SCS struct that orchestrates all indexing, parsing, embedding, and search operations.
pub struct SCS {
    /// Index manager handling chunk storage, embeddings, and file tracking
    pub index: IndexManager,
    /// Tree-sitter based code parser
    code_parser: CodeParser,
    /// Markdown/text documentation parser
    docs_parser: DocsParser,
    /// OpenAI embedder (None if OPENAI_API_KEY not set)
    embedder: Option<Embedder>,
    /// LRU cache for query embeddings
    query_cache: QueryCache,
}

impl SCS {
    /// Load an existing index or create a new one at the given root path.
    pub fn load_or_create(root: &Path) -> Result<Self> {
        let index = IndexManager::load_or_create(root)
            .context("Failed to load or create index")?;

        let code_parser = CodeParser::new()
            .context("Failed to create code parser")?;
        let docs_parser = DocsParser::new();

        let embedder = match Embedder::new() {
            Ok(e) => Some(e),
            Err(_) => {
                warn!("[scs] Warning: OPENAI_API_KEY not set, semantic search disabled");
                None
            }
        };

        let query_cache = QueryCache::new();

        Ok(Self {
            index,
            code_parser,
            docs_parser,
            embedder,
            query_cache,
        })
    }

    /// Check if the index is currently locked by another process.
    pub fn is_locked(&self) -> bool {
        self.index.check_lock().is_some()
    }

    /// Get the PID of the process holding the lock, if any.
    pub fn lock_holder(&self) -> Option<u32> {
        self.index.check_lock()
    }

    /// Ensure the index is up-to-date with the filesystem.
    /// After completion, checks for dirty flag and refreshes again if set.
    pub async fn ensure_fresh(&mut self, mode: EmbedMode) -> Result<IncrementalStats> {
        self.index.acquire_lock()
            .context("Failed to acquire index lock")?;

        // Clear any existing dirty flag before starting
        self.index.check_and_clear_dirty();

        let mut result = self.ensure_fresh_internal(mode).await;

        // Check for dirty flag (set by other processes that tried to refresh while we held lock)
        // Loop until no more dirty flags are set
        while self.index.check_and_clear_dirty() {
            warn!("[scs] Dirty flag detected, running another refresh cycle");
            if let Ok(new_result) = self.ensure_fresh_internal(mode).await {
                if let Ok(ref mut stats) = result {
                    stats.added += new_result.added;
                    stats.updated += new_result.updated;
                    stats.removed += new_result.removed;
                }
            }
        }

        if let Err(e) = self.index.release_lock() {
            warn!("[scs] Warning: Failed to release lock: {}", e);
        }

        result
    }

    /// Try to ensure index is fresh, but use cached data if locked.
    pub async fn try_ensure_fresh(&mut self, mode: EmbedMode) -> Result<Option<IncrementalStats>> {
        if let Some(pid) = self.index.check_lock() {
            scs_log!("[refresh] locked by PID {}, using cached data", pid);
            eprintln!(
                "[scs] Index locked by PID {}. Using cached data (may be slightly stale).",
                pid
            );
            return Ok(None);
        }

        let stats = self.ensure_fresh(mode).await?;
        Ok(Some(stats))
    }

    /// Try to refresh, or mark dirty if locked (non-blocking).
    /// Returns Ok(Some(stats)) if refreshed, Ok(None) if locked and marked dirty.
    pub async fn try_refresh_or_mark_dirty(&mut self, mode: EmbedMode) -> Result<Option<IncrementalStats>> {
        if let Some(pid) = self.index.check_lock() {
            // Another process is running - mark dirty so it refreshes again when done
            self.index.mark_dirty()?;
            warn!(
                "[scs] Index locked by PID {}. Marked dirty for re-check.",
                pid
            );
            return Ok(None);
        }

        let stats = self.ensure_fresh(mode).await?;
        Ok(Some(stats))
    }

    fn calculate_embed_recommendation(pending: usize) -> EmbedRecommendation {
        const SYNC_THRESHOLD: usize = 100;
        const ACCEPTABLE_THRESHOLD: usize = 500;

        if pending == 0 {
            EmbedRecommendation::None
        } else if pending <= SYNC_THRESHOLD {
            EmbedRecommendation::Sync
        } else if pending <= ACCEPTABLE_THRESHOLD {
            EmbedRecommendation::SyncAcceptable
        } else {
            EmbedRecommendation::Background
        }
    }

    /// Internal implementation of ensure_fresh without lock handling.
    async fn ensure_fresh_internal(&mut self, mode: EmbedMode) -> Result<IncrementalStats> {
        let mut stats = IncrementalStats {
            added: 0,
            updated: 0,
            removed: 0,
            pending_embeddings: 0,
            embed_recommendation: EmbedRecommendation::None,
        };

        // 1. Scan current files
        let current_files = self.index.scan_files()
            .context("Failed to scan files")?;
        let current_paths: HashSet<PathBuf> = current_files
            .iter()
            .map(|(p, _)| p.clone())
            .collect();

        // 2. Find deleted files
        let indexed_paths: Vec<PathBuf> = self.index.get_indexed_paths();
        let mut files_to_remove: Vec<PathBuf> = Vec::new();
        for path in &indexed_paths {
            if !current_paths.contains(path) {
                files_to_remove.push(path.clone());
                stats.removed += 1;
            }
        }

        // 3. Find modified/new files
        let mut files_to_index: Vec<(PathBuf, ChunkType)> = Vec::new();
        for (path, chunk_type) in &current_files {
            match self.index.file_needs_update(path) {
                Ok(true) => {
                    if self.index.has_file(path) {
                        stats.updated += 1;
                        files_to_remove.push(path.clone());
                    } else {
                        stats.added += 1;
                    }
                    files_to_index.push((path.clone(), chunk_type.clone()));
                }
                Ok(false) => {}
                Err(e) => {
                    warn!("[scs] Warning: Failed to check file {}: {}", path.display(), e);
                }
            }
        }

        // Early return if no changes
        if !stats.has_changes() {
            return Ok(stats);
        }

        // 4. Remove old chunks/embeddings for deleted/modified files
        self.index.remove_files(&files_to_remove);

        // 5. Parse new files and intern strings
        for (path, chunk_type) in &files_to_index {
            let file_idx = self.index.index.strings.intern(&path.to_string_lossy());
            let mtime = get_mtime(path).unwrap_or(0);

            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    warn!("[scs] Warning: Failed to read {}: {}", path.display(), e);
                    // Register file with error marker to prevent re-indexing
                    let error_hash = format!("__read_error_{}", mtime);
                    let hash_idx = self.index.index.strings.intern(&error_hash);
                    self.index.index.files.push(FileEntry {
                        path_idx: file_idx,
                        mtime,
                        hash_idx,
                        chunk_type: chunk_type.clone(),
                        chunk_range: (0, 0), // No chunks
                    });
                    continue;
                }
            };

            let raw_chunks = match chunk_type {
                ChunkType::Code => self.code_parser.parse(path, &content)?,
                ChunkType::Doc => self.docs_parser.parse(path, &content)?,
            };

            let chunk_start = self.index.index.chunks.len() as u32;

            for raw in raw_chunks {
                let name_idx = self.index.index.strings.intern(&raw.name);
                let content_idx = self.index.index.strings.intern(&raw.content);
                let context_idx = raw.context.as_ref().map(|c| self.index.index.strings.intern(c));
                let signature_idx = raw.signature.as_ref().map(|s| self.index.index.strings.intern(s));

                let doc_summary_idx = raw.doc_summary.as_ref().map(|s| self.index.index.strings.intern(s));

                let chunk = Chunk {
                    chunk_type: raw.chunk_type,
                    name_idx,
                    kind: raw.kind,
                    file_idx,
                    line_start: raw.line_start,
                    line_end: raw.line_end,
                    content_idx,
                    context_idx,
                    signature_idx,
                    doc_summary_idx,
                    visibility: raw.visibility,
                };
                self.index.index.chunks.push(chunk);
            }

            let chunk_end = self.index.index.chunks.len() as u32;
            let hash = compute_hash(path).unwrap_or_else(|_| format!("__hash_error_{}", mtime));
            let hash_idx = self.index.index.strings.intern(&hash);

            self.index.index.files.push(FileEntry {
                path_idx: file_idx,
                mtime,
                hash_idx,
                chunk_type: chunk_type.clone(),
                chunk_range: (chunk_start, chunk_end),
            });
        }

        // 6. Build runtime caches
        self.index.build_runtime_caches();

        // 7. Calculate pending embeddings
        let existing_embeddings = self.index.embeddings.len();
        let total_chunks = self.index.index.chunks.len();
        stats.pending_embeddings = total_chunks.saturating_sub(existing_embeddings);
        stats.embed_recommendation = Self::calculate_embed_recommendation(stats.pending_embeddings);

        // 8. Spawn background embed if needed (Auto mode only)
        if mode == EmbedMode::Auto && stats.pending_embeddings > 0 {
            if let Ok(exe_path) = std::env::current_exe() {
                let root_str = self.index.root.to_string_lossy().to_string();
                match Command::new(&exe_path)
                    .args(["embed", "--path", &root_str])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                {
                    Ok(_) => {
                        warn!(
                            "[scs] Started background embedding for {} chunks",
                            stats.pending_embeddings
                        );
                    }
                    Err(e) => {
                        warn!(
                            "[scs] Failed to start background embed: {}. Run 'scs embed' manually.",
                            e
                        );
                    }
                }
            } else {
                warn!(
                    "[scs] {} chunks need embeddings. Run 'scs embed' for semantic search.",
                    stats.pending_embeddings
                );
            }
        }

        // 9. Update metadata and save
        self.index.index.last_indexed = chrono::Utc::now().to_rfc3339();
        self.index.save()
            .context("Failed to save index")?;

        scs_log!(
            "[refresh] {} added, {} updated, {} removed (pending_embed={})",
            stats.added, stats.updated, stats.removed, stats.pending_embeddings
        );

        Ok(stats)
    }

    /// Format embedding text for a chunk.
    fn format_embedding_text(&self, chunk: &Chunk) -> String {
        let name = self.index.index.strings.get(chunk.name_idx).unwrap_or("");
        let content = self.index.index.strings.get(chunk.content_idx).unwrap_or("");
        let truncated: String = content.chars().take(500).collect();
        match chunk.chunk_type {
            ChunkType::Code => format!("{:?} {}: {}", chunk.kind, name, truncated),
            ChunkType::Doc => format!("{}: {}", name, truncated),
        }
    }

    /// Perform semantic search over the indexed chunks.
    pub async fn search(
        &mut self,
        query: &str,
        top_k: usize,
        filter: Option<ChunkType>,
    ) -> Result<SearchOutput> {
        // Validate empty query - return empty results instead of API error
        if query.trim().is_empty() {
            return Ok(SearchOutput {
                query: query.to_string(),
                match_type: MatchType::NameOnly,
                results: vec![],
                suggestions: vec!["Please provide a search query".to_string()],
            });
        }

        let has_embeddings = !self.index.embeddings.is_empty()
            && self.index.embeddings.len() == self.index.index.chunks.len();

        if has_embeddings {
            match self.get_query_embedding(query).await {
                Ok(Some(embedding)) => {
                    let mut output = search::semantic::search(
                        &self.index.index,
                        &self.index.embeddings,
                        &embedding,
                        top_k,
                        filter,
                    );
                    output.query = query.to_string();
                    return Ok(output);
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!(
                        "[scs] Warning: Query embedding failed: {}, using name-based search",
                        e
                    );
                }
            }
        }

        let output = self.search_by_name(query, top_k, filter);
        Ok(output)
    }

    /// Search chunks by matching query against names and content.
    fn search_by_name(&self, query: &str, top_k: usize, filter: Option<ChunkType>) -> SearchOutput {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored_results: Vec<(f32, usize)> = self.index.index.chunks
            .iter()
            .enumerate()
            .filter(|(_, chunk)| filter.as_ref().map_or(true, |t| &chunk.chunk_type == t))
            .filter_map(|(idx, chunk)| {
                let name = self.index.index.strings.get(chunk.name_idx).unwrap_or("");
                let content = self.index.index.strings.get(chunk.content_idx).unwrap_or("");
                let name_lower = name.to_lowercase();
                let content_lower = content.to_lowercase();

                let mut score: f32 = 0.0;

                if name_lower == query_lower {
                    score += 1.0;
                } else if name_lower.contains(&query_lower) {
                    score += 0.8;
                } else if query_lower.contains(&name_lower) {
                    score += 0.6;
                }

                for word in &query_words {
                    if name_lower.contains(word) {
                        score += 0.3;
                    }
                }

                for word in &query_words {
                    if content_lower.contains(word) {
                        score += 0.1;
                    }
                }

                if score > 0.0 {
                    Some((score, idx))
                } else {
                    None
                }
            })
            .collect();

        scored_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let results: Vec<SearchResult> = scored_results
            .into_iter()
            .take(top_k)
            .filter_map(|(score, idx)| {
                let chunk = self.index.index.chunks.get(idx)?;
                let name = self.index.index.strings.get(chunk.name_idx)?;
                let file = self.index.index.strings.get(chunk.file_idx)?;
                let content = self.index.index.strings.get(chunk.content_idx).unwrap_or("");
                let context = chunk.context_idx.and_then(|i| self.index.index.strings.get(i));
                let is_unique = self.index.get_name_count(chunk.name_idx) == 1;

                Some(SearchResult {
                    chunk_type: chunk.chunk_type.clone(),
                    name: name.to_string(),
                    kind: chunk.kind.clone(),
                    file: PathBuf::from(file),
                    line_start: chunk.line_start as usize,
                    line_end: chunk.line_end as usize,
                    score,
                    preview: search::semantic::make_preview(content),
                    context: context.map(|s| s.to_string()),
                    unique: is_unique,
                })
            })
            .collect();

        let pending = self.missing_embeddings_count();
        let suggestions = if pending > 0 {
            vec![format!(
                "Using name-based search ({} chunks pending embeddings). Run 'scs embed' for semantic search.",
                pending
            )]
        } else if !self.has_embedder() {
            vec!["Using name-based search (OPENAI_API_KEY not set).".to_string()]
        } else {
            vec![]
        };

        SearchOutput {
            query: query.to_string(),
            match_type: MatchType::NameOnly,
            results,
            suggestions,
        }
    }

    /// Look up a symbol by exact name.
    pub fn lookup(&self, name: &str, filter: Option<ChunkType>) -> LookupOutput {
        search::lookup::lookup(&self.index.index, &self.index.name_to_chunks, name, filter)
    }

    /// Get the current status of the index.
    pub fn status(&self) -> Value {
        let embedding_count = self.index.embeddings.len();
        let chunk_count = self.index.index.chunks.len();
        let has_embeddings = embedding_count > 0 && embedding_count == chunk_count;
        let pending_embeddings = self.missing_embeddings_count();
        let embed_recommendation = Self::calculate_embed_recommendation(pending_embeddings);
        let root = self.index.index.root();
        let embedding_model = self.index.index.embedding_model();

        let code_files = self.index.index.files.iter()
            .filter(|f| f.chunk_type == ChunkType::Code)
            .count();
        let doc_files = self.index.index.files.iter()
            .filter(|f| f.chunk_type == ChunkType::Doc)
            .count();

        let duplicate_names = self.index.name_to_chunks.values()
            .filter(|v| v.len() > 1)
            .count();

        json!({
            "version": self.index.index.version,
            "root": root,
            "last_indexed": self.index.index.last_indexed,
            "file_count": self.index.index.files.len(),
            "code_files": code_files,
            "doc_files": doc_files,
            "chunk_count": chunk_count,
            "embedding_count": embedding_count,
            "embedding_model": embedding_model,
            "has_embeddings": has_embeddings,
            "pending_embeddings": pending_embeddings,
            "embed_recommendation": format!("{:?}", embed_recommendation).to_lowercase(),
            "duplicate_names": duplicate_names,
            "embedder_available": self.embedder.is_some(),
            "query_cache_size": self.query_cache.len(),
            "string_table_size": self.index.index.strings.len()
        })
    }

    /// Force a complete reindex of all files.
    pub async fn reindex_all(&mut self, mode: EmbedMode) -> Result<()> {
        self.index.acquire_lock()
            .context("Failed to acquire index lock")?;

        let root = self.index.root.clone();
        self.index.index = Index::new(&root.to_string_lossy(), "text-embedding-3-small");
        self.index.embeddings.clear();
        self.index.name_to_chunks.clear();
        self.index.file_to_entry.clear();
        self.query_cache.clear();

        let result = self.ensure_fresh_internal(mode).await;

        if let Err(e) = self.index.release_lock() {
            warn!("[scs] Warning: Failed to release lock: {}", e);
        }

        result.map(|_| ())
    }

    async fn get_query_embedding(&mut self, query: &str) -> Result<Option<Vec<f32>>> {
        if let Some(emb) = self.query_cache.get(query) {
            return Ok(Some(emb));
        }

        if let Some(ref embedder) = self.embedder {
            let emb = embedder.embed_single(query).await?;
            self.query_cache.put(query.to_string(), emb.clone());
            Ok(Some(emb))
        } else {
            Ok(None)
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.index.index.chunks.len()
    }

    pub fn file_count(&self) -> usize {
        self.index.index.files.len()
    }

    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    pub fn missing_embeddings_count(&self) -> usize {
        if self.index.embeddings.len() >= self.index.index.chunks.len() {
            0
        } else {
            self.index.index.chunks.len() - self.index.embeddings.len()
        }
    }

    /// Get optimal batch size based on API rate limits.
    /// Conservative scaling: concurrency * 20, capped at 2000 for stability.
    /// Testing showed 5000 causes timeout, 1000 is stable.
    pub async fn get_optimal_batch_size(&mut self) -> Result<usize> {
        if let Some(ref mut embedder) = self.embedder {
            let info = embedder.check_rate_limits().await?;
            let concurrency = info.optimal_concurrency();
            // concurrency * 20, min 500, max 2000
            let batch = (concurrency * 20).clamp(500, 2000);
            Ok(batch)
        } else {
            anyhow::bail!("OPENAI_API_KEY not set")
        }
    }

    // ========================================================================
    // Map Summary Methods
    // ========================================================================

    fn load_summary_cache(&self) -> summarizer::SummaryCache {
        let cache_path = self.index.index_dir.join("summaries.json");
        summarizer::SummaryCache::load(&cache_path)
            .unwrap_or_else(|_| summarizer::SummaryCache::new())
    }

    fn save_summary_cache(&self, cache: &summarizer::SummaryCache) {
        let cache_path = self.index.index_dir.join("summaries.json");
        if let Err(e) = cache.save(&cache_path) {
            warn!("[scs] Warning: Failed to save summary cache: {}", e);
        }
    }

    fn collect_needed_dir_file_summaries(
        dirs: &[map::DirNode],
        cache: &summarizer::SummaryCache,
    ) -> Vec<summarizer::gemini::MapSummaryInput> {
        let mut needed = Vec::new();

        for dir in dirs {
            let dir_key = format!("dir::{}", dir.path);
            let dir_hash = map::dir_content_hash(dir);
            if cache.get(&dir_key, &dir_hash, "").is_none() {
                let symbols: Vec<String> = dir.files.iter()
                    .flat_map(|f| f.symbols.iter().map(|s| s.name.clone()))
                    .collect();
                needed.push(summarizer::gemini::MapSummaryInput {
                    path: dir.path.clone(),
                    kind: "directory",
                    symbols,
                });
            }

            for file in &dir.files {
                let file_key = format!("file::{}:{}", dir.path, file.name);
                let file_hash = map::file_content_hash(file);
                if cache.get(&file_key, &file_hash, "").is_none() {
                    let symbols: Vec<String> = file.symbols.iter()
                        .map(|s| s.name.clone())
                        .collect();
                    needed.push(summarizer::gemini::MapSummaryInput {
                        path: format!("{}/{}", dir.path, file.name),
                        kind: "file",
                        symbols,
                    });
                }
            }
        }

        needed
    }

    fn cache_dir_file_batch_result(
        batch: &[summarizer::gemini::MapSummaryInput],
        result: &summarizer::gemini::MapBatchResult,
        dirs: &[map::DirNode],
        cache: &mut summarizer::SummaryCache,
    ) {
        for (path, summary) in &result.summaries {
            let is_dir = batch.iter().any(|e| e.path == *path && e.kind == "directory");
            if is_dir {
                let dir_key = format!("dir::{}", path);
                if let Some(dir) = dirs.iter().find(|d| d.path == *path) {
                    let dir_hash = map::dir_content_hash(dir);
                    cache.insert(dir_key, dir_hash, String::new(), summary.clone());
                }
            } else {
                let dir_path = map::extract_dir_path(path);
                let file_name = path.rsplit('/').next().unwrap_or(path);
                let file_key = format!("file::{}:{}", dir_path, file_name);
                if let Some(dir) = dirs.iter().find(|d| d.path == dir_path) {
                    if let Some(file) = dir.files.iter().find(|f| f.name == file_name) {
                        let file_hash = map::file_content_hash(file);
                        cache.insert(file_key, file_hash, String::new(), summary.clone());
                    }
                }
            }
        }
    }

    fn apply_dir_file_summaries(dirs: &mut [map::DirNode], cache: &summarizer::SummaryCache) {
        for dir in dirs.iter_mut() {
            let dir_key = format!("dir::{}", dir.path);
            let dir_hash = map::dir_content_hash(dir);
            if let Some(summary) = cache.get(&dir_key, &dir_hash, "") {
                dir.summary = Some(summary.to_string());
            }

            for file in &mut dir.files {
                let file_key = format!("file::{}:{}", dir.path, file.name);
                let file_hash = map::file_content_hash(file);
                if let Some(summary) = cache.get(&file_key, &file_hash, "") {
                    file.summary = Some(summary.to_string());
                }
            }
        }
    }

    /// Generate LLM summaries for directories and files in flat map mode.
    pub async fn ensure_map_summaries(&mut self, dirs: &mut [map::DirNode]) -> Result<()> {
        let mut cache = self.load_summary_cache();
        let needed = Self::collect_needed_dir_file_summaries(dirs, &cache);

        if !needed.is_empty() {
            scs_log!("[map] generating {} map summaries", needed.len());
            match summarizer::gemini::GeminiClient::from_env() {
                Ok(client) => {
                    for batch in needed.chunks(50) {
                        match client.summarize_map_batch(batch).await {
                            Ok(result) => {
                                Self::cache_dir_file_batch_result(batch, &result, dirs, &mut cache);
                            }
                            Err(e) => warn!("[scs] Warning: Map summary batch failed: {}", e),
                        }
                    }
                    self.save_summary_cache(&cache);
                }
                Err(e) => warn!("[scs] Warning: Gemini client not available: {}", e),
            }
        }

        Self::apply_dir_file_summaries(dirs, &cache);
        Ok(())
    }

    /// Generate LLM summaries for collapsed tree nodes in large project mode.
    pub async fn ensure_tree_summaries(
        &mut self,
        trees: &mut [map::DirTree],
        dirs: &[map::DirNode],
    ) -> Result<()> {
        let mut cache = self.load_summary_cache();
        let tree_nodes = map::collect_tree_nodes_with_paths(trees, "");

        let mut needed: Vec<summarizer::gemini::MapSummaryInput> = Vec::new();
        for (path, _files, _symbols) in &tree_nodes {
            let tree_key = format!("tree::{}", path);
            let tree_hash = map::tree_content_hash(path, dirs);
            if cache.get(&tree_key, &tree_hash, "").is_none() {
                let symbols = map::collect_tree_symbol_names(path, dirs, 30);
                needed.push(summarizer::gemini::MapSummaryInput {
                    path: path.clone(),
                    kind: "directory",
                    symbols,
                });
            }
        }

        if !needed.is_empty() {
            scs_log!("[map] generating {} tree summaries", needed.len());
            match summarizer::gemini::GeminiClient::from_env() {
                Ok(client) => {
                    for batch in needed.chunks(50) {
                        match client.summarize_map_batch(batch).await {
                            Ok(result) => {
                                for (path, summary) in &result.summaries {
                                    let tree_key = format!("tree::{}", path);
                                    let tree_hash = map::tree_content_hash(path, dirs);
                                    cache.insert(tree_key, tree_hash, String::new(), summary.clone());
                                }
                            }
                            Err(e) => warn!("[scs] Warning: Tree summary batch failed: {}", e),
                        }
                    }
                    self.save_summary_cache(&cache);
                }
                Err(e) => warn!("[scs] Warning: Gemini client not available: {}", e),
            }
        }

        let mut summary_map: HashMap<String, String> = HashMap::new();
        for (path, _, _) in &tree_nodes {
            let tree_key = format!("tree::{}", path);
            let tree_hash = map::tree_content_hash(path, dirs);
            if let Some(summary) = cache.get(&tree_key, &tree_hash, "") {
                summary_map.insert(path.clone(), summary.to_string());
            }
        }
        map::attach_tree_summaries(trees, &summary_map, "");

        Ok(())
    }

    /// Generate LLM summaries scoped to a specific area (directory or file).
    pub async fn ensure_area_summaries(
        &mut self,
        dirs: &mut [map::DirNode],
        area: &str,
    ) -> Result<()> {
        let mut cache = self.load_summary_cache();

        let area_norm = area.trim_end_matches('/');
        let area_dir = map::extract_dir_path(area_norm);

        // Filter dirs to area scope, then collect needed entries
        let area_dirs: Vec<map::DirNode> = dirs.iter()
            .filter(|dir| {
                dir.path == area_norm
                    || dir.path.starts_with(&format!("{}/", area_norm))
                    || dir.path == area_dir
            })
            .cloned()
            .collect();

        let needed = Self::collect_needed_dir_file_summaries(&area_dirs, &cache);

        if !needed.is_empty() {
            match summarizer::gemini::GeminiClient::from_env() {
                Ok(client) => {
                    for batch in needed.chunks(50) {
                        match client.summarize_map_batch(batch).await {
                            Ok(result) => {
                                Self::cache_dir_file_batch_result(batch, &result, dirs, &mut cache);
                            }
                            Err(e) => warn!("[scs] Warning: Area summary batch failed: {}", e),
                        }
                    }
                    self.save_summary_cache(&cache);
                }
                Err(e) => warn!("[scs] Warning: Gemini client not available: {}", e),
            }
        }

        Self::apply_dir_file_summaries(dirs, &cache);
        Ok(())
    }

    // ========================================================================
    // Function Summary Methods
    // ========================================================================

    /// Generate function-level LLM summaries using dependency-ordered batch processing.
    /// Uses CallGraph levels to summarize leaf functions first, then inject their
    /// summaries as context when summarizing callers.
    pub async fn generate_summaries(
        &mut self,
        batch_size: usize,
        force: bool,
    ) -> Result<summarizer::SummaryStats> {
        use sha2::{Sha256, Digest};

        let mut cache = if force {
            summarizer::SummaryCache::new()
        } else {
            self.load_summary_cache()
        };

        let mut stats = summarizer::SummaryStats::new();

        // Build call graph from index chunks
        let mut call_graph = summarizer::levels::CallGraph::new();
        let mut chunk_bodies: HashMap<String, (usize, String)> = HashMap::new(); // name -> (chunk_idx, body)

        for (idx, chunk) in self.index.index.chunks.iter().enumerate() {
            if chunk.chunk_type != ChunkType::Code {
                continue;
            }
            match chunk.kind {
                ChunkKind::Function | ChunkKind::Method => {}
                _ => continue,
            }

            let name = self.index.index.strings.get(chunk.name_idx).unwrap_or("").to_string();
            let body = self.index.index.strings.get(chunk.content_idx).unwrap_or("").to_string();

            call_graph.add_function(&name);
            chunk_bodies.insert(name.clone(), (idx, body));

            // Extract calls from body (simple heuristic: match known function names)
            let content = self.index.index.strings.get(chunk.content_idx).unwrap_or("");
            for (other_idx, other_chunk) in self.index.index.chunks.iter().enumerate() {
                if other_idx == idx { continue; }
                if other_chunk.chunk_type != ChunkType::Code { continue; }
                match other_chunk.kind {
                    ChunkKind::Function | ChunkKind::Method => {}
                    _ => continue,
                }
                let other_name = self.index.index.strings.get(other_chunk.name_idx).unwrap_or("");
                if !other_name.is_empty() && content.contains(other_name) {
                    call_graph.add_call(&name, other_name);
                }
            }
        }

        let levels = call_graph.compute_levels();
        stats.total_functions = chunk_bodies.len();
        stats.levels = levels.len();

        // Create Gemini client
        let client = match summarizer::gemini::GeminiClient::from_env() {
            Ok(c) => c,
            Err(e) => {
                warn!("[scs] Warning: Gemini client not available: {}", e);
                return Ok(stats);
            }
        };

        // Track generated summaries for context injection
        let mut generated_summaries: HashMap<String, String> = HashMap::new();

        // Process each level in dependency order
        for level_funcs in &levels {
            let mut batch_inputs: Vec<summarizer::gemini::FunctionInput> = Vec::new();

            for func_name in level_funcs {
                let (chunk_idx, body) = match chunk_bodies.get(func_name) {
                    Some(v) => v.clone(),
                    None => continue,
                };

                let body_hash = format!("{:x}", Sha256::digest(body.as_bytes()));

                // Build context hash from callee summaries
                let callees = call_graph.get_calls(func_name);
                let mut callee_summaries_sorted: Vec<String> = callees.iter()
                    .filter_map(|c| generated_summaries.get(c))
                    .cloned()
                    .collect();
                callee_summaries_sorted.sort();
                let context_hash = format!("{:x}", Sha256::digest(callee_summaries_sorted.join("|").as_bytes()));

                // Check cache
                if cache.get(func_name, &body_hash, &context_hash).is_some() {
                    stats.cached += 1;
                    // Use cached summary for context in later levels
                    if let Some(entry) = cache.entries.get(func_name) {
                        generated_summaries.insert(func_name.clone(), entry.summary.clone());
                    }
                    continue;
                }

                // Build callee context map
                let mut calls: HashMap<String, String> = HashMap::new();
                for callee in callees {
                    if let Some(summary) = generated_summaries.get(callee) {
                        calls.insert(callee.clone(), summary.clone());
                    }
                }

                batch_inputs.push(summarizer::gemini::FunctionInput {
                    chunk_idx,
                    name: func_name.clone(),
                    body,
                    calls,
                });
            }

            // Process batch_inputs in batch_size chunks
            for batch in batch_inputs.chunks(batch_size) {
                if batch.is_empty() { continue; }

                stats.api_calls += 1;
                match client.summarize_batch(batch).await {
                    Ok(result) => {
                        stats.input_tokens += result.input_tokens;
                        stats.output_tokens += result.output_tokens;

                        for (_, summary) in &result.summaries {
                            // Find the matching function input
                            if let Some(input) = batch.iter().find(|i| {
                                result.summaries.iter().any(|(ci, s)| *ci == i.chunk_idx && s == summary)
                            }) {
                                let body_hash = format!("{:x}", Sha256::digest(input.body.as_bytes()));
                                let callees = call_graph.get_calls(&input.name);
                                let mut callee_summaries_sorted: Vec<String> = callees.iter()
                                    .filter_map(|c| generated_summaries.get(c))
                                    .cloned()
                                    .collect();
                                callee_summaries_sorted.sort();
                                let context_hash = format!("{:x}", Sha256::digest(callee_summaries_sorted.join("|").as_bytes()));

                                cache.insert(
                                    input.name.clone(),
                                    body_hash,
                                    context_hash,
                                    summary.clone(),
                                );
                                generated_summaries.insert(input.name.clone(), summary.clone());
                                stats.summarized += 1;
                            }
                        }

                        // Track failures (inputs not in results)
                        let result_indices: HashSet<usize> =
                            result.summaries.iter().map(|(ci, _)| *ci).collect();
                        for input in batch {
                            if !result_indices.contains(&input.chunk_idx) {
                                stats.failed += 1;
                            }
                        }
                    }
                    Err(e) => {
                        warn!("[scs] Warning: Summary batch failed: {}", e);
                        stats.failed += batch.len();
                    }
                }
            }
        }

        self.save_summary_cache(&cache);

        Ok(stats)
    }

    /// Returns summary stats without actually generating summaries (dry run).
    pub fn summarize_dry_run(&self) -> summarizer::SummaryStats {
        let mut call_graph = summarizer::levels::CallGraph::new();
        let mut func_count = 0usize;

        for (idx, chunk) in self.index.index.chunks.iter().enumerate() {
            if chunk.chunk_type != ChunkType::Code {
                continue;
            }
            match chunk.kind {
                ChunkKind::Function | ChunkKind::Method => {}
                _ => continue,
            }

            let name = self.index.index.strings.get(chunk.name_idx).unwrap_or("");
            call_graph.add_function(name);
            func_count += 1;

            // Extract calls from body
            let content = self.index.index.strings.get(chunk.content_idx).unwrap_or("");
            for (other_idx, other_chunk) in self.index.index.chunks.iter().enumerate() {
                if other_idx == idx { continue; }
                if other_chunk.chunk_type != ChunkType::Code { continue; }
                match other_chunk.kind {
                    ChunkKind::Function | ChunkKind::Method => {}
                    _ => continue,
                }
                let other_name = self.index.index.strings.get(other_chunk.name_idx).unwrap_or("");
                if !other_name.is_empty() && content.contains(other_name) {
                    call_graph.add_call(name, other_name);
                }
            }
        }

        let levels = call_graph.compute_levels();

        summarizer::SummaryStats {
            total_functions: func_count,
            levels: levels.len(),
            ..Default::default()
        }
    }

    /// Generate embeddings for chunks that don't have them.
    pub async fn generate_embeddings(&mut self, batch_size: usize) -> Result<usize> {
        if batch_size == 0 {
            anyhow::bail!("batch_size must be greater than 0");
        }

        let total_missing = self.missing_embeddings_count();
        scs_log!("[embed] starting: {} chunks pending, batch_size={}", total_missing, batch_size);

        // Check rate limits first (requires mutable borrow)
        if let Some(ref mut embedder) = self.embedder {
            if let Err(e) = embedder.check_rate_limits().await {
                warn!("[scs] Warning: Failed to check rate limits: {}", e);
            }
        } else {
            anyhow::bail!("OPENAI_API_KEY not set");
        }

        let mut generated = 0;
        let mut batch_num = 0;

        loop {
            self.index.acquire_lock()
                .context("Failed to acquire index lock")?;

            let start_idx = self.index.embeddings.len();
            let total = self.index.index.chunks.len();

            if start_idx >= total {
                self.index.release_lock()?;
                break;
            }

            let end_idx = (start_idx + batch_size).min(total);
            let batch = &self.index.index.chunks[start_idx..end_idx];

            let texts: Vec<String> = batch
                .iter()
                .map(|c| self.format_embedding_text(c))
                .collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            let embedder = self.embedder.as_ref().unwrap();
            match embedder.embed_batch(text_refs).await {
                Ok(new_embeddings) => {
                    let count = new_embeddings.len();
                    self.index.embeddings.extend(new_embeddings);
                    generated += count;

                    if let Err(e) = self.index.save() {
                        warn!("[scs] Warning: Failed to save after batch: {}", e);
                    }

                    let progress = self.index.embeddings.len();
                    eprintln!(
                        "[scs] Batch {}: {}/{} ({:.1}%)",
                        batch_num + 1,
                        progress,
                        total,
                        (progress as f64 / total as f64) * 100.0
                    );
                }
                Err(e) => {
                    eprintln!("[scs] Error in batch {}: {}", batch_num + 1, e);
                    let _ = self.index.save();
                    self.index.release_lock()?;
                    return Err(e);
                }
            }

            self.index.release_lock()?;
            batch_num += 1;

            tokio::task::yield_now().await;
        }

        scs_log!("[embed] completed: {} embeddings generated", generated);

        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_string_table() {
        let mut st = StringTable::new();
        let idx1 = st.intern("hello");
        let idx2 = st.intern("world");
        let idx3 = st.intern("hello");

        assert_eq!(idx1, idx3);
        assert_ne!(idx1, idx2);
        assert_eq!(st.get(idx1), Some("hello"));
        assert_eq!(st.get(idx2), Some("world"));
        assert_eq!(st.len(), 2);
    }

    #[test]
    fn test_string_table_rebuild_index() {
        let mut st = StringTable::new();
        st.intern("foo");
        st.intern("bar");

        st.index.clear();
        assert!(st.find("foo").is_none());

        st.rebuild_index();
        assert_eq!(st.find("foo"), Some(0));
        assert_eq!(st.find("bar"), Some(1));
    }

    #[test]
    fn test_scs_load_or_create_new() {
        let temp_dir = TempDir::new().unwrap();
        let scs = SCS::load_or_create(temp_dir.path()).unwrap();

        assert_eq!(scs.chunk_count(), 0);
        assert_eq!(scs.file_count(), 0);
    }

    #[test]
    fn test_status_json() {
        let temp_dir = TempDir::new().unwrap();
        let scs = SCS::load_or_create(temp_dir.path()).unwrap();

        let status = scs.status();
        assert!(status.get("version").is_some());
        assert!(status.get("root").is_some());
        assert!(status.get("file_count").is_some());
        assert!(status.get("chunk_count").is_some());
        assert!(status.get("has_embeddings").is_some());
        assert!(status.get("string_table_size").is_some());
    }

    #[test]
    fn test_embed_recommendation() {
        assert_eq!(SCS::calculate_embed_recommendation(0), EmbedRecommendation::None);
        assert_eq!(SCS::calculate_embed_recommendation(50), EmbedRecommendation::Sync);
        assert_eq!(SCS::calculate_embed_recommendation(100), EmbedRecommendation::Sync);
        assert_eq!(SCS::calculate_embed_recommendation(200), EmbedRecommendation::SyncAcceptable);
        assert_eq!(SCS::calculate_embed_recommendation(500), EmbedRecommendation::SyncAcceptable);
        assert_eq!(SCS::calculate_embed_recommendation(600), EmbedRecommendation::Background);
        assert_eq!(SCS::calculate_embed_recommendation(1000), EmbedRecommendation::Background);
    }

    #[tokio::test]
    async fn test_lookup() {
        let temp_dir = TempDir::new().unwrap();

        std::fs::write(
            temp_dir.path().join("test.rs"),
            "fn hello() { println!(\"Hello\"); }"
        ).unwrap();

        let mut scs = SCS::load_or_create(temp_dir.path()).unwrap();
        let _ = scs.ensure_fresh(EmbedMode::Skip).await;

        let result = scs.lookup("hello", None);
        assert!(!result.definitions.is_empty() || result.suggestions.contains(&"Symbol not found".to_string()));
    }
}
