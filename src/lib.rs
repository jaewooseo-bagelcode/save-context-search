use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

pub mod index;
pub mod embeddings;
pub mod parser;
pub mod search;

// Global quiet mode flag
static QUIET_MODE: AtomicBool = AtomicBool::new(false);

/// Set global quiet mode (suppresses warnings)
pub fn set_quiet_mode(quiet: bool) {
    QUIET_MODE.store(quiet, Ordering::SeqCst);
}

/// Check if quiet mode is enabled
pub fn is_quiet() -> bool {
    QUIET_MODE.load(Ordering::SeqCst)
}

/// Print warning only if not in quiet mode
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        if !$crate::is_quiet() {
            eprintln!($($arg)*);
        }
    };
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
}

// ============================================================================
// SCS - Main orchestration struct
// ============================================================================

use std::collections::HashSet;
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

    /// Generate an outline of a file's structure.
    pub fn outline(&self, file: &Path) -> Value {
        search::lookup::outline(&self.index.index, file)
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

    /// Generate embeddings for chunks that don't have them.
    pub async fn generate_embeddings(&mut self, batch_size: usize) -> Result<usize> {
        if batch_size == 0 {
            anyhow::bail!("batch_size must be greater than 0");
        }

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
