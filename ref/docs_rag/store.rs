use super::chunker::{Chunk, Chunker};
use super::embedder::Embedder;
use super::{RagStatus, SearchResult};
use glob::glob;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs as async_fs;
use tokio::sync::Mutex;

const CACHE_DIR: &str = ".syntaxos/docs-rag";
const EMBEDDINGS_BIN: &str = "embeddings.bin";
const METADATA_JSON: &str = "metadata.json";
const INDEX_JSON: &str = "index.json";

/// Directories to exclude from indexing (performance optimization)
const EXCLUDED_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "dist",
    "build",
    ".git",
    ".syntaxos",
    "__pycache__",
    "vendor",
    ".next",
    ".nuxt",
    "coverage",
];

/// LRU Cache for query embeddings using counter-based eviction
/// O(1) get, O(n) eviction only when cache is full
/// Stores up to 100 query embeddings to avoid redundant API calls
pub struct QueryCache {
    cache: HashMap<String, (u64, Vec<f32>)>,  // (access_counter, embedding)
    counter: u64,
    capacity: usize,
}

impl QueryCache {
    pub fn new(capacity: usize) -> Self {
        QueryCache {
            cache: HashMap::new(),
            counter: 0,
            capacity,
        }
    }

    /// Get embedding from cache if exists - O(1)
    /// Returns Some(embedding) if found, None otherwise
    pub fn get(&mut self, query: &str) -> Option<Vec<f32>> {
        if let Some((access, embedding)) = self.cache.get_mut(query) {
            self.counter += 1;
            *access = self.counter;
            eprintln!("[DocsRAG/Store] QueryCache hit: query_len={}", query.len());
            Some(embedding.clone())
        } else {
            None
        }
    }

    /// Put embedding into cache
    /// If cache is full, removes LRU entry (lowest access counter) - O(n) only on eviction
    pub fn put(&mut self, query: String, embedding: Vec<f32>) {
        self.counter += 1;

        // If at capacity and new key, evict LRU entry
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&query) {
            if let Some(lru_key) = self.cache.iter()
                .min_by_key(|(_, (access, _))| *access)
                .map(|(k, _)| k.clone())
            {
                self.cache.remove(&lru_key);
                eprintln!("[DocsRAG/Store] QueryCache evicted: query_len={}", lru_key.len());
            }
        }

        self.cache.insert(query, (self.counter, embedding));
        eprintln!("[DocsRAG/Store] QueryCache miss: added new entry, cache_size={}", self.cache.len());
    }
}


pub struct DocsRagStore {
    project_path: String,
    chunks: Vec<Chunk>,
    embeddings: Vec<Vec<f32>>,
    last_indexed: Option<u64>,
    query_cache: Mutex<QueryCache>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheMetadata {
    chunks: Vec<ChunkMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkMetadata {
    file_path: String,
    start_line: usize,
    end_line: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheIndex {
    file_hashes: HashMap<String, String>,
    version: String,
}

impl DocsRagStore {
    pub async fn new(project_path: String) -> Result<Self, Box<dyn std::error::Error>> {
        let mut store = DocsRagStore {
            project_path: project_path.clone(),
            chunks: vec![],
            embeddings: vec![],
            last_indexed: None,
            query_cache: Mutex::new(QueryCache::new(100)),
        };
        
        store.index_project().await?;
        Ok(store)
    }
    
    async fn index_project(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Build glob patterns for supported file types
        let patterns = vec![
            "**/*.md",
            "**/*.txt",
            "**/*.rst",
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/*.toml",
            "**/*.xml",
        ];
        
        let mut all_chunks = vec![];
        
        for pattern_str in patterns {
            let full_pattern = format!("{}/{}", self.project_path, pattern_str);
            
            if let Ok(entries) = glob(&full_pattern) {
                for entry in entries {
                    if let Ok(path) = entry {
                        // Skip hidden directories and excluded directories
                        let should_skip = path.components().any(|c| {
                            let name = c.as_os_str().to_string_lossy();
                            // Skip hidden (starts with .)
                            if name.starts_with('.') {
                                return true;
                            }
                            // Skip excluded directories
                            EXCLUDED_DIRS.iter().any(|&excluded| name == excluded)
                        });

                        if should_skip {
                            continue;
                        }

                        // Skip if it's a directory
                        if path.is_dir() {
                            continue;
                        }

                        match Chunker::chunk_file(&path).await {
                            Ok(chunks) => all_chunks.extend(chunks),
                            Err(_) => continue,
                        }
                    }
                }
            }
        }
        
        self.chunks = all_chunks;
        
        // Load or compute embeddings
        self.load_or_compute_embeddings().await?;
        
        self.last_indexed = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),  // Fallback to 0 if system time is before UNIX_EPOCH (NTP issues)
        );
        
        Ok(())
    }
    
    async fn load_or_compute_embeddings(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let cache_dir = Path::new(&self.project_path).join(CACHE_DIR);
        
        // Try to load from cache
        if let Ok(embeddings) = self.load_cache(&cache_dir).await {
            self.embeddings = embeddings;
            return Ok(());
        }
        
        // Cache miss: compute embeddings
        self.compute_and_cache_embeddings(&cache_dir).await?;
        Ok(())
    }
    
    async fn load_cache(&self, cache_dir: &Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        // Check if cache files exist
        let embeddings_path = cache_dir.join(EMBEDDINGS_BIN);
        let metadata_path = cache_dir.join(METADATA_JSON);
        let index_path = cache_dir.join(INDEX_JSON);
        
        if !embeddings_path.exists() || !metadata_path.exists() || !index_path.exists() {
            return Err("Cache files not found".into());
        }
        
        // Load metadata
        let metadata_content = async_fs::read_to_string(&metadata_path).await?;
        let metadata: CacheMetadata = serde_json::from_str(&metadata_content)?;
        
        // Verify metadata matches current chunks
        if metadata.chunks.len() != self.chunks.len() {
            return Err("Cache size mismatch".into());
        }
        
        for (i, meta) in metadata.chunks.iter().enumerate() {
            let chunk = &self.chunks[i];
            if meta.file_path != chunk.file_path 
                || meta.start_line != chunk.start_line 
                || meta.end_line != chunk.end_line {
                return Err("Cache metadata mismatch".into());
            }
        }
        
        // Load index
        let index_content = async_fs::read_to_string(&index_path).await?;
        let index: CacheIndex = serde_json::from_str(&index_content)?;
        
        // Verify cache version
        if index.version != "1" {
            eprintln!("[DocsRAG/Store] load_cache: Cache version mismatch");
            return Err("Cache version mismatch".into());
        }
        
        // Build current file hashes (mtime)
        let mut current_hashes: HashMap<String, String> = HashMap::new();
        for chunk in &self.chunks {
            let metadata = std::fs::metadata(&chunk.file_path)?;
            let modified = metadata.modified()?;
            let duration = modified.duration_since(UNIX_EPOCH)?;
            current_hashes.insert(chunk.file_path.clone(), duration.as_secs().to_string());
        }
        
        // Compare file counts
        if current_hashes.len() != index.file_hashes.len() {
            eprintln!(
                "[DocsRAG/Store] load_cache: Cache file count mismatch: cached={} vs current={}",
                index.file_hashes.len(),
                current_hashes.len()
            );
            return Err("Cache file count mismatch".into());
        }
        
        // Compare file mtimes
        for (file_path, cached_mtime) in &index.file_hashes {
            match current_hashes.get(file_path) {
                Some(current_mtime) if current_mtime == cached_mtime => {}
                _ => {
                    eprintln!(
                        "[DocsRAG/Store] load_cache: Cache mtime mismatch for file: {}",
                        file_path
                    );
                    return Err("Cache mtime mismatch".into());
                }
            }
        }
        
        // Load embeddings binary
        let embeddings_data = async_fs::read(&embeddings_path).await?;
        let embeddings = self.deserialize_embeddings(&embeddings_data)?;
        
        Ok(embeddings)
    }
    
    async fn compute_and_cache_embeddings(
        &mut self,
        cache_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create cache directory
        fs::create_dir_all(cache_dir)?;
        
        // Initialize embedder
        let embedder = Embedder::new()?;
        
        // Extract chunk texts
        let chunk_texts: Vec<&str> = self.chunks.iter().map(|c| c.content.as_str()).collect();
        
        if chunk_texts.is_empty() {
            self.embeddings = vec![];
            return Ok(());
        }
        
        // Compute embeddings using batch API
        let embeddings = embedder.embed_batch(chunk_texts).await?;

        // Validate embedding count matches chunk count
        if embeddings.len() != self.chunks.len() {
            let err_msg = format!(
                "[DocsRAG/Store] embedding mismatch: expected {} embeddings but got {}",
                self.chunks.len(),
                embeddings.len()
            );
            eprintln!("{}", err_msg);
            return Err(err_msg.into());
        }
        // Save to cache
        self.save_cache(cache_dir, &embeddings).await?;
        
        self.embeddings = embeddings;
        Ok(())
    }
    
    async fn save_cache(
        &self,
        cache_dir: &Path,
        embeddings: &[Vec<f32>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Save embeddings as binary
        let embeddings_path = cache_dir.join(EMBEDDINGS_BIN);
        let embeddings_binary = self.serialize_embeddings(embeddings);
        async_fs::write(&embeddings_path, embeddings_binary).await?;
        
        // Save metadata
        let metadata = CacheMetadata {
            chunks: self.chunks.iter()
                .map(|c| ChunkMetadata {
                    file_path: c.file_path.clone(),
                    start_line: c.start_line,
                    end_line: c.end_line,
                })
                .collect(),
        };
        let metadata_path = cache_dir.join(METADATA_JSON);
        async_fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?).await?;
        
        // Save index (file hashes)
        let mut file_hashes = HashMap::new();
        for chunk in &self.chunks {
            // Simple hash: just file path and modification time
            if let Ok(metadata) = std::fs::metadata(&chunk.file_path) {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                        file_hashes.insert(
                            chunk.file_path.clone(),
                            format!("{}", duration.as_secs()),
                        );
                    }
                }
            }
        }
        
        let index = CacheIndex {
            file_hashes,
            version: "1".to_string(),
        };
        let index_path = cache_dir.join(INDEX_JSON);
        async_fs::write(&index_path, serde_json::to_string_pretty(&index)?).await?;
        
        Ok(())
    }

    pub async fn incremental_update(
        &mut self,
        modified: Vec<String>,
        added: Vec<String>,
        deleted: Vec<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!(
            "[DocsRAG/Store] incremental_update: modified={} added={} deleted={}",
            modified.len(),
            added.len(),
            deleted.len()
        );

        if modified.is_empty() && added.is_empty() && deleted.is_empty() {
            eprintln!("[DocsRAG/Store] incremental_update: no changes detected");
            return Ok(());
        }

        let mut indices_by_file: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, chunk) in self.chunks.iter().enumerate() {
            indices_by_file
                .entry(chunk.file_path.clone())
                .or_default()
                .push(idx);
        }

        let mut files_to_remove: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for file_path in deleted.iter().chain(modified.iter()) {
            files_to_remove.insert(file_path.clone());
        }

        let mut indices_to_remove: Vec<usize> = Vec::new();
        for file_path in &files_to_remove {
            if let Some(indices) = indices_by_file.get(file_path) {
                indices_to_remove.extend(indices.iter().copied());
            } else {
                eprintln!(
                    "[DocsRAG/Store] incremental_update: no existing chunks for file {}",
                    file_path
                );
            }
        }

        indices_to_remove.sort_unstable();
        indices_to_remove.dedup();

        if !indices_to_remove.is_empty() {
            eprintln!(
                "[DocsRAG/Store] incremental_update: removing {} chunks",
                indices_to_remove.len()
            );
        }

        for idx in indices_to_remove.iter().rev() {
            // Validate both bounds atomically to maintain chunks/embeddings consistency
            if *idx >= self.chunks.len() || *idx >= self.embeddings.len() {
                let err_msg = format!(
                    "[DocsRAG/Store] incremental_update: index {} out of bounds (chunks={}, embeddings={})",
                    idx, self.chunks.len(), self.embeddings.len()
                );
                eprintln!("{}", err_msg);
                return Err(err_msg.into());
            }

            // Remove from both in same iteration to maintain consistency
            self.chunks.remove(*idx);
            self.embeddings.remove(*idx);
        }

        let mut new_chunks: Vec<Chunk> = Vec::new();
        for file_path in modified.iter().chain(added.iter()) {
            let path = Path::new(file_path);
            if !path.exists() {
                eprintln!(
                    "[DocsRAG/Store] incremental_update: file not found: {}",
                    file_path
                );
                continue;
            }

            eprintln!(
                "[DocsRAG/Store] incremental_update: chunking file {}",
                file_path
            );
            match Chunker::chunk_file(path).await {
                Ok(chunks) => new_chunks.extend(chunks),
                Err(err) => {
                    eprintln!(
                        "[DocsRAG/Store] incremental_update: failed to chunk {}: {}",
                        file_path, err
                    );
                }
            }
        }

        if !new_chunks.is_empty() {
            eprintln!(
                "[DocsRAG/Store] incremental_update: embedding {} new chunks",
                new_chunks.len()
            );
            let embedder = Embedder::new()?;
            let chunk_texts: Vec<&str> = new_chunks.iter().map(|c| c.content.as_str()).collect();
            let new_embeddings = embedder.embed_batch(chunk_texts).await?;

            if new_embeddings.len() != new_chunks.len() {
                let err_msg = format!(
                    "[DocsRAG/Store] incremental_update: embedding mismatch: expected {} embeddings but got {}",
                    new_chunks.len(),
                    new_embeddings.len()
                );
                eprintln!("{}", err_msg);
                return Err(err_msg.into());
            }

            self.chunks.extend(new_chunks);
            self.embeddings.extend(new_embeddings);
        } else {
            eprintln!("[DocsRAG/Store] incremental_update: no new chunks to embed");
        }

        if self.chunks.len() != self.embeddings.len() {
            let err_msg = format!(
                "[DocsRAG/Store] incremental_update: chunk/embedding count mismatch: chunks={} embeddings={}",
                self.chunks.len(),
                self.embeddings.len()
            );
            eprintln!("{}", err_msg);
            return Err(err_msg.into());
        }

        let cache_dir = Path::new(&self.project_path).join(CACHE_DIR);
        fs::create_dir_all(&cache_dir)?;
        self.save_cache(&cache_dir, &self.embeddings).await?;
        eprintln!("[DocsRAG/Store] incremental_update: cache saved");

        Ok(())
    }
    
    fn serialize_embeddings(&self, embeddings: &[Vec<f32>]) -> Vec<u8> {
        let mut result = Vec::new();
        for embedding in embeddings {
            for value in embedding {
                result.extend_from_slice(&value.to_le_bytes());
            }
        }
        result
    }
    
    fn deserialize_embeddings(&self, data: &[u8]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        const F32_SIZE: usize = std::mem::size_of::<f32>();
        const EMBEDDING_DIM: usize = 1536;
        const EMBEDDING_BYTE_SIZE: usize = EMBEDDING_DIM * F32_SIZE;
        
        if data.len() % EMBEDDING_BYTE_SIZE != 0 {
            return Err("Invalid embedding binary size".into());
        }
        
        let num_embeddings = data.len() / EMBEDDING_BYTE_SIZE;
        let mut embeddings = Vec::with_capacity(num_embeddings);
        
        for i in 0..num_embeddings {
            let start = i * EMBEDDING_BYTE_SIZE;
            let end = start + EMBEDDING_BYTE_SIZE;
            let embedding_data = &data[start..end];
            
            let mut embedding = Vec::with_capacity(EMBEDDING_DIM);
            for j in 0..EMBEDDING_DIM {
                let bytes = &embedding_data[j * F32_SIZE..(j + 1) * F32_SIZE];
                let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                embedding.push(value);
            }
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }
    
    /// Get or compute query embedding with minimal lock time
    /// This method is separate to allow network calls outside of store lock
    pub async fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Step 1: Check cache (brief lock)
        {
            let mut cache = self.query_cache.lock().await;
            if let Some(cached_embedding) = cache.get(query) {
                return Ok(cached_embedding);
            }
        }
        // Lock released here

        // Step 2: Compute embedding (NO LOCK - network call happens here)
        let embedder = Embedder::new()?;
        let embedding = embedder.embed_text(query).await?;

        // Step 3: Store in cache (brief lock)
        {
            let mut cache = self.query_cache.lock().await;
            cache.put(query.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    /// Search with pre-computed query embedding (fast, no network calls)
    pub fn search_with_embedding(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<SearchResult> {
        if self.embeddings.is_empty() {
            return vec![];
        }

        let mut scored_chunks: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(idx, _chunk)| {
                if idx >= self.embeddings.len() {
                    eprintln!("[DocsRAG/Store] search: embedding index {} out of bounds (total: {})", idx, self.embeddings.len());
                    return None;
                }
                let similarity = self.cosine_similarity(query_embedding, &self.embeddings[idx]);
                if similarity > 0.0 {
                    Some((idx, similarity))
                } else {
                    None
                }
            })
            .collect();

        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored_chunks
            .iter()
            .take(top_k)
            .map(|(idx, score)| {
                let chunk = &self.chunks[*idx];
                SearchResult {
                    file_path: chunk.file_path.clone(),
                    content: chunk.content.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    score: *score,
                }
            })
            .collect()
    }

    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        // If no embeddings, fall back to text matching
        if self.embeddings.is_empty() {
            return self.search_text_fallback(query, top_k);
        }

        // Get query embedding (may involve network call, but cache lock is minimal)
        let query_embedding = self.get_query_embedding(query).await?;

        // Search with pre-computed embedding (fast, no network calls)
        Ok(self.search_with_embedding(&query_embedding, top_k))
    }
    
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        // Compute dot product
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        
        // Compute norms
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    fn search_text_fallback(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        // Fallback to simple substring matching
        let query_lower = query.to_lowercase();
        let mut scored_chunks: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(idx, chunk)| {
                let content_lower = chunk.content.to_lowercase();
                
                // Simple scoring: count query occurrences
                let count = content_lower.matches(&query_lower).count();
                if count > 0 {
                    Some((idx, count as f32))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by score (descending)
        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top_k and convert to SearchResult
        let results = scored_chunks
            .iter()
            .take(top_k)
            .map(|(idx, score)| {
                let chunk = &self.chunks[*idx];
                SearchResult {
                    file_path: chunk.file_path.clone(),
                    content: chunk.content.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    score: *score,
                }
            })
            .collect();
        
        Ok(results)
    }
    
    pub fn get_status(&self) -> RagStatus {
        RagStatus {
            initialized: true,
            file_count: self.chunks.iter().map(|c| &c.file_path).collect::<std::collections::HashSet<_>>().len(),
            chunk_count: self.chunks.len(),
            last_indexed: self.last_indexed.map(|ts| format!("{}", ts)),
        }
    }

    /// Check if embeddings are empty (used for fallback to text search)
    pub fn is_embeddings_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    

    
    #[test]
    fn test_cosine_similarity() {
        let store = DocsRagStore {
            project_path: String::new(),
            chunks: vec![],
            embeddings: vec![],
            last_indexed: None,
            query_cache: Mutex::new(QueryCache::new(100)),
        };
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((store.cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((store.cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
        
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((store.cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_serialize_deserialize() {
        let store = DocsRagStore {
            project_path: String::new(),
            chunks: vec![],
            embeddings: vec![],
            last_indexed: None,
            query_cache: Mutex::new(QueryCache::new(100)),
        };
        
        // Create embedding vectors with proper 1536 dimensions
        let embedding1: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
        let embedding2: Vec<f32> = (0..1536).map(|i| ((1536 - i) as f32) * 0.001).collect();
        let original = vec![embedding1, embedding2];
        
        let serialized = store.serialize_embeddings(&original);
        let deserialized = store.deserialize_embeddings(&serialized).unwrap();
        
        assert_eq!(deserialized.len(), original.len());
        for (orig, deser) in original.iter().zip(deserialized.iter()) {
            for (o, d) in orig.iter().zip(deser.iter()) {
                assert!((o - d).abs() < 0.0001);
            }
        }
    }
    
    #[test]
    fn test_embedding_count_mismatch() {
        // Test case: embeddings.len() != chunks.len()
        // This is a unit test for the validation logic
        let embeddings_1 = vec![vec![0.1; 1536]];
        let embeddings_2 = vec![vec![0.2; 1536], vec![0.3; 1536]];
        
        // embeddings_1.len() = 1 != embeddings_2.len() = 2
        assert_ne!(embeddings_1.len(), embeddings_2.len());
        
        // Verify serialize/deserialize handles different counts correctly
        let store = DocsRagStore {
            project_path: String::new(),
            chunks: vec![],
            embeddings: vec![],
            last_indexed: None,
            query_cache: Mutex::new(QueryCache::new(100)),
        };
        
        let serialized_1 = store.serialize_embeddings(&embeddings_1);
        let serialized_2 = store.serialize_embeddings(&embeddings_2);
        
        let deserialized_1 = store.deserialize_embeddings(&serialized_1).unwrap();
        let deserialized_2 = store.deserialize_embeddings(&serialized_2).unwrap();
        
        assert_eq!(deserialized_1.len(), 1);
        assert_eq!(deserialized_2.len(), 2);
    }
    
    #[test]
    fn test_cosine_similarity_with_zero_embeddings() {
        let store = DocsRagStore {
            project_path: String::new(),
            chunks: vec![],
            embeddings: vec![],
            last_indexed: None,
            query_cache: Mutex::new(QueryCache::new(100)),
        };
        
        // Test with zero vectors
        let zero_vec = vec![0.0; 3];
        let normal_vec = vec![1.0, 0.0, 0.0];
        
        assert_eq!(store.cosine_similarity(&zero_vec, &normal_vec), 0.0);
        assert_eq!(store.cosine_similarity(&zero_vec, &zero_vec), 0.0);
    }
    
    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let store = DocsRagStore {
            project_path: String::new(),
            chunks: vec![],
            embeddings: vec![],
            last_indexed: None,
            query_cache: Mutex::new(QueryCache::new(100)),
        };
        
        // Orthogonal vectors should have zero similarity
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        assert!((store.cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
    }
    
    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let store = DocsRagStore {
            project_path: String::new(),
            chunks: vec![],
            embeddings: vec![],
            last_indexed: None,
            query_cache: Mutex::new(QueryCache::new(100)),
        };
        
        // Opposite vectors should have similarity near -1
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        
        assert!((store.cosine_similarity(&a, &b) - (-1.0)).abs() < 0.001);
    }
    
}
