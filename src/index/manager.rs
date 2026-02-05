//! Index manager with bincode binary format and string interning.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{Context, Result};
use walkdir::WalkDir;

use crate::{ChunkType, Index};
use super::cache::{compute_hash, get_mtime};

/// Index format version
#[allow(dead_code)]
const VERSION: u32 = 2;

/// Embedding model identifier
const EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// Maximum file size to index (1MB)
const MAX_FILE_SIZE: u64 = 1024 * 1024;

/// Embedding dimension for text-embedding-3-small
const EMBEDDING_DIM: usize = 1536;

/// Directories to exclude from scanning
const EXCLUDED_DIRS: &[&str] = &[
    "node_modules",
    "target",
    ".git",
    "dist",
    "build",
    ".scs",
    "__pycache__",
    "vendor",
    ".next",
    ".nuxt",
    "coverage",
    ".venv",
    "venv",
];

/// Code file extensions
const CODE_EXTENSIONS: &[&str] = &["rs", "ts", "tsx", "js", "jsx", "cs", "py"];

/// Documentation file extensions
const DOC_EXTENSIONS: &[&str] = &["md", "mdx", "txt"];

/// IndexManager handles loading, saving, scanning, and locking index operations.
pub struct IndexManager {
    pub root: PathBuf,
    pub index_dir: PathBuf,
    pub index: Index,
    pub embeddings: Vec<Vec<f32>>,
    // Runtime caches (not serialized)
    pub name_to_chunks: HashMap<u32, Vec<u32>>,  // name_idx -> chunk indices
    pub file_to_entry: HashMap<u32, usize>,       // path_idx -> file entry index
}

impl IndexManager {
    /// Load existing index from cache or create a new empty one.
    pub fn load_or_create(root: &Path) -> Result<Self> {
        let root = root.canonicalize().context("Failed to canonicalize root path")?;
        let index_dir = root.join(".scs");

        if !index_dir.exists() {
            fs::create_dir_all(&index_dir).context("Failed to create .scs directory")?;
        }

        let index_bin = index_dir.join("index.bin");
        let embeddings_bin = index_dir.join("embeddings.bin");

        if index_bin.exists() {
            // Load existing index
            let file = File::open(&index_bin).context("Failed to open index.bin")?;
            let reader = BufReader::new(file);
            let mut index: Index = bincode::deserialize_from(reader)
                .context("Failed to deserialize index.bin")?;

            // Rebuild string index for lookups
            index.strings.rebuild_index();

            // Load embeddings
            let embeddings = Self::load_embeddings(&embeddings_bin, index.chunks.len())?;

            let mut manager = Self {
                root,
                index_dir,
                index,
                embeddings,
                name_to_chunks: HashMap::new(),
                file_to_entry: HashMap::new(),
            };
            manager.build_runtime_caches();
            Ok(manager)
        } else {
            // Create new empty index
            let root_str = root.to_string_lossy();
            let index = Index::new(&root_str, EMBEDDING_MODEL);

            Ok(Self {
                root,
                index_dir,
                index,
                embeddings: Vec::new(),
                name_to_chunks: HashMap::new(),
                file_to_entry: HashMap::new(),
            })
        }
    }

    /// Build runtime lookup caches from index data.
    pub fn build_runtime_caches(&mut self) {
        self.name_to_chunks.clear();
        self.file_to_entry.clear();

        // Build name_to_chunks
        for (idx, chunk) in self.index.chunks.iter().enumerate() {
            self.name_to_chunks
                .entry(chunk.name_idx)
                .or_default()
                .push(idx as u32);
        }

        // Build file_to_entry
        for (idx, file) in self.index.files.iter().enumerate() {
            self.file_to_entry.insert(file.path_idx, idx);
        }
    }

    /// Get count of chunks with a given name_idx.
    pub fn get_name_count(&self, name_idx: u32) -> usize {
        self.name_to_chunks.get(&name_idx).map(|v| v.len()).unwrap_or(0)
    }

    /// Get all indexed file paths.
    pub fn get_indexed_paths(&self) -> Vec<PathBuf> {
        self.index.files.iter()
            .filter_map(|f| self.index.strings.get(f.path_idx))
            .map(PathBuf::from)
            .collect()
    }

    /// Check if a file is indexed.
    pub fn has_file(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        if let Some(path_idx) = self.index.strings.find(&path_str) {
            self.file_to_entry.contains_key(&path_idx)
        } else {
            false
        }
    }

    /// Remove files from the index.
    pub fn remove_files(&mut self, paths: &[PathBuf]) {
        let remove_path_indices: Vec<u32> = paths.iter()
            .filter_map(|p| self.index.strings.find(&p.to_string_lossy()))
            .collect();

        if remove_path_indices.is_empty() {
            return;
        }

        // Collect chunk indices to remove
        let mut chunk_indices_to_remove: Vec<usize> = Vec::new();
        let mut file_indices_to_remove: Vec<usize> = Vec::new();

        for path_idx in &remove_path_indices {
            if let Some(&file_idx) = self.file_to_entry.get(path_idx) {
                file_indices_to_remove.push(file_idx);
                let file = &self.index.files[file_idx];
                for chunk_idx in file.chunk_range.0..file.chunk_range.1 {
                    chunk_indices_to_remove.push(chunk_idx as usize);
                }
            }
        }

        // Sort in reverse order for safe removal
        chunk_indices_to_remove.sort_by(|a, b| b.cmp(a));
        file_indices_to_remove.sort_by(|a, b| b.cmp(a));

        // Remove chunks and embeddings
        for idx in &chunk_indices_to_remove {
            if *idx < self.index.chunks.len() {
                self.index.chunks.remove(*idx);
            }
            if *idx < self.embeddings.len() {
                self.embeddings.remove(*idx);
            }
        }

        // Remove file entries
        for idx in &file_indices_to_remove {
            if *idx < self.index.files.len() {
                self.index.files.remove(*idx);
            }
        }

        // Rebuild caches
        self.build_runtime_caches();
    }

    /// Save all cache files to the .scs/ directory atomically.
    pub fn save(&self) -> Result<()> {
        let staging_dir = self.index_dir.join(".staging");
        let backup_dir = self.index_dir.join(".backup");

        // Clean up leftover directories
        if staging_dir.exists() {
            fs::remove_dir_all(&staging_dir).ok();
        }
        if backup_dir.exists() {
            fs::remove_dir_all(&backup_dir).ok();
        }

        fs::create_dir_all(&staging_dir).context("Failed to create staging directory")?;

        // Write index.bin (bincode)
        {
            let path = staging_dir.join("index.bin");
            let file = File::create(&path).context("Failed to create index.bin")?;
            let mut writer = BufWriter::new(&file);
            bincode::serialize_into(&mut writer, &self.index)
                .context("Failed to serialize index")?;
            writer.flush().context("Failed to flush index.bin")?;
            file.sync_all().context("Failed to sync index.bin")?;
        }

        // Write embeddings.bin
        {
            let path = staging_dir.join("embeddings.bin");
            Self::save_embeddings_with_sync(&path, &self.embeddings)?;
        }

        // Atomic swap
        let files = ["index.bin", "embeddings.bin"];

        fs::create_dir_all(&backup_dir).context("Failed to create backup directory")?;

        // Move current files to backup
        for file_name in &files {
            let current = self.index_dir.join(file_name);
            let backup = backup_dir.join(file_name);
            if current.exists() {
                fs::rename(&current, &backup)
                    .with_context(|| format!("Failed to backup {}", file_name))?;
            }
        }

        // Move staging files to current
        for file_name in &files {
            let staged = staging_dir.join(file_name);
            let current = self.index_dir.join(file_name);
            if let Err(e) = fs::rename(&staged, &current) {
                // Rollback
                eprintln!("[scs] Error committing {}, attempting rollback: {}", file_name, e);
                for restore_name in &files {
                    let backup = backup_dir.join(restore_name);
                    let restore_current = self.index_dir.join(restore_name);
                    if backup.exists() && !restore_current.exists() {
                        if let Err(re) = fs::rename(&backup, &restore_current) {
                            eprintln!("[scs] Rollback failed for {}: {}", restore_name, re);
                        }
                    }
                }
                fs::remove_dir_all(&staging_dir).ok();
                return Err(e).with_context(|| format!("Failed to commit {}", file_name));
            }
        }

        // Cleanup
        fs::remove_dir_all(&staging_dir).ok();
        fs::remove_dir_all(&backup_dir).ok();

        Ok(())
    }

    /// Scan the project directory for files to index.
    pub fn scan_files(&self) -> Result<Vec<(PathBuf, ChunkType)>> {
        let mut files = Vec::new();
        let root = &self.root;

        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !Self::is_excluded_relative(e.path(), root))
        {
            let entry = entry.context("Failed to read directory entry")?;

            if entry.file_type().is_dir() {
                continue;
            }

            let path = entry.path();

            if let Ok(metadata) = fs::metadata(path) {
                if metadata.len() > MAX_FILE_SIZE {
                    continue;
                }
            }

            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                let ext_lower = ext.to_lowercase();
                if CODE_EXTENSIONS.contains(&ext_lower.as_str()) {
                    files.push((path.to_path_buf(), ChunkType::Code));
                } else if DOC_EXTENSIONS.contains(&ext_lower.as_str()) {
                    files.push((path.to_path_buf(), ChunkType::Doc));
                }
            }
        }

        Ok(files)
    }

    /// Check if the index is currently locked by another process.
    pub fn check_lock(&self) -> Option<u32> {
        let lock_path = self.index_dir.join("lock");

        if lock_path.exists() {
            if let Ok(content) = fs::read_to_string(&lock_path) {
                if let Ok(pid) = content.trim().parse::<u32>() {
                    if Self::is_process_running(pid) {
                        return Some(pid);
                    }
                }
            }
        }
        None
    }

    /// Acquire an exclusive lock for indexing operations.
    pub fn acquire_lock(&self) -> Result<()> {
        let lock_path = self.index_dir.join("lock");
        let current_pid = process::id();

        if let Some(pid) = self.check_lock() {
            anyhow::bail!(
                "Index is locked by another process (PID: {}). \
                 If this is stale, remove {:?} manually.",
                pid,
                lock_path
            );
        }

        if lock_path.exists() {
            fs::remove_file(&lock_path).context("Failed to remove stale lock file")?;
        }

        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                file.write_all(current_pid.to_string().as_bytes())
                    .context("Failed to write PID to lock file")?;
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if let Some(pid) = self.check_lock() {
                    anyhow::bail!(
                        "Index is locked by another process (PID: {}). Race condition detected.",
                        pid
                    );
                } else {
                    fs::remove_file(&lock_path).context("Failed to remove stale lock file")?;
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&lock_path)
                        .context("Failed to create lock file after retry")?;
                    file.write_all(current_pid.to_string().as_bytes())
                        .context("Failed to write PID to lock file")?;
                    Ok(())
                }
            }
            Err(e) => Err(e).context("Failed to create lock file"),
        }
    }

    /// Release the indexing lock.
    pub fn release_lock(&self) -> Result<()> {
        let lock_path = self.index_dir.join("lock");

        if lock_path.exists() {
            fs::remove_file(&lock_path).context("Failed to remove lock file")?;
        }

        Ok(())
    }

    /// Mark the index as dirty (needs refresh after current operation completes).
    /// Used when a refresh is requested while another process holds the lock.
    pub fn mark_dirty(&self) -> Result<()> {
        let dirty_path = self.index_dir.join("dirty");
        File::create(&dirty_path).context("Failed to create dirty flag")?;
        Ok(())
    }

    /// Check if dirty flag exists and clear it.
    /// Returns true if the flag was set (index needs another refresh).
    pub fn check_and_clear_dirty(&self) -> bool {
        let dirty_path = self.index_dir.join("dirty");
        if dirty_path.exists() {
            let _ = fs::remove_file(&dirty_path);
            true
        } else {
            false
        }
    }

    /// Check if a file has changed since last index.
    pub fn file_needs_update(&self, path: &Path) -> Result<bool> {
        let path_str = path.to_string_lossy();

        if let Some(path_idx) = self.index.strings.find(&path_str) {
            if let Some(&file_idx) = self.file_to_entry.get(&path_idx) {
                let file_entry = &self.index.files[file_idx];
                let current_mtime = get_mtime(path)?;
                if current_mtime == file_entry.mtime {
                    return Ok(false);
                }
                // mtime changed, verify hash
                let current_hash = compute_hash(path)?;
                let stored_hash = self.index.strings.get(file_entry.hash_idx).unwrap_or("");
                return Ok(current_hash != stored_hash);
            }
        }
        // New file
        Ok(true)
    }

    fn is_excluded_relative(path: &Path, root: &Path) -> bool {
        let relative = path.strip_prefix(root).unwrap_or(path);
        relative.components().any(|c| {
            matches!(c, std::path::Component::Normal(name)
                if name.to_str().map(|s| EXCLUDED_DIRS.contains(&s)).unwrap_or(false))
        })
    }

    #[cfg(unix)]
    fn is_process_running(pid: u32) -> bool {
        use std::process::Command;
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[cfg(windows)]
    fn is_process_running(pid: u32) -> bool {
        use std::process::Command;
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {}", pid), "/NH", "/FO", "CSV"])
            .output()
            .map(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let pid_str = format!("\"{}\"", pid);
                stdout.lines().any(|line| {
                    if line.starts_with("INFO:") {
                        return false;
                    }
                    let fields: Vec<&str> = line.split(',').collect();
                    fields.get(1).map(|&p| p == pid_str).unwrap_or(false)
                })
            })
            .unwrap_or(false)
    }

    /// Load embeddings from binary file.
    fn load_embeddings(path: &Path, chunk_count: usize) -> Result<Vec<Vec<f32>>> {
        if chunk_count == 0 {
            return Ok(Vec::new());
        }

        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut file = File::open(path).context("Failed to open embeddings.bin")?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).context("Failed to read embeddings.bin")?;

        if buffer.is_empty() {
            return Ok(Vec::new());
        }

        let expected_size = chunk_count * EMBEDDING_DIM * std::mem::size_of::<f32>();
        if buffer.len() != expected_size {
            // Size mismatch - could be partial embeddings, load what we can
            let actual_count = buffer.len() / (EMBEDDING_DIM * std::mem::size_of::<f32>());
            if actual_count == 0 {
                return Ok(Vec::new());
            }
            eprintln!(
                "[scs] Warning: embeddings.bin size mismatch, loading {} of {} embeddings",
                actual_count, chunk_count
            );
            return Self::parse_embeddings(&buffer, actual_count);
        }

        Self::parse_embeddings(&buffer, chunk_count)
    }

    fn parse_embeddings(buffer: &[u8], count: usize) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * EMBEDDING_DIM * std::mem::size_of::<f32>();
            let mut embedding = Vec::with_capacity(EMBEDDING_DIM);
            for j in 0..EMBEDDING_DIM {
                let offset = start + j * std::mem::size_of::<f32>();
                let bytes = [
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                ];
                embedding.push(f32::from_le_bytes(bytes));
            }
            embeddings.push(embedding);
        }
        Ok(embeddings)
    }

    /// Save embeddings to binary file with fsync.
    fn save_embeddings_with_sync(path: &Path, embeddings: &[Vec<f32>]) -> Result<()> {
        let file = File::create(path).context("Failed to create embeddings.bin")?;
        let mut writer = BufWriter::new(&file);

        for embedding in embeddings {
            for value in embedding {
                writer
                    .write_all(&value.to_le_bytes())
                    .context("Failed to write embedding value")?;
            }
        }

        writer.flush().context("Failed to flush embeddings.bin")?;
        file.sync_all().context("Failed to sync embeddings.bin")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_or_create_new() {
        let temp_dir = TempDir::new().unwrap();
        let manager = IndexManager::load_or_create(temp_dir.path()).unwrap();

        assert!(manager.index_dir.exists());
        assert_eq!(manager.index.version, VERSION);
        assert!(manager.index.chunks.is_empty());
        assert!(manager.index.files.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();

        {
            let mut manager = IndexManager::load_or_create(temp_dir.path()).unwrap();
            // Add some data
            manager.index.strings.intern("test_string");
            manager.save().unwrap();
        }

        {
            let manager = IndexManager::load_or_create(temp_dir.path()).unwrap();
            assert_eq!(manager.index.strings.len(), 3); // root + embedding_model + test_string
        }
    }

    #[test]
    fn test_scan_files() {
        let temp_dir = TempDir::new().unwrap();

        fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();
        fs::write(temp_dir.path().join("README.md"), "# Test").unwrap();
        fs::write(temp_dir.path().join("data.json"), "{}").unwrap();

        let manager = IndexManager::load_or_create(temp_dir.path()).unwrap();
        let files = manager.scan_files().unwrap();

        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|(p, t)| p.ends_with("test.rs") && *t == ChunkType::Code));
        assert!(files.iter().any(|(p, t)| p.ends_with("README.md") && *t == ChunkType::Doc));
    }

    #[test]
    fn test_excluded_dirs() {
        let temp_dir = TempDir::new().unwrap();

        let node_modules = temp_dir.path().join("node_modules");
        fs::create_dir(&node_modules).unwrap();
        fs::write(node_modules.join("package.js"), "module.exports = {}").unwrap();

        fs::write(temp_dir.path().join("main.rs"), "fn main() {}").unwrap();

        let manager = IndexManager::load_or_create(temp_dir.path()).unwrap();
        let files = manager.scan_files().unwrap();

        assert_eq!(files.len(), 1);
        assert!(files[0].0.ends_with("main.rs"));
    }

    #[test]
    fn test_lock() {
        let temp_dir = TempDir::new().unwrap();
        let manager = IndexManager::load_or_create(temp_dir.path()).unwrap();

        manager.acquire_lock().unwrap();
        assert!(manager.index_dir.join("lock").exists());

        manager.release_lock().unwrap();
        assert!(!manager.index_dir.join("lock").exists());
    }

    #[test]
    fn test_embeddings_round_trip() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("embeddings.bin");

        let original = vec![
            vec![0.1f32; EMBEDDING_DIM],
            vec![0.2f32; EMBEDDING_DIM],
        ];

        IndexManager::save_embeddings_with_sync(&path, &original).unwrap();
        let loaded = IndexManager::load_embeddings(&path, 2).unwrap();

        assert_eq!(loaded.len(), 2);
        for (orig, load) in original.iter().zip(loaded.iter()) {
            for (o, l) in orig.iter().zip(load.iter()) {
                assert!((o - l).abs() < 0.0001);
            }
        }
    }
}
