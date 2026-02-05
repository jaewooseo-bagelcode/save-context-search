use std::fs;
use std::path::Path;

use anyhow::Result;
use sha2::{Digest, Sha256};

pub fn get_mtime(path: &Path) -> Result<u64> {
    let metadata = fs::metadata(path)?;
    let modified = metadata.modified()?;
    let duration = modified.duration_since(std::time::UNIX_EPOCH)?;
    Ok(duration.as_secs())
}

pub fn compute_hash(path: &Path) -> Result<String> {
    let content = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}
