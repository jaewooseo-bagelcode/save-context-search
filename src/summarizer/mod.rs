//! LLM-based function summarization module.
//!
//! Uses Gemini 2.5 Flash-Lite via AI Proxy to generate concise function summaries.
//! Implements level-based batch processing for efficient API usage.

pub mod gemini;
pub mod levels;

use serde::{Deserialize, Serialize};

/// Statistics from summarization process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SummaryStats {
    /// Total functions processed
    pub total_functions: usize,
    /// Functions that were cached (skipped)
    pub cached: usize,
    /// Functions that were summarized via API
    pub summarized: usize,
    /// Functions that failed
    pub failed: usize,
    /// Number of API calls made
    pub api_calls: usize,
    /// Total input tokens
    pub input_tokens: usize,
    /// Total output tokens
    pub output_tokens: usize,
    /// Number of levels processed
    pub levels: usize,
}

impl SummaryStats {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Summary cache entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryEntry {
    /// Function name
    pub name: String,
    /// Body hash (for cache invalidation)
    pub body_hash: String,
    /// Context hash (callee summaries hash)
    pub context_hash: String,
    /// Generated summary
    pub summary: String,
}

/// Summary cache stored in .scs/summaries.json
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SummaryCache {
    pub entries: std::collections::HashMap<String, SummaryEntry>,
}

impl SummaryCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load cache from file.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let content = std::fs::read_to_string(path)?;
        let cache: SummaryCache = serde_json::from_str(&content)?;
        Ok(cache)
    }

    /// Save cache to file.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get cached summary if valid.
    pub fn get(&self, name: &str, body_hash: &str, context_hash: &str) -> Option<&str> {
        self.entries.get(name).and_then(|e| {
            if e.body_hash == body_hash && e.context_hash == context_hash {
                Some(e.summary.as_str())
            } else {
                None
            }
        })
    }

    /// Insert or update a summary.
    pub fn insert(&mut self, name: String, body_hash: String, context_hash: String, summary: String) {
        self.entries.insert(
            name.clone(),
            SummaryEntry {
                name,
                body_hash,
                context_hash,
                summary,
            },
        );
    }
}
