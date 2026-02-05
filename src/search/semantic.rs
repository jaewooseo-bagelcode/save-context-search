//! Semantic search using cosine similarity over embeddings.

use simsimd::SpatialSimilarity;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::{ChunkType, Index, MatchType, SearchOutput, SearchResult};

/// Compute cosine similarity between two vectors using simsimd SIMD acceleration.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    if let Some(distance) = <f32 as SpatialSimilarity>::cos(a, b) {
        let similarity = (1.0 - distance) as f32;
        if similarity.is_nan() {
            return 0.0;
        }
        return similarity;
    }

    // Pure Rust fallback
    let mut dot_product = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let similarity = dot_product / (norm_a.sqrt() * norm_b.sqrt());

    if similarity.is_nan() || similarity.is_infinite() {
        return 0.0;
    }

    similarity
}

/// Perform semantic search over chunks using cosine similarity.
pub fn search(
    index: &Index,
    embeddings: &[Vec<f32>],
    query_embedding: &[f32],
    top_k: usize,
    filter: Option<ChunkType>,
) -> SearchOutput {
    if top_k == 0 || index.chunks.is_empty() || embeddings.is_empty() || query_embedding.is_empty() {
        return SearchOutput {
            query: String::new(),
            match_type: MatchType::Semantic,
            results: Vec::new(),
            suggestions: Vec::new(),
        };
    }

    if index.chunks.len() != embeddings.len() {
        eprintln!(
            "[scs] Warning: chunks/embeddings mismatch ({} vs {}). Some results may be missing.",
            index.chunks.len(),
            embeddings.len()
        );
    }

    // Validate embedding dimensions
    if let Some(first_embedding) = embeddings.first() {
        if query_embedding.len() != first_embedding.len() {
            eprintln!(
                "[scs] Error: Query embedding dimension ({}) doesn't match stored embeddings ({}).",
                query_embedding.len(),
                first_embedding.len()
            );
            return SearchOutput {
                query: String::new(),
                match_type: MatchType::Semantic,
                results: Vec::new(),
                suggestions: vec![format!(
                    "Embedding dimension mismatch: query={}, stored={}. Try 'scs reindex'.",
                    query_embedding.len(),
                    first_embedding.len()
                )],
            };
        }
    }

    // Count name occurrences for uniqueness check
    let mut name_counts: HashMap<u32, usize> = HashMap::new();
    let mut scored_chunks: Vec<(usize, f32)> = Vec::new();

    for (idx, chunk) in index.chunks.iter().enumerate() {
        if filter.as_ref().map_or(false, |t| chunk.chunk_type != *t) {
            continue;
        }
        let Some(embedding) = embeddings.get(idx) else { continue };
        let score = cosine_similarity(query_embedding, embedding);
        *name_counts.entry(chunk.name_idx).or_insert(0) += 1;
        scored_chunks.push((idx, score));
    }

    // Sort by score descending
    scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Build results
    let results = scored_chunks
        .iter()
        .take(top_k)
        .filter_map(|(idx, score)| {
            let chunk = index.chunks.get(*idx)?;
            let name = index.strings.get(chunk.name_idx)?;
            let file = index.strings.get(chunk.file_idx)?;
            let content = index.strings.get(chunk.content_idx).unwrap_or("");
            let context = chunk.context_idx.and_then(|i| index.strings.get(i));

            let unique = name_counts
                .get(&chunk.name_idx)
                .map(|count| *count == 1)
                .unwrap_or(true);

            Some(SearchResult {
                chunk_type: chunk.chunk_type.clone(),
                name: name.to_string(),
                kind: chunk.kind.clone(),
                file: PathBuf::from(file),
                line_start: chunk.line_start as usize,
                line_end: chunk.line_end as usize,
                score: *score,
                preview: make_preview(content),
                context: context.map(|s| s.to_string()),
                unique,
            })
        })
        .collect();

    SearchOutput {
        query: String::new(),
        match_type: MatchType::Semantic,
        results,
        suggestions: Vec::new(),
    }
}

/// Create a preview from content by taking first 5 lines.
pub fn make_preview(content: &str) -> String {
    let lines: Vec<&str> = content.lines().take(6).collect();
    let truncated = lines.len() > 5;
    let mut preview = lines.into_iter().take(5).collect::<Vec<_>>().join("\n");
    if truncated {
        preview.push_str("\n...");
    }
    preview
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_make_preview_short() {
        let content = "line1\nline2\nline3";
        let preview = make_preview(content);
        assert_eq!(preview, "line1\nline2\nline3");
    }

    #[test]
    fn test_make_preview_truncated() {
        let content = "line1\nline2\nline3\nline4\nline5\nline6\nline7";
        let preview = make_preview(content);
        assert!(preview.ends_with("..."));
        assert!(preview.contains("line1"));
        assert!(preview.contains("line5"));
        assert!(!preview.contains("line6"));
    }
}
