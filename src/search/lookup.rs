//! Lookup functions for exact symbol resolution.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::{
    ChunkType, Confidence, Definition, Index,
    LookupOutput, MatchType,
};

use super::semantic::make_preview;

/// Parse a potentially qualified name like "Class.Method" into (context, member).
fn parse_qualified_name(name: &str) -> (Option<&str>, &str) {
    if let Some(dot_pos) = name.rfind('.') {
        let context = &name[..dot_pos];
        let member = &name[dot_pos + 1..];
        if context.is_empty() || member.is_empty() {
            return (None, name);
        }
        (Some(context), member)
    } else {
        (None, name)
    }
}

/// Convert a Chunk (with indices) to a Definition (with resolved strings).
fn chunk_to_definition(index: &Index, chunk_idx: u32) -> Option<Definition> {
    let chunk = index.chunks.get(chunk_idx as usize)?;
    let file = index.strings.get(chunk.file_idx)?;
    let content = index.strings.get(chunk.content_idx).unwrap_or("");
    let signature = chunk.signature_idx.and_then(|i| index.strings.get(i).map(|s| s.to_string()));
    let context = chunk.context_idx.and_then(|i| index.strings.get(i).map(|s| s.to_string()));

    Some(Definition {
        chunk_type: chunk.chunk_type.clone(),
        kind: chunk.kind.clone(),
        file: PathBuf::from(file),
        line_start: chunk.line_start as usize,
        line_end: chunk.line_end as usize,
        signature,
        context,
        preview: make_preview(content),
    })
}

/// Look up a symbol or document by exact name.
pub fn lookup(
    index: &Index,
    name_to_chunks: &HashMap<u32, Vec<u32>>,
    name: &str,
    filter: Option<ChunkType>,
) -> LookupOutput {
    let (qualified_context, member_name) = parse_qualified_name(name);

    // Find name in string table
    let name_idx = match index.strings.find(member_name) {
        Some(idx) => idx,
        None => return LookupOutput {
            name: name.to_string(),
            match_type: MatchType::NameOnly,
            confidence: Confidence::Low,
            definitions: vec![],
            suggestions: vec!["Symbol not found".to_string()],
        },
    };

    // Get chunk indices for this name
    let chunk_indices = name_to_chunks.get(&name_idx).cloned().unwrap_or_default();

    // Filter chunks by type and context
    let matching_indices: Vec<u32> = chunk_indices.into_iter()
        .filter(|&idx| {
            if let Some(chunk) = index.chunks.get(idx as usize) {
                // Apply type filter
                if let Some(ref filter_type) = filter {
                    if chunk.chunk_type != *filter_type {
                        return false;
                    }
                }
                // Apply context filter if qualified name
                if let Some(ctx) = qualified_context {
                    match chunk.context_idx {
                        Some(ctx_idx) => index.strings.get(ctx_idx) == Some(ctx),
                        None => false,
                    }
                } else {
                    true
                }
            } else {
                false
            }
        })
        .collect();

    let match_count = matching_indices.len();

    // Convert to definitions
    let definitions: Vec<Definition> = matching_indices
        .iter()
        .filter_map(|&idx| chunk_to_definition(index, idx))
        .collect();

    // Determine confidence and suggestions
    let (confidence, match_type, suggestions) = match match_count {
        0 => (
            Confidence::Low,
            MatchType::NameOnly,
            vec!["Symbol not found".to_string()],
        ),
        1 => (Confidence::High, MatchType::Exact, vec![]),
        n => (
            Confidence::Low,
            MatchType::NameOnly,
            vec![format!(
                "{} homonyms found, use Class.Method to qualify",
                n
            )],
        ),
    };

    LookupOutput {
        name: name.to_string(),
        match_type,
        confidence,
        definitions,
        suggestions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, ChunkKind, StringTable, Visibility};

    fn create_test_index() -> (Index, HashMap<u32, Vec<u32>>) {
        let mut strings = StringTable::new();
        let root_idx = strings.intern("/test");
        let model_idx = strings.intern("test-model");

        let mut index = Index {
            version: 2,
            last_indexed: "2024-01-01".to_string(),
            root_idx,
            embedding_model_idx: model_idx,
            strings,
            chunks: Vec::new(),
            files: Vec::new(),
        };

        // Add test chunks
        let name_idx = index.strings.intern("test_function");
        let file_idx = index.strings.intern("/test/src/main.rs");
        let content_idx = index.strings.intern("fn test_function() {}");

        index.chunks.push(Chunk {
            chunk_type: ChunkType::Code,
            name_idx,
            kind: ChunkKind::Function,
            file_idx,
            line_start: 10,
            line_end: 20,
            content_idx,
            context_idx: None,
            signature_idx: None,
            doc_summary_idx: None,
            visibility: Visibility::Private,
        });

        // Build name_to_chunks
        let mut name_to_chunks = HashMap::new();
        name_to_chunks.insert(name_idx, vec![0]);

        (index, name_to_chunks)
    }

    #[test]
    fn test_parse_qualified_name_with_dot() {
        let (ctx, member) = parse_qualified_name("PlayerController.HandleMovement");
        assert_eq!(ctx, Some("PlayerController"));
        assert_eq!(member, "HandleMovement");
    }

    #[test]
    fn test_parse_qualified_name_without_dot() {
        let (ctx, member) = parse_qualified_name("my_function");
        assert_eq!(ctx, None);
        assert_eq!(member, "my_function");
    }

    #[test]
    fn test_lookup_found() {
        let (index, name_to_chunks) = create_test_index();

        let result = lookup(&index, &name_to_chunks, "test_function", None);

        assert_eq!(result.confidence, Confidence::High);
        assert_eq!(result.definitions.len(), 1);
        assert!(result.suggestions.is_empty());
    }

    #[test]
    fn test_lookup_not_found() {
        let (index, name_to_chunks) = create_test_index();

        let result = lookup(&index, &name_to_chunks, "nonexistent", None);

        assert_eq!(result.confidence, Confidence::Low);
        assert!(result.definitions.is_empty());
        assert_eq!(result.suggestions[0], "Symbol not found");
    }

}
