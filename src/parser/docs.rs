//! Documentation parser for markdown and text files.

use std::path::Path;
use anyhow::Result;
use crate::{ChunkType, ChunkKind, RawChunk};

pub struct DocsParser;

impl DocsParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext {
            "md" | "mdx" => self.parse_markdown(path, content),
            _ => self.parse_text(path, content),
        }
    }

    fn parse_markdown(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let bytes = content.as_bytes();

        // Add document chunk for entire file
        chunks.push(RawChunk {
            chunk_type: ChunkType::Doc,
            name: path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            kind: ChunkKind::Document,
            line_start: 1,
            line_end: lines.len().max(1) as u32,
            byte_start: 0,
            byte_end: bytes.len(),
            content: content.to_string(),
            context: None,
            signature: None,
        });

        // Find sections by headers
        let mut section_start: Option<(usize, String, usize)> = None; // (line, name, byte_start)
        let mut in_code_block = false;
        let mut current_byte = 0usize;

        for (i, line) in lines.iter().enumerate() {
            let line_byte_start = current_byte;

            // Track fenced code blocks
            let trimmed = line.trim_start();
            if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                in_code_block = !in_code_block;
                current_byte += line.len() + 1; // +1 for newline
                continue;
            }

            // Only detect headers outside of code blocks
            if !in_code_block && line.starts_with('#') {
                // End previous section
                if let Some((start_line, name, start_byte)) = section_start.take() {
                    let section_content = lines[start_line..i].join("\n");
                    chunks.push(RawChunk {
                        chunk_type: ChunkType::Doc,
                        name,
                        kind: ChunkKind::Section,
                        line_start: start_line as u32 + 1,
                        line_end: i as u32,
                        byte_start: start_byte,
                        byte_end: line_byte_start,
                        content: section_content,
                        context: None,
                        signature: None,
                                });
                }
                // Start new section
                let name = line.trim_start_matches('#').trim().to_string();
                let name = if name.is_empty() {
                    "Untitled Section".to_string()
                } else {
                    name
                };
                section_start = Some((i, name, line_byte_start));
            }

            current_byte += line.len() + 1; // +1 for newline
        }

        // Warn if code block was never closed
        if in_code_block {
            eprintln!(
                "[scs] Warning: Unclosed code block in {}. Some headers may have been missed.",
                path.display()
            );
        }

        // Handle last section
        if let Some((start_line, name, start_byte)) = section_start {
            let section_content = lines[start_line..].join("\n");
            chunks.push(RawChunk {
                chunk_type: ChunkType::Doc,
                name,
                kind: ChunkKind::Section,
                line_start: start_line as u32 + 1,
                line_end: lines.len() as u32,
                byte_start: start_byte,
                byte_end: bytes.len(),
                content: section_content,
                context: None,
                signature: None,
                });
        }

        Ok(chunks)
    }

    fn parse_text(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        let line_count = content.lines().count().max(1);

        Ok(vec![RawChunk {
            chunk_type: ChunkType::Doc,
            name: path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            kind: ChunkKind::Document,
            line_start: 1,
            line_end: line_count as u32,
            byte_start: 0,
            byte_end: content.len(),
            content: content.to_string(),
            context: None,
            signature: None,
        }])
    }
}

impl Default for DocsParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_markdown_simple() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.md");
        let content = "# Hello\n\nSome content\n\n# World\n\nMore content";

        let chunks = parser.parse(&path, content).unwrap();
        
        // Should have document + 2 sections
        assert!(chunks.len() >= 2);
        
        let hello = chunks.iter().find(|c| c.name == "Hello");
        assert!(hello.is_some());
        assert_eq!(hello.unwrap().kind, ChunkKind::Section);
    }

    #[test]
    fn test_parse_text() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.txt");
        let content = "Just some plain text\nwith multiple lines";

        let chunks = parser.parse(&path, content).unwrap();
        
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Document);
    }
}
