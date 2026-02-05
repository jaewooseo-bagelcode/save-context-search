//! Documentation parser for markdown and text files.

use std::path::Path;

use anyhow::Result;

use crate::{ChunkKind, ChunkType, RawChunk, Visibility};

const CHUNK_SIZE: usize = 500;
const CHUNK_OVERLAP: usize = 100;

pub struct DocsParser;

impl DocsParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "md" | "mdx" => self.parse_markdown(path, content),
            "json" => self.parse_json(path, content),
            "yaml" | "yml" => self.parse_yaml(path, content),
            "toml" => self.parse_toml(path, content),
            _ => self.parse_text_sliding_window(path, content),
        }
    }

    /// Extract file name from path, defaulting to "unknown".
    fn file_name(path: &Path) -> String {
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Create a section chunk with common defaults.
    fn section_chunk(
        name: String,
        content: String,
        line_start: u32,
        line_end: u32,
        byte_start: usize,
        byte_end: usize,
        context: Option<String>,
    ) -> RawChunk {
        RawChunk {
            chunk_type: ChunkType::Doc,
            name,
            kind: ChunkKind::Section,
            line_start,
            line_end,
            byte_start,
            byte_end,
            content,
            context,
            signature: None,
            doc_summary: None,
            visibility: Visibility::Private,
        }
    }

    fn parse_markdown(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let file_name = Self::file_name(path);

        // Add document chunk for entire file
        chunks.push(RawChunk {
            chunk_type: ChunkType::Doc,
            name: file_name,
            kind: ChunkKind::Document,
            line_start: 1,
            line_end: lines.len().max(1) as u32,
            byte_start: 0,
            byte_end: content.len(),
            content: content.to_string(),
            context: None,
            signature: None,
            doc_summary: None,
            visibility: Visibility::Private,
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
                    chunks.push(Self::section_chunk(
                        name,
                        section_content,
                        start_line as u32 + 1,
                        i as u32,
                        start_byte,
                        line_byte_start,
                        None,
                    ));
                }
                // Start new section
                let name = line.trim_start_matches('#').trim();
                let name = if name.is_empty() {
                    "Untitled Section".to_string()
                } else {
                    name.to_string()
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
            chunks.push(Self::section_chunk(
                name,
                section_content,
                start_line as u32 + 1,
                lines.len() as u32,
                start_byte,
                content.len(),
                None,
            ));
        }

        Ok(chunks)
    }

    fn parse_json(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        if content.trim().is_empty() {
            return self.parse_text_sliding_window(path, content);
        }

        let parsed = serde_json::from_str::<serde_json::Value>(content);
        let map = match parsed {
            Ok(serde_json::Value::Object(map)) if !map.is_empty() => map,
            _ => return self.parse_text_sliding_window(path, content),
        };

        let file_name = Self::file_name(path);
        let chunks = map
            .iter()
            .map(|(key, value)| {
                let pretty =
                    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
                let chunk_content = format!("{}:\n{}", key, pretty);
                let line_count = chunk_content.lines().count().max(1);
                Self::section_chunk(
                    key.clone(),
                    chunk_content.clone(),
                    1,
                    line_count as u32,
                    0,
                    chunk_content.len(),
                    Some(file_name.clone()),
                )
            })
            .collect();

        Ok(chunks)
    }

    fn parse_yaml(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        if content.trim().is_empty() {
            return self.parse_text_sliding_window(path, content);
        }

        let file_name = Self::file_name(path);
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_name = String::new();
        let mut start_line = 0usize;
        let mut start_byte = 0usize;
        let mut current_byte = 0usize;
        let mut found_key = false;

        for (idx, line) in lines.iter().enumerate() {
            let is_top_level = !line.is_empty()
                && !line.starts_with(' ')
                && !line.starts_with('\t')
                && !line.starts_with('#')
                && !line.starts_with("---")
                && !line.starts_with("...");

            if is_top_level && !current_name.is_empty() {
                chunks.push(Self::section_chunk(
                    current_name.clone(),
                    current_chunk.trim().to_string(),
                    start_line as u32 + 1,
                    idx as u32,
                    start_byte,
                    current_byte,
                    Some(file_name.clone()),
                ));
                current_chunk.clear();
                start_line = idx;
                start_byte = current_byte;
            }

            if is_top_level {
                let name = line.split(':').next().unwrap_or("").trim();
                current_name = if name.is_empty() {
                    file_name.clone()
                } else {
                    name.to_string()
                };
                found_key = true;
            }

            current_chunk.push_str(line);
            current_chunk.push('\n');
            current_byte += line.len() + 1;
        }

        if !current_chunk.trim().is_empty() && found_key {
            let name = if current_name.is_empty() {
                file_name.clone()
            } else {
                current_name
            };
            chunks.push(Self::section_chunk(
                name,
                current_chunk.trim().to_string(),
                start_line as u32 + 1,
                lines.len().max(1) as u32,
                start_byte,
                content.len(),
                Some(file_name.clone()),
            ));
        }

        if chunks.is_empty() {
            return self.parse_text_sliding_window(path, content);
        }

        Ok(chunks)
    }

    fn parse_toml(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        if content.trim().is_empty() {
            return self.parse_text_sliding_window(path, content);
        }

        let file_name = Self::file_name(path);
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_name = String::new();
        let mut start_line = 0usize;
        let mut start_byte = 0usize;
        let mut current_byte = 0usize;

        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            let is_section_header = trimmed.starts_with('[') && !trimmed.starts_with("[#");

            if is_section_header && !current_chunk.is_empty() {
                let name = if current_name.is_empty() {
                    "root".to_string()
                } else {
                    current_name.clone()
                };
                chunks.push(Self::section_chunk(
                    name,
                    current_chunk.trim().to_string(),
                    start_line as u32 + 1,
                    idx as u32,
                    start_byte,
                    current_byte,
                    Some(file_name.clone()),
                ));
                current_chunk.clear();
                start_line = idx;
                start_byte = current_byte;
            }

            if is_section_header {
                let name = trimmed
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim();
                current_name = if name.is_empty() {
                    "root".to_string()
                } else {
                    name.to_string()
                };
            }

            current_chunk.push_str(line);
            current_chunk.push('\n');
            current_byte += line.len() + 1;
        }

        if !current_chunk.trim().is_empty() {
            let name = if current_name.is_empty() {
                "root".to_string()
            } else {
                current_name
            };
            chunks.push(Self::section_chunk(
                name,
                current_chunk.trim().to_string(),
                start_line as u32 + 1,
                lines.len().max(1) as u32,
                start_byte,
                content.len(),
                Some(file_name.clone()),
            ));
        }

        if chunks.is_empty() {
            return self.parse_text_sliding_window(path, content);
        }

        Ok(chunks)
    }

    fn parse_text_sliding_window(&self, path: &Path, content: &str) -> Result<Vec<RawChunk>> {
        let file_name = Self::file_name(path);

        // Handle empty content
        if content.trim().is_empty() {
            return Ok(vec![RawChunk {
                chunk_type: ChunkType::Doc,
                name: file_name,
                kind: ChunkKind::Document,
                line_start: 1,
                line_end: 1,
                byte_start: 0,
                byte_end: 0,
                content: String::new(),
                context: None,
                signature: None,
                doc_summary: None,
                visibility: Visibility::Private,
            }]);
        }

        // Build character-to-byte index for Unicode-safe slicing
        let char_to_byte: Vec<usize> = content
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(content.len()))
            .collect();
        let char_len = char_to_byte.len().saturating_sub(1);

        // Small content fits in single document chunk
        if char_len <= CHUNK_SIZE {
            let line_count = content.lines().count().max(1);
            return Ok(vec![RawChunk {
                chunk_type: ChunkType::Doc,
                name: file_name,
                kind: ChunkKind::Document,
                line_start: 1,
                line_end: line_count as u32,
                byte_start: 0,
                byte_end: content.len(),
                content: content.to_string(),
                context: None,
                signature: None,
                doc_summary: None,
                visibility: Visibility::Private,
            }]);
        }

        // Split into overlapping chunks
        let mut chunks = Vec::new();
        let mut chunk_num = 0;
        let mut start = 0usize;

        while start < char_len {
            let end = (start + CHUNK_SIZE).min(char_len);
            let start_byte = char_to_byte[start];
            let end_byte = char_to_byte[end];
            let chunk_content = content[start_byte..end_byte].to_string();

            let start_line = content[..start_byte].lines().count();
            let end_line = content[..end_byte].lines().count();

            chunk_num += 1;
            chunks.push(RawChunk {
                chunk_type: ChunkType::Doc,
                name: format!("{} (part {})", file_name, chunk_num),
                kind: ChunkKind::Paragraph,
                line_start: (start_line + 1) as u32,
                line_end: end_line.max(1) as u32,
                byte_start: start_byte,
                byte_end: end_byte,
                content: chunk_content,
                context: Some(file_name.clone()),
                signature: None,
                doc_summary: None,
                visibility: Visibility::Private,
            });

            if end >= char_len {
                break;
            }
            start += CHUNK_SIZE - CHUNK_OVERLAP;
        }

        Ok(chunks)
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

    #[test]
    fn test_parse_json_object() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.json");
        let content = r#"{"name": "test", "version": "1.0", "config": {"key": "value"}}"#;

        let chunks = parser.parse(&path, content).unwrap();
        assert_eq!(chunks.len(), 3);

        let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"name"));
        assert!(names.contains(&"version"));
        assert!(names.contains(&"config"));
    }

    #[test]
    fn test_parse_json_array_fallback() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.json");
        let content = r#"[1, 2, 3]"#;

        let chunks = parser.parse(&path, content).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Document);
    }

    #[test]
    fn test_parse_yaml() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.yaml");
        let content = "name: test\nversion: 1.0\nconfig:\n  key: value";

        let chunks = parser.parse(&path, content).unwrap();
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_parse_yaml_with_comments() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.yaml");
        let content = "# Comment\nname: test\n# Another comment\nversion: 1.0";

        let chunks = parser.parse(&path, content).unwrap();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_parse_toml() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.toml");
        let content = "[package]\nname = \"test\"\n\n[dependencies]\nserde = \"1.0\"";

        let chunks = parser.parse(&path, content).unwrap();
        assert_eq!(chunks.len(), 2);

        let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"package"));
        assert!(names.contains(&"dependencies"));
    }

    #[test]
    fn test_parse_toml_with_root() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.toml");
        let content = "name = \"test\"\nversion = \"1.0\"\n\n[dependencies]";

        let chunks = parser.parse(&path, content).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].name, "root");
    }

    #[test]
    fn test_parse_toml_array_tables() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.toml");
        let content = "[[bin]]\nname = \"foo\"\n\n[[bin]]\nname = \"bar\"";

        let chunks = parser.parse(&path, content).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].name, "bin");
        assert_eq!(chunks[1].name, "bin");
    }

    #[test]
    fn test_parse_text_sliding_window_large() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.txt");
        let content = "a".repeat(1000);

        let chunks = parser.parse(&path, &content).unwrap();
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].kind, ChunkKind::Paragraph);
    }

    #[test]
    fn test_parse_text_sliding_window_small() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.txt");
        let content = "small content";

        let chunks = parser.parse(&path, content).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Document);
    }

    #[test]
    fn test_parse_text_sliding_window_korean() {
        let parser = DocsParser::new();
        let path = PathBuf::from("test.txt");
        let content = "가".repeat(600);

        let chunks = parser.parse(&path, &content).unwrap();
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
        }
    }

    #[test]
    fn test_parse_empty_file() {
        let parser = DocsParser::new();

        let chunks = parser.parse(&PathBuf::from("test.json"), "").unwrap();
        assert_eq!(chunks.len(), 1);

        let chunks = parser.parse(&PathBuf::from("test.yaml"), "").unwrap();
        assert_eq!(chunks.len(), 1);

        let chunks = parser.parse(&PathBuf::from("test.toml"), "").unwrap();
        assert_eq!(chunks.len(), 1);
    }
}
