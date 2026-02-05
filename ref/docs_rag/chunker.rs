use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub file_path: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
}

const CHUNK_SIZE: usize = 500;
const CHUNK_OVERLAP: usize = 100;
const MAX_FILE_SIZE: usize = 1_000_000; // 1MB

pub struct Chunker;

impl Chunker {
    pub async fn chunk_file(path: &Path) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
        let path_str = path.to_string_lossy().to_string();
        let content = fs::read_to_string(path).await?;
        
        // Skip files that are too large
        if content.len() > MAX_FILE_SIZE {
            return Ok(vec![]);
        }
        
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        
        match extension {
            "md" | "rst" | "txt" => Self::chunk_markdown(&content, &path_str),
            "json" => Self::chunk_json(&content, &path_str),
            "yaml" | "yml" => Self::chunk_yaml(&content, &path_str),
            "toml" => Self::chunk_toml(&content, &path_str),
            "xml" => Self::chunk_xml(&content, &path_str),
            _ => Self::chunk_text(&content, &path_str),
        }
    }
    
    fn chunk_markdown(content: &str, path: &str) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
        let mut chunks = vec![];
        let lines: Vec<&str> = content.lines().collect();
        let mut current_chunk = String::new();
        let mut start_line = 0;
        let mut current_line = 0;
        
        for (idx, line) in lines.iter().enumerate() {
            // Check if this line is a header (# level 1-3)
            if (line.starts_with("# ") || line.starts_with("## ") || line.starts_with("### ")) 
                && !current_chunk.is_empty() {
                // Save previous chunk
                chunks.push(Chunk {
                    file_path: path.to_string(),
                    content: current_chunk.trim().to_string(),
                    start_line,
                    end_line: idx - 1,
                });
                current_chunk.clear();
                start_line = idx;
            }
            
            current_chunk.push_str(line);
            current_chunk.push('\n');
            current_line = idx;
        }
        
        // Push final chunk
        if !current_chunk.trim().is_empty() {
            chunks.push(Chunk {
                file_path: path.to_string(),
                content: current_chunk.trim().to_string(),
                start_line,
                end_line: current_line,
            });
        }
        
        Ok(chunks)
    }
    
    fn chunk_json(content: &str, path: &str) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
        match serde_json::from_str::<serde_json::Value>(content) {
            Ok(json) => {
                let mut chunks = vec![];
                if let serde_json::Value::Object(map) = json {
                    for (key, value) in map.iter() {
                        let chunk_content = format!("{}:\n{}", key, value.to_string());
                        chunks.push(Chunk {
                            file_path: path.to_string(),
                            content: chunk_content,
                            start_line: 0,
                            end_line: 0,
                        });
                    }
                }
                Ok(chunks)
            }
            Err(_) => Self::chunk_text(content, path),
        }
    }
    
    fn chunk_yaml(content: &str, path: &str) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
        // Simplified YAML chunking: split by top-level keys (lines starting with non-space)
        let mut chunks = vec![];
        let lines: Vec<&str> = content.lines().collect();
        let mut current_chunk = String::new();
        let mut start_line = 0;
        
        for (idx, line) in lines.iter().enumerate() {
            if !line.is_empty() && !line.starts_with(' ') && !current_chunk.is_empty() {
                chunks.push(Chunk {
                    file_path: path.to_string(),
                    content: current_chunk.trim().to_string(),
                    start_line,
                    end_line: idx - 1,
                });
                current_chunk.clear();
                start_line = idx;
            }
            current_chunk.push_str(line);
            current_chunk.push('\n');
        }
        
        if !current_chunk.trim().is_empty() {
            chunks.push(Chunk {
                file_path: path.to_string(),
                content: current_chunk.trim().to_string(),
                start_line,
                end_line: lines.len() - 1,
            });
        }
        
        Ok(chunks)
    }
    
    fn chunk_toml(content: &str, path: &str) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
        // Simplified TOML chunking: split by section headers [section]
        let mut chunks = vec![];
        let lines: Vec<&str> = content.lines().collect();
        let mut current_chunk = String::new();
        let mut start_line = 0;
        
        for (idx, line) in lines.iter().enumerate() {
            if line.starts_with('[') && !current_chunk.is_empty() {
                chunks.push(Chunk {
                    file_path: path.to_string(),
                    content: current_chunk.trim().to_string(),
                    start_line,
                    end_line: idx - 1,
                });
                current_chunk.clear();
                start_line = idx;
            }
            current_chunk.push_str(line);
            current_chunk.push('\n');
        }
        
        if !current_chunk.trim().is_empty() {
            chunks.push(Chunk {
                file_path: path.to_string(),
                content: current_chunk.trim().to_string(),
                start_line,
                end_line: lines.len() - 1,
            });
        }
        
        Ok(chunks)
    }
    
    fn chunk_xml(content: &str, path: &str) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
        // Simplified XML chunking: use text content split
        Self::chunk_text(content, path)
    }
    
    fn chunk_text(content: &str, path: &str) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
        let mut chunks = vec![];
        let mut char_to_byte = Vec::with_capacity(content.len() + 1);

        for (byte_idx, _) in content.char_indices() {
            char_to_byte.push(byte_idx);
        }
        char_to_byte.push(content.len());

        let char_len = char_to_byte.len().saturating_sub(1);

        if char_len <= CHUNK_SIZE {
            return Ok(vec![Chunk {
                file_path: path.to_string(),
                content: content.to_string(),
                start_line: 0,
                end_line: content.lines().count(),
            }]);
        }

        let mut start = 0;
        while start < char_len {
            let end = std::cmp::min(start + CHUNK_SIZE, char_len);
            let start_byte = char_to_byte[start];
            let end_byte = char_to_byte[end];
            let chunk_str = content[start_byte..end_byte].to_string();

            // Count lines in this chunk
            let start_line = content[..start_byte].lines().count();
            let end_line = content[..end_byte].lines().count();

            chunks.push(Chunk {
                file_path: path.to_string(),
                content: chunk_str,
                start_line,
                end_line,
            });

            start = if start + CHUNK_SIZE >= char_len {
                char_len
            } else {
                start + CHUNK_SIZE - CHUNK_OVERLAP
            };
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunk_markdown() {
        let content = "# Section 1\nContent 1\n## Subsection\nContent 2\n# Section 2\nContent 3";
        let chunks = Chunker::chunk_markdown(content, "test.md").unwrap();
        assert!(chunks.len() > 0);
    }
    
    #[test]
    fn test_chunk_text() {
        let content = "a".repeat(1000);
        let chunks = Chunker::chunk_text(&content, "test.txt").unwrap();
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_chunk_text_korean_multibyte_lines() {
        let line1 = "가".repeat(300);
        let line2 = "나".repeat(300);
        let line3 = "다".repeat(300);
        let content = format!("{}\n{}\n{}", line1, line2, line3);
        let chunks = Chunker::chunk_text(&content, "test.txt").unwrap();
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].start_line, 0);
        assert_eq!(chunks[0].end_line, 2);
        assert_eq!(chunks[1].start_line, 2);
        assert_eq!(chunks[1].end_line, 3);
    }

    #[test]
    fn test_chunk_text_mixed_emoji_korean() {
        let segment = "Hello😀안녕";
        let content = segment.repeat(120);
        let chunks = Chunker::chunk_text(&content, "test.txt").unwrap();
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| !chunk.content.is_empty()));
        assert!(chunks.iter().any(|chunk| chunk.content.contains('😀')));
        assert!(chunks.iter().any(|chunk| chunk.content.contains('안')));
        assert_eq!(chunks[0].start_line, 0);
        assert_eq!(chunks[0].end_line, 1);
    }
}
