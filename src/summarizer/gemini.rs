//! Gemini API client via AI Proxy.
//!
//! Supports batch summarization with JSON response parsing.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Gemini API client configuration.
pub struct GeminiClient {
    base_url: String,
    token: String,
    client: reqwest::Client,
    model: String,
}

/// Request payload for Gemini API.
#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Serialize)]
struct GenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
    temperature: f32,
}

/// Response from Gemini API.
#[derive(Deserialize, Debug)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: Option<CandidateContent>,
}

#[derive(Deserialize, Debug)]
struct CandidateContent {
    parts: Option<Vec<ResponsePart>>,
}

#[derive(Deserialize, Debug)]
struct ResponsePart {
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
}

/// Parsed summary from batch response.
#[derive(Deserialize, Debug)]
pub struct FunctionSummary {
    pub name: String,
    pub summary: String,
}

/// API call result with usage info.
#[derive(Debug)]
pub struct BatchResult {
    /// (chunk_idx, summary_text) pairs
    pub summaries: Vec<(usize, String)>,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl GeminiClient {
    /// Create a new client from environment variables.
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("AI_PROXY_BASE_URL")
            .context("AI_PROXY_BASE_URL environment variable not set")?;
        let token = std::env::var("AI_PROXY_PERSONAL_TOKEN")
            .context("AI_PROXY_PERSONAL_TOKEN environment variable not set")?;

        Self::new(base_url, token)
    }

    /// Create a new client with explicit configuration.
    pub fn new(base_url: String, token: String) -> Result<Self> {
        // Build client with SSL verification disabled (for self-signed certs)
        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            base_url,
            token,
            client,
            model: "gemini-2.5-flash-lite".to_string(),
        })
    }

    /// Get the API endpoint URL.
    fn endpoint(&self) -> String {
        format!(
            "{}/google/v1beta/models/{}:generateContent",
            self.base_url, self.model
        )
    }

    /// Summarize a batch of functions.
    ///
    /// Returns (chunk_idx, summary) pairs matched by input order.
    pub async fn summarize_batch(
        &self,
        functions: &[FunctionInput],
    ) -> Result<BatchResult> {
        if functions.is_empty() {
            return Ok(BatchResult {
                summaries: vec![],
                input_tokens: 0,
                output_tokens: 0,
            });
        }

        let prompt = build_batch_prompt(functions);
        let response = self.call_api(&prompt).await?;

        // Match API response to inputs by name, resolving duplicates by order
        let raw_summaries = response.summaries;
        let mut matched: Vec<(usize, String)> = Vec::new();
        let mut used: Vec<bool> = vec![false; raw_summaries.len()];

        for input in functions {
            // Find first unused response with matching name
            if let Some(pos) = raw_summaries.iter().enumerate().position(|(i, s)| {
                !used[i] && s.0 == input.name
            }) {
                used[pos] = true;
                matched.push((input.chunk_idx, raw_summaries[pos].1.clone()));
            }
        }

        Ok(BatchResult {
            summaries: matched,
            input_tokens: response.input_tokens,
            output_tokens: response.output_tokens,
        })
    }

    /// Batch summarize directories and files for the project map.
    pub async fn summarize_map_batch(
        &self,
        entries: &[MapSummaryInput],
    ) -> Result<MapBatchResult> {
        if entries.is_empty() {
            return Ok(MapBatchResult {
                summaries: vec![],
                input_tokens: 0,
                output_tokens: 0,
            });
        }

        let prompt = build_map_batch_prompt(entries);
        let response = self.call_api(&prompt).await?;

        // Match responses by path
        let mut matched: Vec<(String, String)> = Vec::new();
        let mut used: Vec<bool> = vec![false; response.summaries.len()];

        for input in entries {
            if let Some(pos) = response.summaries.iter().enumerate().position(|(i, s)| {
                !used[i] && s.0 == input.path
            }) {
                used[pos] = true;
                matched.push((input.path.clone(), response.summaries[pos].1.clone()));
            }
        }

        Ok(MapBatchResult {
            summaries: matched,
            input_tokens: response.input_tokens,
            output_tokens: response.output_tokens,
        })
    }

    /// Call the Gemini API with retry logic.
    async fn call_api(&self, prompt: &str) -> Result<RawApiResult> {
        let max_retries = 3;
        let mut last_error = None;

        for attempt in 0..max_retries {
            match self.call_api_once(prompt).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let error_str = e.to_string();
                    // Retry on rate limit (429) or server error (500)
                    if error_str.contains("429") || error_str.contains("rate")
                        || error_str.contains("500") || error_str.contains("INTERNAL")
                    {
                        let delay = Duration::from_secs(2u64.pow(attempt as u32));
                        eprintln!(
                            "[summarizer] Retrying in {:?} (attempt {}/{}): {}",
                            delay,
                            attempt + 1,
                            max_retries,
                            &error_str[..error_str.len().min(80)]
                        );
                        tokio::time::sleep(delay).await;
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown error after retries")))
    }

    /// Single API call without retry.
    async fn call_api_once(&self, prompt: &str) -> Result<RawApiResult> {
        let request = GeminiRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: prompt.to_string(),
                }],
            }],
            generation_config: GenerationConfig {
                max_output_tokens: 8000,  // Batch 50 × ~160 chars scaled summaries
                temperature: 0.3,
            },
        };

        let response = self
            .client
            .post(&self.endpoint())
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Gemini API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Gemini API error {}: {}", status, body);
        }

        let gemini_response: GeminiResponse = response
            .json()
            .await
            .context("Failed to parse Gemini API response")?;

        // Extract text from response
        let text = gemini_response
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .and_then(|c| c.parts)
            .and_then(|p| p.into_iter().next())
            .and_then(|p| p.text)
            .unwrap_or_default();

        // Parse JSON from response
        let summaries = parse_json_response(&text)?;

        // Extract usage info
        let (input_tokens, output_tokens) = gemini_response
            .usage_metadata
            .map(|u| {
                (
                    u.prompt_token_count.unwrap_or(0) as usize,
                    u.candidates_token_count.unwrap_or(0) as usize,
                )
            })
            .unwrap_or((0, 0));

        Ok(RawApiResult {
            summaries,
            input_tokens,
            output_tokens,
        })
    }
}

/// Raw API result before matching to chunk indices.
#[derive(Debug)]
struct RawApiResult {
    summaries: Vec<(String, String)>,  // (name, summary)
    input_tokens: usize,
    output_tokens: usize,
}

/// Function input for batch summarization.
#[derive(Debug, Clone)]
pub struct FunctionInput {
    pub chunk_idx: usize,                // chunk index for response matching
    pub name: String,
    pub body: String,
    pub calls: HashMap<String, String>,  // callee name -> summary
}

/// Dir/File summary input for map batch summarization.
#[derive(Debug, Clone)]
pub struct MapSummaryInput {
    pub path: String,         // "src/parser" or "src/parser/code.rs"
    pub kind: &'static str,   // "directory" or "file"
    pub symbols: Vec<String>, // symbol names in this dir/file
}

/// Map batch result.
#[derive(Debug)]
pub struct MapBatchResult {
    pub summaries: Vec<(String, String)>,  // (path, summary)
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Build the prompt for batch summarization.
fn build_batch_prompt(functions: &[FunctionInput]) -> String {
    let mut prompt = String::new();
    prompt.push_str("Summarize each function. Scale summary length with function complexity:\n");
    prompt.push_str("- Short/trivial functions (< 10 lines): ~40 chars, e.g. \"Returns the cache size.\"\n");
    prompt.push_str("- Medium functions (10-50 lines): ~80 chars, one sentence with key detail\n");
    prompt.push_str("- Long/complex functions (50+ lines): up to 160 chars, include key mechanism\n");
    prompt.push_str("Focus on WHAT it does and key mechanisms (e.g. via API call, using parallel processing).\n");
    prompt.push_str("Return JSON array: [{\"name\": \"fn_name\", \"summary\": \"...\"}]\n\n");
    prompt.push_str("Functions:\n\n");

    for (i, func) in functions.iter().enumerate() {
        let line_count = func.body.lines().count();
        prompt.push_str(&format!("{}. {} ({} lines)\n", i + 1, func.name, line_count));

        if func.calls.is_empty() {
            prompt.push_str("   Calls: (none)\n");
        } else {
            let calls_str: Vec<String> = func
                .calls
                .iter()
                .map(|(name, summary)| format!("{} (\"{}\")", name, summary))
                .collect();
            prompt.push_str(&format!("   Calls: {}\n", calls_str.join(", ")));
        }

        // Truncate body to ~1500 chars (Unicode-safe)
        let body: String = if func.body.chars().count() > 1500 {
            let truncated: String = func.body.chars().take(1500).collect();
            format!("{}...", truncated)
        } else {
            func.body.clone()
        };
        prompt.push_str(&format!("   ```\n{}\n   ```\n\n", body));
    }

    prompt
}

/// Build the prompt for map batch summarization (dirs/files).
fn build_map_batch_prompt(entries: &[MapSummaryInput]) -> String {
    let mut prompt = String::new();
    prompt.push_str("Summarize each entry's purpose in 1 concise phrase (~50 chars max).\n");
    prompt.push_str("Rules:\n");
    prompt.push_str("- Focus on WHAT it does and key technology/mechanism (e.g. \"tree-sitter code parsing\", \"OpenAI batch embedding\")\n");
    prompt.push_str("- Be specific, not generic. Bad: \"Code parsing functions.\" Good: \"Tree-sitter AST parsing for 5 languages\"\n");
    prompt.push_str("- No trailing period. No articles (a/an/the) at start.\n");
    prompt.push_str("Return JSON array: [{\"name\": \"path\", \"summary\": \"...\"}]\n\n");
    prompt.push_str("Entries:\n\n");

    for (i, entry) in entries.iter().enumerate() {
        prompt.push_str(&format!("{}. [{}] {}\n", i + 1, entry.kind, entry.path));
        if entry.symbols.is_empty() {
            prompt.push_str("   Symbols: (none)\n");
        } else {
            prompt.push_str(&format!("   Symbols: {}\n", entry.symbols.join(", ")));
        }
        prompt.push('\n');
    }

    prompt
}

/// Parse JSON array from response text into (name, summary) pairs.
fn parse_json_response(text: &str) -> Result<Vec<(String, String)>> {
    // Find JSON array in response (may have markdown code blocks)
    let json_start = text.find('[').unwrap_or(0);
    let json_end = text.rfind(']').map(|i| i + 1).unwrap_or(text.len());

    if json_start >= json_end {
        anyhow::bail!("No JSON array found in response: {}", text);
    }

    let json_str = &text[json_start..json_end];
    let summaries: Vec<FunctionSummary> =
        serde_json::from_str(json_str).context("Failed to parse JSON response")?;

    Ok(summaries.into_iter().map(|s| (s.name, s.summary)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_batch_prompt() {
        let functions = vec![
            FunctionInput {
                chunk_idx: 0,
                name: "foo".to_string(),
                body: "println!(\"hello\");".to_string(),
                calls: HashMap::new(),
            },
            FunctionInput {
                chunk_idx: 1,
                name: "bar".to_string(),
                body: "foo(); x + 1".to_string(),
                calls: [("foo".to_string(), "Prints hello".to_string())]
                    .into_iter()
                    .collect(),
            },
        ];

        let prompt = build_batch_prompt(&functions);
        assert!(prompt.contains("1. foo"));
        assert!(prompt.contains("2. bar"));
        assert!(prompt.contains("Calls: (none)"));
        assert!(prompt.contains("foo (\"Prints hello\")"));
    }

    #[test]
    fn test_parse_json_response() {
        let text = r#"Here are the summaries:
```json
[{"name": "foo", "summary": "Does something"}, {"name": "bar", "summary": "Does another thing"}]
```"#;

        let summaries = parse_json_response(text).unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].0, "foo");
        assert_eq!(summaries[0].1, "Does something");
    }

    #[test]
    fn test_parse_json_response_plain() {
        let text = r#"[{"name": "test", "summary": "Test function"}]"#;
        let summaries = parse_json_response(text).unwrap();
        assert_eq!(summaries.len(), 1);
    }

    #[test]
    fn test_build_map_batch_prompt() {
        let entries = vec![
            MapSummaryInput {
                path: "src/parser".to_string(),
                kind: "directory",
                symbols: vec!["parse_function".to_string(), "parse_class".to_string()],
            },
            MapSummaryInput {
                path: "src/parser/code.rs".to_string(),
                kind: "file",
                symbols: vec!["parse_function".to_string()],
            },
        ];

        let prompt = build_map_batch_prompt(&entries);
        assert!(prompt.contains("[directory] src/parser"));
        assert!(prompt.contains("[file] src/parser/code.rs"));
        assert!(prompt.contains("parse_function, parse_class"));
    }
}
