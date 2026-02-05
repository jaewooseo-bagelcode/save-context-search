use anyhow::{Context, Result};
use futures::{stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

const EMBEDDING_MODEL: &str = "text-embedding-3-small";
pub const EMBEDDING_DIM: usize = 1536;
const BATCH_SIZE: usize = 100;
const MAX_ATTEMPTS: usize = 3;
const BACKOFF_BASE_MS: u64 = 1_000;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const API_URL: &str = "https://api.openai.com/v1/embeddings";

// Default concurrency if tier check fails
const DEFAULT_CONCURRENCY: usize = 5;

/// Rate limit info from OpenAI API response headers
#[derive(Debug, Clone)]
pub struct RateLimitInfo {
    pub requests_per_minute: usize,
    pub tokens_per_minute: usize,
}

impl RateLimitInfo {
    /// Calculate optimal concurrency based on rate limits (RPM and TPM)
    pub fn optimal_concurrency(&self) -> usize {
        // RPM-based: conservative RPM / 140
        // Tier 1 (500 RPM): 5, Tier 2 (3500 RPM): 25, Tier 5 (10000 RPM): 50
        let rpm_based = self.requests_per_minute / 140;

        // TPM safety cap: only limit if TPM is very restrictive (< 1M)
        // ~8,000 tokens per request, ~30 req/min per concurrent slot
        let tpm_cap = if self.tokens_per_minute < 1_000_000 {
            self.tokens_per_minute / 8_000 / 30
        } else {
            usize::MAX // High TPM tiers: don't limit by TPM
        };

        rpm_based.min(tpm_cap).clamp(5, 50)
    }
}

pub struct Embedder {
    client: Client,
    api_key: String,
    concurrency: usize,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY environment variable not set")?;

        let client = Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()?;

        Ok(Self {
            client,
            api_key,
            concurrency: DEFAULT_CONCURRENCY,
        })
    }

    /// Check rate limits by making a minimal API call
    pub async fn check_rate_limits(&mut self) -> Result<RateLimitInfo> {
        let request = EmbeddingRequest {
            model: EMBEDDING_MODEL.to_string(),
            input: vec!["test".to_string()],
        };

        let response = self.client
            .post(API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to check rate limits")?;

        let headers = response.headers();

        let rpm = headers
            .get("x-ratelimit-limit-requests")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(500); // Default to conservative limit

        let tpm = headers
            .get("x-ratelimit-limit-tokens")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(150_000);

        let info = RateLimitInfo {
            requests_per_minute: rpm,
            tokens_per_minute: tpm,
        };

        // Update concurrency based on rate limits
        self.concurrency = info.optimal_concurrency();
        eprintln!(
            "[scs] Rate limits: {} RPM, {} TPM → {} concurrent requests",
            rpm, tpm, self.concurrency
        );

        Ok(info)
    }

    /// Get current concurrency setting
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// Set concurrency manually (for testing)
    pub fn set_concurrency(&mut self, concurrency: usize) {
        self.concurrency = concurrency.clamp(1, 100);
    }

    pub async fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(vec![text]).await?;
        results.into_iter().next()
            .context("Empty embedding response")
    }

    /// Embed texts in parallel batches
    pub async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let chunks: Vec<Vec<String>> = texts
            .chunks(BATCH_SIZE)
            .map(|chunk| chunk.iter().map(|s| s.to_string()).collect())
            .collect();

        let num_chunks = chunks.len();
        if num_chunks == 0 {
            return Ok(vec![]);
        }

        // For small batches, use sequential processing
        if num_chunks <= 2 {
            let mut all_embeddings = Vec::new();
            for chunk in chunks {
                let chunk_refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                let embeddings = self.embed_batch_internal(&chunk_refs).await?;
                all_embeddings.extend(embeddings);
            }
            return Ok(all_embeddings);
        }

        // Parallel processing with controlled concurrency
        let client = Arc::new(self.client.clone());
        let api_key = Arc::new(self.api_key.clone());

        let results: Vec<Result<(usize, Vec<Vec<f32>>)>> = stream::iter(
            chunks.into_iter().enumerate()
        )
        .map(|(idx, chunk)| {
            let client = Arc::clone(&client);
            let api_key = Arc::clone(&api_key);
            async move {
                let embeddings = Self::embed_batch_internal_static(
                    &client,
                    &api_key,
                    &chunk,
                ).await?;
                Ok((idx, embeddings))
            }
        })
        .buffer_unordered(self.concurrency)
        .collect()
        .await;

        // Sort results by index and flatten
        let mut indexed_results: Vec<(usize, Vec<Vec<f32>>)> = Vec::new();
        for result in results {
            indexed_results.push(result?);
        }
        indexed_results.sort_by_key(|(idx, _)| *idx);

        let all_embeddings: Vec<Vec<f32>> = indexed_results
            .into_iter()
            .flat_map(|(_, embeddings)| embeddings)
            .collect();

        Ok(all_embeddings)
    }

    async fn embed_batch_internal(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        Self::embed_batch_internal_static(&self.client, &self.api_key, &texts_owned).await
    }

    async fn embed_batch_internal_static(
        client: &Client,
        api_key: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let request = EmbeddingRequest {
            model: EMBEDDING_MODEL.to_string(),
            input: texts.to_vec(),
        };

        for attempt in 1..=MAX_ATTEMPTS {
            let response = client
                .post(API_URL)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();

                    if status == 429 {
                        // Rate limit
                        if attempt < MAX_ATTEMPTS {
                            let delay = BACKOFF_BASE_MS * (1 << (attempt - 1));
                            eprintln!("[scs] Rate limited, retrying in {}ms...", delay);
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                            continue;
                        }
                        anyhow::bail!("Rate limit exceeded after {} attempts", MAX_ATTEMPTS);
                    }

                    if !status.is_success() {
                        let error_text = resp.text().await.unwrap_or_default();
                        anyhow::bail!("OpenAI API error {}: {}", status, error_text);
                    }

                    let body: EmbeddingResponse = resp.json().await?;

                    // Validate response count matches request
                    if body.data.len() != texts.len() {
                        anyhow::bail!(
                            "Embedding count mismatch: expected {}, got {}",
                            texts.len(),
                            body.data.len()
                        );
                    }

                    // Sort by index to ensure correct order (API may return out of order)
                    let mut data = body.data;
                    data.sort_by_key(|d| d.index);

                    // Extract embeddings in sorted order
                    return Ok(data.into_iter().map(|d| d.embedding).collect());
                }
                Err(e) => {
                    if e.is_timeout() && attempt < MAX_ATTEMPTS {
                        let delay = BACKOFF_BASE_MS * (1 << (attempt - 1));
                        eprintln!("[scs] Timeout, retrying in {}ms...", delay);
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        anyhow::bail!("Failed after {} attempts", MAX_ATTEMPTS)
    }
}
