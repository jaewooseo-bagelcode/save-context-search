use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::time::Duration;
use tokio::time::sleep;

const EMBEDDING_DIM: usize = 1536;
const EMBEDDING_MODEL: &str = "text-embedding-3-small";
const EMBEDDING_URL: &str = "https://api.openai.com/v1/embeddings";
const BATCH_SIZE: usize = 100;
const MAX_ATTEMPTS: usize = 3;
const RATE_LIMIT_DELAY_MS: u64 = if cfg!(test) { 10 } else { 1_000 };
const BACKOFF_BASE_DELAY_MS: u64 = if cfg!(test) { 10 } else { 1_000 };
const REQUEST_TIMEOUT_SECS: u64 = if cfg!(test) { 1 } else { 30 };

#[derive(Clone)]
pub struct Embedder {
    client: Client,
    api_key: String,
    base_url: String,
}

impl Embedder {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| "Missing OpenAI API key: set OPENAI_API_KEY environment variable")?;

        Ok(Self {
            client: Client::new(),
            api_key,
            base_url: EMBEDDING_URL.to_string(),
        })
    }

    #[cfg(test)]
    fn new_with_base_url(base_url: String, api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }

    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let request = EmbeddingRequest {
            model: EMBEDDING_MODEL,
            input: text,
        };

        let response = self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "OpenAI embedding request failed with status {}: {}",
                status, body
            )
            .into());
        }

        let response_body: EmbeddingResponse = response.json().await?;
        let embedding = response_body
            .data
            .into_iter()
            .next()
            .ok_or("Missing embedding data in response")?
            .embedding;

        if embedding.len() != EMBEDDING_DIM {
            return Err(format!(
                "Expected embedding dimension {}, got {}",
                EMBEDDING_DIM,
                embedding.len()
            )
            .into());
        }

        Ok(embedding)
    }

    pub async fn embed_batch(
        &self,
        texts: Vec<&str>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let total_batches = (texts.len() + BATCH_SIZE - 1) / BATCH_SIZE;

        for batch_idx in 0..total_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = std::cmp::min(start + BATCH_SIZE, texts.len());
            let batch_texts: Vec<String> = texts[start..end].iter().map(|s| s.to_string()).collect();

            // Exponential backoff retry
            let mut attempt = 0;
            loop {
                attempt += 1;

                let batch_input: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
                let request = EmbeddingBatchRequest {
                    model: EMBEDDING_MODEL,
                    input: batch_input,
                };

                let response = self
                    .client
                    .post(&self.base_url)
                    .bearer_auth(&self.api_key)
                    .json(&request)
                    .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
                    .send()
                    .await;

                match response {
                    Ok(resp) => {
                        if resp.status() == StatusCode::TOO_MANY_REQUESTS {
                            if attempt < MAX_ATTEMPTS {
                                let backoff_delay = BACKOFF_BASE_DELAY_MS * (1 << (attempt - 1));
                                sleep(Duration::from_millis(backoff_delay)).await;
                                continue;
                            } else {
                                return Err("Max retries exceeded for rate limit".into());
                            }
                        }

                        if !resp.status().is_success() {
                            let status = resp.status();
                            let body = resp.text().await.unwrap_or_default();
                            return Err(format!(
                                "OpenAI embedding request failed with status {}: {}",
                                status, body
                            )
                            .into());
                        }

                        let response_body: EmbeddingResponse = resp.json().await?;

                        // Store embeddings in correct positions
                        for data in response_body.data {
                            let original_idx = start + data.index;
                            if original_idx < results.len() {
                                results[original_idx] = Some(data.embedding);
                            }
                        }

                        break;
                    }
                    Err(e) if e.is_timeout() => {
                        if attempt < MAX_ATTEMPTS {
                            let backoff_delay = BACKOFF_BASE_DELAY_MS * (1 << (attempt - 1));
                            sleep(Duration::from_millis(backoff_delay)).await;
                            continue;
                        } else {
                            return Err(format!("Request timeout after {} attempts", MAX_ATTEMPTS).into());
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            }

            // Rate limiting: sleep between batches
            if batch_idx < total_batches - 1 {
                sleep(Duration::from_millis(RATE_LIMIT_DELAY_MS)).await;
            }
        }

        // Collect embeddings - fail if any is missing
        let mut output = Vec::with_capacity(texts.len());
        for (idx, opt) in results.into_iter().enumerate() {
            match opt {
                Some(embedding) => output.push(embedding),
                None => {
                    return Err(format!(
                        "Embedding failed for text at index {}: no result returned",
                        idx
                    ).into());
                }
            }
        }

        Ok(output)
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Serialize)]
struct EmbeddingBatchRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embed_text_success() {
        let mut server = mockito::Server::new_async().await;
        let embedding = vec![0.1_f32; EMBEDDING_DIM];
        let body = serde_json::json!({
            "data": [
                {
                    "embedding": embedding,
                    "index": 0
                }
            ]
        });

        let mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let embedder = Embedder::new_with_base_url(
            format!("{}/v1/embeddings", server.url()),
            "test-key".to_string(),
        );

        let result = embedder.embed_text("hello world").await.unwrap();
        assert_eq!(result.len(), EMBEDDING_DIM);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_embed_text_api_failure() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(500)
            .with_body("server error")
            .create_async()
            .await;

        let embedder = Embedder::new_with_base_url(
            format!("{}/v1/embeddings", server.url()),
            "test-key".to_string(),
        );

        let result = embedder.embed_text("hello world").await;
        assert!(result.is_err());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_embed_batch_success() {
        let mut server = mockito::Server::new_async().await;
        let embedding_a = vec![0.1_f32; EMBEDDING_DIM];
        let embedding_b = vec![0.2_f32; EMBEDDING_DIM];
        let body = serde_json::json!({
            "data": [
                { "embedding": embedding_a, "index": 0 },
                { "embedding": embedding_b, "index": 1 }
            ]
        });

        let mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let embedder = Embedder::new_with_base_url(
            format!("{}/v1/embeddings", server.url()),
            "test-key".to_string(),
        );

        let result = embedder
            .embed_batch(vec!["hello", "world"])
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0][0] - 0.1).abs() < 1e-6);
        assert!((result[1][0] - 0.2).abs() < 1e-6);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_embed_batch_rate_limit_retry() {
        let mut server = mockito::Server::new_async().await;
        let rate_limit_mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(429)
            .with_body("rate limit")
            .create_async()
            .await;

        let embedding = vec![0.3_f32; EMBEDDING_DIM];
        let body = serde_json::json!({
            "data": [
                { "embedding": embedding, "index": 0 }
            ]
        });
        let success_mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let embedder = Embedder::new_with_base_url(
            format!("{}/v1/embeddings", server.url()),
            "test-key".to_string(),
        );

        let result = embedder.embed_batch(vec!["hello"]).await.unwrap();
        assert_eq!(result.len(), 1);
        rate_limit_mock.assert_async().await;
        success_mock.assert_async().await;
    }
}
