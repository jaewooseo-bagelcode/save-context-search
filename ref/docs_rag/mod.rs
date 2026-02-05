pub mod chunker;
pub mod embedder;
pub mod store;
pub mod watcher;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use once_cell::sync::Lazy;
use tauri::{AppHandle, Emitter};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub file_path: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RagStatus {
    pub initialized: bool,
    pub file_count: usize,
    pub chunk_count: usize,
    pub last_indexed: Option<String>,
}

/// Event payload for DocsRAG status updates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocsRagStatusEvent {
    pub project_path: String,
    pub status: RagStatus,
}

/// Emit DocsRAG status update event
fn emit_status_event(app: &AppHandle, project_path: &str, status: RagStatus) {
    let payload = DocsRagStatusEvent {
        project_path: project_path.to_string(),
        status,
    };
    if let Err(e) = app.emit("docs-rag:status", payload) {
        eprintln!("[DocsRAG] Failed to emit status event: {}", e);
    }
}

static STORES: Lazy<RwLock<HashMap<String, Arc<RwLock<store::DocsRagStore>>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

static WATCHERS: Lazy<RwLock<HashMap<String, watcher::FileWatcher>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Global query cache - separate from store to avoid holding RwLock during network calls
static QUERY_CACHE: Lazy<Mutex<store::QueryCache>> =
    Lazy::new(|| Mutex::new(store::QueryCache::new(100)));

#[tauri::command]
pub async fn init_docs_rag(project_path: String) -> Result<(), String> {
    let store = store::DocsRagStore::new(project_path.clone())
        .await
        .map_err(|e| format!("Failed to initialize DocsRAG: {}", e))?;

    let mut stores = STORES.write().await;
    stores.insert(project_path, Arc::new(RwLock::new(store)));
    Ok(())
}

#[tauri::command]
pub fn auto_init_docs_rag(app: AppHandle, project_path: String) {
    tauri::async_runtime::spawn(async move {
        eprintln!("[DocsRAG] auto_init_docs_rag START (background): {}", project_path);

        // Convert error to String before the await to satisfy Send bound
        let store_result: Result<_, String> = store::DocsRagStore::new(project_path.clone())
            .await
            .map_err(|e| format!("{}", e));

        match store_result {
            Ok(store) => {
                // Get status before inserting
                let status = store.get_status();

                let mut stores = STORES.write().await;
                let store_arc = Arc::new(RwLock::new(store));
                stores.insert(project_path.clone(), store_arc.clone());
                eprintln!("[DocsRAG] auto_init_docs_rag DONE: {}", project_path);

                // Emit status event (initialized)
                emit_status_event(&app, &project_path, status);

                // Start FileWatcher for incremental updates (in separate task to avoid Send issues)
                let project_path_for_watcher = project_path.clone();
                let app_for_watcher = app.clone();
                tokio::spawn(async move {
                    start_file_watcher(project_path_for_watcher, store_arc, app_for_watcher).await;
                });
            }
            Err(e) => {
                eprintln!("[DocsRAG] auto_init_docs_rag FAILED: {}", e);
            }
        }
    });
}

/// Start FileWatcher for a project and handle incremental updates
async fn start_file_watcher(project_path: String, store_arc: Arc<RwLock<store::DocsRagStore>>, app: AppHandle) {
    let project_path_clone = project_path.clone();
    let store_clone = store_arc.clone();
    let app_clone = app.clone();

    // Create callback that triggers incremental indexing on file changes
    let on_change = move |changes: watcher::FileChanges| {
        let project_path_inner = project_path_clone.clone();
        let store = store_clone.clone();
        let app_inner = app_clone.clone();

        tokio::spawn(async move {
            eprintln!(
                "[DocsRAG] File changes detected: {} modified, {} deleted",
                changes.modified.len(),
                changes.deleted.len()
            );

            // Pass modified files as both modified and added (watcher doesn't distinguish)
            // Deleted files are now properly tracked
            let modified = changes.modified;
            let added = Vec::new();
            let deleted = changes.deleted;

            // Perform incremental update
            let mut store_guard = store.write().await;
            match store_guard.incremental_update(modified, added, deleted).await {
                Ok(()) => {
                    eprintln!("[DocsRAG] Incremental update completed successfully");
                    // Emit status event after update
                    let status = store_guard.get_status();
                    emit_status_event(&app_inner, &project_path_inner, status);
                }
                Err(e) => {
                    eprintln!("[DocsRAG] Incremental update failed: {}", e);
                }
            }
        });
    };

    // Create the FileWatcher - handle potential error
    let watcher_result = watcher::FileWatcher::new(&project_path, on_change);
    match watcher_result {
        Ok(file_watcher) => {
            let mut watchers = WATCHERS.write().await;
            watchers.insert(project_path.clone(), file_watcher);
            eprintln!("[DocsRAG] FileWatcher started for: {}", project_path);
        }
        Err(e) => {
            eprintln!("[DocsRAG] Failed to start FileWatcher: {}", e);
        }
    }
}

#[tauri::command]
pub async fn search_docs(
    project_path: String,
    query: String,
    top_k: usize,
) -> Result<Vec<SearchResult>, String> {
    let store_arc = {
        let stores = STORES.read().await;
        stores.get(&project_path).cloned()
    }
    .ok_or_else(|| "DocsRAG not initialized for this project".to_string())?;

    // Step 1: Check if store has embeddings (brief RwLock)
    let has_embeddings = {
        let store = store_arc.read().await;
        !store.is_embeddings_empty()
    };
    // RwLock released here

    if !has_embeddings {
        // Fall back to text search
        let store = store_arc.read().await;
        return store.search(&query, top_k)
            .await
            .map_err(|e| format!("Search failed: {}", e));
    }

    // Step 2: Get query embedding from GLOBAL cache (NO RwLock held)
    let query_embedding = get_or_compute_query_embedding(&query).await
        .map_err(|e| format!("Failed to get query embedding: {}", e))?;
    // No locks held during network call!

    // Step 3: Search with pre-computed embedding (brief RwLock)
    let store = store_arc.read().await;
    Ok(store.search_with_embedding(&query_embedding, top_k))
}

/// Get query embedding from global cache or compute via API
/// This function does NOT hold any RwLock - only the cache Mutex briefly
async fn get_or_compute_query_embedding(query: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    // Step 1: Check cache (brief Mutex lock)
    {
        let mut cache = QUERY_CACHE.lock().await;
        if let Some(embedding) = cache.get(query) {
            return Ok(embedding);
        }
    }
    // Mutex released here

    // Step 2: Compute embedding (NO LOCKS - network call)
    let embedder = embedder::Embedder::new()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
    let embedding = embedder.embed_text(query).await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;

    // Step 3: Store in cache (brief Mutex lock)
    {
        let mut cache = QUERY_CACHE.lock().await;
        cache.put(query.to_string(), embedding.clone());
    }

    Ok(embedding)
}

/// Stop DocsRAG for a project and cleanup resources
/// Call this when closing a project to prevent memory leaks
#[tauri::command]
pub async fn stop_docs_rag(project_path: String) -> Result<(), String> {
    eprintln!("[DocsRAG] Stopping DocsRAG for: {}", project_path);

    // Remove and drop FileWatcher (triggers channel close and debounce_handler exit)
    {
        let mut watchers = WATCHERS.write().await;
        if watchers.remove(&project_path).is_some() {
            eprintln!("[DocsRAG] FileWatcher removed for: {}", project_path);
        }
    }

    // Remove store
    {
        let mut stores = STORES.write().await;
        if stores.remove(&project_path).is_some() {
            eprintln!("[DocsRAG] Store removed for: {}", project_path);
        }
    }

    Ok(())
}

#[tauri::command]
pub async fn get_docs_rag_status(project_path: String) -> Result<RagStatus, String> {
    let store_arc = {
        let stores = STORES.read().await;
        stores.get(&project_path).cloned()
    }
    .ok_or_else(|| "DocsRAG not initialized for this project".to_string())?;

    let store = store_arc.read().await;
    Ok(store.get_status())
}
