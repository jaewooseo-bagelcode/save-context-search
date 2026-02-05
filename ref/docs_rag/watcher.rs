use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Debounce delay in milliseconds
const DEBOUNCE_MS: u64 = 500;

/// Supported file extensions for watching
const WATCHED_EXTENSIONS: &[&str] = &["md", "json", "yaml", "yml", "toml", "xml", "txt", "rst"];

/// Directories to exclude from watching
const EXCLUDED_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "dist",
    "build",
    ".git",
    ".syntaxos",
    "__pycache__",
    "vendor",
    ".next",
    ".nuxt",
    "coverage",
];

/// Represents a file change event with its type
#[derive(Debug, Clone)]
pub enum FileChangeEvent {
    Modified(String),  // File modified or created
    Deleted(String),   // File deleted
}

/// Result of debounced file changes, separated by event type
#[derive(Debug, Default)]
pub struct FileChanges {
    pub modified: Vec<String>,
    pub deleted: Vec<String>,
}

/// FileWatcher detects changes to documentation files and triggers indexing
/// Uses tokio channels for async debounce handling
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    project_path: String,
}

impl Drop for FileWatcher {
    fn drop(&mut self) {
        eprintln!("[FileWatcher] Dropping watcher for: {}", self.project_path);
    }
}

impl FileWatcher {
    /// Create a new FileWatcher for the given project path
    ///
    /// # Arguments
    /// * `project_path` - Root path to watch for changes
    /// * `on_change` - Callback function invoked with FileChanges after debounce
    ///
    /// # Debounce Behavior
    /// Multiple rapid file changes are coalesced into a single callback invocation
    /// after 500ms of inactivity. This prevents excessive re-indexing.
    pub fn new<F>(project_path: &str, on_change: F) -> Result<Self, String>
    where
        F: Fn(FileChanges) + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::unbounded_channel::<FileChangeEvent>();

        let project_path_str = project_path.to_string();

        let on_change = Arc::new(on_change);
        tokio::spawn(debounce_handler(rx, on_change));

        let event_handler = move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    match event.kind {
                        // Handle modifications and creations
                        EventKind::Modify(_) | EventKind::Create(_) => {
                            for path in &event.paths {
                                if should_watch_file(path) {
                                    let abs_path = path.to_string_lossy().to_string();
                                    let _ = tx.send(FileChangeEvent::Modified(abs_path));
                                }
                            }
                        }
                        // Handle deletions
                        EventKind::Remove(_) => {
                            for path in &event.paths {
                                // For deletions, check extension only (file may not exist)
                                if should_watch_extension(path) {
                                    let abs_path = path.to_string_lossy().to_string();
                                    eprintln!("[FileWatcher] Detected deletion: {}", abs_path);
                                    let _ = tx.send(FileChangeEvent::Deleted(abs_path));
                                }
                            }
                        }
                        // Handle renames (old path deleted, new path added)
                        EventKind::Access(_) => {} // Ignore access events
                        EventKind::Other => {} // Ignore other events
                        _ => {}
                    }
                }
                Err(e) => {
                    eprintln!("[FileWatcher] Error: {}", e);
                }
            }
        };

        let mut watcher = RecommendedWatcher::new(event_handler, Default::default())
            .map_err(|e| format!("Failed to create watcher: {}", e))?;

        watcher
            .watch(Path::new(&project_path_str), RecursiveMode::Recursive)
            .map_err(|e| format!("Failed to watch path: {}", e))?;

        eprintln!(
            "[FileWatcher] Started watching: {} (debounce: {}ms, events: modify/create/delete)",
            project_path_str, DEBOUNCE_MS
        );

        Ok(FileWatcher {
            _watcher: watcher,
            project_path: project_path_str,
        })
    }
}

/// Background task that handles debouncing file changes
async fn debounce_handler<F>(mut rx: mpsc::UnboundedReceiver<FileChangeEvent>, on_change: Arc<F>)
where
    F: Fn(FileChanges) + Send + Sync + 'static,
{
    let mut pending_modified: HashSet<String> = HashSet::new();
    let mut pending_deleted: HashSet<String> = HashSet::new();

    loop {
        let first = rx.recv().await;
        match first {
            None => {
                eprintln!("[FileWatcher] Channel closed, stopping debounce handler");
                break;
            }
            Some(event) => {
                collect_event(&mut pending_modified, &mut pending_deleted, event);
            }
        }

        // Start debounce window
        loop {
            let timeout = tokio::time::sleep(tokio::time::Duration::from_millis(DEBOUNCE_MS));
            tokio::pin!(timeout);

            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Some(event) => {
                            collect_event(&mut pending_modified, &mut pending_deleted, event);
                        }
                        None => {
                            // Channel closed - flush remaining and exit
                            flush_changes(&mut pending_modified, &mut pending_deleted, &on_change);
                            return;
                        }
                    }
                }
                _ = &mut timeout => {
                    flush_changes(&mut pending_modified, &mut pending_deleted, &on_change);
                    break;
                }
            }
        }
    }
}

/// Collect an event into the appropriate set
fn collect_event(
    modified: &mut HashSet<String>,
    deleted: &mut HashSet<String>,
    event: FileChangeEvent,
) {
    match event {
        FileChangeEvent::Modified(path) => {
            // If file was deleted then modified, remove from deleted
            deleted.remove(&path);
            modified.insert(path);
        }
        FileChangeEvent::Deleted(path) => {
            // If file was modified then deleted, remove from modified
            modified.remove(&path);
            deleted.insert(path);
        }
    }
}

/// Flush pending changes and invoke callback
fn flush_changes<F>(
    modified: &mut HashSet<String>,
    deleted: &mut HashSet<String>,
    on_change: &Arc<F>,
)
where
    F: Fn(FileChanges) + Send + Sync + 'static,
{
    if modified.is_empty() && deleted.is_empty() {
        return;
    }

    let changes = FileChanges {
        modified: modified.drain().collect(),
        deleted: deleted.drain().collect(),
    };

    eprintln!(
        "[FileWatcher] Debounce flush: {} modified, {} deleted",
        changes.modified.len(),
        changes.deleted.len()
    );

    on_change(changes);
}

/// Check if a file should be watched based on extension and path
fn should_watch_file(path: &Path) -> bool {
    if !should_watch_extension(path) {
        return false;
    }

    // Check if any path component is hidden or excluded
    for component in path.components() {
        if let Some(name) = component.as_os_str().to_str() {
            if name.starts_with('.') {
                return false;
            }
            if EXCLUDED_DIRS.contains(&name) {
                return false;
            }
        }
    }

    true
}

/// Check if file extension should be watched (used for delete events where file may not exist)
fn should_watch_extension(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            return WATCHED_EXTENSIONS.contains(&ext_str);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_should_watch_valid_file() {
        let path = PathBuf::from("/project/docs/readme.md");
        assert!(should_watch_file(&path));
    }

    #[test]
    fn test_should_not_watch_unsupported_extension() {
        let path = PathBuf::from("/project/docs/script.js");
        assert!(!should_watch_file(&path));
    }

    #[test]
    fn test_should_not_watch_excluded_dir() {
        let path = PathBuf::from("/project/node_modules/package/readme.md");
        assert!(!should_watch_file(&path));
    }

    #[test]
    fn test_should_not_watch_hidden_file() {
        let path = PathBuf::from("/project/.hidden/readme.md");
        assert!(!should_watch_file(&path));
    }

    #[test]
    fn test_should_not_watch_git_dir() {
        let path = PathBuf::from("/project/.git/config");
        assert!(!should_watch_file(&path));
    }

    #[test]
    fn test_should_not_watch_target_dir() {
        let path = PathBuf::from("/project/target/debug/readme.md");
        assert!(!should_watch_file(&path));
    }

    #[test]
    fn test_should_watch_extension() {
        assert!(should_watch_extension(Path::new("test.md")));
        assert!(should_watch_extension(Path::new("test.json")));
        assert!(!should_watch_extension(Path::new("test.js")));
        assert!(!should_watch_extension(Path::new("test")));
    }

    #[test]
    fn test_collect_event_modified_then_deleted() {
        let mut modified = HashSet::new();
        let mut deleted = HashSet::new();

        // File modified
        collect_event(&mut modified, &mut deleted, FileChangeEvent::Modified("test.md".to_string()));
        assert!(modified.contains("test.md"));
        assert!(!deleted.contains("test.md"));

        // Then deleted
        collect_event(&mut modified, &mut deleted, FileChangeEvent::Deleted("test.md".to_string()));
        assert!(!modified.contains("test.md"));
        assert!(deleted.contains("test.md"));
    }

    #[test]
    fn test_collect_event_deleted_then_recreated() {
        let mut modified = HashSet::new();
        let mut deleted = HashSet::new();

        // File deleted
        collect_event(&mut modified, &mut deleted, FileChangeEvent::Deleted("test.md".to_string()));
        assert!(deleted.contains("test.md"));

        // Then recreated (modified event)
        collect_event(&mut modified, &mut deleted, FileChangeEvent::Modified("test.md".to_string()));
        assert!(modified.contains("test.md"));
        assert!(!deleted.contains("test.md"));
    }
}
