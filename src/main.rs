//! scs - Context-efficient semantic search CLI for code
//!
//! This CLI provides semantic search, symbol lookup, dependency tracking,
//! optimized for token-efficient codebase exploration.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use save_context_search::{ChunkType, EmbedMode, SCS, map};

/// Context-efficient semantic search for code
#[derive(Parser)]
#[command(name = "scs")]
#[command(about = "Context-efficient semantic search for code")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Project root directory (defaults to current directory)
    #[arg(long, global = true)]
    path: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Semantic search for code and documentation
    Search {
        /// Search query
        query: String,

        /// Number of results to return (must be > 0)
        #[arg(long, short = 't', default_value = "5")]
        top: usize,

        /// Filter by type: code, doc
        #[arg(long, short)]
        filter: Option<String>,
    },

    /// Look up a symbol by exact name
    Lookup {
        /// Symbol name (supports Class.Method notation)
        name: String,

        /// Filter by type: code, doc
        #[arg(long, short)]
        filter: Option<String>,
    },

    /// Show index status
    Status,

    /// Update index incrementally
    Refresh {
        /// Suppress output
        #[arg(long, short)]
        quiet: bool,

        /// Skip embedding generation (no background embed spawned)
        #[arg(long)]
        no_embed: bool,
    },

    /// Force full reindex
    Reindex {
        /// Skip embedding generation (faster, but no semantic search)
        #[arg(long)]
        no_embed: bool,
    },

    /// Generate embeddings for indexed chunks (can run in background)
    Embed {
        /// Batch size (auto-scaled based on API tier if not specified)
        #[arg(long, short)]
        batch: Option<usize>,
    },

    /// Generate project map for AI context
    Map {
        /// Max tokens for output (default: 2000)
        #[arg(long, default_value = "2000")]
        max_tokens: usize,

        /// Focus on a specific directory or file (shows it at max depth)
        #[arg(long)]
        area: Option<String>,
    },

    /// Generate LLM summaries for functions
    Summarize {
        /// Batch size (functions per API call)
        #[arg(long, short, default_value = "50")]
        batch: usize,

        /// Force regeneration (ignore cache)
        #[arg(long)]
        force: bool,

        /// Dry run (show plan without API calls)
        #[arg(long)]
        dry_run: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Determine project root (canonicalize to absolute path for consistent matching)
    let root = match cli.path {
        Some(p) => std::fs::canonicalize(&p)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or(p)),
        None => std::env::current_dir().context("Failed to get current directory")?,
    };

    // Load or create SCS instance
    let mut scs = SCS::load_or_create(&root).context("Failed to initialize SCS")?;

    match cli.command {
        Commands::Search { query, top, filter } => {
            // Try to ensure index is fresh (auto mode for search)
            // If locked (e.g., background embedding), use cached data
            let stats_opt = scs.try_ensure_fresh(EmbedMode::Auto).await.context("Failed to refresh index")?;

            // Warn if embeddings are pending
            if let Some(stats) = stats_opt {
                if stats.pending_embeddings > 0 {
                    eprintln!(
                        "[scs] Note: {} chunks pending embeddings. Run 'scs embed' for better search.",
                        stats.pending_embeddings
                    );
                }
            } else {
                // Using cached data due to lock - check if we have embeddings
                let pending = scs.missing_embeddings_count();
                if pending > 0 {
                    eprintln!(
                        "[scs] Note: {} chunks pending embeddings (background process running).",
                        pending
                    );
                }
            }

            let filter_type = parse_filter(&filter)?;
            let results = scs
                .search(&query, top, filter_type)
                .await
                .context("Search failed")?;

            // Output as JSON
            println!(
                "{}",
                serde_json::to_string_pretty(&results).context("Failed to serialize results")?
            );
        }

        Commands::Lookup { name, filter } => {
            // Lookup doesn't need embeddings, try to refresh but use cache if locked
            scs.try_ensure_fresh(EmbedMode::Skip).await.context("Failed to refresh index")?;

            let filter_type = parse_filter(&filter)?;
            let results = scs.lookup(&name, filter_type);

            println!(
                "{}",
                serde_json::to_string_pretty(&results).context("Failed to serialize results")?
            );
        }

        Commands::Status => {
            let status = scs.status();
            println!(
                "{}",
                serde_json::to_string_pretty(&status).context("Failed to serialize status")?
            );
        }

        Commands::Refresh { quiet, no_embed } => {
            // Set quiet mode to suppress warnings
            save_context_search::set_quiet_mode(quiet);

            // Determine embedding mode (default: Auto spawns background embed)
            let mode = if no_embed { EmbedMode::Skip } else { EmbedMode::Auto };

            // Non-blocking: if locked, mark dirty and exit
            let stats_opt = scs.try_refresh_or_mark_dirty(mode).await.context("Failed to refresh index")?;

            if !quiet {
                match stats_opt {
                    Some(stats) if stats.has_changes() => {
                        let embed_status = if no_embed {
                            " (no embeddings)".to_string()
                        } else if stats.pending_embeddings > 0 {
                            " (background embed started)".to_string()
                        } else {
                            String::new()
                        };
                        eprintln!(
                            "Index updated: {} added, {} updated, {} removed{}",
                            stats.added, stats.updated, stats.removed, embed_status
                        );
                    }
                    Some(_) => {
                        eprintln!("Index is up to date");
                    }
                    None => {
                        eprintln!("Index locked, marked dirty for later refresh");
                    }
                }
            }
        }

        Commands::Reindex { no_embed } => {
            let mode = if no_embed { EmbedMode::Skip } else { EmbedMode::Auto };
            eprintln!("Rebuilding index{}...", if no_embed { " (no embeddings)" } else { "" });
            scs.reindex_all(mode).await.context("Failed to reindex")?;
            eprintln!("Index rebuilt successfully");
        }

        Commands::Embed { batch } => {
            let missing = scs.missing_embeddings_count();
            if missing == 0 {
                eprintln!("All chunks already have embeddings");
                return Ok(());
            }

            // Auto-scale batch size based on API tier if not specified
            let batch_size = match batch {
                Some(b) => b,
                None => {
                    // Get optimal batch size based on rate limits (concurrency * 100)
                    let optimal = scs.get_optimal_batch_size().await.unwrap_or(5000);
                    eprintln!("[scs] Auto-scaled batch size: {}", optimal);
                    optimal
                }
            };

            eprintln!("Generating embeddings for {} chunks (batch size: {})...", missing, batch_size);
            let generated = scs.generate_embeddings(batch_size).await.context("Failed to generate embeddings")?;
            eprintln!("Generated {} embeddings", generated);
        }

        Commands::Map { max_tokens, area } => {
            // Ensure index is fresh (no embedding needed for map)
            scs.try_ensure_fresh(EmbedMode::Skip).await.context("Failed to refresh index")?;

            if scs.index.index.chunks.is_empty() {
                eprintln!("[scs] No symbols found in index");
                return Ok(());
            }

            let root_str = root.to_string_lossy().to_string();
            let dirs = map::build_hierarchy(&scs.index.index, &root_str);

            if let Some(ref area_str) = area {
                // Zoom-in mode: --area specified
                let zoom = map::detect_zoom_level(area_str, &dirs);
                let mut dirs = dirs;

                match zoom {
                    map::ZoomLevel::File | map::ZoomLevel::Directory => {
                        // Generate summaries only for area-scoped dirs/files
                        if let Err(e) = scs.ensure_area_summaries(&mut dirs, area_str).await {
                            if !save_context_search::is_quiet() {
                                eprintln!("[scs] Note: {}", e);
                            }
                        }
                    }
                    _ => {}
                }

                let config = map::MapConfig { max_tokens, area };
                let output = map::generate_map_from_dirs(dirs, &scs.index.index, &root_str, &config);

                if output.is_empty() {
                    eprintln!("[scs] No code symbols found in index");
                } else {
                    println!("{}", output);
                }
            } else {
                // No --area: full project map
                let use_tree = map::would_use_tree_mode(&dirs, max_tokens);

                if use_tree {
                    // Large project: tree mode with collapsed node summaries
                    let mut trees = map::build_dir_tree(&dirs);
                    for tree in &mut trees {
                        map::collapse_single_children(tree);
                    }

                    if let Err(e) = scs.ensure_tree_summaries(&mut trees, &dirs).await {
                        if !save_context_search::is_quiet() {
                            eprintln!("[scs] Note: {}", e);
                        }
                    }

                    let project_name = std::path::Path::new(&root_str)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("project");

                    let lang_counts = dirs.iter()
                        .flat_map(|d| &d.files)
                        .fold(std::collections::HashMap::new(), |mut m, f| {
                            *m.entry(f.lang).or_insert(0usize) += f.symbols.iter().map(|s| 1 + s.members.len()).sum::<usize>();
                            m
                        });
                    let primary_lang = lang_counts.into_iter()
                        .max_by_key(|(_, c)| *c)
                        .map(|(l, _)| l)
                        .unwrap_or(map::SourceLang::Other);
                    let total_symbols: usize = dirs.iter().map(|d| d.symbol_count).sum();

                    let output = map::render_tree_map(&trees, project_name, primary_lang, total_symbols, max_tokens);
                    println!("{}", output);
                } else {
                    // Small/medium project: flat mode with dir/file summaries
                    let mut dirs = dirs;
                    if let Err(e) = scs.ensure_map_summaries(&mut dirs).await {
                        if !save_context_search::is_quiet() {
                            eprintln!("[scs] Note: {}", e);
                        }
                    }

                    let config = map::MapConfig { max_tokens, area: None };
                    let output = map::generate_map_from_dirs(dirs, &scs.index.index, &root_str, &config);

                    if output.is_empty() {
                        eprintln!("[scs] No code symbols found in index");
                    } else {
                        println!("{}", output);
                    }
                }
            }
        }

        Commands::Summarize { batch, force, dry_run } => {
            // Ensure index is fresh
            scs.try_ensure_fresh(EmbedMode::Skip).await.context("Failed to refresh index")?;

            if dry_run {
                let stats = scs.summarize_dry_run();
                eprintln!("=== Summarize Dry Run ===");
                eprintln!("Total functions: {}", stats.total_functions);
                eprintln!("Levels: {}", stats.levels);
                eprintln!("Batch size: {}", batch);
                eprintln!("Estimated API calls: {}", (stats.total_functions + batch - 1) / batch);
                eprintln!();
                eprintln!("Run without --dry-run to generate summaries.");
                return Ok(());
            }

            eprintln!("Generating LLM summaries (batch size: {})...", batch);
            let stats = scs.generate_summaries(batch, force).await
                .context("Failed to generate summaries")?;

            eprintln!();
            eprintln!("=== Summary Statistics ===");
            eprintln!("Total functions: {}", stats.total_functions);
            eprintln!("Cached (skipped): {}", stats.cached);
            eprintln!("Summarized: {}", stats.summarized);
            eprintln!("Failed: {}", stats.failed);
            eprintln!("API calls: {}", stats.api_calls);
            eprintln!("Input tokens: {}", stats.input_tokens);
            eprintln!("Output tokens: {}", stats.output_tokens);
        }
    }

    Ok(())
}

/// Detect if the project is primarily Rust or TypeScript/JavaScript
/// Parse filter string to ChunkType
fn parse_filter(filter: &Option<String>) -> Result<Option<ChunkType>> {
    match filter {
        None => Ok(None),
        Some(s) => match s.to_lowercase().as_str() {
            "code" => Ok(Some(ChunkType::Code)),
            "doc" | "docs" => Ok(Some(ChunkType::Doc)),
            _ => anyhow::bail!("Invalid filter: '{}'. Use 'code' or 'doc'", s),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_filter_none() {
        let result = parse_filter(&None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_filter_code() {
        let result = parse_filter(&Some("code".to_string())).unwrap();
        assert_eq!(result, Some(ChunkType::Code));
    }

    #[test]
    fn test_parse_filter_doc() {
        let result = parse_filter(&Some("doc".to_string())).unwrap();
        assert_eq!(result, Some(ChunkType::Doc));
    }

    #[test]
    fn test_parse_filter_docs() {
        let result = parse_filter(&Some("docs".to_string())).unwrap();
        assert_eq!(result, Some(ChunkType::Doc));
    }

    #[test]
    fn test_parse_filter_case_insensitive() {
        let result = parse_filter(&Some("CODE".to_string())).unwrap();
        assert_eq!(result, Some(ChunkType::Code));

        let result = parse_filter(&Some("Doc".to_string())).unwrap();
        assert_eq!(result, Some(ChunkType::Doc));
    }

    #[test]
    fn test_parse_filter_invalid() {
        let result = parse_filter(&Some("invalid".to_string()));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid filter"));
        assert!(err.contains("invalid"));
    }

    #[test]
    fn verify_cli() {
        // Verify CLI parses correctly
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
