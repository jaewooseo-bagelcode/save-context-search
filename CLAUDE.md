# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**save-context-search (scs)** is a context-efficient semantic search CLI tool for Claude Code that reduces token consumption by 90%+ when exploring codebases. It replaces the glob→grep→read workflow with intelligent symbol extraction and semantic search.

**Status**: Pre-release research implementation. The specification is in `coderag-spec.md` (1,596 lines).

**Tech Stack**: Rust CLI + tree-sitter (code parsing) + OpenAI text-embedding-3-small + simsimd (vector similarity)

## Build Commands

```bash
cargo build --release    # Build optimized binary (output: target/release/scs)
cargo build              # Debug build
cargo test               # Run all tests
cargo clippy             # Lint
cargo fmt                # Format
```

The release profile uses LTO and stripping for minimal binary size.

## Environment Setup

```bash
export OPENAI_API_KEY=sk-...   # Required for embeddings
```

## Architecture

```
save-context-search/
├── coderag-spec.md              # Complete specification (canonical source)
├── plugin/                      # Claude Code plugin (skills, hooks, bundled binary)
└── src/
    ├── main.rs                  # CLI entry (7 subcommands)
    ├── lib.rs                   # Core data structures
    ├── index/manager.rs         # Lazy incremental indexing
    ├── parser/code.rs           # tree-sitter code parsing
    ├── parser/docs.rs           # Document parsing (MD, JSON, YAML, TOML, XML)
    ├── embeddings/openai.rs     # OpenAI batch embedding
    └── search/semantic.rs       # Cosine similarity search
```

## CLI Commands (per specification)

| Command | Purpose | Example |
|---------|---------|---------|
| `scs search <query>` | Semantic search | `scs search "player movement" --top 5` |
| `scs lookup <name>` | Exact symbol/doc lookup | `scs lookup "PlayerController"` |
| `scs deps <name>` | References & calls | `scs deps "HandleMovement" --direction calls` |
| `scs outline <file>` | File structure (replaces full read) | `scs outline src/player.rs` |
| `scs status` | Index health | `scs status` |
| `scs refresh` | Incremental update | `scs refresh --quiet` |
| `scs reindex` | Force full rebuild | `scs reindex --path .` |

## Core Data Model

**Unified Chunk**: Single structure for code symbols and doc sections
- `ChunkType`: Code | Doc
- `ChunkKind`: Class, Function, Method, Section, Document, etc.
- Includes: file, line_start, line_end, content, context (parent class/section), signature

**Confidence Levels** (for lookup results):
- High: Unique symbol, one definition
- Medium: Multiple definitions but distinguishable
- Low: Many homonyms, needs confirmation (use `ClassName.MethodName` to qualify)

## Supported Languages & Formats

**Code** (via tree-sitter): Rust (.rs), TypeScript/TSX, JavaScript/JSX, C# (.cs), Python (.py)
**Docs**: Markdown (.md, .mdx), JSON, YAML (.yaml, .yml), TOML, XML, RST, Plain text (.txt)
**Ignored**: node_modules, target, dist, build, .git, .scs

## Caching

Cache location: `.scs/` directory
- `meta.json` - Index metadata
- `chunks.json` - Parsed chunks
- `file_hashes.json` - mtime + SHA256 tracking
- `embeddings.bin` - Binary vectors (bincode)

Lazy incremental indexing: Only re-indexes modified files (mtime-based).

## Key Implementation Details

**Embedding**: text-embedding-3-small (1,536 dims), batch size 100, 30s timeout, exponential backoff on rate limits

**Search**: Cosine similarity via simsimd, returns top-K with scores

**Reference Resolution**: Name-based only (no type inference). Use dot notation for disambiguation: `Player.Move` instead of just `Move`

## Workflow Example

Instead of:
```
glob **/*.rs → grep "symbol" → Read entire files (9,200+ tokens)
```

Use:
```bash
scs lookup "PlayerController"  # Returns exact location (~950 tokens)
Read /path/to/file:15-120      # Read only needed lines
```

