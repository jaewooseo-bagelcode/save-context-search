# SCS (Save Context Search)

Context-efficient semantic search CLI for Claude Code that reduces token consumption by **90%+** when exploring codebases.

## Why SCS?

Traditional codebase exploration in Claude Code:
```
glob **/*.rs → grep "symbol" → read files → ~9,200 tokens
```

With SCS:
```
scs lookup "symbol" → ~950 tokens
```

SCS replaces the glob→grep→read workflow with intelligent symbol extraction and semantic search.

## Features

- **Semantic Search** - Find code by meaning, not just text matching
- **Symbol Lookup** - Exact symbol definitions with confidence levels
- **File Outline** - View file structure without reading full content
- **Incremental Indexing** - Only re-indexes changed files (mtime-based)
- **Non-blocking Refresh** - Dirty flag mechanism for concurrent access
- **Claude Code Plugin** - Auto-refresh hooks and Grep/Glob interception

## Installation

### Prerequisites

- Rust toolchain (for building)
- OpenAI API key (for semantic search embeddings)

### Build

```bash
git clone https://github.com/jaewooseo/save-context-search.git
cd save-context-search
./build-plugin.sh
```

### Install as Claude Code Plugin

The plugin is located in `./plugin` directory after building.

**Option 1: Use --plugin-dir (for testing)**
```bash
claude --plugin-dir ./plugin
```

**Option 2: Add to local marketplace**

Create `~/.claude-plugin/marketplace.json`:
```json
{
  "name": "local",
  "owner": { "name": "Your Name", "email": "you@email.com" },
  "metadata": { "description": "Local plugins", "version": "1.0.0" },
  "plugins": [
    {
      "name": "scs",
      "source": "/path/to/save-context-search/plugin",
      "description": "Context-efficient semantic search"
    }
  ]
}
```

Then enable via `/plugin` in Claude Code.

### Environment Variables

```bash
export OPENAI_API_KEY=sk-...  # Required for semantic search
```

## Usage

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `scs search` | Semantic search | `scs search "user authentication"` |
| `scs lookup` | Symbol lookup | `scs lookup "PlayerController"` |
| `scs outline` | File structure | `scs outline src/main.rs` |
| `scs status` | Index health | `scs status` |
| `scs refresh` | Incremental update | `scs refresh` |
| `scs embed` | Generate embeddings | `scs embed` |
| `scs reindex` | Full rebuild | `scs reindex` |

### Search

Find code by meaning:

```bash
scs search "handle user login"
scs search "error handling retry" --top 10
scs search "database connection" --filter code
```

### Lookup

Find exact symbol definitions:

```bash
scs lookup "PlayerController"
scs lookup "Player.Update"  # Qualified name for disambiguation
```

### Outline

View file structure without reading content:

```bash
scs outline src/main.rs
```

## Plugin Features

When installed as a Claude Code plugin, SCS provides:

### Auto-Refresh Hooks

- **SessionStart** - Index refreshes when Claude Code starts
- **UserPromptSubmit** - Index refreshes before each prompt

### Grep/Glob Interception

PreToolUse hook suggests SCS before using Grep or Glob:

```
"Before using Grep or Glob for code exploration, consider if SCS would be more efficient..."
```

## Supported Languages

**Code** (via tree-sitter):
- Rust (.rs)
- TypeScript/TSX (.ts, .tsx)
- JavaScript/JSX (.js, .jsx)
- Python (.py)
- C# (.cs)

**Documentation**:
- Markdown (.md, .mdx)
- Plain text (.txt)

## Architecture

```
save-context-search/
├── src/
│   ├── main.rs              # CLI entry
│   ├── lib.rs               # Core logic + SCS struct
│   ├── index/manager.rs     # Incremental indexing
│   ├── parser/              # tree-sitter + markdown parsing
│   ├── embeddings/          # OpenAI API client
│   └── search/              # Semantic + lookup search
└── plugin/
    ├── .claude-plugin/
    │   ├── plugin.json      # Plugin manifest
    │   └── marketplace.json # For local marketplace
    ├── bin/scs              # Bundled binary
    ├── hooks/hooks.json     # Auto-refresh + PreToolUse
    └── skills/scs/          # Skill definition
```

## How It Works

1. **Indexing** - Parses code with tree-sitter, extracts symbols (functions, classes, methods)
2. **Embedding** - Generates OpenAI text-embedding-3-small vectors (1,536 dims)
3. **Search** - Cosine similarity search via simsimd
4. **Caching** - Stores index in `.scs/` directory with mtime-based invalidation

## Token Efficiency

| Operation | Traditional | SCS |
|-----------|-------------|-----|
| Find symbol definition | ~9,200 tokens | ~950 tokens |
| Explore file structure | ~2,000 tokens | ~100 tokens |
| Search implementations | ~15,000 tokens | ~1,500 tokens |

## Configuration

Index stored in `.scs/` directory:
- `index.bin` - Indexed chunks and metadata
- `embeddings.bin` - Vector embeddings
- `lock` - Process lock file
- `dirty` - Dirty flag for deferred refresh

## Development

```bash
cargo build           # Debug build
cargo build --release # Release build
cargo test            # Run tests
cargo clippy          # Lint
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
