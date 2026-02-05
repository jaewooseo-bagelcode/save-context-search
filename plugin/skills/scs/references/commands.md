# SCS Command Reference

Complete reference for all SCS commands with options, output formats, and advanced usage.

## search

Semantic search across indexed code and documentation.

### Synopsis

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs search "<query>" [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--top N` | Number of results to return | 5 |
| `--filter code\|doc` | Filter by chunk type | all |
| `--path <PATH>` | Project root directory | current directory |

### Output Format

```json
{
  "query": "user authentication",
  "match_type": "semantic",
  "results": [
    {
      "chunk_type": "code",
      "name": "authenticate_user",
      "kind": "function",
      "file": "src/auth.rs",
      "line_start": 45,
      "line_end": 78,
      "score": 0.89,
      "preview": "pub fn authenticate_user(credentials: &Credentials) -> Result<User>",
      "context": "AuthModule",
      "unique": true
    }
  ],
  "suggestions": []
}
```

### Match Types

- `semantic` - Embedding-based similarity search (requires OPENAI_API_KEY)
- `name_only` - Fallback text matching when embeddings unavailable

### Examples

```bash
# Basic semantic search
${CLAUDE_PLUGIN_ROOT}/bin/scs search "handle user login"

# Limit results
${CLAUDE_PLUGIN_ROOT}/bin/scs search "error handling" --top 3

# Search only code (exclude docs)
${CLAUDE_PLUGIN_ROOT}/bin/scs search "database connection pool" --filter code

# Search only documentation
${CLAUDE_PLUGIN_ROOT}/bin/scs search "API usage guide" --filter doc
```

---

## lookup

Find symbol definitions by exact name match.

### Synopsis

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "<name>" [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--filter code\|doc` | Filter by chunk type | all |
| `--path <PATH>` | Project root directory | current directory |

### Qualified Names

Use dot notation to disambiguate symbols with same name:

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "Player.Update"      # Method Update in Player class
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "Enemy.Update"       # Method Update in Enemy class
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "GameManager.Start"  # Start method in GameManager
```

### Output Format

```json
{
  "name": "PlayerController",
  "confidence": "high",
  "definitions": [
    {
      "chunk_type": "code",
      "kind": "class",
      "file": "src/player.rs",
      "line_start": 10,
      "line_end": 150,
      "signature": "pub struct PlayerController",
      "context": null
    }
  ],
  "suggestions": []
}
```

### Confidence Levels

- `high` - Unique symbol, single definition found
- `medium` - Multiple definitions but distinguishable by context
- `low` - Many homonyms, use qualified name for disambiguation

### Examples

```bash
# Find a class
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "UserService"

# Find a function
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "parse_config"

# Disambiguate with qualified name
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "Config.load"

# Search only in code files
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "README" --filter doc
```

---

## outline

Display file structure without reading full content.

### Synopsis

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs outline "<file_path>" [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--path <PATH>` | Project root directory | current directory |

### Output Format

```json
{
  "file": "src/main.rs",
  "symbols": [
    {
      "name": "main",
      "kind": "function",
      "line_start": 15,
      "line_end": 45,
      "signature": "fn main() -> Result<()>"
    },
    {
      "name": "Config",
      "kind": "struct",
      "line_start": 50,
      "line_end": 65,
      "signature": "pub struct Config"
    }
  ]
}
```

### Supported File Types

**Code:**
- Rust (.rs)
- TypeScript/TSX (.ts, .tsx)
- JavaScript/JSX (.js, .jsx)
- Python (.py)
- C# (.cs)

**Documentation:**
- Markdown (.md, .mdx)
- Plain text (.txt)

### Examples

```bash
# Outline a source file
${CLAUDE_PLUGIN_ROOT}/bin/scs outline "src/lib.rs"

# Outline a markdown document
${CLAUDE_PLUGIN_ROOT}/bin/scs outline "docs/API.md"

# Absolute path
${CLAUDE_PLUGIN_ROOT}/bin/scs outline "/path/to/project/src/main.rs"
```

---

## status

Check index health and statistics.

### Synopsis

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs status [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--path <PATH>` | Project root directory | current directory |

### Output Format

```json
{
  "root": "/path/to/project",
  "chunk_count": 1234,
  "embedding_count": 1234,
  "file_count": 89,
  "code_chunks": 1100,
  "doc_chunks": 134
}
```

### Health Indicators

- `chunk_count == embedding_count` - Embeddings complete, semantic search available
- `embedding_count == 0` - Run `${CLAUDE_PLUGIN_ROOT}/bin/scs embed` to enable semantic search
- `embedding_count < chunk_count` - Partial embeddings, run `${CLAUDE_PLUGIN_ROOT}/bin/scs refresh`

---

## refresh

Update index incrementally (changed files only).

### Synopsis

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs refresh [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-q, --quiet` | Suppress output | false |
| `--no-embed` | Skip embedding generation | false |
| `--sync` | Force synchronous embedding | false |
| `--path <PATH>` | Project root directory | current directory |

### Behavior

- Detects changed files by mtime
- Non-blocking: if another process holds lock, marks dirty and exits
- Dirty flag mechanism ensures eventual consistency

### Examples

```bash
# Standard refresh
${CLAUDE_PLUGIN_ROOT}/bin/scs refresh

# Quiet mode (for hooks)
${CLAUDE_PLUGIN_ROOT}/bin/scs refresh --quiet

# Refresh without updating embeddings
${CLAUDE_PLUGIN_ROOT}/bin/scs refresh --no-embed
```

---

## embed

Generate embeddings for indexed chunks.

### Synopsis

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs embed [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --batch <N>` | Batch size for API calls | auto-scaled |
| `--path <PATH>` | Project root directory | current directory |

### Examples

```bash
# Generate embeddings (auto batch size)
${CLAUDE_PLUGIN_ROOT}/bin/scs embed

# Custom batch size
${CLAUDE_PLUGIN_ROOT}/bin/scs embed --batch 50
```

---

## reindex

Force full reindex (rebuild from scratch).

### Synopsis

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs reindex [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--no-embed` | Skip embedding generation | false |
| `--path <PATH>` | Project root directory | current directory |

### When to Use

- After major codebase restructuring
- When index appears corrupted
- To reset embedding state

### Examples

```bash
# Full reindex with embeddings
${CLAUDE_PLUGIN_ROOT}/bin/scs reindex

# Reindex without embeddings (faster)
${CLAUDE_PLUGIN_ROOT}/bin/scs reindex --no-embed
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | For semantic search |

## Index Location

Index files stored in `.scs/` directory:
- `index.bin` - Indexed chunks and metadata
- `embeddings.bin` - Vector embeddings
- `lock` - Process lock file
- `dirty` - Dirty flag for deferred refresh
