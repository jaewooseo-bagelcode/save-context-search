# SCS (Save Context Search) Skill Guide

Context-efficient semantic search CLI for Claude Code. Replaces glob→grep→read workflow with intelligent symbol extraction and semantic search, achieving 90%+ token savings.

## Quick Start

```bash
# Search (auto-selects semantic or name-based)
scs search "player movement" --top 5

# Exact symbol lookup
scs lookup PlayerController

# Find callers/callees
scs deps HandleMovement --direction both

# File structure
scs outline src/player.rs

# Index status
scs status
```

## Commands

| Command | Purpose | Needs Embeddings |
|---------|---------|------------------|
| `search <query>` | Semantic/name search | Auto-fallback |
| `lookup <name>` | Exact symbol lookup | No |
| `deps <name>` | Callers and callees | No |
| `outline <file>` | File structure | No |
| `status` | Index health | No |
| `refresh` | Incremental update | Optional |
| `embed` | Generate embeddings | N/A |

## Automatic Behaviors

### 1. Search Auto-Fallback

`scs search` automatically selects the best method:

```
임베딩 있음 → match_type: "semantic" (유사도 기반)
임베딩 없음 → match_type: "name_only" (이름 기반, 자동 폴백)
```

**Example output without embeddings:**
```json
{
  "query": "player movement",
  "match_type": "name_only",
  "results": [...],
  "suggestions": ["Using name-based search (500 chunks pending). Run 'scs embed' for semantic search."]
}
```

### 2. Adaptive Embedding

`scs refresh` automatically decides embedding strategy by chunk count:

| Chunks | Strategy | Time |
|--------|----------|------|
| ≤ 500 | Sync embedding | ~30s - 3min |
| > 500 | Skip + tip message | - |

Use `--sync` to force sync embedding (errors if >500 chunks).
Use `--no-embed` to skip embedding entirely.

### 3. Lock Handling

Read operations (`lookup`, `deps`, `outline`, `search`) work even when index is locked:

```
scs embed (background) → lock held
scs lookup Foo → [scs] Index locked by PID 123. Using cached data.
                 → Returns results from cache ✅
```

Write operations (`refresh`, `reindex`, `embed`) fail if locked:
```
scs refresh → Error: Index is locked by another process (PID: 123)
```

### 4. Batch-Level Lock Release

`scs embed` releases lock between batches, allowing other operations:

```
Batch 1 → save → unlock → yield → lock → Batch 2 → ...
```

## Status Fields

```json
{
  "chunk_count": 3000,
  "embedding_count": 3000,
  "has_embeddings": true,
  "pending_embeddings": 0,
  "embed_recommendation": "none",
  "embedder_available": true
}
```

| Field | Description |
|-------|-------------|
| `has_embeddings` | All chunks have embeddings |
| `pending_embeddings` | Chunks without embeddings |
| `embed_recommendation` | `none`, `sync`, `sync_acceptable`, `background` |

## Skill Integration Pattern

### Simple Pattern (Recommended)

Just use `scs search` - it handles everything automatically:

```markdown
## Search Codebase

1. Run search:
   ```bash
   scs search "<query>" --top 5
   ```

2. Check `match_type` in results:
   - `semantic`: High-quality similarity search
   - `name_only`: Name-based fallback (still useful)

3. For exact symbols, use:
   ```bash
   scs lookup "ClassName.MethodName"
   ```

4. For call relationships:
   ```bash
   scs deps "FunctionName" --direction both
   ```
```

## Output Formats

### Search Result

```json
{
  "query": "error handling",
  "match_type": "semantic",
  "results": [
    {
      "name": "handleError",
      "kind": "function",
      "file": "/src/utils.ts",
      "line_start": 45,
      "line_end": 60,
      "score": 0.82,
      "preview": "function handleError(err: Error) {\n  ...",
      "context": "ErrorUtils",
      "unique": true
    }
  ],
  "suggestions": []
}
```

### Lookup Result

```json
{
  "name": "PlayerController",
  "match_type": "exact",
  "confidence": "high",
  "definitions": [
    {
      "kind": "class",
      "file": "/src/player.ts",
      "line_start": 10,
      "line_end": 150,
      "signature": "class PlayerController {",
      "preview": "..."
    }
  ],
  "suggestions": []
}
```

### Deps Result

```json
{
  "name": "handleMovement",
  "definition": {...},
  "references": [
    {
      "file": "/src/game.ts",
      "line": 45,
      "kind": "call",
      "context": "GameLoop",
      "snippet": "this.handleMovement(input)"
    }
  ],
  "calls": [
    {"name": "validateInput", "line": 12},
    {"name": "applyPhysics", "line": 15}
  ]
}
```

## Confidence Levels

| Confidence | Meaning | Action |
|------------|---------|--------|
| `high` | Unique symbol, exact match | Use directly |
| `medium` | Multiple but distinguishable | Review definitions |
| `low` | Many homonyms | Use `Class.Method` notation |

## Supported Languages

**Code** (tree-sitter): Rust, TypeScript, JavaScript, C#, Python

**Docs**: Markdown (.md, .mdx), Plain text (.txt)

## Cache Location

```
.scs/
├── index.bin       # All metadata (bincode, includes chunks, files, call_graph, string_table)
├── embeddings.bin  # Vector embeddings
└── lock            # Process lock (when indexing)
```

## Environment

```bash
export OPENAI_API_KEY=sk-...  # Required for semantic search
```

Without API key, `scs search` falls back to name-based search automatically.

## Performance

| Operation | Time |
|-----------|------|
| Incremental refresh | 1-2 seconds |
| Full index (3000 files, no embed) | ~45 seconds |
| Embedding 100 chunks | ~30 seconds |
| Search query | < 1 second |
| Lookup/Deps/Outline | Instant |
