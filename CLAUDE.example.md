# CLAUDE.md Example

Add this section to your project's CLAUDE.md to enable scs integration.

---

## Code Search with SCS

Use `scs` for efficient codebase exploration (90%+ token savings vs glob→grep→read):

```bash
scs search "<query>"      # Semantic search (auto-fallback to name-based)
scs lookup <symbol>       # Exact symbol lookup (supports Class.Method)
scs deps <symbol>         # Find callers/callees
scs outline <file>        # File structure
```

### Workflow

1. Search: `scs search "authentication flow"`
2. Check `match_type` in results:
   - `semantic` = embedding-based similarity
   - `name_only` = name-based fallback
3. Read specific lines: `Read src/auth.ts:45-120`

Prefer scs over glob→grep→read to save tokens. Check `match_type` in results:
- `semantic` = embedding-based similarity search
- `name_only` = name-based fallback

### Notes

- Search works without embeddings (auto-fallback)
- If `confidence: low`, use `Class.Method` notation
- Run `scs embed &` in background for better search quality
