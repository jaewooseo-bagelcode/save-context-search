---
name: scs
description: "**[현재 프로젝트 탐색용]** 코드 검색, 심볼 찾기, 파일 구조 확인 시 사용. 'search code', 'find where X is implemented', 'lookup symbol', 'show file outline', 코드 탐색 요청 시 자동 호출. **중요: Grep/Glob/Read 대신 SCS를 먼저 사용 - 토큰 90%+ 절감.**"
version: 0.1.0
---

# SCS - Save Context Search (현재 프로젝트 전용)

**⚠️ 현재 작업 중인 프로젝트의 코드 탐색에 SCS를 우선 사용하세요.**

Semantic code search CLI로 glob→grep→read 워크플로우를 대체합니다.

**Binary:** `${CLAUDE_PLUGIN_ROOT}/bin/scs`

## 언제 SCS를 사용하나요?

| 기존 방식 (비효율) | SCS 방식 (권장) |
|------------------|----------------|
| `Glob **/*.rs` → `Grep "symbol"` → `Read` | `scs lookup "symbol"` |
| 여러 파일 Read로 구조 파악 | `scs outline "file.rs"` |
| Grep으로 의미 기반 검색 시도 | `scs search "user auth"` |

**토큰 절감:** ~9,200 → ~950 토큰 (90%+)

## Commands

Execute via Bash tool with full path.

### search - Semantic Search

Find code by meaning, not just text matching.

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs search "<query>" [--top N] [--filter code|doc]
```

**Examples:**
```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs search "user authentication"
${CLAUDE_PLUGIN_ROOT}/bin/scs search "error handling retry" --top 10
${CLAUDE_PLUGIN_ROOT}/bin/scs search "database connection" --filter code
```

**When to use:** Finding implementations by concept, locating related code, understanding how features work.

---

### lookup - Symbol Lookup

Find exact symbol definitions by name.

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "<name>" [--filter code|doc]
```

**Examples:**
```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "PlayerController"
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "PlayerController.Update"  # qualified name
${CLAUDE_PLUGIN_ROOT}/bin/scs lookup "handleRequest"
```

**When to use:** When the exact symbol name is known and its location/signature is needed.

---

### outline - File Structure

Get file structure without reading full content.

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs outline "<file_path>"
```

**Examples:**
```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs outline "src/main.rs"
${CLAUDE_PLUGIN_ROOT}/bin/scs outline "README.md"
```

**When to use:** Before reading a large file - inspect structure first, then read only relevant lines.

---

### status - Index Health

Check index statistics and health.

```bash
${CLAUDE_PLUGIN_ROOT}/bin/scs status
```

**When to use:** Verify index readiness, check embedding completion, debug search issues.

---

## Output Format

All commands return JSON for easy parsing.

## Token Efficiency

| Traditional | SCS |
|-------------|-----|
| glob **/*.rs → grep "symbol" → read files | scs lookup "symbol" |
| ~9,200 tokens | ~950 tokens |

## Requirements

- `OPENAI_API_KEY` environment variable for embeddings (semantic search)

## Additional Resources

For detailed usage and advanced patterns, consult:
- **`references/commands.md`** - Complete command reference with all options
