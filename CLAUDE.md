# CLAUDE.md

## What is this

**scs** — Claude Code 플러그인으로 동작하는 코드베이스 탐색 CLI. 프로젝트 맵, 의미 검색, 심볼 룩업을 제공하여 glob→grep→read 대비 토큰 90%+ 절감.

Rust CLI + tree-sitter + OpenAI embeddings + Gemini Flash summaries.

## Build & Test

```bash
cargo build --release    # target/release/scs (LTO + strip)
cargo test               # 89 tests
```

환경변수: `OPENAI_API_KEY` (embeddings), `GEMINI_API_KEY` (LLM summaries)

## 소스 구조

```
src/
├── main.rs              # CLI 진입점: map, search, lookup + 유지보수 명령
├── lib.rs               # SCS 코어: 인덱싱 오케스트레이션, ensure_*_summaries()
├── map.rs               # 프로젝트 맵: DirNode/FileNode/DirTree, 줌인, 렌더링
├── index/manager.rs     # 증분 인덱싱, 파일 스캔, lock/dirty 메커니즘
├── parser/
│   ├── code.rs          # tree-sitter 파싱 (Rust/TS/JS/C#/Python)
│   ├── docs.rs          # 문서 파싱 (MD/JSON/YAML/TOML/XML/TXT)
│   └── queries.rs       # tree-sitter 쿼리 상수 (언어별)
├── search/
│   ├── semantic.rs      # 코사인 유사도 검색 (simsimd)
│   └── lookup.rs        # 이름 기반 심볼 룩업
├── embeddings/openai.rs # OpenAI batch embedding + rate limit
└── summarizer/
    ├── mod.rs           # SummaryCache (.scs/summaries.json)
    ├── gemini.rs        # Gemini API 클라이언트 + batch prompt
    └── levels.rs        # 함수→요약 레벨 기반 배치 처리
```

```
plugin/
├── .claude-plugin/plugin.json   # 플러그인 매니페스트
├── bin/scs                      # 번들 바이너리 (심볼릭 링크)
├── hooks/
│   ├── hooks.json               # SessionStart + UserPromptSubmit
│   └── session-start.sh         # refresh + map 주입
└── skills/scs/
    ├── SKILL.md                 # 에이전트 워크플로우 가이드
    └── references/commands.md   # 명령 레퍼런스
```

## 핵심 데이터 흐름

```
파일 → tree-sitter 파싱 → Chunk(심볼) → StringTable 인터닝 → Index(bincode)
                                              ↓
                                    OpenAI embedding → embeddings.bin
                                    Gemini summary  → summaries.json
```

### CLI 명령 (3개 핵심 + 유지보수)

| 명령 | 용도 |
|------|------|
| `scs map [--area <path>]` | 프로젝트 구조 맵 + 줌인 (핵심) |
| `scs search <query>` | 임베딩 유사도 검색 |
| `scs lookup <name>` | 정확한 심볼 위치 |
| `scs refresh/reindex/embed/summarize/status` | 인덱스 유지보수 |

## 코드 수정 시 알아야 할 것

### StringTable (문자열 인터닝)
모든 Chunk의 파일 경로, 심볼 이름은 `u32` 인덱스로 저장. `strings.intern("name")` → `u32`, `strings.get(idx)` → `&str`. 메모리 효율 + 빠른 비교.

### map.rs 렌더링 패턴
- `build_hierarchy()` → `Vec<DirNode>` (index → 계층 구조)
- 소형 프로젝트: `generate_map_from_dirs()` (flat mode, dir/file 요약)
- 대형 프로젝트: `build_dir_tree()` → `render_tree_map()` (tree mode, collapsing)
- 줌인: `render_with_area()` + `ZoomLevel` enum (Project/Directory/File)
- 출력 형식: `// comment-style` (토큰 효율적, JSON 아님)

### SummaryCache
`.scs/summaries.json` — 모든 LLM 요약 저장. 키: `file::path`, `dir::path`, `tree::path`, `func_name`. body_hash로 무효화. FORMAT_VERSION=3.

### 인덱싱 동시성
- `lock` 파일로 동시 접근 방지
- `dirty` 파일로 지연 갱신 (lock 중 요청 → dirty 마킹 → 다음 기회에 처리)
- `try_ensure_fresh()`: lock이면 캐시 사용, `ensure_fresh()`: 대기

### 언어 추가 시
1. `Cargo.toml`에 `tree-sitter-{lang}` 추가
2. `parser/queries.rs`에 쿼리 상수 추가
3. `parser/code.rs`의 `SupportedLanguage` enum + `parse_with_queries()` 분기

### 테스트 패턴
각 모듈 하단에 `#[cfg(test)] mod tests`. `create_test_index()` 헬퍼로 테스트용 Index 생성. `tempfile` 크레이트로 임시 디렉토리.
