# SCS (Save Context Search)

Claude Code 플러그인 — 프로젝트 맵, 의미 검색, 심볼 룩업으로 코드 탐색 토큰을 **90%+** 절감.

## Why SCS?

기존 Claude Code 코드 탐색:
```
glob **/*.rs → grep "symbol" → read files → ~9,200 tokens
```

SCS:
```
scs map → scs lookup "symbol" → read lines → ~950 tokens
```

## Features

- **Project Map** — 프로젝트 구조를 LLM 요약과 함께 한눈에 파악. 디렉토리/파일 단위 줌인 지원
- **Semantic Search** — 임베딩 기반 의미 검색으로 이름을 모르는 코드도 찾기
- **Symbol Lookup** — 정확한 심볼 위치(file:line) + 시그니처 반환
- **Incremental Indexing** — mtime 기반 변경 파일만 재인덱싱
- **Auto-refresh** — 매 프롬프트마다 인덱스 자동 갱신 (11ms~200ms)

## Installation

### Prerequisites

- Rust toolchain
- `OPENAI_API_KEY` — 임베딩 생성
- `GEMINI_API_KEY` — LLM 요약 (map 기능)

### Build & Install

```bash
git clone https://github.com/jaewooseo/save-context-search.git
cd save-context-search
cargo build --release
cp target/release/scs plugin/bin/scs
```

### Claude Code Plugin 등록

```bash
# Option 1: 직접 지정
claude --plugin-dir ./plugin

# Option 2: /plugins 명령으로 설치
```

## Usage

### Agent Workflow

세션 시작 시 프로젝트 맵이 자동 주입됩니다:

```
① Orient   — 맵으로 프로젝트 구조 파악 (자동)
② Navigate — scs map --area <dir>  → 관심 모듈로 줌인
③ Locate   — scs lookup <name>     → 정확한 위치
             scs search <concept>  → 의미 검색
④ Read     — Read file:L1-L2       → 필요한 줄만
```

### Commands

#### map — 프로젝트 맵 + 줌인

```bash
scs map                          # 전체 프로젝트 구조
scs map --area src/parser        # 디렉토리 줌인: 파일 + 요약
scs map --area src/parser/code.rs  # 파일 줌인: 함수 + 시그니처
```

출력 예시:
```
// Project: my-project (Rust, 367 symbols)
// src/parser/  — 3 files, 15 symbols (Code and doc parsing)
//   code.rs  — 8 — Parses source code via tree-sitter
//   docs.rs  — 5 — Document format parsing
```

#### search — 의미 검색

```bash
scs search "handle user login"
scs search "error handling" --top 10
scs search "database connection" --filter code
```

#### lookup — 심볼 위치

```bash
scs lookup "PlayerController"
scs lookup "Player.Update"       # ClassName.Method으로 동명 구분
```

#### Maintenance

```bash
scs refresh          # 증분 갱신 (자동 실행됨)
scs reindex          # 전체 재인덱싱
scs embed            # 임베딩 생성
scs summarize        # 함수 LLM 요약 생성
scs status           # 인덱스 상태
```

## Plugin Architecture

```
plugin/
├── .claude-plugin/
│   ├── plugin.json          # 매니페스트 (v0.2.0)
│   └── marketplace.json     # 마켓플레이스 등록
├── bin/scs                  # 번들 바이너리
└── hooks/
    ├── hooks.json           # SessionStart + UserPromptSubmit
    └── session-start.sh     # refresh + map + workflow 주입
```

### Hooks

| Event | 동작 | 시간 |
|-------|------|------|
| **SessionStart** | `refresh` + `map` + workflow 가이드 주입 | ~800ms |
| **UserPromptSubmit** | `refresh --quiet` (인덱스 갱신) | 11~200ms |

## Supported Languages

**Code** (tree-sitter): Rust, TypeScript/TSX, JavaScript/JSX, Python, C#

**Docs**: Markdown, JSON, YAML, TOML, XML, Plain text

## How It Works

1. **Parse** — tree-sitter로 코드 심볼 추출 (함수, 클래스, 메서드)
2. **Index** — StringTable 인터닝 + bincode 직렬화 (`.scs/index.bin`)
3. **Embed** — OpenAI text-embedding-3-small (1,536차원)
4. **Summarize** — Gemini Flash로 dir/file/function 요약 생성
5. **Map** — 계층 구조 렌더링 (소형: flat mode, 대형: tree mode + collapsing)
6. **Search** — simsimd 코사인 유사도 검색

## Performance

| 프로젝트 | 심볼 수 | map 모드 | 첫 실행 | 캐시 |
|---------|---------|---------|---------|------|
| SCS | 367 | flat | 2.8s | 10ms |
| RumiBot | 767 | flat | 3s | 10ms |
| SyntaxOS | 1.7K | flat | 12s | 10ms |
| upg-client | 49K | tree | 57s | 0.2s |
| PBI | 32K | tree | 44s | 1.4s |

## Cache

```
.scs/
├── index.bin        # 심볼 인덱스 (bincode)
├── embeddings.bin   # 벡터 임베딩 (bincode)
├── summaries.json   # LLM 요약 캐시
├── lock             # 프로세스 락
└── dirty            # 지연 갱신 플래그
```

## Development

```bash
cargo build           # Debug build
cargo build --release # Release build (LTO + strip)
cargo test            # 89 tests
cargo clippy          # Lint
```

## License

MIT
