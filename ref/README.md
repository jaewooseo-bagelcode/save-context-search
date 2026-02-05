# DocsRAG - 문서 기반 RAG 시스템

프로젝트 문서를 자동으로 인덱싱하고 시맨틱 검색을 제공하는 Rust 모듈입니다.

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DocsRAG System                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  mod.rs  │───▶│ store.rs │───▶│embedder.rs│───▶│  OpenAI  │      │
│  │ (Entry)  │    │ (Vector) │    │  (API)   │    │   API    │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│       │               │                                              │
│       │               │                                              │
│       ▼               ▼                                              │
│  ┌──────────┐    ┌──────────┐                                       │
│  │watcher.rs│    │chunker.rs│                                       │
│  │(FileWatch)│    │ (Parse)  │                                       │
│  └──────────┘    └──────────┘                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 파일 구조

```
docs_rag/
├── mod.rs       # 모듈 진입점, Tauri 명령어 정의
├── store.rs     # 벡터 저장소, 캐시, 검색 로직
├── embedder.rs  # OpenAI Embedding API 클라이언트
├── chunker.rs   # 문서 청킹 (MD, JSON, YAML, TOML, XML)
└── watcher.rs   # 파일 시스템 감시 (실시간 업데이트)
```

## 핵심 컴포넌트

### 1. mod.rs (진입점)

**주요 기능:**
- Tauri 명령어 등록 (`#[tauri::command]`)
- 전역 상태 관리 (STORES, WATCHERS, QUERY_CACHE)
- 프로젝트별 DocsRAG 인스턴스 관리

**Tauri 명령어:**
| 명령어 | 설명 |
|--------|------|
| `init_docs_rag(project_path)` | 동기 초기화 |
| `auto_init_docs_rag(project_path)` | 비동기 백그라운드 초기화 |
| `search_docs(project_path, query, top_k)` | 시맨틱 검색 |
| `get_docs_rag_status(project_path)` | 상태 조회 |
| `stop_docs_rag(project_path)` | 리소스 정리 |

**이벤트:**
- `docs-rag:status` - 인덱싱 상태 변경 시 발행

### 2. store.rs (벡터 저장소)

**핵심 구조체:**
```rust
pub struct DocsRagStore {
    project_path: String,
    chunks: Vec<Chunk>,           // 문서 청크
    embeddings: Vec<Vec<f32>>,    // 벡터 임베딩
    last_indexed: Option<u64>,    // 마지막 인덱싱 시간
    query_cache: Mutex<QueryCache>, // 쿼리 캐시
}
```

**캐시 시스템:**
- 위치: `{project_path}/.syntaxos/docs-rag/`
- 파일:
  - `embeddings.bin` - 바이너리 임베딩 데이터
  - `metadata.json` - 청크 메타데이터
  - `index.json` - 파일 해시 (mtime 기반)

**검색 방식:**
1. **시맨틱 검색** - 임베딩이 있으면 코사인 유사도 사용
2. **텍스트 폴백** - 임베딩 없으면 단순 substring 매칭

**제외 디렉토리:**
```rust
const EXCLUDED_DIRS: &[&str] = &[
    "node_modules", "target", "dist", "build", ".git",
    ".syntaxos", "__pycache__", "vendor", ".next", ".nuxt", "coverage"
];
```

### 3. embedder.rs (임베딩 API)

**설정:**
| 상수 | 값 | 설명 |
|------|-----|------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI 모델 |
| `EMBEDDING_DIM` | 1536 | 벡터 차원 |
| `BATCH_SIZE` | 100 | 배치 크기 |
| `MAX_ATTEMPTS` | 3 | 재시도 횟수 |
| `REQUEST_TIMEOUT_SECS` | 30 | 타임아웃 |

**환경 변수:**
```bash
OPENAI_API_KEY=sk-xxx  # 필수
```

**재시도 로직:**
- Rate Limit (429) 시 지수 백오프
- 타임아웃 시 재시도

### 4. chunker.rs (문서 청킹)

**지원 포맷:**
| 확장자 | 청킹 전략 |
|--------|----------|
| `.md`, `.rst`, `.txt` | 헤더 기준 분할 (`#`, `##`, `###`) |
| `.json` | 최상위 키 기준 분할 |
| `.yaml`, `.yml` | 들여쓰기 없는 라인 기준 분할 |
| `.toml` | `[section]` 헤더 기준 분할 |
| `.xml` | 고정 크기 분할 |
| 기타 | 고정 크기 + 오버랩 분할 |

**청킹 파라미터:**
```rust
const CHUNK_SIZE: usize = 500;      // 문자 수
const CHUNK_OVERLAP: usize = 100;   // 오버랩
const MAX_FILE_SIZE: usize = 1_000_000;  // 1MB 제한
```

### 5. watcher.rs (파일 감시)

**동작 방식:**
1. `notify` crate로 파일 시스템 이벤트 감지
2. 500ms 디바운싱으로 중복 이벤트 병합
3. 변경 감지 시 `incremental_update` 호출

**감시 확장자:**
```rust
const WATCHED_EXTENSIONS: &[&str] = &[
    "md", "json", "yaml", "yml", "toml", "xml", "txt", "rst"
];
```

**이벤트 타입:**
- `Modified` - 파일 수정/생성
- `Deleted` - 파일 삭제

## 데이터 흐름

### 초기화 흐름

```
auto_init_docs_rag(project_path)
    │
    ▼
DocsRagStore::new()
    │
    ├─▶ index_project()
    │       │
    │       ├─▶ glob 패턴으로 파일 수집
    │       ├─▶ Chunker::chunk_file() 각 파일 청킹
    │       └─▶ load_or_compute_embeddings()
    │               │
    │               ├─▶ 캐시 로드 시도 (load_cache)
    │               │       └─▶ mtime 검증
    │               │
    │               └─▶ 캐시 미스 시 compute_and_cache_embeddings()
    │                       └─▶ Embedder::embed_batch()
    │
    ▼
FileWatcher::new() - 실시간 감시 시작
```

### 검색 흐름

```
search_docs(project_path, query, top_k)
    │
    ▼
임베딩 존재 여부 확인
    │
    ├─▶ 임베딩 있음: get_or_compute_query_embedding()
    │       │
    │       ├─▶ QUERY_CACHE 확인
    │       └─▶ 캐시 미스 시 Embedder::embed_text()
    │               │
    │               ▼
    │       search_with_embedding() - 코사인 유사도
    │
    └─▶ 임베딩 없음: search_text_fallback() - substring 매칭
```

### 증분 업데이트 흐름

```
FileWatcher 이벤트 감지
    │
    ▼
debounce_handler (500ms 대기)
    │
    ▼
incremental_update(modified, added, deleted)
    │
    ├─▶ 삭제/수정 파일의 기존 청크 제거
    ├─▶ 수정/추가 파일 재청킹
    ├─▶ 새 청크 임베딩 계산
    └─▶ 캐시 저장
```

## 의존성

### Cargo.toml 추가 필요

```toml
[dependencies]
# 비동기 런타임
tokio = { version = "1", features = ["full"] }

# HTTP 클라이언트
reqwest = { version = "0.11", features = ["json"] }

# 직렬화
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# 파일 시스템 감시
notify = "6"

# 파일 패턴 매칭
glob = "0.3"

# 전역 상태
once_cell = "1"

# (테스트용)
mockito = "1"
```

## 이식 가이드

### 1. 모듈 등록

```rust
// lib.rs 또는 main.rs
pub mod docs_rag;

// Tauri 앱에서
fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            docs_rag::init_docs_rag,
            docs_rag::auto_init_docs_rag,
            docs_rag::search_docs,
            docs_rag::get_docs_rag_status,
            docs_rag::stop_docs_rag,
        ])
        .run(tauri::generate_context!())
        .expect("error running app");
}
```

### 2. 환경 변수 설정

```bash
export OPENAI_API_KEY=sk-xxx
```

### 3. 프론트엔드 연동 (TypeScript)

```typescript
import { invoke } from '@tauri-apps/api/core';

// 초기화 (백그라운드)
await invoke('auto_init_docs_rag', { projectPath: '/path/to/project' });

// 검색
const results = await invoke<SearchResult[]>('search_docs', {
  projectPath: '/path/to/project',
  query: '검색어',
  topK: 5,
});

// 타입 정의
interface SearchResult {
  filePath: string;
  content: string;
  startLine: number;
  endLine: number;
  score: number;
}
```

### 4. 이벤트 리스닝

```typescript
import { listen } from '@tauri-apps/api/event';

await listen('docs-rag:status', (event) => {
  console.log('DocsRAG status:', event.payload);
});
```

## 커스터마이징 포인트

### 1. 지원 파일 형식 추가

`store.rs`의 `index_project()` 패턴 수정:
```rust
let patterns = vec![
    "**/*.md",
    "**/*.txt",
    // 추가 패턴...
];
```

`watcher.rs`의 `WATCHED_EXTENSIONS` 수정:
```rust
const WATCHED_EXTENSIONS: &[&str] = &["md", "json", /* ... */];
```

`chunker.rs`에 새 청킹 전략 추가:
```rust
fn chunk_file(path: &Path) -> Result<Vec<Chunk>, ...> {
    match extension {
        "new_ext" => Self::chunk_new_format(&content, &path_str),
        // ...
    }
}
```

### 2. 임베딩 모델 변경

`embedder.rs` 수정:
```rust
const EMBEDDING_DIM: usize = 3072;  // text-embedding-3-large
const EMBEDDING_MODEL: &str = "text-embedding-3-large";
```

### 3. 캐시 위치 변경

`store.rs` 수정:
```rust
const CACHE_DIR: &str = ".my-app/rag-cache";
```

## 성능 고려사항

1. **대규모 프로젝트**: `EXCLUDED_DIRS`를 적절히 설정하여 불필요한 파일 제외
2. **임베딩 비용**: 캐시가 유효하면 API 호출 안 함 (mtime 기반 검증)
3. **메모리**: 청크와 임베딩이 메모리에 유지됨 - 대규모 프로젝트에서는 주의
4. **동시성**: `RwLock` 사용으로 읽기 작업은 동시 실행 가능

## 테스트

```bash
# 단위 테스트 실행
cargo test --package your-package -- docs_rag

# 특정 모듈 테스트
cargo test --package your-package -- docs_rag::store::tests
cargo test --package your-package -- docs_rag::chunker::tests
cargo test --package your-package -- docs_rag::watcher::tests
cargo test --package your-package -- docs_rag::embedder::tests
```

## 라이선스

원본 프로젝트(SyntaxOS)의 라이선스를 따릅니다.
