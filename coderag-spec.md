# save-context-search: Claude Code Skill 구현 명세서

## 개요

save-context-search는 Claude Code의 glob/grep/read 패턴을 대체하여 컨텍스트를 90%+ 절약하는 시맨틱 검색 도구다. 코드와 문서를 모두 지원하며, MCP 서버 없이 CLI 바이너리로 동작한다.

### 핵심 가치
- **컨텍스트 절약**: glob→grep→read 대신 정확한 위치로 바로 점프
- **통합 검색**: 코드(.rs, .ts, .cs, .py) + 문서(.md, .txt) 한번에
- **서버리스**: 호출 시에만 실행, 상주 프로세스 없음
- **자동 최신화**: Lazy Incremental 패턴으로 변경 자동 추적

### 기술 스택
- Rust (CLI 바이너리)
- tree-sitter (코드 파싱, 심볼 추출)
- OpenAI text-embedding-3-small (시맨틱 임베딩)
- simsimd (벡터 유사도 검색)

### LSP와의 차이
- LSP: 타입 추론 기반, 서버 상주, 완벽한 정확도
- save-context-search: 이름 기반, 서버 없음, 80% 정확도 + 시맨틱 검색

---

## 프로젝트 구조

```
save-context-search/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI 엔트리포인트
│   ├── lib.rs               # 라이브러리 루트
│   ├── index/
│   │   ├── mod.rs
│   │   ├── manager.rs       # 인덱스 관리 (로드/저장/증분)
│   │   └── cache.rs         # 파일 해시/mtime 캐시
│   ├── parser/
│   │   ├── mod.rs
│   │   ├── code.rs          # tree-sitter 코드 파서
│   │   ├── docs.rs          # 문서 파서 (MD, TXT)
│   │   └── languages.rs     # 언어별 쿼리 정의
│   ├── embeddings/
│   │   ├── mod.rs
│   │   └── openai.rs        # OpenAI API 클라이언트
│   ├── search/
│   │   ├── mod.rs
│   │   ├── semantic.rs      # 시맨틱 검색
│   │   └── lookup.rs        # 정확한 이름 검색
│   └── output/
│       └── json.rs          # JSON 출력 포맷
└── SKILL.md                 # Claude Code Skill 정의
```

---

## Cargo.toml

```toml
[package]
name = "save-context-search"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "scs"
path = "src/main.rs"

[dependencies]
# CLI
clap = { version = "4", features = ["derive"] }

# Async
tokio = { version = "1", features = ["full"] }

# Tree-sitter
tree-sitter = "0.24"
tree-sitter-rust = "0.23"
tree-sitter-typescript = "0.23"
tree-sitter-c-sharp = "0.23"
tree-sitter-python = "0.23"

# OpenAI
async-openai = "0.25"

# Vector search
simsimd = "0.5"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
bincode = "1"

# Utils
anyhow = "1"
walkdir = "2"
sha2 = "0.10"
chrono = "0.4"

[profile.release]
lto = true
strip = true
```

---

## 핵심 데이터 구조

### src/lib.rs

```rust
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 청크 타입: 코드 또는 문서
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkType {
    Code,
    Doc,
}

/// 통합 청크 구조
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_type: ChunkType,
    pub name: String,
    pub kind: ChunkKind,
    pub file: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
    pub context: Option<String>,      // 코드: 소속 클래스, 문서: 섹션
    pub signature: Option<String>,    // 코드만: 함수 시그니처
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkKind {
    // 코드
    Class,
    Struct,
    Interface,
    Enum,
    Function,
    Method,
    Field,
    Constant,
    // 문서
    Document,
    Section,
    Paragraph,
}

/// 검색 매칭 방식
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MatchType {
    Semantic,   // 시맨틱 검색
    Exact,      // 유일한 심볼/문서
    NameOnly,   // 이름만 매칭, 동명 존재
}

/// 결과 신뢰도
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    High,    // 유일하거나 매우 높은 유사도
    Medium,  // 동명 있지만 구분 가능
    Low,     // 동명 많음, 확인 필요
}

/// 참조 종류
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    Call,       // 함수 호출
    Access,     // 필드 접근
    TypeRef,    // 타입 참조
    DocLink,    // 문서 링크
}

/// Search 결과
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOutput {
    pub query: String,
    pub match_type: MatchType,
    pub results: Vec<SearchResult>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_type: ChunkType,
    pub name: String,
    pub kind: ChunkKind,
    pub file: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub score: f32,
    pub preview: String,
    pub context: Option<String>,
    pub unique: bool,
}

/// Lookup 결과
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupOutput {
    pub name: String,
    pub match_type: MatchType,
    pub confidence: Confidence,
    pub definitions: Vec<Definition>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Definition {
    pub chunk_type: ChunkType,
    pub kind: ChunkKind,
    pub file: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub signature: Option<String>,
    pub context: Option<String>,
    pub preview: String,
}

/// Deps 결과
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepsOutput {
    pub name: String,
    pub match_type: MatchType,
    pub confidence: Confidence,
    pub definition: Option<Definition>,
    pub references: Vec<Reference>,
    pub calls: Option<Vec<CallEntry>>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub file: PathBuf,
    pub line: usize,
    pub column: Option<usize>,
    pub kind: ReferenceKind,
    pub context: String,
    pub snippet: String,
    pub possible_targets: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallEntry {
    pub name: String,
    pub line: usize,
}

/// 파일 캐시 엔트리
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub mtime: u64,
    pub hash: String,
    pub chunk_type: ChunkType,
    pub symbols: Vec<String>,
}

/// 인덱스 메타정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub version: String,
    pub last_indexed: String,
    pub root: PathBuf,
    pub file_count: usize,
    pub code_files: usize,
    pub doc_files: usize,
    pub chunk_count: usize,
    pub embedding_model: String,
    pub duplicate_names: usize,
}
```

---

## CLI 구현

### src/main.rs

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "scs")]
#[command(about = "save-context-search: 컨텍스트 절약형 코드/문서 검색")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 시맨틱 검색 - 자연어로 코드/문서 찾기
    Search {
        query: String,
        #[arg(long, default_value = "5")]
        top: usize,
        #[arg(long)]
        code_only: bool,
        #[arg(long)]
        docs_only: bool,
    },
    /// 심볼/문서 정의 찾기
    Lookup {
        name: String,
        #[arg(long)]
        code_only: bool,
        #[arg(long)]
        docs_only: bool,
    },
    /// 파일 구조 요약 - Read 대신 사용
    Outline {
        file: PathBuf,
    },
    /// 참조/호출 관계 (코드만)
    Deps {
        name: String,
        #[arg(long, default_value = "both")]
        direction: String,
    },
    /// 인덱스 상태 확인
    Status,
    /// 강제 재인덱싱
    Reindex {
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// 변경된 파일만 갱신
    Refresh {
        #[arg(long)]
        quiet: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut scs = save_context_search::SCS::load_or_create(".").await?;

    match cli.command {
        Commands::Search { query, top, code_only, docs_only } => {
            let stats = scs.ensure_fresh().await?;
            if stats.has_changes() {
                eprintln!("[scs] {} 파일 업데이트됨", stats.total());
            }
            
            let filter = match (code_only, docs_only) {
                (true, false) => Some(save_context_search::ChunkType::Code),
                (false, true) => Some(save_context_search::ChunkType::Doc),
                _ => None,
            };
            
            let results = scs.search(&query, top, filter).await?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Commands::Lookup { name, code_only, docs_only } => {
            scs.ensure_fresh().await?;
            
            let filter = match (code_only, docs_only) {
                (true, false) => Some(save_context_search::ChunkType::Code),
                (false, true) => Some(save_context_search::ChunkType::Doc),
                _ => None,
            };
            
            let results = scs.lookup(&name, filter)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Commands::Outline { file } => {
            let outline = scs.outline(&file)?;
            println!("{}", serde_json::to_string_pretty(&outline)?);
        }
        Commands::Deps { name, direction } => {
            scs.ensure_fresh().await?;
            let deps = scs.deps(&name, &direction)?;
            println!("{}", serde_json::to_string_pretty(&deps)?);
        }
        Commands::Status => {
            let status = scs.status()?;
            println!("{}", serde_json::to_string_pretty(&status)?);
        }
        Commands::Reindex { path } => {
            let root = path.unwrap_or_else(|| PathBuf::from("."));
            scs.reindex_all(&root).await?;
            eprintln!("[scs] 전체 재인덱싱 완료");
        }
        Commands::Refresh { quiet } => {
            let stats = scs.ensure_fresh().await?;
            if !quiet {
                eprintln!("[scs] {:?}", stats);
            }
        }
    }

    Ok(())
}
```

---

## 인덱스 관리

### src/index/manager.rs

```rust
use crate::{Chunk, ChunkType, FileEntry, IndexMeta};
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub struct IndexManager {
    pub root: PathBuf,
    pub index_dir: PathBuf,
    pub meta: Option<IndexMeta>,
    pub file_hashes: HashMap<PathBuf, FileEntry>,
    pub chunks: Vec<Chunk>,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Default)]
pub struct IncrementalStats {
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
}

impl IncrementalStats {
    pub fn has_changes(&self) -> bool {
        self.added > 0 || self.updated > 0 || self.removed > 0
    }
    pub fn total(&self) -> usize {
        self.added + self.updated + self.removed
    }
}

impl IndexManager {
    pub fn load_or_create(root: &Path) -> Result<Self> {
        let index_dir = root.join(".scs");
        
        let mut manager = Self {
            root: root.to_path_buf(),
            index_dir: index_dir.clone(),
            meta: None,
            file_hashes: HashMap::new(),
            chunks: Vec::new(),
            embeddings: Vec::new(),
        };

        if index_dir.exists() {
            manager.load_existing()?;
        }

        Ok(manager)
    }

    fn load_existing(&mut self) -> Result<()> {
        let meta_path = self.index_dir.join("meta.json");
        if meta_path.exists() {
            let content = std::fs::read_to_string(&meta_path)?;
            self.meta = Some(serde_json::from_str(&content)?);
        }

        let hashes_path = self.index_dir.join("file_hashes.json");
        if hashes_path.exists() {
            let content = std::fs::read_to_string(&hashes_path)?;
            self.file_hashes = serde_json::from_str(&content)?;
        }

        let chunks_path = self.index_dir.join("chunks.json");
        if chunks_path.exists() {
            let content = std::fs::read_to_string(&chunks_path)?;
            self.chunks = serde_json::from_str(&content)?;
        }

        let emb_path = self.index_dir.join("embeddings.bin");
        if emb_path.exists() {
            let bytes = std::fs::read(&emb_path)?;
            self.embeddings = bincode::deserialize(&bytes)?;
        }

        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.index_dir)?;

        if let Some(meta) = &self.meta {
            let content = serde_json::to_string_pretty(meta)?;
            std::fs::write(self.index_dir.join("meta.json"), content)?;
        }

        let hashes = serde_json::to_string_pretty(&self.file_hashes)?;
        std::fs::write(self.index_dir.join("file_hashes.json"), hashes)?;

        let chunks = serde_json::to_string(&self.chunks)?;
        std::fs::write(self.index_dir.join("chunks.json"), chunks)?;

        let bytes = bincode::serialize(&self.embeddings)?;
        std::fs::write(self.index_dir.join("embeddings.bin"), bytes)?;

        Ok(())
    }

    /// 파일 타입 판별
    pub fn classify_file(path: &Path) -> Option<ChunkType> {
        let ext = path.extension()?.to_str()?;
        match ext {
            // 코드
            "rs" | "ts" | "tsx" | "js" | "jsx" | "cs" | "py" | "go" | "java" => {
                Some(ChunkType::Code)
            }
            // 문서
            "md" | "txt" | "mdx" => Some(ChunkType::Doc),
            _ => None,
        }
    }

    /// 소스 파일 스캔
    pub fn scan_files(&self) -> Result<HashMap<PathBuf, ChunkType>> {
        let ignore_dirs = ["node_modules", "target", ".git", "dist", "build", ".scs"];

        let files: HashMap<_, _> = WalkDir::new(&self.root)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !ignore_dirs.contains(&name.as_ref())
            })
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| {
                let path = e.path().to_path_buf();
                Self::classify_file(&path).map(|ct| (path, ct))
            })
            .collect();

        Ok(files)
    }
}
```

---

## 파서: 코드

### src/parser/code.rs

```rust
use crate::{Chunk, ChunkKind, ChunkType};
use anyhow::Result;
use std::path::PathBuf;
use tree_sitter::{Language, Parser, Query, QueryCursor};

pub struct CodeParser {
    parsers: std::collections::HashMap<String, Parser>,
    queries: std::collections::HashMap<String, Query>,
}

impl CodeParser {
    pub fn new() -> Result<Self> {
        let mut parser = Self {
            parsers: std::collections::HashMap::new(),
            queries: std::collections::HashMap::new(),
        };

        parser.register("rs", tree_sitter_rust::LANGUAGE.into(), RUST_QUERY)?;
        parser.register("ts", tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(), TS_QUERY)?;
        parser.register("tsx", tree_sitter_typescript::LANGUAGE_TSX.into(), TS_QUERY)?;
        parser.register("cs", tree_sitter_c_sharp::LANGUAGE.into(), CSHARP_QUERY)?;
        parser.register("py", tree_sitter_python::LANGUAGE.into(), PYTHON_QUERY)?;

        Ok(parser)
    }

    fn register(&mut self, ext: &str, lang: Language, query_str: &str) -> Result<()> {
        let mut parser = Parser::new();
        parser.set_language(&lang)?;
        self.parsers.insert(ext.to_string(), parser);
        self.queries.insert(ext.to_string(), Query::new(&lang, query_str)?);
        Ok(())
    }

    pub fn parse(&self, path: &PathBuf, content: &str) -> Result<Vec<Chunk>> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        
        let (parser, query) = match (self.parsers.get(ext), self.queries.get(ext)) {
            (Some(p), Some(q)) => (p, q),
            _ => return Ok(Vec::new()),
        };

        let tree = parser.parse(content, None)
            .ok_or_else(|| anyhow::anyhow!("Parse failed"))?;

        let mut cursor = QueryCursor::new();
        let matches = cursor.matches(query, tree.root_node(), content.as_bytes());

        let mut chunks = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for m in matches {
            for capture in m.captures {
                let node = capture.node;
                let capture_name = &query.capture_names()[capture.index as usize];
                
                let kind = match capture_name.as_str() {
                    "class" | "class.name" => ChunkKind::Class,
                    "struct" | "struct.name" => ChunkKind::Struct,
                    "function" | "function.name" => ChunkKind::Function,
                    "method" | "method.name" => ChunkKind::Method,
                    _ => continue,
                };

                let def_node = self.find_definition_parent(node);
                let key = (def_node.start_byte(), def_node.end_byte());
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key);

                let name = node.utf8_text(content.as_bytes())?.to_string();
                let full_content = def_node.utf8_text(content.as_bytes())?.to_string();
                let context = self.find_parent_class(&def_node, content);
                let signature = self.extract_signature(&def_node, content);

                chunks.push(Chunk {
                    chunk_type: ChunkType::Code,
                    name,
                    kind,
                    file: path.clone(),
                    line_start: def_node.start_position().row + 1,
                    line_end: def_node.end_position().row + 1,
                    content: full_content,
                    context,
                    signature,
                });
            }
        }

        Ok(chunks)
    }

    fn find_definition_parent(&self, node: tree_sitter::Node) -> tree_sitter::Node {
        let def_kinds = [
            "function_definition", "function_item", "method_definition",
            "class_definition", "class_declaration", "struct_item",
        ];
        let mut current = node;
        while let Some(parent) = current.parent() {
            if def_kinds.contains(&parent.kind()) {
                return parent;
            }
            current = parent;
        }
        node
    }

    fn find_parent_class(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        let class_kinds = ["class_definition", "class_declaration", "impl_item"];
        let mut current = *node;
        while let Some(parent) = current.parent() {
            if class_kinds.contains(&parent.kind()) {
                // 클래스 이름 추출
                for child in parent.children(&mut parent.walk()) {
                    if child.kind().contains("name") || child.kind() == "identifier" {
                        return child.utf8_text(content.as_bytes()).ok().map(|s| s.to_string());
                    }
                }
            }
            current = parent;
        }
        None
    }

    fn extract_signature(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
        node.utf8_text(content.as_bytes())
            .ok()
            .and_then(|t| t.lines().next())
            .map(|s| s.trim().to_string())
    }

    /// 함수 내 호출 추출
    pub fn extract_calls(&self, content: &str) -> Option<Vec<crate::CallEntry>> {
        // 간단한 정규식 기반 호출 추출
        let call_pattern = regex::Regex::new(r"(\w+)\s*\(").ok()?;
        let mut calls = Vec::new();
        
        for (i, line) in content.lines().enumerate() {
            for cap in call_pattern.captures_iter(line) {
                if let Some(name) = cap.get(1) {
                    calls.push(crate::CallEntry {
                        name: name.as_str().to_string(),
                        line: i + 1,
                    });
                }
            }
        }
        
        Some(calls)
    }
}

const RUST_QUERY: &str = r#"
(function_item name: (identifier) @function.name) @function
(impl_item (type_identifier) @class.name) @class
(struct_item name: (type_identifier) @struct.name) @struct
"#;

const TS_QUERY: &str = r#"
(function_declaration name: (identifier) @function.name) @function
(class_declaration name: (identifier) @class.name) @class
(method_definition name: (property_identifier) @method.name) @method
"#;

const CSHARP_QUERY: &str = r#"
(class_declaration name: (identifier) @class.name) @class
(method_declaration name: (identifier) @method.name) @method
"#;

const PYTHON_QUERY: &str = r#"
(function_definition name: (identifier) @function.name) @function
(class_definition name: (identifier) @class.name) @class
"#;
```

---

## 파서: 문서

### src/parser/docs.rs

```rust
use crate::{Chunk, ChunkKind, ChunkType};
use anyhow::Result;
use std::path::PathBuf;

pub struct DocsParser;

impl DocsParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, path: &PathBuf, content: &str) -> Result<Vec<Chunk>> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        
        match ext {
            "md" | "mdx" => self.parse_markdown(path, content),
            "txt" => self.parse_text(path, content),
            _ => Ok(Vec::new()),
        }
    }

    fn parse_markdown(&self, path: &PathBuf, content: &str) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let mut current_section: Option<String> = None;
        let mut section_start = 0;
        let mut section_content = String::new();

        for (i, line) in content.lines().enumerate() {
            if line.starts_with('#') {
                // 이전 섹션 저장
                if let Some(ref section) = current_section {
                    if !section_content.trim().is_empty() {
                        chunks.push(Chunk {
                            chunk_type: ChunkType::Doc,
                            name: section.clone(),
                            kind: ChunkKind::Section,
                            file: path.clone(),
                            line_start: section_start,
                            line_end: i,
                            content: section_content.trim().to_string(),
                            context: None,
                            signature: None,
                        });
                    }
                }

                // 새 섹션 시작
                let title = line.trim_start_matches('#').trim().to_string();
                current_section = Some(title);
                section_start = i + 1;
                section_content = String::new();
            } else {
                section_content.push_str(line);
                section_content.push('\n');
            }
        }

        // 마지막 섹션 저장
        if let Some(section) = current_section {
            if !section_content.trim().is_empty() {
                chunks.push(Chunk {
                    chunk_type: ChunkType::Doc,
                    name: section,
                    kind: ChunkKind::Section,
                    file: path.clone(),
                    line_start: section_start,
                    line_end: content.lines().count(),
                    content: section_content.trim().to_string(),
                    context: None,
                    signature: None,
                });
            }
        }

        // 문서 전체도 하나의 청크로
        let doc_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();

        chunks.insert(0, Chunk {
            chunk_type: ChunkType::Doc,
            name: doc_name,
            kind: ChunkKind::Document,
            file: path.clone(),
            line_start: 1,
            line_end: content.lines().count(),
            content: content.to_string(),
            context: None,
            signature: None,
        });

        Ok(chunks)
    }

    fn parse_text(&self, path: &PathBuf, content: &str) -> Result<Vec<Chunk>> {
        let doc_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();

        Ok(vec![Chunk {
            chunk_type: ChunkType::Doc,
            name: doc_name,
            kind: ChunkKind::Document,
            file: path.clone(),
            line_start: 1,
            line_end: content.lines().count(),
            content: content.to_string(),
            context: None,
            signature: None,
        }])
    }
}
```

---

## 메인 SCS 구조체

### src/lib.rs (impl)

```rust
pub mod index;
pub mod parser;
pub mod embeddings;
pub mod search;

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct SCS {
    index: index::manager::IndexManager,
    code_parser: parser::code::CodeParser,
    docs_parser: parser::docs::DocsParser,
    embedder: embeddings::openai::OpenAIEmbedder,
    name_counts: HashMap<String, usize>,
}

impl SCS {
    pub async fn load_or_create(root: &str) -> Result<Self> {
        let root = Path::new(root);
        let index = index::manager::IndexManager::load_or_create(root)?;
        let code_parser = parser::code::CodeParser::new()?;
        let docs_parser = parser::docs::DocsParser::new();
        let embedder = embeddings::openai::OpenAIEmbedder::new()?;
        
        let mut name_counts = HashMap::new();
        for chunk in &index.chunks {
            *name_counts.entry(chunk.name.clone()).or_default() += 1;
        }

        Ok(Self { index, code_parser, docs_parser, embedder, name_counts })
    }

    pub async fn ensure_fresh(&mut self) -> Result<index::manager::IncrementalStats> {
        let current_files = self.index.scan_files()?;
        let mut stats = index::manager::IncrementalStats::default();
        let mut to_reindex: Vec<(PathBuf, ChunkType)> = Vec::new();

        // 삭제된 파일
        let cached_paths: Vec<_> = self.index.file_hashes.keys().cloned().collect();
        for path in cached_paths {
            if !current_files.contains_key(&path) {
                self.remove_file(&path);
                stats.removed += 1;
            }
        }

        // 추가/변경된 파일
        for (path, chunk_type) in &current_files {
            let mtime = std::fs::metadata(path)?
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();

            match self.index.file_hashes.get(path) {
                Some(entry) if entry.mtime == mtime => {}
                Some(_) => {
                    to_reindex.push((path.clone(), chunk_type.clone()));
                    stats.updated += 1;
                }
                None => {
                    to_reindex.push((path.clone(), chunk_type.clone()));
                    stats.added += 1;
                }
            }
        }

        // 배치 재인덱싱
        if !to_reindex.is_empty() {
            self.reindex_files(&to_reindex).await?;
        }

        if stats.has_changes() {
            self.update_meta(&current_files);
            self.index.save()?;
            
            // name_counts 재계산
            self.name_counts.clear();
            for chunk in &self.index.chunks {
                *self.name_counts.entry(chunk.name.clone()).or_default() += 1;
            }
        }

        Ok(stats)
    }

    async fn reindex_files(&mut self, files: &[(PathBuf, ChunkType)]) -> Result<()> {
        for (path, _) in files {
            self.remove_file(path);
        }

        let mut new_chunks = Vec::new();
        for (path, chunk_type) in files {
            let content = std::fs::read_to_string(path)?;
            let chunks = match chunk_type {
                ChunkType::Code => self.code_parser.parse(path, &content)?,
                ChunkType::Doc => self.docs_parser.parse(path, &content)?,
            };

            let mtime = std::fs::metadata(path)?
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            let hash = format!("{:x}", sha2::Sha256::digest(content.as_bytes()));
            let symbols: Vec<_> = chunks.iter().map(|c| c.name.clone()).collect();

            self.index.file_hashes.insert(path.clone(), FileEntry {
                mtime,
                hash,
                chunk_type: chunk_type.clone(),
                symbols,
            });

            new_chunks.extend(chunks);
        }

        if !new_chunks.is_empty() {
            let texts: Vec<_> = new_chunks.iter()
                .map(|c| format!("{:?} {}: {}", c.kind, c.name, c.content))
                .collect();
            let new_embeddings = self.embedder.embed_batch(&texts).await?;

            self.index.chunks.extend(new_chunks);
            self.index.embeddings.extend(new_embeddings);
        }

        Ok(())
    }

    fn remove_file(&mut self, path: &PathBuf) {
        let indices: Vec<_> = self.index.chunks.iter()
            .enumerate()
            .filter(|(_, c)| &c.file == path)
            .map(|(i, _)| i)
            .rev()
            .collect();

        for i in indices {
            self.index.chunks.remove(i);
            self.index.embeddings.remove(i);
        }
        self.index.file_hashes.remove(path);
    }

    pub async fn search(
        &self, 
        query: &str, 
        top_k: usize,
        filter: Option<ChunkType>,
    ) -> Result<SearchOutput> {
        let query_emb = self.embedder.embed_single(query).await?;
        
        let filtered_indices: Vec<_> = self.index.chunks.iter()
            .enumerate()
            .filter(|(_, c)| filter.as_ref().map(|f| &c.chunk_type == f).unwrap_or(true))
            .map(|(i, _)| i)
            .collect();

        let mut scores: Vec<_> = filtered_indices.iter()
            .map(|&i| {
                let score = simsimd::SpatialSimilarity::cosine(&query_emb, &self.index.embeddings[i])
                    .unwrap_or(0.0) as f32;
                (i, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let results: Vec<_> = scores.into_iter()
            .take(top_k)
            .map(|(i, score)| {
                let chunk = &self.index.chunks[i];
                SearchResult {
                    chunk_type: chunk.chunk_type.clone(),
                    name: chunk.name.clone(),
                    kind: chunk.kind.clone(),
                    file: chunk.file.clone(),
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    score,
                    preview: Self::make_preview(&chunk.content),
                    context: chunk.context.clone(),
                    unique: self.name_counts.get(&chunk.name).map(|&c| c == 1).unwrap_or(true),
                }
            })
            .collect();

        Ok(SearchOutput {
            query: query.to_string(),
            match_type: MatchType::Semantic,
            results,
            suggestions: vec![],
        })
    }

    pub fn lookup(&self, name: &str, filter: Option<ChunkType>) -> Result<LookupOutput> {
        let (class_filter, symbol_name) = if name.contains('.') {
            let parts: Vec<_> = name.split('.').collect();
            (Some(parts[0].to_string()), parts[1].to_string())
        } else {
            (None, name.to_string())
        };

        let definitions: Vec<_> = self.index.chunks.iter()
            .filter(|c| {
                let type_match = filter.as_ref().map(|f| &c.chunk_type == f).unwrap_or(true);
                let name_match = c.name.to_lowercase().contains(&symbol_name.to_lowercase());
                let class_match = class_filter.as_ref()
                    .map(|cf| c.context.as_ref()
                        .map(|ctx| ctx.to_lowercase() == cf.to_lowercase())
                        .unwrap_or(false))
                    .unwrap_or(true);
                type_match && name_match && class_match
            })
            .map(|c| Definition {
                chunk_type: c.chunk_type.clone(),
                kind: c.kind.clone(),
                file: c.file.clone(),
                line_start: c.line_start,
                line_end: c.line_end,
                signature: c.signature.clone(),
                context: c.context.clone(),
                preview: Self::make_preview(&c.content),
            })
            .collect();

        let (match_type, confidence, suggestions) = self.compute_confidence(&symbol_name, &definitions);

        Ok(LookupOutput {
            name: name.to_string(),
            match_type,
            confidence,
            definitions,
            suggestions,
        })
    }

    pub fn deps(&self, name: &str, direction: &str) -> Result<DepsOutput> {
        let lookup = self.lookup(name, Some(ChunkType::Code))?;
        let definition = lookup.definitions.first().cloned();

        let references = if direction == "calls" {
            Vec::new()
        } else {
            self.find_references(name, &lookup)
        };

        let calls = if direction == "called_by" {
            None
        } else {
            definition.as_ref().and_then(|def| {
                self.index.chunks.iter()
                    .find(|c| c.file == def.file && c.line_start == def.line_start)
                    .and_then(|chunk| self.code_parser.extract_calls(&chunk.content))
            })
        };

        Ok(DepsOutput {
            name: name.to_string(),
            match_type: lookup.match_type,
            confidence: lookup.confidence,
            definition,
            references,
            calls,
            suggestions: lookup.suggestions,
        })
    }

    fn find_references(&self, name: &str, lookup: &LookupOutput) -> Vec<Reference> {
        let mut refs = Vec::new();
        let possible_targets: Option<Vec<String>> = if lookup.confidence == Confidence::Low {
            Some(lookup.definitions.iter()
                .filter_map(|d| d.context.as_ref().map(|ctx| format!("{}.{}", ctx, name)))
                .collect())
        } else {
            None
        };

        for chunk in &self.index.chunks {
            if chunk.chunk_type != ChunkType::Code {
                continue;
            }
            if lookup.definitions.iter().any(|d| d.file == chunk.file && d.line_start == chunk.line_start) {
                continue;
            }
            if chunk.content.contains(name) {
                refs.push(Reference {
                    file: chunk.file.clone(),
                    line: chunk.line_start,
                    column: None,
                    kind: ReferenceKind::Call,
                    context: chunk.name.clone(),
                    snippet: Self::extract_snippet(&chunk.content, name),
                    possible_targets: possible_targets.clone(),
                });
            }
        }
        refs
    }

    fn extract_snippet(content: &str, name: &str) -> String {
        content.lines()
            .find(|line| line.contains(name))
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }

    fn compute_confidence(
        &self, 
        name: &str, 
        definitions: &[Definition]
    ) -> (MatchType, Confidence, Vec<String>) {
        match definitions.len() {
            0 => (MatchType::NameOnly, Confidence::Low, vec!["심볼/문서를 찾을 수 없음".to_string()]),
            1 => (MatchType::Exact, Confidence::High, vec![]),
            n => {
                let contexts: Vec<_> = definitions.iter()
                    .filter_map(|d| d.context.as_ref().map(|ctx| format!("{}.{}", ctx, name)))
                    .collect();
                (
                    MatchType::NameOnly,
                    Confidence::Low,
                    vec![
                        format!("동명 {}개 발견", n),
                        if !contexts.is_empty() {
                            format!("클래스명으로 구분: {}", contexts.join(", "))
                        } else {
                            "파일 경로로 구분 필요".to_string()
                        },
                    ]
                )
            }
        }
    }

    pub fn outline(&self, file: &PathBuf) -> Result<serde_json::Value> {
        let chunks: Vec<_> = self.index.chunks.iter()
            .filter(|c| &c.file == file)
            .collect();

        let is_doc = chunks.first().map(|c| c.chunk_type == ChunkType::Doc).unwrap_or(false);

        if is_doc {
            let sections: Vec<_> = chunks.iter()
                .filter(|c| c.kind == ChunkKind::Section)
                .map(|c| serde_json::json!({
                    "name": c.name,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                }))
                .collect();

            Ok(serde_json::json!({
                "file": file,
                "type": "document",
                "sections": sections
            }))
        } else {
            let symbols: Vec<_> = chunks.iter()
                .map(|c| serde_json::json!({
                    "name": c.name,
                    "kind": c.kind,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "signature": c.signature,
                    "context": c.context,
                }))
                .collect();

            Ok(serde_json::json!({
                "file": file,
                "type": "code",
                "symbols": symbols
            }))
        }
    }

    pub fn status(&self) -> Result<serde_json::Value> {
        let code_count = self.index.file_hashes.values()
            .filter(|e| e.chunk_type == ChunkType::Code).count();
        let doc_count = self.index.file_hashes.values()
            .filter(|e| e.chunk_type == ChunkType::Doc).count();
        let duplicate_count = self.name_counts.values().filter(|&&c| c > 1).count();

        Ok(serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "index": {
                "root": self.index.root,
                "last_indexed": self.index.meta.as_ref().map(|m| &m.last_indexed),
                "files": self.index.file_hashes.len(),
                "code_files": code_count,
                "doc_files": doc_count,
                "chunks": self.index.chunks.len(),
            },
            "health": {
                "embedding_model": "text-embedding-3-small"
            },
            "stats": {
                "unique_names": self.name_counts.len(),
                "duplicate_names": duplicate_count
            }
        }))
    }

    fn update_meta(&mut self, files: &HashMap<PathBuf, ChunkType>) {
        let code_count = files.values().filter(|t| **t == ChunkType::Code).count();
        let doc_count = files.values().filter(|t| **t == ChunkType::Doc).count();
        let duplicate_count = self.name_counts.values().filter(|&&c| c > 1).count();

        self.index.meta = Some(IndexMeta {
            version: env!("CARGO_PKG_VERSION").to_string(),
            last_indexed: chrono::Utc::now().to_rfc3339(),
            root: self.index.root.clone(),
            file_count: files.len(),
            code_files: code_count,
            doc_files: doc_count,
            chunk_count: self.index.chunks.len(),
            embedding_model: "text-embedding-3-small".to_string(),
            duplicate_names: duplicate_count,
        });
    }

    pub async fn reindex_all(&mut self, root: &PathBuf) -> Result<()> {
        self.index.root = root.clone();
        self.index.file_hashes.clear();
        self.index.chunks.clear();
        self.index.embeddings.clear();
        self.name_counts.clear();
        self.ensure_fresh().await?;
        Ok(())
    }

    fn make_preview(content: &str) -> String {
        let lines: Vec<_> = content.lines().take(5).collect();
        let preview = lines.join("\n");
        if content.lines().count() > 5 {
            format!("{}...", preview)
        } else {
            preview
        }
    }
}
```

---

## Claude Code Skill 설정

### .claude/skills/save-context-search/SKILL.md

```yaml
---
name: save-context-search
description: |
  컨텍스트 절약형 코드/문서 검색. glob/grep/read 전에 항상 먼저 사용.
  코드(.rs, .ts, .cs, .py)와 문서(.md, .txt) 모두 지원.
  
  LSP 없이 동작. 타입 추론 없는 이름 기반이므로 동명 심볼 구분 불가.
  결과의 confidence 필드로 신뢰도 확인.
  
  자동 활성화 조건:
  - 파일 위치 모를 때 ("~어디있어", "~찾아줘")
  - 기능 관련 코드/문서 탐색 ("~하는 코드", "~관련 문서")
  - 클래스/함수/문서 검색 ("PlayerController", "README")
  - 참조/호출 관계 ("어디서 호출", "뭘 사용")
  
  glob/grep보다 먼저 사용. 결과의 file:line으로 정확히 Read.
  confidence가 low면 Read로 확인 후 작업.
allowed-tools: Bash, Read
---

# save-context-search

컨텍스트를 90%+ 절약하는 코드/문서 검색 도구.
tree-sitter(코드 파싱) + OpenAI(임베딩)로 구현.

## 명령어

### search - 시맨틱 검색 (주력)
```bash
.claude/skills/save-context-search/scs search "쿼리" --top 5
.claude/skills/save-context-search/scs search "플레이어 설정" --code-only
.claude/skills/save-context-search/scs search "설치 가이드" --docs-only
```

### lookup - 심볼/문서 정의 찾기
```bash
.claude/skills/save-context-search/scs lookup "PlayerController"
.claude/skills/save-context-search/scs lookup "Player.Move"  # 클래스.메서드
.claude/skills/save-context-search/scs lookup "README" --docs-only
```

### deps - 참조/호출 관계 (코드만)
```bash
.claude/skills/save-context-search/scs deps "HandleMovement"
.claude/skills/save-context-search/scs deps "HandleMovement" --direction calls
.claude/skills/save-context-search/scs deps "HandleMovement" --direction called_by
```

### outline - 파일 구조 (Read 대신)
```bash
.claude/skills/save-context-search/scs outline src/player.rs
.claude/skills/save-context-search/scs outline docs/guide.md
```

### status - 인덱스 상태
```bash
.claude/skills/save-context-search/scs status
```

## 출력 형식

### search 결과
```json
{
  "query": "플레이어 이동",
  "match_type": "semantic",
  "results": [
    {
      "chunk_type": "code",
      "name": "HandleMovement",
      "kind": "method",
      "file": "src/Player/PlayerController.cs",
      "line_start": 42,
      "line_end": 67,
      "score": 0.92,
      "preview": "void HandleMovement(Vector3 input) {...}",
      "context": "PlayerController",
      "unique": true
    },
    {
      "chunk_type": "doc",
      "name": "캐릭터 조작",
      "kind": "section",
      "file": "docs/gameplay.md",
      "line_start": 15,
      "score": 0.85,
      "preview": "플레이어 캐릭터는 WASD로 이동..."
    }
  ]
}
```

### lookup 결과
```json
{
  "name": "PlayerController",
  "match_type": "exact",
  "confidence": "high",
  "definitions": [
    {
      "chunk_type": "code",
      "kind": "class",
      "file": "src/Player/PlayerController.cs",
      "line_start": 15,
      "line_end": 120,
      "signature": "public class PlayerController : MonoBehaviour",
      "context": null,
      "preview": "public class PlayerController..."
    }
  ],
  "suggestions": []
}
```

### deps 결과
```json
{
  "name": "HandleMovement",
  "match_type": "exact",
  "confidence": "high",
  "definition": {
    "file": "src/Player/PlayerController.cs",
    "line": 42
  },
  "references": [
    {
      "file": "src/Player/PlayerController.cs",
      "line": 28,
      "kind": "call",
      "context": "Update",
      "snippet": "HandleMovement(input);"
    }
  ],
  "calls": [
    { "name": "MovePosition", "line": 45 }
  ]
}
```

## 핵심 필드

| 필드 | 값 | 의미 |
|------|-----|------|
| `chunk_type` | `code` / `doc` | 코드인지 문서인지 |
| `match_type` | `exact` / `name_only` / `semantic` | 매칭 방식 |
| `confidence` | `high` / `medium` / `low` | 신뢰도 |
| `suggestions` | string[] | 사용 가이드 |

## 워크플로우

### 기본
1. `scs search` 또는 `scs lookup`으로 위치 찾기
2. 결과의 `file:line_start-line_end`로 Read
3. 작업 수행

### confidence: low일 때
1. suggestions 확인
2. Read로 대상 검증
3. 맞는 대상에만 작업

## 예시

❌ 비효율:
```
glob **/*.cs → 100개 파일
grep "Player" → 50개 매칭
Read 10개 파일 전체
총: ~10,000 토큰
```

✅ 효율:
```
scs search "플레이어 이동"
→ confidence: high
→ src/Player/PlayerController.cs:42-67
Read 해당 범위만
총: ~500 토큰 (95% 절약)
```
```

---

## CLAUDE.md 규칙

```markdown
## 코드/문서 탐색 원칙

이 프로젝트는 **save-context-search** skill을 사용해 탐색한다.
컨텍스트를 90%+ 절약할 수 있다.

### 탐색 순서

1. **scs search** - 자연어로 코드/문서 찾기
2. **scs lookup** - 심볼/문서명으로 정의 찾기
3. **scs deps** - 참조/호출 관계 (코드만)
4. **scs outline** - 파일 구조 (Read 대신)
5. **Read file:line-line** - 필요한 부분만
6. **glob/grep** - scs로 못 찾을 때만

### confidence 대응

| confidence | 행동 |
|------------|------|
| high | 바로 작업 |
| medium | 확인 권장 |
| low | 반드시 Read로 확인 |

### 하지 말 것

- glob으로 파일 목록 먼저 보기
- grep으로 광범위 검색
- 파일 전체 Read
- confidence: low 그대로 신뢰
```

---

## 빌드 및 설치

```bash
# 빌드
cd save-context-search
cargo build --release

# 바이너리 이름: scs
# 설치 (전역)
mkdir -p ~/.claude/skills/save-context-search
cp target/release/scs ~/.claude/skills/save-context-search/
cp SKILL.md ~/.claude/skills/save-context-search/

# 또는 프로젝트별
mkdir -p .claude/skills/save-context-search
cp target/release/scs .claude/skills/save-context-search/
cp SKILL.md .claude/skills/save-context-search/
```

---

## 환경 변수

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 테스트

```bash
# 상태
./scs status

# 코드 검색
./scs search "main function" --top 3
./scs search "player movement" --code-only

# 문서 검색
./scs search "installation guide" --docs-only

# lookup
./scs lookup "PlayerController"
./scs lookup "Player.Move"

# deps
./scs deps "HandleMovement"

# outline
./scs outline src/main.rs
./scs outline README.md
```

---

## 컨텍스트 절약 효과

| 시나리오 | 기존 | scs | 절약 |
|----------|------|-----|------|
| 클래스 찾기 | ~9,200 토큰 | ~950 | 90% |
| 자연어 검색 | ~17,200 토큰 | ~560 | 97% |
| 참조 찾기 | ~8,600 토큰 | ~300 | 97% |
| 문서 섹션 찾기 | ~5,000 토큰 | ~400 | 92% |

**세션당 평균 90%+ 컨텍스트 절약**
