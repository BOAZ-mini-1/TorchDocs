# TorchDocs
- Directory architecture (draft)
- raw 폴더는 용량이 커서 충돌이 일어나므로 업로드 하지 않았음 (.gitignore 참고)

```
TorchDocs/
├─ .gitignore/
├─ data/
│  ├─ raw/                 # HTML 원본 (version 별 ZIP 파일)
│  │  ├─ pytorch_docs_2.8.zip
│  │  └─ ...
│  └─ processed/           # 전처리 완료된 version별 jsonl 파일
│     ├─ torchdocs_2.8_chunks.jsonl
│     └─ ...
├─ preprocessing.py        # HTML 원본 전처리 스크립트
├─ qa_chunks.py            # .jsonl 파일 품질 검사 스크립트
├─ script_guide.txt        # 커맨드 라인 입력 가이드
│
├─ indexes/
│  ├─ chroma/              # Q&A index (vectorDB)
│  └─ snippets_chroma/     # snippet index (vectorDB)
├─ src/
│  ├─ crawl.py             # (optional) URL 리스트 받아 HTML 저장
│  ├─ transform.py         # HTML→Markdown/Text, 잡음 제거, 메타 추출
│  ├─ make_chunks.py       # 문서→청킹(JSONL 생성), 코드블록 추출
│  ├─ build_index.py       # chunks/snippets → vectorDB 구축
│  ├─ rag_qa.py            # Q&A mode chain (w/ version filter)
│  └─ rag_snippet.py       # snippet mode chain
├─ app/
│  └─ ui.py                # Streamlit/Gradio UI (모드 토글, 버전 선택, 인용 표시)
├─ configs/
│  └─ default.yaml         # 경로/파라미터(chuck_size, k값 등)
├─ .env.example            # OPENAI_API_KEY=...
├─ requirements.txt
└─ README.md
```
