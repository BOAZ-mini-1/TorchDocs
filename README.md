# TorchDocs
- Directory architecture (draft)

```
TorchDocs/
├─ data/
│  ├─ raw/                 # HTML 원본 (그대로 저장)
│  │  └─ stable/2.4/...    # 버전별/섹션별 파일 구조
│  └─ processed/
│     ├─ documents.jsonl   # 문서 메타 
│     ├─ chunks.jsonl      # 청킹 결과 (검색 단위)
│     ├─ snippets.jsonl    # code snippet만 모은 파일
│     └─ embeddings.npy    # 임베딩 결과
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
