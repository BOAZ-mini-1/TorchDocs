# TorchDocs
- Directory architecture (draft)
- raw, embeddings 폴더는 용량이 커서 충돌이 일어나므로 업로드 하지 않았음 (.gitignore 참고)

## Current File Structure
```
.
├── README.md
├── data
│   └── processed
│       ├── token_analysis.txt
│       ├── torchdocs_2.4_chunks_e5.jsonl
│       ├── torchdocs_2.5_chunks_e5.jsonl
│       ├── torchdocs_2.6_chunks_e5.jsonl
│       ├── torchdocs_2.7_chunks_e5.jsonl
│       └── torchdocs_2.8_chunks_e5.jsonl
├── doc
│   └── offical_tut.ipynb
├── intfloat_e5_large_v2
│   ├── embeddings
│   │   ├── embeddings_torchdocs_2.4_chunks_e5_intfloat_e5_large_v2_mean.npy
│   │   ├── embeddings_torchdocs_2.5_chunks_e5_intfloat_e5_large_v2_mean.npy
│   │   ├── embeddings_torchdocs_2.6_chunks_e5_intfloat_e5_large_v2_mean.npy
│   │   ├── embeddings_torchdocs_2.7_chunks_e5_intfloat_e5_large_v2_mean.npy
│   │   ├── embeddings_torchdocs_2.8_chunks_e5_intfloat_e5_large_v2_mean.npy
│   │   ├── id_mapping_torchdocs_2.4_chunks_e5_intfloat_e5_large_v2_mean.json
│   │   ├── id_mapping_torchdocs_2.5_chunks_e5_intfloat_e5_large_v2_mean.json
│   │   ├── id_mapping_torchdocs_2.6_chunks_e5_intfloat_e5_large_v2_mean.json
│   │   ├── id_mapping_torchdocs_2.7_chunks_e5_intfloat_e5_large_v2_mean.json
│   │   └── id_mapping_torchdocs_2.8_chunks_e5_intfloat_e5_large_v2_mean.json
│   └── index
│       ├── faiss.index
│       ├── faiss.index.zip
│       ├── global_ids.json
│       └── stats.json
├── make_embeddings.py
├── preprocessing.py
├── qa_chunks.py
├── retriever.py
├── script_guide.txt
├── scripts
│   ├── 00_make_eval_pool.py
│   ├── 01_generate_qas.py
│   ├── 02_run_retreiver.py
│   ├── 03_run_generator.py
│   ├── 04_build_ragas_dataset.py
│   └── 05_run_ragas.py
└── token_analyzer.py
```


```
TorchDocs/
├─ .gitignore/
├─ data/
│  ├─ raw/                   # HTML 원본 (version 별 ZIP 파일)
│  │  ├─ pytorch_docs_2.8.zip
│  │  └─ ...
│  ├─ processed/            # 전처리 완료된 version별 jsonl 파일
│  │  ├─ token_analysis.txt # 각 json 파일이 512 토큰을 넘는지 검사한 결과 로그
│  │  ├─ torchdocs_2.8_chunks_e5.jsonl
│  │  └─ ...
│  └── eval/                # 평가 데이터셋(RAGAS 합성 or 수작업)
│       ├── ragas_synth.jsonl
│       └── human_gold.jsonl
├─ preprocessing.py         # HTML 원본 전처리 스크립트
├─ qa_chunks.py             # .jsonl 파일 품질 검사 스크립트
├─ token_analyzer.py        # 토큰 수 초과 검사
├─ script_guide.txt         # CLI 입력 가이드
├─ make_embeddings.py       # embeddings 생성
├── embeddings/
│   ├── intfloat_e5_large_v2/                      # 임베딩 모델별 디렉토리
│   │   ├── embeddings_2.4_e5_large_v2_mean.npy    # 문서 임베딩
│   │   ├── ...
│   │   ├── id_2.4_e5_large_v2_mean.json           # id
│   │   ├── ...
│   │   ├── metadata_2.4_e5_large_v2_mean.parquet  # 메타데이터(title, url, etc.)
│   │   └── ...
│   ├── BAAI_bge_base_en/
|   └─ ...
├── generator/
│   └── generator.py
│
│ # ==== 여기까지 현재 완료(8/??) ====
├── index/
│   └── e5-large-v2/
│       ├── faiss.index           # FAISS 인덱스(FlatIP/HNSW/IVF 등)
│       └── stats.json            # 차원, ntotal, 빌드 파라미터
├── retrieval/
│   ├── configs/
│   │   └── default.yaml          # k, mmr, reranker 등 하이퍼파라미터
│   ├── search.py                 # 1차 검색(FAISS/BM25)
│   ├── mmr.py                    # 다양성 제어
│   ├── rerank.py                 # cross-encoder / bge-reranker
│   └── pipeline.py               # retrieve(query) → candidates 정식 API
├── evaluation/
│   ├── build_synth_dataset.py    # ragas 합성 평가셋 생성
│   ├── run_ragas.py              # ragas 실행 스크립트
│   └── reports/                  # 지표 CSV/JSON, 그래프
├── scripts/                      # CLI: 인덱스/평가/배치 유틸
│   ├── build_embeddings.py
│   ├── build_faiss.py
│   ├── batch_retrieve.py
│   └── ablation_sweep.py
├── configs/
│   └── project.yaml              # 공통 경로/모델명/장치 설정
├── env/                          # 환경(선택)
│   └── environment.yml
├── tests/                        # 유닛 테스트
└── README.md

```
