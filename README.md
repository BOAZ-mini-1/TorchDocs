# TorchDocs
- Directory architecture (draft)
- raw, embeddings 폴더는 용량이 커서 충돌이 일어나므로 업로드 하지 않았음 (.gitignore 참고)

## Current File Structure
```
TorchDocs/
│
├── data/
│   └── processed/
│       ├── token_analysis.txt # 각 json 파일이 512 토큰을 넘는지 검사한 결과 로그
│       ├── torchdocs_2.8_chunks_e5.jsonl
│       └── ...
│   
├── intfloat_e5_large_v2/
│   ├── embeddings/
│   │   ├── embeddings_torchdocs_2.8_chunks_e5_intfloat_e5_large_v2_mean.npy  # 문서 임베딩
│   │   └── ...
│   │   ├── id_mapping_torchdocs_2.8_chunks_e5_intfloat_e5_large_v2_mean.json # id
│   │   └── ...
│   │
│   └── index/
│       ├── faiss.index
│       ├── faiss.index.zip  # 원본 faiss.index의 용량으로 인한 github용 압축 처리
│       ├── global_ids.json
│       └── stats.json
│
├── preprocessing.py         # HTML 원본 전처리 스크립트
├── qa_chunks.py             # .jsonl 파일 품질 검사 스크립트
├── token_analyzer.py        # 토큰 수 초과 검사
├── script_guide.txt         # CLI 입력 가이드
├── make_embeddings.py       # embeddings 생성
├── retriever.py
│
├── scripts/
│   ├── 00_make_eval_pool.py
│   ├── 01_generate_qas.py
│   ├── 02_run_retriever.py
│   ├── 03_run_generator.py
│   ├── 04_build_ragas_dataset.py
│   └── 05_run_ragas.py
│
├── doc/
│   └── offical_tut.ipynb
├── .gitignore
├── README.md
```

