# 1. 전처리 코드 실행 스크립트

```bash
python preprocessing.py \
  --zip data/raw/pytorch_docs_2.8.zip \
  --out data/processed/torchdocs_2.8_chunks_e5.jsonl \
  --chunker hf \
  --hf_model intfloat/e5-large-v2 \
  --hf_max_len 512 \
  --hf_reserve 10 \
  --max_tokens 510 \
  --overlap 60 \
  --default_version 2.8 --force_version 2.8 \
  --ignore_canonical --max_code_lines 20
```

# 2. 각 jsonl 파일 품질 검사

```bash
python qa_chunks.py data/processed/torchdocs_2.8_chunks.jsonl
```

# ================DEMO용 CLI (미완성 상태이니 just 예시.)================

```bash
# 0) (데모 머신에) 인덱스 배치
ls intfloat_e5_large_v2/index/faiss.index  # 존재 여부 체크 

# 1) 품질 검사 (이미 완료함)
python qa_chunks.py data/processed/torchdocs_2.8_chunks_e5.jsonl
python token_analyzer.py --jsonl data/processed/torchdocs_2.8_chunks_e5.jsonl

# 2) Retriever(질문 하나로 빠른 시연)
python -c "from retriever import search; 
print(search('How does autograd compute gradients in PyTorch?', top_k=5))"

# 3) Generator(동일 질문을 컨텍스트와 함께)
python -c "from retriever import search; from generator import generate; 
docs=search('How does autograd compute gradients in PyTorch?', top_k=5); 
print(generate('How does autograd compute gradients in PyTorch?', [d['content'] for d in docs]))"

# 4) RAGAS(배치 평가)
python scripts/00_make_eval_pool.py --version 2.8 --pool-size 100
python scripts/01_generate_qas.py --n-per-chunk 2 --fallback
python scripts/02_run_retriever.py --top-k 5 --use-faiss-fallback --version 2.8
python scripts/03_run_generator.py
python scripts/04_build_ragas_dataset.py --drop-missing --min-contexts 1
python scripts/05_run_ragas.py --dry-run


```
