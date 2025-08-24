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

---

# 2. 각 jsonl 파일 품질 검사

```bash
python qa_chunks.py data/processed/torchdocs_2.8_chunks.jsonl
```

---

# 3. Retriever 데모 검사

### 3-1. 단일 질문
```bash
python scripts/02_run_retriever.py --q "How does autograd compute gradients in PyTorch 2.8?" --top-k 5 --fetch-k 40 --out data/eval/02_contexts.jsonl
head -n1 data/eval/02_contexts.jsonl | jq .
```

### 3-2. 버전 자동 파싱 확인 (v2.6로 입력했을 때 regex가 잘 먹는지?)
```bash
python scripts/02_run_retriever.py --q "In PyTorch v2.6, how to register a custom autograd Function?" --top-k 5 --out data/eval/02_contexts_v26.jsonl
head -n1 data/eval/02_contexts_v26.jsonl | jq .
```

### 3-3. (optional) CrossEncoder 리랭킹
```bash
python scripts/02_run_retriever.py --q "torch.compile limitations in 2.8" --use-cross --top-k 5 --out data/eval/02_contexts_ce.jsonl
```

---

# Generator 데모 검사

### 4-1. (필요시) retriever를 교정 필터/리랭커로 다시 실행

```bash
python scripts/02_run_retriever.py --q "How does autograd compute gradients in PyTorch 2.8?" --top-k 5 --fetch-k 60 --use-cross --out data/eval/02_contexts.jsonl
```

### 4-2. generator
```bash
python scripts/03_run_generator.py --contexts data/eval/02_contexts.jsonl --k 4
head -n1 data/eval/03_answers.jsonl | jq .
```

# 5. RAGAS pipeline

```bash
# 0) 풀 만들기
python scripts/00_make_eval_pool.py --version 2.8 --pool-size 100

# 1) Q/A 생성 (일단 fallback)
python scripts/01_generate_qas.py --in data/eval/00_pool.jsonl --n-per-chunk 2 --out data/eval/01_qas.jsonl

# 2) Retriever 
python scripts/02_run_retriever.py --top-k 5 --fetch-k 60 --use-cross --out data/eval/02_contexts.jsonl --q "How does autograd compute gradients in PyTorch 2.8?"
#   배치로 하고 싶으면 --questions-jsonl data/eval/01_qas.jsonl 로 돌려도 됨

# 3) Generator
export GEN_MODEL="Qwen/Qwen2.5-3B-Instruct"   # CPU면 1.5B/3B 권장, CUDA 있으면 7B도 OK
python scripts/03_run_generator.py --contexts data/eval/02_contexts.jsonl --k 4

# 4) RAGAS용 합치기
python scripts/04_build_ragas_dataset.py --drop-missing --min-contexts 1

# 5) RAGAS 평가
python scripts/05_run_ragas.py --in data/eval/ragas_synth.jsonl --report data/eval/ragas_report.json


```

# ===========DEMO용 CLI (미완성 상태이니 just 예시.)===========

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
