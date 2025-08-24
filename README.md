# TorchDocs
- Directory architecture (draft)
- raw, embeddings 폴더는 용량이 커서 충돌이 일어나므로 업로드 하지 않았음 (.gitignore 참고)

### Location Notices
- 인덱스: intfloat_e5_large_v2/index/faiss.index (필수 배치)
- id 매핑: intfloat_e5_large_v2/embeddings/id_mapping_*.json
- retriever 출력: data/eval/02_contexts.jsonl
- generator 출력: data/eval/03_answers.jsonl
- ragas 묶음: data/eval/ragas_synth.jsonl

## RAG Pipeline 정리
1. 코퍼스 준비 → 전처리/청크
- PyTorch 문서들을 `.jsonl`로 청크화 (text_for_embedding, metadata{title,url,version,...}).

2. 임베딩 & 인덱싱 (FAISS)
- `intfloat/e5-large-v2`로 임베딩 → faiss.index 저장 + id_mapping_*.json(인덱스→로컬 jsonl id 매핑).

3. 질의 → 검색(retriever)
- 쿼리를 e5 규칙("query: ...")으로 임베딩 → FAISS top-N → (버전필터) → MMR/리랭커 → context list로 반환.

4. 생성(generator)
- 상위 k 컨텍스트를 프롬프트에 붙여 LLM(Qwen 등)으로 답변 생성(+간단 인라인 [#1],[#2] 인용).

5. 평가(RAGAS)
(Q, A, contexts, ground_truth) 통합 데이터셋을 만들고, RAGAS 메트릭(정확성/정합성/정밀도/재현율)으로 점수 산출.



---

## Current Folder Structure
```
TorchDocs/
│
├── data/
│   └── processed/
│   │   ├── token_analysis.txt # 각 json 파일이 512 토큰을 넘는지 검사한 결과 로그
│   │   ├── torchdocs_2.8_chunks_e5.jsonl
│   │   └── ...
│   └── eval/
│       ├── 02_contexts.jsonl  # retriever output
│       ├── 03_answers.jsonl   # generator output
│       └── ragas_synth.jsonl  # ragas (예정)
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
├── make_embeddings.py       # embeddings 생성
├── retriever.py
├── generator.py
│
├── scripts/
│   ├── qa_chunks.py             # .jsonl 파일 품질 검사 스크립트
│   ├── token_analyzer.py        # 토큰 수 초과 검사
│   ├── 00_make_eval_pool.py
│   ├── 01_generate_qas.py
│   ├── 02_run_retriever.py
│   ├── 03_run_generator.py
│   ├── 04_build_ragas_dataset.py
│   └── 05_run_ragas.py
│
├── doc/
│   └── offical_tut.ipynb
├── backups/
│   ├── retriever_backup.py
│   └── generator_backup.py
├── .gitignore
├── README.md
```

---

## Script guide for each files

<details>
  <summary> CLI guide (Click)</summary>

### 1. 전처리 코드 실행 스크립트

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

### 2. 각 jsonl 파일 품질 검사

```bash
python qa_chunks.py data/processed/torchdocs_2.8_chunks.jsonl
```

### 3. Retriever 데모 검사

1. 단일 질문
```bash
python scripts/02_run_retriever.py --q "How does autograd compute gradients in PyTorch 2.8?" --top-k 5 --fetch-k 40 --out data/eval/02_contexts.jsonl
head -n1 data/eval/02_contexts.jsonl | jq .
```

2. 버전 자동 파싱 확인 (v2.6로 입력했을 때 regex가 잘 먹는지?)
```bash
python scripts/02_run_retriever.py --q "In PyTorch v2.6, how to register a custom autograd Function?" --top-k 5 --out data/eval/02_contexts_v26.jsonl
head -n1 data/eval/02_contexts_v26.jsonl | jq .
```

3. (optional) CrossEncoder 리랭킹
```bash
python scripts/02_run_retriever.py --q "torch.compile limitations in 2.8" --use-cross --top-k 5 --out data/eval/02_contexts_ce.jsonl
```

### 4. Generator 데모 검사

1. (필요 시) retriever를 교정 필터/리랭커로 다시 실행

```bash
python scripts/02_run_retriever.py --q "How does autograd compute gradients in PyTorch 2.8?" --top-k 5 --fetch-k 60 --use-cross --out data/eval/02_contexts.jsonl
```

2. generator
```bash
python scripts/03_run_generator.py --contexts data/eval/02_contexts.jsonl --k 4
head -n1 data/eval/03_answers.jsonl | jq .
```

### 5. RAGAS pipeline

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

### ===========DEMO용 CLI (미완성 상태이니 just 예시.)===========

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
</details>

---

# script/(00~05) 설명

## 00\_make\_eval\_pool.py — "평가 후보 청크 풀 만들기"

- **입력**: 버전별 전처리 산출물 jsonl (ex. `data/processed/torchdocs_2.8_chunks_e5.jsonl`)
- **역할**: 평가용으로 말이 되는 청크를 샘플링해 풀(`00_pool.jsonl`)을 생성
  - 너무 짧거나, 지나치게 길고 잡음 많은 블록은 걸러냄
- **출력(jsonl 각 줄)**:

  ```json
  {
    "id": "<local_id>",
    "version": "2.8",
    "title": "...",
    "url": "pytorch_docs_2.8/...",
    "content": "<text_for_embedding 또는 content>"
  }
  ```

---

## 01\_generate\_qas.py — "평가용 질문/정답 만들기"

- **입력**: `00_pool.jsonl`
- **역할**: 각 청크에서 **Q/A 페어**를 생성
  - LLM이 있으면 간단 템플릿으로 생성, 없으면 **fallback 규칙**으로 무난한 Q/A 1-2개 생성
- **출력(jsonl 각 줄)**:

  ```json
  {
    "question": "How does autograd work?",
    "ground_truth": "Autograd builds a computation graph ...",
    "version": "2.8",
    "source_id": "<local_id>",      // 추적용
    "source_title": "...",
    "source_url": "..."
  }
  ```
- **주의 사항**: ground\_truth가 **컨텍스트 기반**인지(즉, hallucination X), 질문이 너무 광범위하지 않은지 - 이게 부실하면 나중에 "answer\_correctness"가 왜곡되기 때문 

---

## 02\_run\_retriever.py — "질문별 컨텍스트 검색"

- **입력**:
  - 단일 질문 `--q "..."` 또는
  - 배치 `--questions-jsonl data/eval/01_qas.jsonl`
- **역할**:

  1. 질문에서 버전 표기 자동 파싱("2.8", "v2.6" 등) -> 없으면 기본값(2.8)
  2. e5 임베딩 -> FAISS top-N.
  3. **인덱스 -> 로컬 jsonl id** 변환(여기서 `id_mapping_*.json` 사용)
  4. 버전 필터(정규화해서 "2.8"로 맞춤, 안 맞으면 무필터 재시도)
  5. **MMR**로 중복 억제(+ 선택: CrossEncoder 리랭커)

- **출력(jsonl 각 줄)**:

  ```json
  {
    "question": "...",
    "version": "2.8",
    "contexts": [
      {"id":"<local_id>","url":"...","version":"2.8","title":"...","content":"...","score":0.81},
      ...
    ]
  }
  ```

---

## 03\_run\_generator.py — "컨텍스트로 답변 생성"

- **입력**: `02_contexts.jsonl`
- **역할**: 각 질문별 상위 k 컨텍스트를 프롬프트에 붙여 **Qwen 등으로 답변** 생성
  - system/user 템플릿은 팀 백업 톤을 살림(근거 기반, [#i] 표기, 250~300 단어 제한 등)
  - 최대 프롬프트 토큰 예산/슬라이스를 걸어 OOM 방지

- **출력(jsonl 각 줄)**:

  ```json
  {
    "question": "...",
    "version": "2.8",
    "answer": "... (Markdown, [#1],[#2] 인용 포함)",
    "used_ctx_ids": ["id1","id2","id3","id4"],
    "used_refs": [{"ref_num":1,"id":"id1","url":"...","version":"2.8","title":"..."}],
    "k": 4
  }
  ```

- **주의 사항**:
  * 환경 문제: `transformers`가 `accelerate/bitsandbytes`를 로드하며 충돌 → \*\*CPU 경로(가벼운 모델)\*\*로 먼저 확인하거나 Colab 사용 (라이브러리 충돌나는지 지수 컴에서 계속 안돼요 지금... 코랩으로 해야 할듯)
  * 너무 긴 컨텍스트/너무 높은 k → LLM 입력 초과/질 저하

---

## 04\_build\_ragas\_dataset.py — "RAGAS 입력 포맷으로 합치기"

- **입력**:
  * `01_qas.jsonl`(Q, ground\_truth)
  * `02_contexts.jsonl`(retrieved contexts)
  * `03_answers.jsonl`(generated answer)

- **역할**: 질문을 키로 **조인**해서 RAGAS가 먹는 포맷으로 **정규화**
- **출력(jsonl 각 줄)** → 최종 평가 입력:

  ```json
  {
    "question": "...",
    "answer": "...",
    "contexts": ["ctx text 1", "ctx text 2", ...],   // 문자열 리스트
    "ground_truth": "..."
  }
  ```

  *(메타 필드는 선택적으로 함께 저장 가능)*

- **주의 사항**:
  - `contexts`는 **문자열 리스트**여야 함 (객체 리스트 x)
  * 질문 키가 파일 간 정확히 매칭되는지(공백/문자 차이로 join 실패 → drop-missing 옵션 확인)

---

## 05\_run\_ragas.py — "metric 계산"
- **입력**: `ragas_synth.jsonl`
- **역할**: RAGAS 실행(LLM 판사 + 임베딩) → 점수 산출
- **사용 메트릭 (핵심 4개)**
  - **answer\_correctness**: 답변 vs ground\_truth를 LLM이 판별(문장/의미 일치)
  - **faithfulness**: 답변이 제공된 contexts에 근거했는가? (환상/외삽 여부)
  - **context\_precision**: 가져온 컨텍스트 중 관련 있는 것의 비율 (낮으면 **탐색 정확도** 부족)
  - **context\_recall**: 근거로 필요한 정보를 얼마나 회수했는가 (낮으면 **회수율**/청크/임베딩 문제가 의심)

- **출력**: report 최종 JSON(sample)

  ```json
  {
    "n": 180,
    "scores": {
      "answer_correctness": 0.62,
      "faithfulness": 0.71,
      "context_precision": 0.48,
      "context_recall": 0.59
    }
  }
  ```

- **해석/디버깅 가이드(by GPT)**

  * **precision↓, recall↑** → 많이 가져오지만 잡음 많음 ⇒ retriever 후처리/리랭커/코드 패널티.
  * **precision↑, recall↓** → 정밀하지만 놓침 ⇒ `fetch_k↑`, MMR λ 조정(0.4\~0.7), chunk 길이/stride 조정.
  * **faithfulness↓, precision↑** → 근거는 맞는데 답변이 외삽 ⇒ 생성 프롬프트를 “컨텍스트 외 금지”로 강화, `temperature↓`(0.1\~0.2).
  * **answer\_correctness↓** → ground\_truth 품질/coverage 문제 가능 ⇒ 01단계 LLM Q/A 생성 품질 개선 or n-per-chunk=3로 다양성↑.

---



