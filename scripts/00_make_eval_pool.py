# 0. 평가 pool 만들기
import json, random
import numpy as np
from collections import defaultdict

random.seed(2025)

MIN_LEN, MAX_LEN = 200, 1200  # 너무 짧거나 너무 긴 청크 제외 (200-1200자)
PER_DOC_MAX = 5
POOL_SIZE = 300  # 최종 청크 수(질문은 이후 청크당 1~2개 생성)

for version in np.arange(2.4, 2.9, 0.1):
    with open(f"data/processed/torchdocs_{version}_chunks_e5.jsonl", "r") as f:
        chunks = [json.loads(l) for l in f]

# 길이 필터
chunks = [c for c in chunks if MIN_LEN <= len(c["text_for_embedding"]) <= MAX_LEN]

# 문서별 상한
by_doc = defaultdict(list)
for c in chunks:
    by_doc[c["doc_id"]].append(c)
pool = []
for doc_id, arr in by_doc.items():
    random.shuffle(arr)
    pool.extend(arr[:PER_DOC_MAX])

# 전체 규모 제한
random.shuffle(pool)
pool = pool[:POOL_SIZE]

with open("eval/00_eval_pool.jsonl", "w") as f:
    for c in pool:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

'''
{"qid":"q_000001","chunk_id":"c_0001",
 "question":"ATen은 PyTorch에서 어떤 역할을 하나요?",
 "ground_truth":"ATen은 PyTorch의 핵심 텐서 라이브러리로 Python/C++ 인터페이스의 기반이 된다."}
'''