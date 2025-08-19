# 2. Retriever 실행(진짜 시스템 성능 반영)

# 생성된 질문으로 현재 retriever를 호출해 top‑k 문맥을 수집
# chunk_id 매칭이 가능하면 나중에 gold vs retrieved 비교가 쉬움

import json

TOP_K = 5

def retrieve(query, top_k=TOP_K):
    # retriever 연결부. 반환 [{"chunk_id": "...", "text": "..."}] * top_k
    raise NotImplementedError

out = []
with open("eval/01_qas_seed.jsonl") as f:
    for l in f:
        item = json.loads(l)
        hits = retrieve(item["question"], top_k=TOP_K)
        out.append({"qid": item["qid"], "retrieved": hits})

with open("eval/02_retrieval_logs.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
