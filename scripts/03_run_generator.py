# 3. Generator 실행(Llama‑3.1‑8B‑Instruct)

# retrieved 문맥을 하나의 컨텍스트로 묶어 프롬프트에 넣고 생성
# 온도 낮게(예: temperature=0.2) -> 재현성 up /판사LLM 혼동 low

import json

def generate_answer(question, contexts):
    # Llama 3.1 8B Instruct 호출부
    # system: "다음 context만을 근거로 간결하고 정확히 답하세요. 모르면 '모름'이라고 답하라."
    # user: f"Question: {question}\n\nContext:\n{ctx_block}"
    raise NotImplementedError

# qid -> contexts 매핑
ctx_map = {}
with open("eval/02_retrieval_logs.jsonl") as f:
    for l in f:
        r = json.loads(l)
        ctx_map[r["qid"]] = [c["text"] for c in r["retrieved"]]

out = []
with open("eval/01_qas_seed.jsonl") as f:
    for l in f:
        qa = json.loads(l)
        ctxs = ctx_map.get(qa["qid"], [])
        ans = generate_answer(qa["question"], ctxs)
        out.append({"qid": qa["qid"], "answer": ans})

with open("eval/03_generation_logs.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
