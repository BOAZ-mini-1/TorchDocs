# 4. RAGAS 데이터셋 빌드

# 최소 필수 컬럼: question, answer, contexts
# 권장: ground_truth 포함(있으면 Correctness 등 확장 지표 가능)

import json

qa_map, ctx_map, ans_map = {}, {}, {}

with open("eval/01_qas_seed.jsonl") as f:
    for l in f:
        x = json.loads(l); qa_map[x["qid"]] = x

with open("eval/02_retrieval_logs.jsonl") as f:
    for l in f:
        x = json.loads(l); ctx_map[x["qid"]] = [c["text"] for c in x["retrieved"]]

with open("eval/03_generation_logs.jsonl") as f:
    for l in f:
        x = json.loads(l); ans_map[x["qid"]] = x["answer"]

with open("eval/04_ragas_dataset.jsonl", "w") as f:
    for qid, qa in qa_map.items():
        record = {
            "question": qa["question"],
            "answer": ans_map.get(qid, ""),
            "ground_truth": qa["ground_truth"],
            "contexts": ctx_map.get(qid, [])
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
