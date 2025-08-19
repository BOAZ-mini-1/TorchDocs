# 5. RAGAS 실행(Qwen2.5‑14B‑Instruct + e5‑large‑v2)

# 지표: faithfulness, answer_relevance, context_recall, context_precision

# Correctness는 ground_truth 품질이 좋으면 추가

import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
# 필요시: from ragas.metrics import answer_correctness
# LLM/Ebd 어댑터는 환경에 맞는 래퍼 사용(OpenAI 호환/transformers 등)

def load_llm_qwen():
    # Qwen2.5-14B-Instruct를 judge로 쓰는 래퍼 생성
    # 예: ragas.integrations.openai.OpenAIChat(model="...")
    raise NotImplementedError

def load_embeddings_e5():
    # sentence-transformers(e5-large-v2) 임베딩 래퍼
    raise NotImplementedError

rows = []
with open("eval/04_ragas_dataset.jsonl") as f:
    for l in f:
        rows.append(json.loads(l))

df = pd.DataFrame(rows)  # cols: question, answer, ground_truth, contexts(list[str])

llm = load_llm_qwen()
emb = load_embeddings_e5()

res = evaluate(
    dataset=df,
    metrics=[faithfulness, answer_relevance, context_recall, context_precision],
    llm=llm,
    embeddings=emb,
)
# res는 metric별 점수와 항목별 점수까지 갖고 있는 객체/DF (버전에 따라 다름)

with open("eval/05_ragas_report.json", "w") as f:
    f.write(res.to_json() if hasattr(res, "to_json") else json.dumps(res, ensure_ascii=False))
