# 1. chunk → 질문/정답 자동 생성

# 생성 기준: "해당 chunk만 읽으면 답할 수 있는, 단일‑스팬 또는 짧은 서술형 문제"
# 한 청크당 1-2문항 권장(중복 개념/표현 피하기)
# ground_truth는 chunk에서 직접 근거가 나오도록 (추상화/의견 금지)

# 프롬프트 핵심 규칙
# 질문은 명확/구체적(모호한 "설명하라" 금지)
# 정답은 짧고 사실적(1-3문장)
# 반드시 해당 chunk에서 답을 찾을 수 있어야 함

import json, uuid

def llm_generate_qas(text, n=2):
    # LLM에 맞추어 수정 필요 (qwen)
    # 반환: [{"question": "...", "ground_truth": "..."} ...]
    # 프롬프트는 "이 텍스트만 근거로..." 규칙을 강하게 명시
    raise NotImplementedError

out = []
with open("eval/00_eval_pool.jsonl") as f:
    for line in f:
        chunk = json.loads(line)
        qas = llm_generate_qas(chunk["text_for_embedding"], n=2)
        for qa in qas:
            out.append({
                "qid": f"q_{uuid.uuid4().hex[:8]}",
                "chunk_id": chunk["chunk_id"],
                "question": qa["question"],
                "ground_truth": qa["ground_truth"]
            })

with open("eval/01_qas_seed.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 필요한 검수: ground_truth가 질문과 같은 문장 복붙인지, ground_truth 길이>400자 같은 케이스 있다면 drop할 것
'''
{"qid":"q_000001","retrieved":[
  {"chunk_id":"c_0001","text":"ATen is fundamentally a tensor library..."},
  {"chunk_id":"c_0932","text":"The Tensor class defines many operations..."}
]}
'''