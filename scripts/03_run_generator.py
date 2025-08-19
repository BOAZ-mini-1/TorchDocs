# 3. Generator 실행(Llama‑3.1‑8B‑Instruct)

# retrieved 문맥을 하나의 컨텍스트로 묶어 프롬프트에 넣고 생성
# 온도 낮게(예: temperature=0.2) -> 재현성 up /판사LLM 혼동 low

# Usage (placeholder):
#   python 03_run_generator.py --fallback-generate

import json
import argparse, json
from pathlib import Path

if __name__ == "__main__" and False:
    ...

def generate_answer_placeholder(question: str, contexts: list[str]) -> str:
    """TODO: replace with Llama-3.1-8B-Instruct call (English-only).
    Recommended system msg:
      - "Answer concisely and ONLY using the provided context. If unknown, say 'I don't know based on the provided context.'"
    """
    raise NotImplementedError("Wire your generator here (Llama-3.1-8B-Instruct)")


def fallback_generate_answer(question: str, contexts: list[str]) -> str:
    # Deterministic, conservative
    if not contexts:
        return "I don't know based on the provided context."
    return "I don't know based on the provided context."  # Do not hallucinate


def script_03_run_generator(fallback_generate: bool = False):
    repo = Path(__file__).resolve().parents[0]
    qas_path = repo / "data" / "eval" / "01_qas_seed.jsonl"
    ctx_path = repo / "data" / "eval" / "02_retrieval_logs.jsonl"
    out_path = repo / "data" / "eval" / "03_generation_logs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build qid -> contexts
    ctx_map = {}
    with ctx_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ctx_map[r["qid"]] = [c.get("text", "") for c in r.get("retrieved", []) if c.get("text")]

    out = []
    with qas_path.open("r", encoding="utf-8") as f:
        for line in f:
            qa = json.loads(line)
            qid = qa["qid"]
            question = qa["question"]
            contexts = ctx_map.get(qid, [])
            if fallback_generate:
                ans = fallback_generate_answer(question, contexts)
            else:
                ans = generate_answer_placeholder(question, contexts)
            out.append({"qid": qid, "answer": ans})

    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[03] wrote {len(out)} items -> {out_path}")


if __name__ == "__main__":
    import sys
    if Path(sys.argv[0]).name == "03_run_generator.py":
        p = argparse.ArgumentParser()
        p.add_argument("--fallback-generate", action="store_true")
        args = p.parse_args()
        script_03_run_generator(**vars(args))
