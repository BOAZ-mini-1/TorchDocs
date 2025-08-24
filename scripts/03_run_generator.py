# 3. Generator 실행(Llama‑3.1‑8B‑Instruct)

# file: scripts/03_run_generator.py
import sys, json, argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from generator import generate

"""
입력: 02_contexts.jsonl (각 줄: {"question":..., "version":..., "contexts":[...]})
출력: 03_answers.jsonl (각 줄: {"question","version","answer","used_ctx_ids","used_refs"})
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=str, default="data/eval/02_contexts.jsonl")
    ap.add_argument("--out", type=str, default="data/eval/03_answers.jsonl")
    ap.add_argument("--k", type=int, default=4)  # 상위 k개 컨텍스트만 사용
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.contexts, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question","")
            version = obj.get("version")
            ctxs = (obj.get("contexts") or [])[: args.k]
            if not ctxs:
                ans = {
                    "question": q,
                    "version": version,
                    "answer": "(no contexts) I don't know.",
                    "used_ctx_ids": [],
                    "used_refs": [],
                }
            else:
                out = generate(q, ctxs)
                ans = {
                    "question": q,
                    "version": version,
                    **out
                }
            w.write(json.dumps(ans, ensure_ascii=False) + "\n")
            n += 1

    print(f"[generator] wrote: {args.out} ({n} lines)")

if __name__ == "__main__":
    main()
