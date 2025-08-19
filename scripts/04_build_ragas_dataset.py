# 4. RAGAS 데이터셋 빌드
# Usage:
#   python 04_build_ragas_dataset.py --drop-missing --min-contexts 1

import json
import argparse, json
from pathlib import Path

if __name__ == "__main__" and False:
    ...


def script_04_build_ragas_dataset(
    drop_missing: bool = True,
    min_contexts: int = 1,
):
    repo = Path(__file__).resolve().parents[0]
    qas_path = repo / "data" / "eval" / "01_qas_seed.jsonl"
    ctx_path = repo / "data" / "eval" / "02_retrieval_logs.jsonl"
    ans_path = repo / "data" / "eval" / "03_generation_logs.jsonl"
    out_path = repo / "data" / "eval" / "ragas_synth.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qa_map, ctx_map, ans_map = {}, {}, {}

    with qas_path.open("r", encoding="utf-8") as f:
        for l in f:
            x = json.loads(l)
            qa_map[x["qid"]] = x

    with ctx_path.open("r", encoding="utf-8") as f:
        for l in f:
            x = json.loads(l)
            ctx = []
            for c in x.get("retrieved", []):
                t = (c.get("text") or "").strip()
                if t:
                    ctx.append(t)
            # de-dup conservative
            seen = set()
            uniq = []
            for t in ctx:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            ctx_map[x["qid"]] = uniq

    with ans_path.open("r", encoding="utf-8") as f:
        for l in f:
            x = json.loads(l)
            ans_map[x["qid"]] = x.get("answer", "")

    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for qid, qa in qa_map.items():
            rec = {
                "question": qa.get("question", ""),
                "answer": ans_map.get(qid, ""),
                "ground_truth": qa.get("ground_truth", ""),
                "contexts": ctx_map.get(qid, []),
            }
            if drop_missing:
                if not rec["question"] or not rec["answer"]:
                    continue
                if len(rec["contexts"]) < min_contexts:
                    continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[04] wrote {kept} items → {out_path}")


if __name__ == "__main__":
    import sys
    if Path(sys.argv[0]).name == "04_build_ragas_dataset.py":
        p = argparse.ArgumentParser()
        p.add_argument("--drop-missing", action="store_true")
        p.add_argument("--min-contexts", type=int, default=1)
        args = p.parse_args()
        script_04_build_ragas_dataset(**vars(args))

