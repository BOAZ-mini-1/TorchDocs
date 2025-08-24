# 4. RAGAS 데이터셋 빌드
# Usage:
#   python scripts/04_build_ragas_dataset.py \
#       --qas data/eval/01_qas.jsonl \
#       --contexts data/eval/02_contexts.jsonl \
#       --answers data/eval/03_answers.jsonl \
#       --out data/eval/ragas_synth.jsonl \
#       --drop-missing --min-contexts 1
import argparse, json
from pathlib import Path

def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l: continue
            yield json.loads(l)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qas", type=str, default="data/eval/01_qas.jsonl")
    ap.add_argument("--contexts", type=str, default="data/eval/02_contexts.jsonl")
    ap.add_argument("--answers", type=str, default="data/eval/03_answers.jsonl")
    ap.add_argument("--out", type=str, default="data/eval/ragas_synth.jsonl")
    ap.add_argument("--drop-missing", action="store_true")
    ap.add_argument("--min-contexts", type=int, default=1)
    args = ap.parse_args()

    # 인덱싱
    q2gt = {}
    for o in iter_jsonl(args.qas):
        q = o.get("question")
        if not q: continue
        q2gt[q] = o

    q2ctx = {}
    for o in iter_jsonl(args.contexts):
        q = o.get("question")
        ctxs = o.get("contexts") or []
        q2ctx[q] = ctxs

    q2ans = {}
    for o in iter_jsonl(args.answers):
        q = o.get("question")
        if not q: continue
        q2ans[q] = o

    n_write = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for q, gt in q2gt.items():
            ctxs = q2ctx.get(q) or []
            if len(ctxs) < args.min_contexts and args.drop-missing:
                continue

            ans_obj = q2ans.get(q) or {}
            answer = ans_obj.get("answer") or ""
            # ragas는 문자열 리스트의 contexts를 선호한다네요
            ctx_texts = []
            for c in ctxs:
                txt = c.get("content") or c.get("text") or ""
                if txt:
                    ctx_texts.append(txt)

            row = {
                "question": q,
                "answer": answer,
                "contexts": ctx_texts,
                "ground_truth": gt.get("ground_truth") or "",
                # (선택) 추적용 메타
                "meta": {
                    "version": gt.get("version"),
                    "source_id": gt.get("source_id"),
                    "source_url": gt.get("source_url"),
                    "used_refs": ans_obj.get("used_refs"),
                }
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n"); n_write += 1

    print(f"[ragas-build] wrote {args.out} (n={n_write})")

if __name__ == "__main__":
    main()
