# 0. 평가 pool 만들기
# Usage:
#   python scripts/00_make_eval_pool.py --version 2.8 --pool-size 300 --out data/eval/00_pool.jsonl
import argparse, json, random
from pathlib import Path

def pick_jsonl_for_version(ver: str):
    cand = [
        f"data/processed/torchdocs_{ver}_chunks_e5.jsonl",
    ]
    for p in cand:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"No jsonl for version {ver}: {cand}")

def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)

def is_reasonable_chunk(o):
    txt = (o.get("text_for_embedding") or o.get("content") or "").strip()
    if len(txt) < 200:           # 너무 짧으면 제외
        return False
    if txt.count("\n") > 200:    # 지나치게 긴 나열 페이지 제외
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=str, default="2.8")
    ap.add_argument("--pool-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/eval/00_pool.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    src = pick_jsonl_for_version(args.version)

    rows = [o for o in iter_jsonl(src) if is_reasonable_chunk(o)]
    if len(rows) == 0:
        raise SystemExit("No eligible chunks")

    # 랜덤 샘플링
    sample = random.sample(rows, k=min(args.pool_size, len(rows)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for o in sample:
            md = o.get("metadata", {}) or {}
            out = {
                "id": o.get("id"),
                "version": str(md.get("version") or args.version),
                "title": md.get("title") or "",
                "url": md.get("url") or "",
                "content": o.get("text_for_embedding") or o.get("content") or "",
            }
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[pool] wrote {args.out} (n={len(sample)}) from {src}")

if __name__ == "__main__":
    main()
