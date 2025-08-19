# 0. 평가 pool 만들기
# Usage example:
#   python 00_make_eval_pool.py \
#     --version 2.8 \
#     --pool-size 300 \
#     --min-chars 200 --max-chars 1200 \
#     --per-doc-max 5

if __name__ == "__main__" and False:
    ...

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def _guess_doc_key(meta: dict) -> str:
    """Build a per-document key for stratified sampling.
    Prefer path; fallback to title; fallback to url.
    """
    if not meta:
        return "unknown_doc"
    for k in ["path", "title", "url"]:
        v = meta.get(k)
        if v:
            return f"{meta.get('version','')}//{v}"
    return f"{meta.get('version','')}//unknown_doc"


def script_00_make_eval_pool(
    version: str = "2.8",
    pool_size: int = 300,
    min_chars: int = 200,
    max_chars: int = 1200,
    per_doc_max: int = 5,
    lang: str = "en",
    seed: int = 2025,
):
    random.seed(seed)
    root = Path(__file__).resolve().parents[0]
    repo = root  # this file is expected to live in TorchDocs/ by default
    processed = repo / "data" / "processed" / f"torchdocs_{version}_chunks_e5.jsonl"
    out_path = repo / "data" / "eval" / "00_eval_pool.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load chunks
    chunks = []
    with processed.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("text_for_embedding") or obj.get("content") or ""
            meta = obj.get("metadata", {})
            # Filter by lang if present
            if lang and meta.get("lang") and meta.get("lang") != lang:
                continue
            if len(t) < min_chars or len(t) > max_chars:
                continue
            chunks.append({
                "id": obj.get("id"),
                "text": t,
                "metadata": meta,
            })

    # Bucket by document key
    by_doc = defaultdict(list)
    for c in chunks:
        key = _guess_doc_key(c.get("metadata", {}))
        by_doc[key].append(c)

    # Cap per-doc & shuffle
    pool = []
    for key, arr in by_doc.items():
        random.shuffle(arr)
        pool.extend(arr[:per_doc_max])

    random.shuffle(pool)
    pool = pool[:pool_size]

    with out_path.open("w", encoding="utf-8") as f:
        for c in pool:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[00] wrote {len(pool)} items -> {out_path}")


if __name__ == "__main__":
    import sys
    if Path(sys.argv[0]).name == "00_make_eval_pool.py":
        p = argparse.ArgumentParser()
        p.add_argument("--version", default="2.8")
        p.add_argument("--pool-size", type=int, default=300)
        p.add_argument("--min-chars", type=int, default=200)
        p.add_argument("--max-chars", type=int, default=1200)
        p.add_argument("--per-doc-max", type=int, default=5)
        p.add_argument("--lang", default="en")
        p.add_argument("--seed", type=int, default=2025)
        args = p.parse_args()
        script_00_make_eval_pool(**vars(args))
