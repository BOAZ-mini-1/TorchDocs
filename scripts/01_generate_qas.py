# 1. chunk → 질문/정답 자동 생성

# 생성 기준: "해당 chunk만 읽으면 답할 수 있는, 단일‑스팬 또는 짧은 서술형 문제"
# 한 청크당 1-2문항 권장(중복 개념/표현 피하기)
# ground_truth는 chunk에서 직접 근거가 나오도록 (추상화/의견 금지)

# 프롬프트 핵심 규칙
# 질문은 명확/구체적(모호한 "설명하라" 금지)
# 정답은 짧고 사실적(1-3문장)
# 반드시 해당 chunk에서 답을 찾을 수 있어야 함

# Usage example (LLM not wired yet; fallback will create simple QAs):
#   python 01_generate_qas.py --n-per-chunk 2 --fallback

if __name__ == "__main__" and False:
    ...

import argparse, json, re, uuid
from pathlib import Path


def _first_sentences(text: str, max_sents: int = 2) -> str:
    """Return up to N short sentences as a naive ground_truth fallback."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[:max_sents])[:600]


def llm_generate_qas_english(text: str, n: int = 2):
    """TODO: plug LLM call here

    The prompt should instruct:
      - Ask specific, answerable questions using ONLY this chunk.
      - Produce short factual answers (1-3 sentences), no opinions.
      - English only.

    Return structure: List[{"question": str, "ground_truth": str}]
    """
    raise NotImplementedError("Connect your LLM (e.g., Qwen/Llama) to generate English QAs.")


def fallback_generate_qas(text: str, n: int = 1):
    # Very conservative: one generic question + extractive ground_truth.
    title = None
    m = re.search(r"^\s*([A-Za-z0-9 _#:\-]{3,})\s*$", text.splitlines()[0])
    if m:
        title = m.group(1).strip("# ")
    q = f"What is the main point of this passage{f' about {title}' if title else ''}?"
    gt = _first_sentences(text, max_sents=2)
    return [{"question": q, "ground_truth": gt}]


def script_01_generate_qas(
    n_per_chunk: int = 2,
    fallback: bool = False,
):
    repo = Path(__file__).resolve().parents[0]
    pool_path = repo / "data" / "eval" / "00_eval_pool.jsonl"
    out_path = repo / "data" / "eval" / "01_qas_seed.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = []
    with pool_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            if fallback:
                qas = fallback_generate_qas(text, n=1)
            else:
                qas = llm_generate_qas_english(text, n_per_chunk)
            for qa in qas:
                out.append({
                    "qid": f"q_{uuid.uuid4().hex[:8]}",
                    "chunk_id": item.get("id"),
                    "question": qa["question"],
                    "ground_truth": qa["ground_truth"],
                })

    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            # light rule checks
            if not r["question"] or not r["ground_truth"]:
                continue
            if len(r["ground_truth"]) > 1000:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[01] wrote {len(out)} items -> {out_path}")


if __name__ == "__main__":
    import sys
    if Path(sys.argv[0]).name == "01_generate_qas.py":
        p = argparse.ArgumentParser()
        p.add_argument("--n-per-chunk", type=int, default=2)
        p.add_argument("--fallback", action="store_true")
        args = p.parse_args()
        script_01_generate_qas(**vars(args))
