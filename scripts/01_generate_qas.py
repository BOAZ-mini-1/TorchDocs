# 1. chunk → 질문/정답 자동 생성
# Usage:
#   python scripts/01_generate_qas.py --in data/eval/00_pool.jsonl --n-per-chunk 2 --out data/eval/01_qas.jsonl
#   (옵션) --llm Qwen/Qwen2.5-3B-Instruct  → LLM 생성 / 생략 시 규칙기반 fallback
import argparse, json, re, random
from pathlib import Path
from typing import List, Dict, Any, Optional

def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l: continue
            yield json.loads(l)

def sent_split(txt: str) -> List[str]:
    # 단순 분할
    parts = re.split(r'(?<=[.!?])\s+', txt.strip())
    return [p for p in parts if p]

def fallback_qas_one(chunk: Dict[str, Any], n: int = 2) -> List[Dict[str, Any]]:
    """llm 오류 시 rule-based fallback 구현 - 타이틀/첫문장을 근거로 간단 Q/A 생성"""
    title = chunk.get("title") or ""
    ctx = chunk.get("content") or ""
    sents = sent_split(ctx)
    s0 = sents[0] if sents else ctx[:200]

    cands = []
    if title:
        cands.append(("What is {}?".format(title.strip().rstrip("#").strip()),
                      f"{title} is described as follows: {s0}"))
    if "autograd" in ctx.lower():
        cands.append(("How does autograd work?",
                      "Autograd builds a computation graph and computes gradients via reverse-mode autodiff. " + s0))
    if "tensor" in ctx.lower():
        cands.append(("What is a Tensor in PyTorch?",
                      "A Tensor is a multi-dimensional array with automatic differentiation support. " + s0))

    # 채워 넣기
    if not cands:
        cands.append((
            "Summarize the main idea of this section.",
            s0
        ))

    random.shuffle(cands)
    out = []
    for q, a in cands[:n]:
        out.append({
            "question": q.strip(),
            "ground_truth": a.strip(),
            "version": chunk.get("version"),
            "source_id": chunk.get("id"),
            "source_title": chunk.get("title"),
            "source_url": chunk.get("url"),
        })
    return out

# ------- LLM 생성기 --------
def llm_generate_qas_english(chunks: List[Dict[str, Any]], model_name: Optional[str], per_chunk: int) -> List[Dict[str, Any]]:
    # 모델이 제공되면 간단 템플릿으로 Q/A 생성. (없으면 fallback)
    if not model_name:
        res = []
        for c in chunks:
            res.extend(fallback_qas_one(c, n=per_chunk))
        return res

    # transformers pipeline 로딩 (가벼운 모델)
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")\
          .to("cuda" if __import__("torch").cuda.is_available() else "cpu")
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256)

    prompt_tpl = (
        "You are a question generation bot. Given the context below, write {k} helpful Q/A pairs in English.\n"
        "Return as JSON lines with keys: question, answer.\n\n[CONTEXT]\n{ctx}\n"
    )

    res = []
    for c in chunks:
        ctx = (c.get("content") or "")[:1200]
        prompt = prompt_tpl.format(k=per_chunk, ctx=ctx)
        out = gen(prompt)[0]["generated_text"]
        # 매우 보수적으로 파싱: { "question": ..., "answer": ... } 라인이 보이면 캡쳐
        matches = re.findall(r'{"question"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"}', out)
        if matches:
            for q, a in matches[:per_chunk]:
                res.append({
                    "question": q.strip(),
                    "ground_truth": a.strip(),
                    "version": c.get("version"),
                    "source_id": c.get("id"),
                    "source_title": c.get("title"),
                    "source_url": c.get("url"),
                })
        else:
            # 파싱 실패 시 fallback
            res.extend(fallback_qas_one(c, n=per_chunk))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="data/eval/00_pool.jsonl")
    ap.add_argument("--out", type=str, default="data/eval/01_qas.jsonl")
    ap.add_argument("--n-per-chunk", type=int, default=2)
    ap.add_argument("--llm", type=str, default=None)  # Qwen/Qwen2.5-3B-Instruct
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    chunks = list(iter_jsonl(args.inp))
    qas = llm_generate_qas_english(chunks, args.llm, args.n_per_chunk)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for row in qas:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[qas] wrote {args.out} (n={len(qas)})")

if __name__ == "__main__":
    main()
