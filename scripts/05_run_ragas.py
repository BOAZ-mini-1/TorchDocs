# 5. RAGAS 실행(Qwen2.5‑14B‑Instruct)

import json, argparse, torch
from pathlib import Path

from datasets import Dataset
from ragas.metrics import answer_correctness, faithfulness, context_precision, context_recall
from ragas import evaluate

# HF 파이프라인 기반 LLM/임베딩
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

def build_llm(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=512)

def build_embedder(name="intfloat/e5-large-v2"):
    return SentenceTransformer(name)

def load_ragas_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            # 기대 형식: {"question","answer","contexts":[{"content":...},...], ...}
            ctx_texts = []
            for c in o.get("contexts") or []:
                # 04_build_ragas_dataset.py에서 이미 정규화되어 있을 것
                text = c.get("content") or c.get("text") or ""
                if text:
                    ctx_texts.append(text)
            if not (o.get("question") and o.get("answer") and ctx_texts):
                continue
            rows.append({
                "question": o["question"],
                "answer": o["answer"],
                "contexts": ctx_texts,
                # ground_truth가 있으면 같이 넣기 (없으면 metrics 일부만)
                "ground_truth": o.get("ground_truth") or "",
            })
    return Dataset.from_list(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--report", type=str, default="data/eval/ragas_report.json")
    ap.add_argument("--judge", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--embed", type=str, default="intfloat/e5-large-v2")
    args = ap.parse_args()

    ds = load_ragas_jsonl(args.inp)
    if len(ds) == 0:
        raise SystemExit("No rows to evaluate in the input dataset.")

    llm_pipe = build_llm(args.judge)
    emb = build_embedder(args.embed)

    # ragas evaluate
    result = evaluate(
        ds,
        metrics=[answer_correctness, faithfulness, context_precision, context_recall],
        llm=llm_pipe,
        embeddings=emb,
    )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({
            "n": len(ds),
            "scores": {k: float(v) for k, v in result["scores"].items()},
        }, f, ensure_ascii=False, indent=2)

    print("[ragas] wrote:", args.report)
    print(result)

if __name__ == "__main__":
    main()
