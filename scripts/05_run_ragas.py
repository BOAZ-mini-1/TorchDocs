# 5. RAGAS 실행(Qwen2.5‑14B‑Instruct + e5‑large‑v2)

# Usage (dry run to validate dataset columns):
#   python 05_run_ragas.py --dry-run
#
# When ready to evaluate (wire judge & embeddings):
#   python 05_run_ragas.py \
#     --judge qwen2.5-14b-instruct \
#     --emb intfloat/e5-large-v2 \
#     --metrics faithfulness answer_relevance context_recall context_precision

import json
import argparse, json
from pathlib import Path

if __name__ == "__main__" and False:
    ...

import pandas as pd
from typing import List

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
except Exception:
    evaluate = None
    faithfulness = answer_relevance = context_precision = context_recall = None


def init_judge_llm(model_name: str):
    """TODO: Return a RAGAS-compatible LLM client for the evaluator (judge).
    Target: Qwen2.5-14B-Instruct.

    Options:
    1) If you use LangChain wrappers, pass that LLM here (ragas accepts langchain LLMs).
    2) Or use ragas' native integrations if available in your version.

    For now this is a hard placeholder to ensure you explicitly wire it.
    """
    raise NotImplementedError("Provide a judge LLM compatible with RAGAS (e.g., Qwen2.5-14B-Instruct)")


class E5LangChainLikeEmbeddings:
    """Minimal adapter exposing embed_documents / embed_query for ragas.
    Uses sentence-transformers under the hood.
    """
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for E5 embeddings")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # E5 passage encoding: prepend "passage: " if your corpus was encoded that way.
        # If your existing embeddings did NOT include the prefix, keep it consistent.
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()


def script_05_run_ragas(
    dataset_filename: str = "ragas_synth.jsonl",
    report_filename: str = "05_ragas_report.json",
    judge: str = "qwen2.5-14b-instruct",
    emb: str = "intfloat/e5-large-v2",
    metrics: List[str] = None,
    max_records: int = 0,
    dry_run: bool = False,
):
    if metrics is None:
        metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]

    repo = Path(__file__).resolve().parents[0]
    ds_path = repo / "data" / "eval" / dataset_filename
    out_path = repo / "data" / "eval" / report_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in ds_path.open("r", encoding="utf-8").read().splitlines() if l.strip()]
    if max_records and max_records > 0:
        rows = rows[:max_records]
    df = pd.DataFrame(rows)

    required_cols = {"question", "answer", "contexts"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if dry_run:
        print(f"[05] dataset rows: {len(df)}; columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
        return

    if evaluate is None:
        raise RuntimeError("ragas is not installed or failed to import. Please install ragas >= 0.1.")

    # Build metric objects
    metric_objs = []
    for m in metrics:
        if m == "faithfulness":
            metric_objs.append(faithfulness)
        elif m == "answer_relevance":
            metric_objs.append(answer_relevance)
        elif m == "context_precision":
            metric_objs.append(context_precision)
        elif m == "context_recall":
            metric_objs.append(context_recall)
        else:
            raise ValueError(f"Unknown metric: {m}")

    # Init judge LLM and embeddings
    llm = init_judge_llm(judge)
    emb_client = E5LangChainLikeEmbeddings(emb)

    # Evaluate
    res = evaluate(
        dataset=df,
        metrics=metric_objs,
        llm=llm,
        embeddings=emb_client,
    )

    # Save a JSON result; try a couple of common accessors
    payload = None
    for attr in ("to_json", "model_dump_json", "to_dict"):
        if hasattr(res, attr):
            try:
                payload = getattr(res, attr)()
                break
            except Exception:
                pass
    if payload is None:
        # Fallback: compute aggregated means if possible
        try:
            means = {m: float(getattr(res, m)) for m in metrics if hasattr(res, m)}
            payload = json.dumps({"metrics": means}, ensure_ascii=False)
        except Exception:
            payload = json.dumps({"note": "RAGAS result serialization fallback"}, ensure_ascii=False)

    with out_path.open("w", encoding="utf-8") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            f.write(json.dumps(payload, ensure_ascii=False))

    print(f"[05] wrote report -> {out_path}")


if __name__ == "__main__":
    import sys
    if Path(sys.argv[0]).name == "05_run_ragas.py":
        p = argparse.ArgumentParser()
        p.add_argument("--dataset-filename", default="ragas_synth.jsonl")
        p.add_argument("--report-filename", default="05_ragas_report.json")
        p.add_argument("--judge", default="qwen2.5-14b-instruct")
        p.add_argument("--emb", default="intfloat/e5-large-v2")
        p.add_argument("--metrics", nargs="*", default=["faithfulness", "answer_relevance", "context_recall", "context_precision"])
        p.add_argument("--max-records", type=int, default=0)
        p.add_argument("--dry-run", action="store_true")
        args = p.parse_args()
        script_05_run_ragas(**vars(args))
