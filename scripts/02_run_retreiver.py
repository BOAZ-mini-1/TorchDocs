# 2. Retriever 실행

# Usage (FAISS fallback with your precomputed e5 embeddings):
#   python 02_run_retriever.py \
#     --top-k 5 \
#     --use-faiss-fallback \
#     --version 2.8 \
#     --query-encoder intfloat/e5-large-v2 \
#     --mmr-lambda 0.5 --mmr-fetch-k 20

if __name__ == "__main__" and False:
    ...

import json
import numpy as np
import argparse, json, faiss
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def _load_id_mapping(path: Path) -> dict:
    """Support two formats:
    1) {"id_mapping": [{"index": 0, "id": "..."}, ...]}
    2) ["id0", "id1", ...] or {"0": "id0", ...}
    Returns: {index: id}
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "id_mapping" in data:
        return {int(x["index"]): x["id"] for x in data["id_mapping"]}
    if isinstance(data, list):
        return {i: v for i, v in enumerate(data)}
    if isinstance(data, dict):
        try:
            return {int(k): v for k, v in data.items()}
        except Exception:
            pass
    raise ValueError("Unsupported id mapping format")


def _build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    # Use cosine (normalize + inner product)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def _mmr_rerank(qv: np.ndarray, cand_idx: list, cand_vs: np.ndarray, top_k: int, lam: float = 0.5):
    """Simple MMR over candidate indices.
    qv: (D,) normalized; cand_vs: (C, D) normalized.
    Returns: list of selected indices (subset of cand_idx) length top_k
    """
    selected = []
    not_selected = list(range(len(cand_idx)))
    # Precompute q·d
    q_sims = cand_vs @ qv
    while not_selected and len(selected) < top_k:
        best_j = None
        best_score = -1e9
        for j in not_selected:
            d_sim = q_sims[j]
            if not selected:
                mmr = d_sim
            else:
                # diversity term: max sim to any selected
                div = max(cand_vs[j] @ cand_vs[s] for s in selected)
                mmr = lam * d_sim - (1 - lam) * div
            if mmr > best_score:
                best_score = mmr
                best_j = j
        selected.append(best_j)
        not_selected.remove(best_j)
    return [cand_idx[j] for j in selected]


def retrieve_placeholder(query: str, top_k: int):
    """Team retriever adapter used when --use-team-retriever is set at CLI.
    Falls back to NotImplementedError if team module is unavailable.
    
    This adapter will try to:
      1) call retriever.search_with_mmr(query, ...)
      2) optionally call retriever.rerank_with_cross_encoder(...)
    Environment-configurable via optional globals if present in this module:
      _TEAM_USE_RERANKER (bool), _MMR_FETCH_K (int), _MMR_LAMBDA (float), _VERSION_FILTER (str|None)
    """
    try:
        import retriever as team_retriever # 태완이 짜놓은 retriever - 모듈로 불러온거라 경로 체크 필요 
    except Exception as e:
        raise NotImplementedError(f"Team retriever not available: {e}") # 혹시 모를 fallback 처리

    # Pull optional globals if set by the caller (script_02_run_retriever)
    mmr_fetch_k = globals().get("_MMR_FETCH_K", 20)
    mmr_lambda = globals().get("_MMR_LAMBDA", 0.5)
    version_filter = globals().get("_VERSION_FILTER", None)
    team_use_reranker = globals().get("_TEAM_USE_RERANKER", False)

    # Step 1: candidate selection with MMR (fetch more than top_k for rerank)
    initial = team_retriever.search_with_mmr(
        query,
        k=max(top_k, mmr_fetch_k),
        fetch_k=mmr_fetch_k,
        lambda_mult=mmr_lambda,
        version=version_filter,
    )

    # Step 2: optional cross-encoder rerank
    if team_use_reranker:
        final = team_retriever.rerank_with_cross_encoder(query, initial, top_n=top_k)
    else:
        final = initial[:top_k]

    # Map to pipeline format
    outputs = []
    for item in final:
        doc = (item.get("doc_info") or {})
        _id = doc.get("id")
        _text = doc.get("text_for_embedding") or doc.get("content") or ""
        score = float(item.get("rerank_score", -item.get("distance", 0.0)))
        outputs.append({"id": _id, "text": _text, "score": score})
    return outputs



def script_02_run_retriever(
    top_k: int = 5,
    use_faiss_fallback: bool = False,
    version: str = "2.8",
    query_encoder: str = "intfloat/e5-large-v2",
    mmr_lambda: float = 0.5,
    mmr_fetch_k: int = 20,
):
    repo = Path(__file__).resolve().parents[0]
    qas_path = repo / "data" / "eval" / "01_qas_seed.jsonl"
    out_path = repo / "data" / "eval" / "02_retrieval_logs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If fallback: load embeddings & mapping & texts
    if use_faiss_fallback:
        emb_dir = repo / "embeddings" / "intfloat_e5_large_v2"
        vec_path = emb_dir / f"embeddings_{version}_e5_large_v2_mean.npy"
        id_path = emb_dir / f"id_{version}_e5_large_v2_mean.json"
        processed = repo / "data" / "processed" / f"torchdocs_{version}_chunks_e5.jsonl"
        vecs = np.load(vec_path).astype("float32")
        id_map = _load_id_mapping(id_path)
        # id_by_idx, text_by_id
        id_by_idx = {i: id_map[i] for i in range(len(id_map))}
        text_by_id = {}
        with processed.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text_by_id[obj.get("id")] = obj.get("text_for_embedding") or obj.get("content") or ""
        index = _build_faiss_index(vecs)
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for --use-faiss-fallback")
        q_model = SentenceTransformer(query_encoder)

    out = []
    with qas_path.open("r", encoding="utf-8") as f:
        for line in f:
            qa = json.loads(line)
            qid = qa["qid"]
            query = qa["question"]

            if use_faiss_fallback:
                qv = q_model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")[0]
                # Fetch a candidate pool
                D, I = index.search(qv.reshape(1, -1), mmr_fetch_k)
                cand_idx = list(I[0])
                cand_idx = [i for i in cand_idx if i >= 0]
                cand_vs = vecs[cand_idx]
                # Already normalized; MMR select top_k
                sel_idx = _mmr_rerank(qv, cand_idx, cand_vs, top_k=top_k, lam=mmr_lambda)
                retrieved = []
                for i in sel_idx:
                    id_ = id_by_idx[i]
                    txt = text_by_id.get(id_, "")
                    # IP score approximates cosine
                    score = float(qv @ vecs[i])
                    retrieved.append({"id": id_, "text": txt, "score": score})
            else:
                hits = retrieve_placeholder(query, top_k)
                retrieved = []
                for h in hits:
                    retrieved.append({"id": h.get("id"), "text": h.get("text", ""), "score": h.get("score", 0.0)})

            out.append({"qid": qid, "retrieved": retrieved})

    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[02] wrote {len(out)} items -> {out_path}")


if __name__ == "__main__":
    import sys
    if Path(sys.argv[0]).name == "02_run_retriever.py":
        p = argparse.ArgumentParser()
        p.add_argument("--top-k", type=int, default=5)
        p.add_argument("--use-faiss-fallback", action="store_true")
        p.add_argument("--version", default="2.8")
        p.add_argument("--query-encoder", default="intfloat/e5-large-v2")
        p.add_argument("--mmr-lambda", type=float, default=0.5)
        p.add_argument("--mmr-fetch-k", type=int, default=20)
        args = p.parse_args()
        script_02_run_retriever(**vars(args))

