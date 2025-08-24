# 태완오빠가 작성한 코드 디벨롭 버전
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
import re, json, glob, faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Embedding / Reranker
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from sentence_transformers import CrossEncoder  # e.g., cross-encoder/ms-marco-MiniLM-L-6-v2
    HAS_CE = True
except Exception:
    HAS_CE = False

# ----------- Paths & defaults -----------
BASE_PATH = os.getenv("E5_BASE_PATH", "intfloat_e5_large_v2")
INDEX_PATH = f"{BASE_PATH}/index/faiss.index"
GLOBAL_IDS_PATH = f"{BASE_PATH}/index/global_ids.json"
IDMAP_GLOB = f"{BASE_PATH}/embeddings/id_mapping_*chunks_e5_intfloat_e5_large_v2_mean.json"
PROCESSED_GLOB = "data/processed/*.jsonl"                         # TorchDocs JSONL들
DEFAULT_VERSION = os.getenv("TORCHDOCS_DEFAULT_VERSION", "2.8")   # default version : 2.8

# ----------- Lazy singletons -----------
_index: Optional[faiss.Index] = None
_global_ids: Optional[List[str]] = None
_global_to_local: Optional[Dict[str, str]] = None
_id2doc: Optional[Dict[str, Dict]] = None
_idx2local_cache: Dict[str, List[str]] = {}  # key=version -> list[index] = local_id

_embedder: Optional[HuggingFaceEmbeddings] = None
_reranker: Optional[object] = None  # CrossEncoder

# ----------- Utils -----------
VERSION_PATTERNS = [                 # 일단 내가 생각나는 형식만... 적어봄
    r"\b([0-9]\.[0-9])\b",           # "2.8"
    r"\bv([0-9]\.[0-9])\b",          # "v2.8"
    r"\bPyTorch\s*([0-9]\.[0-9])\b", # "PyTorch 2.8"
    r"\bversion\s*([0-9]\.[0-9])\b", # "version 2.8"
]

def parse_version_from_query(q: str) -> Optional[str]:
    q = q.strip()
    for pat in VERSION_PATTERNS:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None

def ensure_index() -> faiss.Index:
    global _index
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        _index = faiss.read_index(INDEX_PATH)
    return _index

def ensure_global_ids() -> List[str]:
    global _global_ids
    if _global_ids is None:
        with open(GLOBAL_IDS_PATH, "r", encoding="utf-8") as f:
            _global_ids = json.load(f)
    return _global_ids

def ensure_id2doc() -> Dict[str, Dict]:
    """
    JSONL의 각 row는 최소 다음 키들을 가진다고 가정하고 시작:
      - id (문서/청크 고유 ID)
      - text_for_embedding (generator에 넣을 본문)
      - metadata: { url, title, version, ... }
    """
    global _id2doc
    if _id2doc is None:
        _id2doc = {}
        for p in glob.glob(PROCESSED_GLOB):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    doc_id = obj.get("id")
                    if not doc_id:
                        continue
                    _id2doc[doc_id] = obj
    return _id2doc


# new (3차 시도)
def _pick_idmap_path_for_version(ver: str) -> str:
    """
    버전에 맞는 id_mapping 파일을 선택.
    파일명이 'torchdocs_{ver}_chunks_e5_' 를 포함하는 것을 우선 선택.
    없으면 가장 최신/가장 긴 파일 하나를 fallback으로 고름.
    """
    cand = []
    for p in glob.glob(IDMAP_GLOB):
        if f"torchdocs_{ver}_" in p:
            cand.append(p)
    if cand:
        # 버전에 정확히 매칭되는 첫 파일
        cand.sort()
        return cand[0]

    # fallback: 아무거나 하나 (가장 최근/긴 파일 우선)
    all_maps = glob.glob(IDMAP_GLOB)
    if not all_maps:
        raise FileNotFoundError(f"No id_mapping json found under {IDMAP_GLOB}")
    # 길이/이름 기준 정렬
    all_maps.sort(key=lambda x: (len(x), x))
    return all_maps[-1]


def ensure_index_to_local_id(ver: str) -> List[str]:
    """
    주어진 버전에 대한 index -> local_id 리스트를 만든다.
    id_mapping[i]['id'] 가 곧 i번째 벡터의 로컬 jsonl id.
    """
    ver = str(ver)
    if ver in _idx2local_cache:
        return _idx2local_cache[ver]

    p = _pick_idmap_path_for_version(ver)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("id_mapping")
    if not isinstance(items, list):
        raise ValueError(f"id_mapping format unexpected in {p}")

    # index 기반이므로 바로 배열로 만든다(안전하게 길이 추정)
    total = data.get("metadata", {}).get("total_items")
    if isinstance(total, int) and total > 0:
        arr = [None] * total
    else:
        # total이 없으면 index 최대값+1 로 길이를 잡자
        max_idx = max(int(it.get("index", -1)) for it in items)
        arr = [None] * (max_idx + 1)

    for it in items:
        i = int(it["index"])
        lid = str(it["id"])
        if 0 <= i < len(arr):
            arr[i] = lid

    # 누락분(간혹 None)이 있으면 필터
    # (실제 검색에서 i가 None인 곳에 걸리면 그냥 건너뜀)
    _idx2local_cache[ver] = arr
    return arr


def ensure_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        # sentence-transformers/intfloat-e5-large-v2와 호환되는 Community Wrapper
        _embedder = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            # model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # cosine/IP 호환? (사실 뭔지 모르는데 일단 넣음)
        )
    return _embedder

# reranker도 추가했어요 
def ensure_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _reranker
    if _reranker is None and HAS_CE:
        _reranker = CrossEncoder(model_name)
    return _reranker

# 질문 임베딩 작업
def e5_query_embed(query: str) -> np.ndarray:
    emb = ensure_embedder().embed_query(f"query: {query}")
    return np.array(emb, dtype="float32")

def mmr(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    cand_ids: List[str],
    top_k: int = 5,
    lambda_mult: float = 0.5,
) -> List[int]:
    """
    간단한 MMR 구현 함수 
    query_vec: (D,), cand_vecs: (C, D), cand_ids: len C
    return: 선택된 cand 인덱스 리스트(길이 top_k)
    """
    # 유사도
    q = query_vec.reshape(1, -1)  # (1, D)
    sim_to_query = (cand_vecs @ query_vec)  # (C,)  (normalize=True 이므로 코사인 유사도)
    chosen = []
    remaining = list(range(len(cand_ids)))

    while remaining and len(chosen) < top_k:
        if not chosen:
            # 가장 유사한 것부터
            best = int(np.argmax(sim_to_query))
            chosen.append(best)
            remaining.remove(best)
            continue
        # 다중성(서로 유사한 것)을 억제
        max_div = []
        for j in remaining:
            # j와 이미 선택된 것들 간의 최대 유사도
            if len(chosen) == 1:
                max_sim_sel = float(cand_vecs[j] @ cand_vecs[chosen[0]])
            else:
                sims = cand_vecs[j] @ cand_vecs[chosen].T  # (len(chosen),)
                max_sim_sel = float(np.max(sims))
            score = lambda_mult * sim_to_query[j] - (1 - lambda_mult) * max_sim_sel
            max_div.append(score)
        best_j = remaining[int(np.argmax(max_div))]
        chosen.append(best_j)
        remaining.remove(best_j)

    return chosen

# metadata 끌고온 최종 return schema
def build_doc_record(doc: Dict, score: float) -> Dict:
    md = doc.get("metadata", {}) or {}
    return {
        "id": doc.get("id"),
        "url": md.get("url"),
        "version": str(md.get("version") or ""),
        "title": md.get("title"),
        "content": doc.get("text_for_embedding") or doc.get("content") or "",
        "score": float(score),
    }

# ----------- Public API -----------
def search(
    query: str,
    top_k: int = 5,
    fetch_k: int = 50,
    version: Optional[str] = None,
    use_cross_encoder: bool = False,
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    mmr_lambda: float = 0.5,
) -> List[Dict]:
    """
    1) e5 쿼리 임베딩
    2) FAISS에서 후보 fetch_k개
    3) 버전 필터 (기본은 DEFAULT_VERSION; 질문에 버전 명시되면 override)
    4) MMR로 다양성 반영 후 top_k 선택
    5) (optional) CrossEncoder로 rerank
    6) 표준 Doc record 반환
    """
    # 0) 버전 결정
    parsed = parse_version_from_query(query)
    target_version = (version or parsed or DEFAULT_VERSION)

    # 1) 임베딩 & FAISS 검색
    index = ensure_index()
    gids = ensure_global_ids()      # <- global_id 리스트
    id2doc = ensure_id2doc()
    idx2lid = ensure_index_to_local_id(target_version)  # target_version은 "2.8" 같은 포맷


    qv = e5_query_embed(query)  # (D,)
    qv_norm = qv / (np.linalg.norm(qv) + 1e-8)
    D = qv_norm.reshape(1, -1)  # (1, D)
    # FAISS 인덱스는 IP 기준이므로, 쿼리도 normalize 한 채로 search
    sims, idxs = index.search(D.astype("float32"), fetch_k)  # sims: (1, fetch_k)
    idxs = idxs[0].tolist()
    sims = sims[0].tolist()

    # 2) 후보 수집(+ version filter)
    cand_ids: List[str] = []
    cand_docs: List[Dict] = []
    cand_scores: List[float] = []

    for i, s in zip(idxs, sims):
        if i < 0 or i >= len(idx2lid):
            continue
        local_id = idx2lid[i]
        if not local_id:
            continue
        doc = id2doc.get(local_id)
        if not doc:
            continue

        md = doc.get("metadata", {}) or {}
        ver = str(md.get("version") or "")
        if target_version and ver and ver != str(target_version):
            continue
        cand_ids.append(local_id)
        cand_docs.append(doc)
        cand_scores.append(float(s))

    if not cand_docs:
        return []

    # 3) cand_doc들의 벡터 재구성(필요 시)
    #   여기서는 FAISS 인덱스가 이미 normalize 임베딩을 쥐고 있으므로,
    #   간단히 재호출 없이 쿼리-문서 유사도(sims)만으로 MMR 적용 가능.
    #   다만 cand_vecs가 필요한 구현이면 아래처럼 다시 얻어올 수 있도록
    #   별도 저장(이번 최소구현은 sims만으로 충분).
    # → 간단 구현: sims만 이용해 MMR 흉내 내려면 cand 간 유사도가 필요해서
    #   여기서는 embedder로 candidate content 임베딩을 한 번 더 구함(상위 fetch_k라 비용 OK).
    embedder = ensure_embedder()
    # e5 규칙상 문서 임베딩은 "passage: ..." 접두를 붙이는 게 권장된다고 하네유.
    passages = [f"passage: {d.get('text_for_embedding') or ''}" for d in cand_docs]
    cand_vecs = np.array(embedder.embed_documents(passages), dtype="float32")  # (C, D)
    # 이미 normalize_embeddings=True 이므로 단위 벡터일 것
    qv_n = qv_norm.astype("float32")
    sel_idx = mmr(qv_n, cand_vecs, cand_ids, top_k=top_k, lambda_mult=mmr_lambda)

    selected = [ (cand_docs[i], float(cand_vecs[i] @ qv_n)) for i in sel_idx ]
    # 4) (optional) CrossEncoder rerank
    if use_cross_encoder and HAS_CE:
        reranker = ensure_reranker(cross_encoder_name)
        pairs = [[query, d.get("text_for_embedding") or ""] for d, _ in selected]
        ce_scores = reranker.predict(pairs).tolist()
        ranked = sorted(
            [(d, s, ce) for (d, s), ce in zip(selected, ce_scores)],
            key=lambda x: x[2], reverse=True
        )
        return [build_doc_record(d, score=ce) for (d, _s, ce) in ranked]
    else:
        # MMR 점수(= q·d 유사도)를 score로 사용
        ranked = sorted(selected, key=lambda x: x[1], reverse=True)
        return [build_doc_record(d, score=s) for (d, s) in ranked]


if __name__ == "__main__":
    # quick smoke test
    q = "How does autograd compute gradients in PyTorch 2.8?"
    results = search(q, top_k=5, fetch_k=40, version=None, use_cross_encoder=False)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['title']} (v{r['version']})  score={r['score']:.3f}")
        print(f"    {r['url']}")
