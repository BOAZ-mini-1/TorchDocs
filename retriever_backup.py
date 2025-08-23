import faiss
import json
import glob
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

base_path = "intfloat_e5_large_v2"
index_path = f"{base_path}/index/faiss.index"
global_ids_path = f"{base_path}/index/global_ids.json"

original_content_paths = glob.glob("data/processed/*.jsonl")

try:
    index = faiss.read_index(index_path)
    print(f"Total vectors in FAISS: {index.ntotal}")
except Exception as e:
    print(f"faiss index shit: {e}")
    exit()

try:
    with open(global_ids_path, 'r', encoding='utf-8') as f:
        global_ids_data = json.load(f)
    actual_ids_list = global_ids_data['ids']
except Exception as e:
    print(f"global id open failed: {e}")
    exit()

# DocStore 생성
doc_store = {}
try:
    if not original_content_paths:
        raise FileNotFoundError("File doenst exists(original jsonL file")
    else:
        for content_path in original_content_paths:
            with open(content_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    if 'id' in doc:
                        doc_store[doc['id']] = doc
        print(f"jsonL length: {len(doc_store)}")
except Exception as e:
    print(f"Failed for some reason: {e}")
    exit()

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")


# Search
def fetch_candidates(query_vector: np.ndarray, fetch_k: int, **kwargs):
    """
    Fetches at least `fetch_k` candidates that match the filter criteria.
    It repeatedly queries the index with a larger k until enough matches are found.
    """
    version_filter = kwargs.get("version")

    # If no filter is applied, we can use the simple, original logic.
    if not version_filter:
        distances, indices = index.search(np.array([query_vector], dtype=np.float32), fetch_k)
        candidates = []
        for i in range(len(indices[0])):
            vector_index = indices[0][i]
            if vector_index < 0: continue
            doc_id = actual_ids_list[vector_index]
            candidates.append({
                "distance": distances[0][i],
                "doc_info": doc_store.get(doc_id),
                "vector_index": vector_index,
                "vector": index.reconstruct(int(vector_index))
            })
        return candidates

    # --- Logic for when a filter IS applied ---
    candidates = []
    # Start by searching for more documents than we need
    k_to_search = fetch_k * 2
    processed_indices = set()

    # Loop until we have enough candidates or have searched the entire index
    while len(candidates) < fetch_k:
        # Safety break: stop if we are asking for more documents than exist in the index
        if k_to_search > index.ntotal:
            k_to_search = index.ntotal

        distances, indices = index.search(np.array([query_vector], dtype=np.float32), k_to_search)

        initial_candidate_count = len(candidates)

        for i in range(len(indices[0])):
            vector_index = int(indices[0][i])

            if vector_index < 0 or vector_index in processed_indices:
                continue

            processed_indices.add(vector_index)
            doc_id = actual_ids_list[vector_index]
            doc_info = doc_store.get(doc_id)

            # Apply the version filter
            if doc_info and doc_info.get('metadata', {}).get('version') == version_filter:
                candidates.append({
                    "distance": distances[0][i],
                    "doc_info": doc_info,
                    "vector_index": vector_index,
                    "vector": index.reconstruct(vector_index)
                })

                # If we have collected enough matching documents, we can stop.
                if len(candidates) == fetch_k:
                    break

        # If we have searched the entire index or a full search pass found no new candidates, break the loop.
        if k_to_search == index.ntotal or len(candidates) == initial_candidate_count:
            if len(candidates) < fetch_k:
                print(
                    f"Warning: Searched all {index.ntotal} documents but only found {len(candidates)} that match the filter.")
            break

        # If we still need more, double the search size for the next iteration
        k_to_search = min(k_to_search * 2, index.ntotal)

    return candidates


def search_with_mmr(query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5, version: str = None):
    """
    Retrieves documents using a two-step MMR approach.

    Args:
        query (str): The search query.
        k (int): The final number of documents to return.
        fetch_k (int): The number of initial candidates to fetch.
        lambda_mult (float): The lambda parameter for MMR (0=max diversity, 1=max relevance).
        version (str, optional): The version to filter documents by. Defaults to None.
    """
    # Embed the query first
    query_vector = embeddings.embed_query(query)

    # Pass the version parameter to fetch_candidates
    candidates = fetch_candidates(query_vector, fetch_k, version=version)
    if not candidates:
        return []

    candidate_vectors = np.array([c['vector'] for c in candidates])

    # Calculate relevance scores (query -> candidate similarity)
    # Cosine similarity is a standard choice for MMR. 1 is most similar, -1 is least.
    relevance_scores = cosine_similarity([query_vector], candidate_vectors)[0]

    selected_indices = []
    candidate_pool_indices = list(range(len(candidates)))

    # Start with the most relevant document
    best_initial_idx = np.argmax(relevance_scores)
    selected_indices.append(candidate_pool_indices.pop(best_initial_idx))

    while len(selected_indices) < k and candidate_pool_indices:
        mmr_scores = {}

        # Get vectors of documents already selected
        selected_vectors = np.array([candidates[i]['vector'] for i in selected_indices])

        # Iterate through remaining candidates to find the next best one
        for idx in candidate_pool_indices:
            cand_vector = candidates[idx]['vector'].reshape(1, -1)

            # Calculate diversity score (max similarity between the candidate and the selected set)
            diversity_score = np.max(cosine_similarity(cand_vector, selected_vectors))

            # Calculate MMR score
            relevance = relevance_scores[idx]
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_score
            mmr_scores[idx] = mmr_score

        # Select the document with the highest MMR score
        if not mmr_scores:
            break
        best_next_idx = max(mmr_scores, key=mmr_scores.get)
        selected_indices.append(best_next_idx)
        candidate_pool_indices.remove(best_next_idx)

    # We return the original candidate dicts, which already have the L2 distance.
    final_results = [candidates[i] for i in selected_indices]

    # Clean up the output to match your original format
    return [{"distance": res["distance"], "doc_info": res["doc_info"]} for res in final_results]


# ReRanking
def rerank_with_cross_encoder(query: str, search_results: list, top_n: int = 5):
    reranker_model = CrossEncoder('BAAI/bge-reranker-large')

    pairs = []
    for result in search_results:
        if result.get('doc_info') and result['doc_info'].get('content'):
            pairs.append([query, result['doc_info']['content']])
        else:
            pairs.append([query, ""])

    scores = reranker_model.predict(pairs)

    for i in range(len(search_results)):
        search_results[i]['rerank_score'] = scores[i]

    reranked_results = sorted(search_results, key=lambda x: x['rerank_score'], reverse=True)

    return reranked_results[:top_n]


def search_and_rerank_pipeline(
    query: str,
    fetch_k: int = 50,
    mmr_k: int = 20,
    top_n: int = 5,
    lambda_mult: float = 0.5,
    version: str = None
):
    """
    Performs a full search and rerank pipeline.

    Args:
        query (str): The search query from the user.
        fetch_k (int): The number of initial candidates to fetch from FAISS.
        mmr_k (int): The number of candidates to select using MMR.
        top_n (int): The final number of documents to return after reranking.
        lambda_mult (float): The lambda parameter for MMR (0=max diversity, 1=max relevance).
        version (str, optional): The version to filter documents by. Defaults to None.

    Returns:
        list: A list of the top N reranked documents.
    """
    # Step 1: Retrieve initial candidates using MMR
    initial_candidates = search_with_mmr(
        query=query,
        k=mmr_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        version=version
    )

    # Step 2: Rerank the candidates to get the final results
    final_results = rerank_with_cross_encoder(
        query=query,
        search_results=initial_candidates,
        top_n=top_n
    )

    return final_results


test_query = ("how autograd is computed inside pytorch?")

final_results = search_and_rerank_pipeline(test_query)

# 검색된 각 문서의 모든 정보를 JSON 형식으로 예쁘게 출력
for i, result in enumerate(final_results, 1):
    print(f"=============== 문서 {i} ================")
    print(f"L2 거리 (Distance): {result['distance']:.4f}")
    print(f"관련성 점수 (Rerank Score): {result['rerank_score']:.4f}\n") # 리랭크 점수 출력
    print(json.dumps(result['doc_info'], indent=4, ensure_ascii=False))
    print("=" * 40 + "\n")