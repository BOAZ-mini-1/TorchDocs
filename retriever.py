import faiss
import json
import glob
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

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
def fetch_candidates(query_vector: np.ndarray, fetch_k: int):
    """
    Step 1: Fetch a larger pool of candidate documents using raw similarity search.
    This is a modified version of your original search function.
    """
    distances, indices = index.search(np.array([query_vector], dtype=np.float32), fetch_k)

    candidates = []
    for i in range(fetch_k):
        vector_index = indices[0][i]
        if vector_index < 0: # FAISS returns -1 for out of bounds
            continue
        if vector_index < len(actual_ids_list):
            doc_id = actual_ids_list[vector_index]
            distance = distances[0][i]
            doc_info = doc_store.get(doc_id)
            # We also need the vector for MMR calculations
            doc_vector = index.reconstruct(int(vector_index))
            candidates.append({
                "distance": distance,
                "doc_info": doc_info,
                "vector_index": vector_index, # Keep track of the original index in FAISS
                "vector": doc_vector
            })
        else:
            print(f"{vector_index} doenst exist")
    return candidates


def search_with_mmr(query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5):
    """
    Retrieves documents using a two-step MMR approach.
    """
    # Embed the query first
    query_vector = embeddings.embed_query(query)

    candidates = fetch_candidates(query_vector, fetch_k)
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


# Main
test_query = "how to tell if an object’s code is from a torch.package?"
search_results = search_with_mmr(test_query, k=3, fetch_k=10, lambda_mult=0.5)

print(f"Q: {test_query}\n")
print("Result")
# 검색된 각 문서의 모든 정보를 JSON 형식으로 예쁘게 출력
for i, result in enumerate(search_results, 1):
    print(f"=============== 문서 {i} ================")
    print(f"유사도 거리 (Distance): {result['distance']:.4f}\n")
    print(json.dumps(result['doc_info'], indent=4, ensure_ascii=False))
    print("=" * 35 + "\n")