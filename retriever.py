import faiss
import json
import glob
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

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

# Serach
def search_similar_documents(query: str, k: int = 5):
    query_vector = embeddings.embed_query(query)
    distances, indices = index.search(np.array([query_vector], dtype=np.float32), k)

    results = []
    for i in range(k):
        vector_index = indices[0][i]
        if vector_index < len(actual_ids_list):
            doc_id = actual_ids_list[vector_index]
            distance = distances[0][i]
            doc_info = doc_store.get(doc_id)
            results.append({"distance": distance, "doc_info": doc_info})
        else:
            print(f"{vector_index} doenst exist")
    return results

# Main
test_query = "how to tell if an object’s code is from a torch.package?"
search_results = search_similar_documents(test_query, k=3)

print(f"Q: {test_query}\n")
print("Result")
# 검색된 각 문서의 모든 정보를 JSON 형식으로 예쁘게 출력
for i, result in enumerate(search_results, 1):
    print(f"=============== 문서 {i} ================")
    print(f"유사도 거리 (Distance): {result['distance']:.4f}\n")
    print(json.dumps(result['doc_info'], indent=4, ensure_ascii=False))
    print("=" * 35 + "\n")