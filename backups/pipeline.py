
from generator_backup import generate_from_rerank
from retriever_backup import search_and_rerank_pipeline


test_query = ("how autograd is computed inside pytorch?")

temp = search_and_rerank_pipeline(test_query)

print(temp)

res = generate_from_rerank(test_query, temp)
print(res["answer"])
print(res["used_refs"])
