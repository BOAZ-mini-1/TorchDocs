# generator/generate.py
# - 입력: retriever/reranker가 만든 reranked_results
#   예: [{"distance":..., "rerank_score":..., "doc_info": {...}}, ...]
# - 출력: {
#     "answer": str,
#     "used_ctx_ids": [..],
#     "used_refs": [{"ref_num":1,"id":"...","url":"...","version":"...","title":"..."}],
#     "k": int
#   }
# - 모델: Qwen/Qwen2.5-7B-Instruct

from typing import List, Dict, Any, Optional
import sys
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def seed_all(seed=42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# CUDA 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[디바이스 설정] {device} 사용")

# ---- 모델 로드 ----
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    # pad_token 설정
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 디바이스별 로드 설정
    load_kwargs = {}
    if torch.cuda.is_available():
        load_kwargs.update({
            "device_map": "auto",
            "torch_dtype": torch.bfloat16
        })
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    
    if device == "cpu" and not torch.cuda.is_available():
        model = model.to(device)
        
except Exception as e:
    print(
        f"[모델 로드 실패] {e}\n"
        "- 모델 이름이 올바른지 확인\n"
        "- 네트워크 연결 상태 확인\n",
        file=sys.stderr
    )
    raise

# # ============================ ver1 ============================
# # 모델 프롬프트
# SYSTEM_MSG = ( 
#     "You are a assistant for junior ai developer who uses pytorch framework. "
#     "Respond as a helpful, concise chatbot in Markdown. "
#     "Strictly ground every factual statement in the provided contexts and cite them inline as [#1], [#2], ... "
#     "If the answer is uncertain or not supported, say so explicitly and suggest the closest supported guidance. "
#     "Prefer the most recent PyTorch version when contexts disagree, but state the version you cite. "
#     "If the user specifies a version, only use documents matching that version (metadata.version). "
#     "Do NOT invent APIs or behavior not present in the contexts. "
#     "Keep code minimal and runnable. "
#     "Language: English."
# )

# # 사용자 프롬프트
# USER_TEMPLATE = """ # User Question  
# {query}

# # What you MUST do
# - Write the answer in **clear Markdown** with this structure:
#   1) **Summary** — 2–3 bullet points with the direct answer.
#   2) **Explanation** — brief reasoning and when to use it, citing sources inline like [#1].
#   3) **Example** — a minimal, runnable code snippet (if relevant).
#   4) **Notes & Pitfalls** — edge cases, version differences (mention version numbers), and common mistakes.
# - Only use information present in the reference documents. If evidence is insufficient, write: *"The provided references don't fully support X."* and stop.
# - When multiple versions appear, prefer the **newest version** but **name the version(s)** you used.
# - **If the user specifies a version, only use documents from that version. If no documents match, say so explicitly.**
# - Keep the final answer under ~250–300 words unless the user asked for more.

# """

# ============================ ver2 ============================
# 시스템 프롬프트
SYSTEM_MSG = ( 
    "당신은 PyTorch 프레임워크를 사용하는 주니어 AI 개발자를 돕는 조수입니다. "
    "항상 간결하게 **마크다운** 형식으로 대답하세요. "
    "모든 사실은 반드시 제공된 컨텍스트에 근거해야 하며, 출처를 [#1], [#2] 형식으로 표시하세요. "
    "만약 답변이 불확실하거나 자료에 없는 경우, 그렇게 명시하고 가장 근접한 근거를 제안하세요. "
    "컨텍스트에 서로 다른 버전이 있으면 최신 PyTorch 버전을 우선 사용하되, 어떤 버전인지 반드시 밝히세요. "
    "사용자가 특정 버전을 지정하면 해당 버전 문서(metadata.version)만 사용하세요. "
    "존재하지 않는 API나 동작은 절대 만들어내지 마세요. "
    "코드는 최소한으로 간단하고 실행 가능하도록 작성하세요. "
    "답변은 반드시 한국어로 작성하세요."
)

# 사용자 프롬프트
'''
사용자 질문 + RAG 컨텍스트 기반 답변 구조 강제
요약 / 설명 / 예제 / 주의사항 4단계 구성
Evidence 부족 시 명확히 표시
출력은 한국어
'''
USER_TEMPLATE = """# 사용자 질문  
{query}

# 답변 작성 지침
- 답변은 **명확한 마크다운** 구조로 작성하세요:
  1) **요약** — 핵심 답변을 2~3개 불릿 포인트로 정리
  2) **설명** — 간단한 이유와 활용 상황, [#1]처럼 출처 표시
  3) **예제** — 실행 가능한 최소 코드 스니펫 (필요한 경우)
  4) **주의사항** — 버전 차이, 엣지 케이스, 흔한 실수 등을 정리
- 참고 문서에 있는 정보만 사용하세요. 충분한 근거가 없다면: *"제공된 문서에는 X에 대한 충분한 근거가 없습니다."* 라고만 작성하고 멈추세요.
- 여러 버전이 있으면 **최신 버전**을 우선하되, 사용한 버전을 명시하세요.
- 사용자가 특정 버전을 지정하면 해당 버전 문서만 사용하세요. 문서가 없으면 "해당 버전에 맞는 문서를 찾을 수 없습니다."라고 하세요.
- 답변은 특별히 요구하지 않는 한 **250~300단어 이내**로 유지하세요.
- 답변은 반드시 한국어로 작성하세요.
"""


def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def _format_one_context(i: int, doc_info: Dict[str, Any]) -> str:
    """
    reranker 결과의 doc_info를 사람이 읽는 컨텍스트 블록으로 변환.
    기대 doc_info:
      {
        "id": "...",
        "content": "...",
        "code_blocks": [{"lang": "...", "code": "..."}],
        "metadata": {"title": "...", "url": "...", "version": "...", ...}
      }
    """
    cid = str(doc_info.get("id", f"ctx_{i}"))
    meta = doc_info.get("metadata") or {}
    title = meta.get("title") or ""
    url = meta.get("url") or ""
    version = meta.get("version") or ""

    content = doc_info.get("content") or doc_info.get("text_for_embedding") or ""
    content = _shorten(content, 1200)

    # 코드 스니펫 1~2개(각 300자)만 포함
    code_blocks = []
    for b in (doc_info.get("code_blocks") or [])[:2]:
        code_snip = _shorten(b.get("code", ""), 300)
        if code_snip:
            code_blocks.append(f"```{b.get('lang','text')}\n{code_snip}\n```")
    code_section = "\n".join(code_blocks).strip()

    meta_line = []
    if title:   meta_line.append(f"title='{title}'")
    if version: meta_line.append(f"version={version}")
    if url:     meta_line.append(f"url={url}")
    meta_line = (" (" + ", ".join(meta_line) + ")") if meta_line else ""

    block = f"- [#{i}] (id: {cid}){meta_line}\n{content}\n"
    if code_section:
        block += f"\nExample code:\n{code_section}\n"
    return block

def select_contexts(
    reranked_results: List[Dict[str, Any]], 
    k: int = 6, 
    required_version: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    버전 필터링 및 상위 k개 컨텍스트 선별
    """
    items = reranked_results
    
    # 버전 필터링
    if required_version:
        items = [
            x for x in items 
            if (x.get("doc_info", {}).get("metadata", {}) or {}).get("version") == required_version
        ]
        if not items:
            # 버전 불일치 시 빈 리스트 반환
            return []
    
    # rerank_score 우선, 없으면 distance 역순으로 정렬
    items = sorted(
        items, 
        key=lambda x: (-(x.get("rerank_score") or 0.0), x.get("distance") or 0.0)
    )
    
    return items[:k]

def build_ctx_block_from_rerank(selected_results: List[Dict[str, Any]]) -> str:
    """
    선별된 결과를 LLM 프롬프트용 텍스트로 직렬화
    """
    blocks = []
    for i, item in enumerate(selected_results, start=1):
        doc = item.get("doc_info") or {}
        blocks.append(_format_one_context(i, doc))
    return "\n".join(blocks)

def truncate_to_token_budget(text: str, tokenizer, max_tokens: int) -> str:
    """
    토큰 예산에 맞게 텍스트 잘라내기
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def ensure_references_footer(answer: str, num_refs: int) -> str:
    """
    References 섹션이 없으면 추가
    """
    if "References used:" in answer:
        return answer
    if num_refs > 0:
        idxs = ", ".join(f"[#{i}]" for i in range(1, num_refs + 1))
        return answer.rstrip() + f"\n\nReferences used: {idxs}"
    return answer

def generate_from_rerank(
    query: str,
    reranked_results: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    k: int = 6,
    required_version: Optional[str] = None,
    max_prompt_tokens: int = 28000,
) -> Dict[str, Any]:
    """
    rerank 상위 k 결과를 그대로 받아 Qwen2.5에 투입해 답변 생성.
    """
    # 컨텍스트 선별 (버전 필터링 포함)
    selected_results = select_contexts(reranked_results, k, required_version)
    
    # 버전 필터링 후 결과가 없으면 처리
    if required_version and not selected_results:
        return {
            "answer": f"*지정된 버전 '{required_version}'에 해당하는 문서를 찾을 수 없습니다.*",
            "used_ctx_ids": [],
            "used_refs": [],
            "k": 0,
        }
    
    ctx_block = build_ctx_block_from_rerank(selected_results)
    user_prompt = USER_TEMPLATE.format(query=query, ctx_block=ctx_block)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_prompt},
    ]
    
    # Qwen의 chat template 적용
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    
    # 토큰 예산 관리
    prompt = truncate_to_token_budget(prompt, tokenizer, max_prompt_tokens)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # 생성 파라미터 설정
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": 1.05,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        out = model.generate(input_ids=input_ids, **gen_kwargs)

    answer = tokenizer.batch_decode(
        out[:, input_ids.shape[-1]:], skip_special_tokens=True
    )[0].strip()

    # References 섹션 보장
    answer = ensure_references_footer(answer, len(selected_results))

    # 하위 호환: 사용한 컨텍스트 ID 목록
    used_ctx_ids = []
    # 프런트엔드 하이퍼링크용 상세 참조 목록
    used_refs = []
    for i, item in enumerate(selected_results, start=1):
        doc = item.get("doc_info") or {}
        meta = doc.get("metadata") or {}
        doc_id = doc.get("id")
        used_ctx_ids.append(str(doc_id) if doc_id is not None else f"ctx_{i}")
        used_refs.append({
            "ref_num": i,                      # 본문 [#i]와 매칭
            "id": str(doc_id) if doc_id is not None else "",
            "url": meta.get("url", ""),
            "version": meta.get("version", ""),
            "title": meta.get("title", ""),
        })

    return {
        "answer": answer,
        "used_ctx_ids": used_ctx_ids,
        "used_refs": used_refs,
        "k": len(selected_results),  # 실제로 투입한 컨텍스트 개수
    }

# ============================

# # (선택) 모듈 단일 테스트용 - retriever/reranker 결과 가정
# if __name__ == "__main__":
#     demo_query = "How can I check if an object's code came from torch.package?"
#     demo_reranked = [{
#         "distance": 0.91, "rerank_score": 7.8,
#         "doc_info": {
#             "id": "29f6...",
#             "content": "To check if an object's code came from torch.package, use torch.package.is_from_package(...).",
#             "code_blocks": [{"lang": "python", "code": "assert is_from_package(mod) ..."}],
#             "metadata": {"title": "Distinguishing between packaged and non-packaged code",
#                          "url": "pytorch_docs_2.7/...", "version": "2.7"}
#         }
#     }]
#     res = generate_from_rerank(demo_query, demo_reranked)
#     print(res["answer"])
#     print(res["used_refs"])
