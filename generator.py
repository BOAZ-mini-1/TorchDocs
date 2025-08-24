# 미지가 작성한 코드 디벨롭 및 수정 버전
from typing import List, Dict, Any, Optional
import os, sys, random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# LOAD_4BIT = os.getenv("GEN_4BIT", "1") == "1"

# generator 답변 다양한게 좋을 거 같은데 seed를 왜 굳이..? 넣어야 하는진 이해를 못했다만.. 일단 똑같이 넣었어요 
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

_tokenizer = None
_model = None

def ensure_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    if _model is None:
        # accelerate/4bit 경로를 피해서 로드 (제 환경에서는 accelerate 끌어오느라 계속 오류가 나더라고예,,)
        use_cuda = torch.cuda.is_available()
        load_kwargs = {"trust_remote_code": True}
        # CUDA면 bfloat16로 로드(메모리 절약)하고 CPU면 dtype 지정 안 함.
        if use_cuda:
            load_kwargs["torch_dtype"] = torch.bfloat16

        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)

        # 장치 배치 (accelerate 없이)
        _model = _model.to("cuda" if use_cuda else "cpu")
    return _tokenizer, _model


SYSTEM_MSG = (
    "You are a assistant for junior ai developer who uses pytorch framework. "
    "Respond as a helpful, concise chatbot in Markdown. "
    "Strictly ground every factual statement in the provided contexts and cite them inline as [#1], [#2], ... "
    "If the answer is uncertain or not supported, say so explicitly and suggest the closest supported guidance. "
    "Prefer the most recent PyTorch version when contexts disagree, but state the version you cite. "
    "If the user specifies a version, only use documents matching that version (metadata.version). "
    "Do NOT invent APIs or behavior not present in the contexts. "
    "Keep code minimal and runnable. "
    "Language: English."
)

USER_TEMPLATE = """# User Question  
{query}

# What you MUST do
- Write the answer in **clear Markdown** with this structure:
  1) **Summary** — 2-3 bullet points with the direct answer.
  2) **Explanation** — brief reasoning and when to use it, citing sources inline like [#1].
  3) **Example** — a minimal, runnable code snippet (if relevant).
  4) **Notes & Pitfalls** — edge cases, version differences (mention version numbers), and common mistakes.
  5) **References** — `References used: [#1, #3, ...]`.
- Only use information present in the reference documents. If evidence is insufficient, write: *"The provided references don't fully support X."* and stop.
- When multiple versions appear, prefer the **newest version** but **name the version(s)** you used.
- **If the user specifies a version, only use documents from that version. If no documents match, say so explicitly.**
- Keep the final answer under ~250-300 words unless the user asked for more.
- **Do not include raw URLs in the answer; cite as [#n] only.**

# Reference Documents (use only what's needed)
{ctx_block}
"""

# ----- 헬퍼 -----
def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def _format_one_context(i: int, ctx: Dict[str, Any]) -> str:
    """
    retriever 출력 스키마를 백업본의 컨텍스트 블록 형식으로 변환
    ctx: {"id","title","url","version","content","score"}
    """
    cid = str(ctx.get("id", f"ctx_{i}"))
    title = ctx.get("title") or ""
    url = ctx.get("url") or ""
    version = ctx.get("version") or ""
    content = _shorten(ctx.get("content") or "", 1200)

    meta_line = []
    if title:   meta_line.append(f"title='{title}'")
    if version: meta_line.append(f"version={version}")
    if url:     meta_line.append(f"url={url}")
    meta_line = (" (" + ", ".join(meta_line) + ")") if meta_line else ""

    block = f"- [#{i}] (id: {cid}){meta_line}\n{content}\n"
    return block

def build_ctx_block_from_contexts(contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        blocks.append(_format_one_context(i, c))
    return "\n".join(blocks)

def truncate_to_token_budget(text: str, tokenizer, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def ensure_references_footer(answer: str, num_refs: int) -> str:
    if "References used:" in answer:
        return answer
    if num_refs > 0:
        idxs = ", ".join(f"[#{i}]" for i in range(1, num_refs + 1))
        return answer.rstrip() + f"\n\nReferences used: {idxs}"
    return answer

# ----- Public API (scripts/03_run_generator.py가 호출하는 부분) -----
def generate(
    question: str,
    contexts: List[Dict[str, Any]],
    *,
    k: int = 4,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    max_prompt_tokens: int = 28000,
) -> Dict[str, Any]:
    """
    retriever 출력 컨텍스트를 받아 Qwen으로 답변 생성
    반환: {"answer","used_ctx_ids","used_refs","k"}
    """
    tok, model = ensure_model()

    # 상위 k개만 사용 + 각 문서 1200자 내로 이미 슬라이스됨
    ctxs = (contexts or [])[:k]
    if not ctxs:
        return {
            "answer": "(no contexts) I don't know.",
            "used_ctx_ids": [],
            "used_refs": [],
            "k": 0,
        }

    ctx_block = build_ctx_block_from_contexts(ctxs)
    user_prompt = USER_TEMPLATE.format(query=question, ctx_block=ctx_block)

    # chat 템플릿 적용
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt = f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    # 토큰 예산 관리
    prompt = truncate_to_token_budget(prompt, tok, max_prompt_tokens)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    full = tok.decode(out[0], skip_special_tokens=True)
    # 간단히 assistant 파트만 추출
    answer = full.split(user_prompt)[-1].strip() if user_prompt in full else full.strip()
    answer = ensure_references_footer(answer, len(ctxs))

    used_refs = []
    used_ids = []
    for i, c in enumerate(ctxs, 1):
        used_ids.append(c.get("id"))
        used_refs.append({
            "ref_num": i,
            "id": c.get("id") or "",
            "url": c.get("url") or "",
            "version": c.get("version") or "",
            "title": c.get("title") or "",
        })

    return {"answer": answer, "used_ctx_ids": used_ids, "used_refs": used_refs, "k": len(ctxs)}

if __name__ == "__main__":
    seed_all(42)
    demo_q = "How does autograd compute gradients in PyTorch?"
    demo_ctxs = [{
        "id": "X",
        "title": "Autograd mechanics",
        "url": "pytorch_docs_2.8/autograd.html",
        "version": "2.8",
        "content": "Autograd builds a computation graph of tensor ops and performs reverse-mode automatic differentiation when calling backward().",
        "score": 0.9
    }]
    print(generate(demo_q, demo_ctxs)["answer"][:600])
