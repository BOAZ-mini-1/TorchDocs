# --- put these 2 lines at the very top ---
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # block TensorFlow
os.environ["TRANSFORMERS_NO_FLAX"] = "1" # block JAX/Flax (optional)

import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

# 중요: sentence_transformers는 전혀 import하지 않고 토크나이저만 사용!
from transformers import AutoTokenizer


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSONL parse error at line {line_no}: {e}")
                continue


def pick_text(obj: Dict[str, Any], prefer: str, fallbacks=("text", "content", "chunk", "page_content")) -> Optional[str]:
    if prefer in obj and isinstance(obj[prefer], str):
        return obj[prefer]
    for k in fallbacks:
        if k in obj and isinstance(obj[k], str):
            return obj[k]
    return None


def main():
    parser = argparse.ArgumentParser(description="Count tokens per chunk and report >512.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to input .jsonl file")
    # sentence-transformers 계열 토크나이저 이름 사용 가능 (임베딩은 안 함)
    parser.add_argument("--tokenizer", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HF tokenizer repo name (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--text-field", type=str, default="text_for_embedding",
                        help="Field name to read text from (fallbacks: text, content, chunk, page_content)")
    parser.add_argument("--max-len", type=int, default=512, help="Threshold for long chunks")
    parser.add_argument("--show-top", type=int, default=10, help="Show top-N longest chunks (by token length)")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found")

    # fast tokenizer 권장
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    total = 0
    over = 0
    lengths = []
    longest = []  # list of tuples (tok_len, idx, preview)

    for idx, obj in enumerate(iter_jsonl(jsonl_path), 1):
        text = pick_text(obj, args.text_field)
        if text is None:
            # 텍스트가 없으면 스킵 (필요시 카운트 로직 조정 가능)
            continue

        # add_special_tokens=True -> 실제 모델 입력 길이 기준
        # truncation=False -> 실제 길이를 그대로 계산
        enc = tokenizer(text, add_special_tokens=True, truncation=False)
        tok_len = len(enc["input_ids"])
        lengths.append(tok_len)
        total += 1
        if tok_len > args.max_len:
            over += 1

        # 상위 N개 긴 청크 추리기 (간단한 방식)
        preview = text[:120].replace("\n", " ")
        longest.append((tok_len, idx, preview))

    # 정렬하여 상위 N개
    longest.sort(key=lambda x: x[0], reverse=True)
    topN = longest[: max(args.show_top, 0)]

    # 결과 출력
    print("=== Token Length Report ===")
    print(f"File           : {jsonl_path}")
    print(f"Tokenizer      : {args.tokenizer}")
    print(f"Text field     : {args.text_field}")
    print(f"Max length     : {args.max_len}")
    print(f"Total chunks   : {total}")
    print(f"> {args.max_len} tokens: {over}  ({(over/total*100 if total else 0):.2f}%)")
    if lengths:
        print(f"Min/Max/Avg    : {min(lengths)} / {max(lengths)} / {sum(lengths)/len(lengths):.2f}")
    print()

    if topN:
        print(f"=== Top {len(topN)} longest chunks ===")
        for tok_len, idx, preview in topN:
            print(f"[#{idx}] tokens={tok_len}  preview={preview!r}")
    else:
        print("No chunks or no text found.")

# python token_analyzer.py --jsonl data/processed/torchdocs_2.8_chunks_e5.jsonl
if __name__ == "__main__":
    main()
