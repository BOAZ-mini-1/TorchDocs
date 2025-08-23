# 2. Retriever 실행
import os
os.environ["DISABLE_TF"] = "1"
os.environ["DISABLE_FLAX"] = "1"
import sys
import json
import argparse
from pathlib import Path

# 로컬 모듈 import
BASE_DIR = Path(__file__).resolve().parents[1]  # .../TorchDocs
sys.path.append(str(BASE_DIR))
from retriever import search

from retriever import parse_version_from_query, DEFAULT_VERSION


"""
질문 리스트를 받아 context 결과를 JSONL로 저장하는 파일
Input:
  --questions-jsonl : {"question": "..."} 형태의 JSONL (선택)
  --q               : 단일 질문 문자열 (선택)
  --top-k           : 최종 반환 문맥 개수 (기본: 5)
  --fetch-k         : FAISS에서 가져올 후보 개수 (기본: 50)
  --version         : 버전 필터링. 미지정 시 기본 2.8, 혹은 질문에서 자동 파싱
  --use-cross       : CrossEncoder rerank 사용 여부 (기본 false)
Output:
  --out             : 결과 JSONL 경로 (기본: data/eval/02_contexts.jsonl)

출력 JSONL 스키마(한 줄당):
{
  "question": "...",
  "version": "2.8",
  "contexts": [
      {"id","url","version","title","content","score"},
      ...
  ]
}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-jsonl", type=str, default=None)
    ap.add_argument("--q", type=str, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-k", type=int, default=50)
    ap.add_argument("--version", type=str, default=None)
    ap.add_argument("--use-cross", action="store_true")
    ap.add_argument("--out", type=str, default="data/eval/02_contexts.jsonl")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    questions = []
    if args.questions_jsonl:
        with open(args.questions_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = obj.get("question")
                if q:
                    questions.append(q)

    if args.q:
        questions.append(args.q)

    if not questions:
        print("No questions provided. Use --q or --questions-jsonl", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8") as w:
        for q in questions:
            resolved_v = args.version or parse_version_from_query(q) or DEFAULT_VERSION
            docs = search(
                q,
                top_k=args.top_k,
                fetch_k=args.fetch_k,
                version=resolved_v,
                use_cross_encoder=args.use_cross,
            )
            out = {
                "question": q,
                "version": resolved_v,  # None일 수 있음 (이 경우 retriever가 자동 판별)
                "contexts": docs
            }
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[retriever] wrote: {args.out}")

if __name__ == "__main__":
    main()

