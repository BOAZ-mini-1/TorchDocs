#!/usr/bin/env python3
import re, json, hashlib, zipfile, argparse
from datetime import datetime

# Try BeautifulSoup (더 정확한 HTML 파싱). 없으면 정규식 fallback.
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

# Pipeline summary:
# ZIP 내부의 HTML open
# > h2/h3 기준 섹션 나누기(1차 chunking) 
# > 섹션이 길면 단어 수 기준으로 분할(2차 chunking w/ overlap)
# > embedding용 문자열 만들기: title+content+code 일부 
# > 메타데이터/ID 붙여서 NDJSON(.jsonl)으로 저장 


# ------------------ Helpers ------------------
# 추후 db 작업을 위한 고유 ID 생성 : id = sha1(f"{url}#{anchor}|{chunk_index}|{version}")
def sha1(s: str) -> str:
    import hashlib as _h
    return _h.sha1(s.encode("utf-8")).hexdigest()

# 제목 같은 문자열을 regex로 URL-friendly 하게 정리
# e.g. 소문자/공백 -> hyphen/특수문자 제거 / section anchor가 없을 때 대체제로 사용
def slugify(text: str) -> str:
    text = re.sub(r"[\s/]+", "-", (text or "").strip().lower())
    text = re.sub(r"[^a-z0-9\-._]", "", text)
    return text.strip("-_.") or "section"

# 문서 텍스트에서 자질구레한 곁다리 더러운 기호? 정리 : 열어보니까 ¶, [source] 이런거 있길래 제거하고 공백 축약
def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("¶", " ")
    s = re.sub(r"\[source\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 경로에 따라서 대략적으로 api/tutorial/note를 판별하는 휴리스틱 
# if문 읽어보면 기능은 이해 갈꺼고 코드 돌려보니까 처리가 잘 안되는지? 다 note로 나옴,,ㅠ 사실 별로 안중요하긴 함
def guess_doc_type(url_or_path: str) -> str:
    u = (url_or_path or "").lower()
    if "/generated/" in u or "/nn/" in u or "/torch." in u:
        return "api"
    if "/tutorials/" in u:
        return "tutorial"
    return "note"

# 본문에서 torch.xx 같은 심볼명을 regex로 찾아 리스트로 return (나중에 re-ranking할 때 사용하기 위함) 
def extract_symbols(text: str):
    return list(sorted(set(re.findall(r"(torch\.[A-Za-z0-9_\.]+)", text or ""))))

# 토큰 근사치로 단어 수 기준 분할하기 
# 참고: default는 아래 파라미터처럼 해놓긴 했는데 .jsonl 파싱할 때는 인자 각각 350, 60으로 주었음!
# 지금은 "단어 수" 기반 근사인 상태이고, 나중에 모델 토크나이저(tiktoken 등)로 바꾸면 더 정확도가 높아질 것임. 이 점을 고려해서 먼저 이렇게 진행하고 후에 실제 모델 토크나이저를 도입한 후의 정확도와 비교 실험하는 것도 좋은 연구가 될 듯.
def split_by_tokens(text: str, max_tokens=400, overlap=60):
    """단어 단위 근사 분할. 필요시 tiktoken으로 교체 가능."""
    words = (text or "").split()
    if not words: return [""]
    chunks, i = [], 0
    step = max(1, max_tokens - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_tokens]))
        i += step
    return chunks

# 임베딩 모델 입력 문자열 조립하기~ (지피띠니가 아예 임베딩용 텍스트를 json에 미리 만들어 놓으면 효율적이라고 해서...ㅇ.ㅇ)
# title, breadcrumbs, content+code_blocks의 앞부분만(max_code_lines 줄)
# 앞부분만 자른 이유는 code가 너무 길면 임베딩 코튼 폭주하니까 앞부분만 잘라서 안정화 작업
def build_embed_text(title, breadcrumbs, content, code_blocks, max_code_lines=40):
    """임베딩 입력 문자열(title + breadcrumbs + content + 코드 스니펫 일부)."""
    crumbs = " > ".join(breadcrumbs) if breadcrumbs else ""
    code_joined = ""
    if code_blocks:
        lines = []
        for b in code_blocks:
            snippet = (b.get("code") or "").splitlines()[:max_code_lines]
            lines.append("\n".join(snippet))
        code_joined = "\n\n".join(lines)
    parts = [title, crumbs, content, code_joined]
    return "\n\n".join([p for p in parts if p]).strip()

# HTML의 <link rel="canonical">에서 정규 URL을 추출
# --ignore_canonical 옵션 : 무시하고 ZIP 내부 경로를 URL로 사용
def canonical_url_from_html(html: str):
    m = re.search(r'rel=["\']canonical["\']\s+href=["\']([^"\']+)["\']', html, flags=re.I)
    return m.group(1) if m else None

# ------------------ HTML -> Sections Extraction ------------------

# HTML을 파싱해서 h2/h3 헤더 기준으로 section 자르기
def sections_from_html(html: str):
    """
    returns: [{level, title, anchor, text, code_blocks}]

    각 섹션에 대해:
    - title: 헤더 텍스트
    - anchor: 헤더의 id 또는 headerlink에서 뽑은 앵커(없으면 slugify(title))
    - text: 헤더 이후 다음 H2/H3 전까지의 본문 텍스트를 모아 정리(clean_text)
    - code_blocks: 섹션 내 <pre> 코드블록을 찾아 언어 추정(lang) + 코드로 수집
    - H2/H3가 하나도 없으면 H1 또는 전체 본문을 단일 섹션으로 리턴
    - BeautifulSoup이 없으면 정규식 fallback으로 대략 분리
    """
    if HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.select_one("main") or soup

        headers = main.find_all(["h2", "h3"])
        out = []
        for h in headers:
            level = int(h.name[1])
            title = h.get_text(" ", strip=True) or "Section"
            anchor = h.get("id")
            if not anchor:
                a = h.find("a", attrs={"class":"headerlink"})
                if a and a.get("href", "").startswith("#"):
                    anchor = a["href"][1:]

            # 다음 H2/H3 전까지 수집
            nodes, sib = [], h.next_sibling
            while sib and not (getattr(sib, "name", None) in ["h2", "h3"]):
                nodes.append(sib); sib = sib.next_sibling

            text_parts, code_blocks = [], []
            for node in nodes:
                name = getattr(node, "name", "")
                if name in ["pre", "code", "div"]:
                    pre = node
                    if name == "div":
                        pre = node.find("pre")
                    if pre:
                        code_text = pre.get_text("\n", strip=True)
                        lang = "text"
                        cls = (pre.get("class") or []) + (node.get("class") or [])
                        for c in cls:
                            m = re.search(r"language-([a-zA-Z0-9_]+)", c)
                            if m: lang = m.group(1)
                        if code_text:
                            code_blocks.append({"lang": lang, "code": code_text})
                    else:
                        txt = node.get_text(" ", strip=True) if hasattr(node, "get_text") else str(node).strip()
                        if txt: text_parts.append(txt)
                else:
                    txt = node.get_text(" ", strip=True) if hasattr(node, "get_text") else str(node).strip()
                    if txt: text_parts.append(txt)

            text = clean_text("\n".join(text_parts))
            out.append({
                "level": level,
                "title": title,
                "anchor": anchor or slugify(title),
                "text": text,
                "code_blocks": code_blocks
            })

        if out:
            return out

        # Fallback: H2/H3가 없으면 H1/전체
        h1 = (main.find("h1").get_text(" ", strip=True) if main.find("h1") else "Document")
        body_text = clean_text(main.get_text(" ", strip=True))
        return [{
            "level": 1,
            "title": h1,
            "anchor": slugify(h1),
            "text": body_text,
            "code_blocks": []
        }]

    # BeautifulSoup이 없으면 regex 기반 처리
    sections = []
    for m in re.split(r"(?i)</h[23]>", html):
        hdr = re.search(r"(?i)<h([23])[^>]*>(.*?)$", m, flags=re.S)
        if not hdr: 
            continue
        level = int(hdr.group(1))
        title = re.sub(r"<[^>]+>", " ", hdr.group(2)).strip() or "Section"
        content = re.sub(r"<[^>]+>", " ", m).strip()
        sections.append({
            "level": level,
            "title": title,
            "anchor": slugify(title),
            "text": clean_text(content),
            "code_blocks": []
        })
    if not sections:
        text = clean_text(re.sub(r"<[^>]+>", " ", html))
        sections = [{
            "level": 1,
            "title": "Document",
            "anchor": "document",
            "text": text,
            "code_blocks": []
        }]
    return sections

# ------------------ Main processing ------------------
# Main function
def process_zip(zip_path: str,
                out_jsonl: str,
                limit_files: int = 0,
                max_tokens: int = 400,
                overlap: int = 60,
                default_version: str = "2.8",
                force_version: str = None,
                ignore_canonical: bool = False,
                max_code_lines: int = 40):
    """
    ZIP 안의 HTML을 읽어 H2/H3 기준 섹션화 → 토큰 분할 → NDJSON으로 저장.
    return: (처리한 HTML 파일 수, 생성한 청크 수)


    """
    count_files = 0
    items = 0

    with zipfile.ZipFile(zip_path, "r") as z, open(out_jsonl, "w", encoding="utf-8") as out:
        # 1. ZIP을 열고 HTML 파일 목록 정렬
        names = [n for n in z.namelist() if n.lower().endswith((".html", ".htm"))]
        names.sort()

        for name in names:
            if limit_files and count_files >= limit_files:
                break

            html_bytes = z.read(name)
            try:
                html = html_bytes.decode("utf-8", errors="ignore")
            except Exception:
                count_files += 1
                continue

            # 2. 각 HTML 파일을 읽어 URL 선택
            # default는 canonical URL, --ignore_canonical이면 무시하고 ZIP 경로 사용
            url = name if ignore_canonical else (canonical_url_from_html(html) or name)

            # 3. Version 결정: force > URL 파싱 > default
            # --force_version가 있으면 무조건 그 값 > 아니면 URL에서 /docs/{ver}/ 패턴을 뽑아 쓰고 > 없으면 --default_version
            version = force_version or default_version
            if not force_version:
                mver = re.search(r"/docs/([0-9]+\.[0-9]+|stable|nightly)/", url)
                if mver:
                    version = mver.group(1)

            # 4. 문서 타입 추정: guess_doc_type(url)
            doc_type = guess_doc_type(url)
            # 5. sections_from_html로 H2/H3 기준 섹션화
            sections = sections_from_html(html)

            for sec in sections:
                base_title = sec["title"]
                base_anchor = sec["anchor"] or slugify(base_title)
                base_text = sec["text"] or ""
                base_code = sec.get("code_blocks", [])

                # 6. 각 섹션 텍스트를 split_by_tokens로 2차 분할(overlap 포함), chunk_index 부여
                chunks = split_by_tokens(base_text, max_tokens=max_tokens, overlap=overlap)
                for idx, txt in enumerate(chunks):
                    content = txt.strip()
                    code_blocks = base_code if idx == 0 else []  # 코드 중복 방지

                    # 7. 빈 content & 코드도 없는 빈 청크면 skip 
                    # content가 비어도 코드블록이 있으면 유지
                    if not content and not code_blocks:
                        continue

                    breadcrumbs = [] 
                    # 8. build_embed_text로 임베딩 입력 문자열 생성(코드는 --max_code_lines 줄만 포함)
                    text_for_embedding = build_embed_text(
                        base_title, breadcrumbs, content, code_blocks, max_code_lines=max_code_lines
                    )

                    # 9. metadata 채우기
                    metadata = {
                        "url": url,                          # 원문 출처 링크 (현재는 --ignore_canonical 사용중이라 'pytorch_docs_2.8/cppdocs_.html' 이렇게 로컬 ZIP 경로가 들어가 있음. 실서비스에서는 https://pytorch.org/...#anchor로 변환 필요) 
                        "path": name,                        # ZIP 내부 파일 경로 (디버깅 시 원본 파일 추적 용도로 넣음)
                        "version": version,
                        "lang": "en",
                        "doc_type": doc_type,                # 대략적인 문서의 타입(api/tutorial/note) : 검색 범위 축소를 위해 넣었지만 큰 비중을 차지하진 않아서 제거 가능
                        "title": base_title,                 # 해당 section의 제목
                        "breadcrumbs": breadcrumbs,          # 상위 카테고리 경로 : h1>h2>h3>... (embedding context 유지를 위함)
                        "section_anchor": base_anchor,       # 문서 내 anchor : 정확한 섹션 링크 생성 용도(클릭하면 바로 그 위치로 이동하도록)
                        "chunk_index": idx,                  # 같은 section에서 몇 번째 chunk인지 (답변 조립할 때 인접 청크 병합 용도 & 리랭킹로직, 즉 같은 섹션에서 2개 이상 뽑히는 것을 방지)
                        "split_policy": {"by": "tokens", "max": max_tokens, "overlap": overlap},  # 분할 설정 기록 > 이건 이후 parameter sweeping하면서 experiment할 때 관리하기 편할려고 넣었음
                        "num_tokens": len(content.split()),  # content의 대략 token/단어 수 (context window 관리를 위해, 즉 답변 조립 시 초과 방지)
                        "symbol_tags": extract_symbols(content),
                        "created_at": datetime.utcnow().isoformat() + "Z" # 생성 시각인데 이건 future work 처럼 이후 데이터 파이프라인 추적하거나 old/new 데이터 동시 운용 사 최신에 가중치를 두기 위해 사용할 용도 ㅇㅇ
                    }

                    # 10. 벡터DB용 고유 id 생성: sha1(f"{url}#{anchor}|{idx}|{version}")
                    _id = sha1(f"{url}#{base_anchor}|{idx}|{version}")

                    # 11. 한 청크(= 한 라인)씩 NDJSON(.jsonl)로 출력
                    out_obj = {
                        "id": _id,
                        "content": content,
                        "code_blocks": code_blocks,
                        "text_for_embedding": text_for_embedding,
                        "metadata": metadata
                    }
                    out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    items += 1

            count_files += 1

    return count_files, items

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to ZIP that contains HTML files")
    ap.add_argument("--out", required=True, help="Output JSONL (NDJSON) path")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of HTML files (0 = all)")
    ap.add_argument("--max_tokens", type=int, default=400, help="Token-ish chunk size (by words)")
    ap.add_argument("--overlap", type=int, default=60, help="Overlap between chunks (by words)")
    ap.add_argument("--default_version", type=str, default="2.8", help="Fallback version label")
    ap.add_argument("--force_version", type=str, default=None, help="Force all chunks to this version (overrides URL/default)")
    ap.add_argument("--ignore_canonical", action="store_true", help="Use ZIP path instead of <link rel=canonical> URL")
    ap.add_argument("--max_code_lines", type=int, default=40, help="Max lines of code to include in embedding text")
    args = ap.parse_args()

    n_files, n_items = process_zip(
        zip_path=args.zip,
        out_jsonl=args.out,
        limit_files=args.limit,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        default_version=args.default_version,
        force_version=args.force_version,
        ignore_canonical=args.ignore_canonical,
        max_code_lines=args.max_code_lines
    )
    print(f"Processed files: {n_files}, chunks written: {n_items}")

if __name__ == "__main__":
    main()
