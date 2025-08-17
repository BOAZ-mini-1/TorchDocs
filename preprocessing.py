#!/usr/bin/env python3
import re, json, hashlib, zipfile, argparse
from datetime import datetime

# Try BeautifulSoup; fallback to regex if unavailable
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

# Pipeline summary:
# ZIP 내부의 HTML open
# > h2/h3 기준 섹션 나누기(1차 chunking) 
# > 섹션이 길면 Hugging Face 토크나이저로 실제 서브워드 토큰을 세어 512 이하 맞춤 (new - split_by_hf_tokenizer())
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

# 문서 텍스트에서 자질구레한 기호 ¶, [source] 제거 및 공백 축약
def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("¶", " ")
    s = re.sub(r"\[source\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 경로에 따라서 대략적으로 api/tutorial/note를 판별하는 휴리스틱 
# if문 읽어보면 기능은 이해 갈꺼고 코드 돌려보니까 처리가 잘 안되는지? 다 note로 나옴,,ㅠ 사실 별로 안중요하긴 하다
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

def build_code_snippet(code_blocks, max_code_lines=40):
    if not code_blocks: return ""
    lines = []
    for b in code_blocks:
        snippet = (b.get("code") or "").splitlines()[:max_code_lines]
        lines.append("\n".join(snippet))
    return "\n\n".join(lines)

# 임베딩 모델 입력 문자열 조립하기
# title, breadcrumbs, content+code_blocks의 앞부분만(max_code_lines 줄)
# 앞부분만 자른 이유는 code가 너무 길면 임베딩 코튼 폭주하기 때문에 앞부분만 잘라서 안정화 작업
def build_embed_text(title, breadcrumbs, content, code_blocks, max_code_lines=40):
    crumbs = " > ".join(breadcrumbs) if breadcrumbs else ""
    code_joined = build_code_snippet(code_blocks, max_code_lines=max_code_lines)
    parts = [title, crumbs, content, code_joined]
    return "\n\n".join([p for p in parts if p]).strip()

# HTML의 <link rel="canonical">에서 정규 URL을 추출
# --ignore_canonical 옵션 : 무시하고 ZIP 내부 경로를 URL로 사용
def canonical_url_from_html(html: str):
    m = re.search(r'rel=["\']canonical["\']\s+href=["\']([^"\']+)["\']', html, flags=re.I)
    return m.group(1) if m else None

# ------------------ HTML -> Sections ------------------
# HTML을 파싱해서 h2/h3 헤더 기준으로 section 자르기
def sections_from_html(html: str):
    # returns: [{level, title, anchor, text, code_blocks}]

    # 각 섹션에 대해:
    # - title: 헤더 텍스트
    # - anchor: 헤더의 id 또는 headerlink에서 뽑은 앵커(없으면 slugify(title))
    # - text: 헤더 이후 다음 H2/H3 전까지의 본문 텍스트를 모아 정리(clean_text)
    # - code_blocks: 섹션 내 <pre> 코드블록을 찾아 언어 추정(lang) + 코드로 수집
    # - H2/H3가 하나도 없으면 H1 또는 전체 본문을 단일 섹션으로 리턴
    # - BeautifulSoup이 없으면 정규식 fallback으로 대략 분리

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
            out.append({"level": level, "title": title, "anchor": anchor or slugify(title),
                        "text": text, "code_blocks": code_blocks})
        if out:
            return out

        h1 = (main.find("h1").get_text(" ", strip=True) if main.find("h1") else "Document")
        body_text = clean_text(main.get_text(" ", strip=True))
        return [{"level":1, "title":h1, "anchor":slugify(h1), "text": body_text, "code_blocks": []}]

    # fallback (regex)
    sections = []
    for m in re.split(r"(?i)</h[23]>", html):
        hdr = re.search(r"(?i)<h([23])[^>]*>(.*?)$", m, flags=re.S)
        if not hdr: continue
        level = int(hdr.group(1))
        title = re.sub(r"<[^>]+>", " ", hdr.group(2)).strip() or "Section"
        content = re.sub(r"<[^>]+>", " ", m).strip()
        sections.append({"level": level, "title": title, "anchor": slugify(title),
                         "text": clean_text(content), "code_blocks": []})
    if not sections:
        text = clean_text(re.sub(r"<[^>]+>", " ", html))
        sections = [{"level":1, "title":"Document", "anchor":"document", "text": text, "code_blocks": []}]
    return sections

# ------------------ 2차 청킹: 단어 기반(기존) ------------------
# 토큰 근사치로 단어 수 기준 분할하기
def split_by_words(text: str, max_tokens=400, overlap=60):
    words = (text or "").split()
    if not words: return [""]
    chunks, i = [], 0
    step = max(1, max_tokens - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_tokens]))
        i += step
    return chunks

# ------------------ 2차 청킹: HF 토크나이저 기반(NEW) ------------------
def split_by_hf_tokenizer_with_context(title: str,
                                       breadcrumbs: list,
                                       content: str,
                                       code_blocks: list,
                                       tokenizer,
                                       max_len: int = 512,
                                       overlap: int = 60,
                                       max_code_lines: int = 20,
                                       reserve_special: int = 2):
    """
    `prefix(title/breadcrumbs) + content + suffix(코드 일부) 조합`이
    add_special_tokens=True 기준 max_len을 넘지 않도록 content를 토큰 단위로 쪼개는 작업
    - 첫 청크: suffix(코드) 포함
    - 이후 청크: suffix 없음
    """
    prefix = "\n\n".join([p for p in [title, " > ".join(breadcrumbs) if breadcrumbs else ""] if p])
    suffix = build_code_snippet(code_blocks, max_code_lines=max_code_lines)  # first chunk only

    # encode lengths (add_special_tokens=False로 "실 텍스트 토큰"만 계산)
    enc = tokenizer
    prefix_ids = enc.encode(prefix, add_special_tokens=False) if prefix else []
    suffix_ids = enc.encode(suffix, add_special_tokens=False) if suffix else []
    content_ids = enc.encode(content or "", add_special_tokens=False)

    # 실효 최대치(특수토큰 여유)
    eff_max = max_len - max(0, reserve_special)

    # 각 청크에서 content에 배정 가능한 토큰 수
    first_allow = eff_max - len(prefix_ids) - len(suffix_ids)
    next_allow  = eff_max - len(prefix_ids)

    # 만약 코드 때문에 first_allow가 너무 작으면(<=0), 코드 생략
    use_suffix = True
    if first_allow <= 0:
        use_suffix = False
        first_allow = eff_max - len(prefix_ids)

    first_allow = max(1, first_allow)
    next_allow  = max(1, next_allow)

    # 오버랩은 토큰 단위
    step_first = max(1, first_allow - overlap)
    step_next  = max(1, next_allow  - overlap)

    out = []
    i = 0
    chunk_idx = 0
    while i < len(content_ids):
        allow = first_allow if chunk_idx == 0 else next_allow
        step  = step_first if chunk_idx == 0 else step_next
        piece_ids = content_ids[i:i+allow]
        piece_text = enc.decode(piece_ids, skip_special_tokens=True)

        # text_for_embedding 생성
        if chunk_idx == 0 and use_suffix and suffix:
            tfe = "\n\n".join([p for p in [prefix, piece_text, suffix] if p])
            code_for_chunk = code_blocks  # 첫 청크에만 코드 유지
        else:
            tfe = "\n\n".join([p for p in [prefix, piece_text] if p])
            code_for_chunk = []

        # 안전장치: 최종 문자열이 정말 max_len 이내인지 확인(넘치면 마지막 문장 살짝 잘라냄)
        # add_special_tokens=True로 실사용 길이 측정
        if len(enc.encode(tfe, add_special_tokens=True)) > max_len:
            # 아주 긴 경우를 위해 보수적으로 조금 더 줄임
            # (일반적으로 여기 안 걸리지만, 토크나이저 규칙에 따라 걸릴 수 있기 때문)
            shrink_ids = enc.encode(piece_text, add_special_tokens=False)
            budget = max(1, allow - 8)  # 여유 8토큰
            piece_text = enc.decode(shrink_ids[:budget], skip_special_tokens=True)
            if chunk_idx == 0 and use_suffix and suffix:
                tfe = "\n\n".join([p for p in [prefix, piece_text, suffix] if p])
            else:
                tfe = "\n\n".join([p for p in [prefix, piece_text] if p])

        out.append({
            "content": piece_text,
            "text_for_embedding": tfe,
            "code_blocks": code_for_chunk,
            "embed_num_tokens": len(enc.encode(tfe, add_special_tokens=True))  # HF 기준 길이 기록
        })

        i += step
        chunk_idx += 1

    if not out:
        # content가 비고, 코드도 없는 케이스 예외 처리 (혹시 모르니까 ㅇ.ㅇ)
        base_tfe = "\n\n".join([p for p in [prefix, suffix] if p]) or title
        out = [{"content": "", "text_for_embedding": base_tfe,
                "code_blocks": code_blocks if suffix else [],
                "embed_num_tokens": len(enc.encode(base_tfe, add_special_tokens=True))}]
    return out

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
                max_code_lines: int = 20,
                chunker: str = "hf",           # 기본값: HF 토크나이저 기반
                hf_model: str = "intfloat/e5-large-v2",
                hf_max_len: int = 512,
                hf_reserve: int = 10):
    """
    HTML ZIP -> 섹션(H2/H3) -> 2차 청킹 -> NDJSON
    chunker:
      - "hf": HF 토크나이저 기반 (임베딩 모델 길이 보장)
      - "words": 단어 기반(이전 방식)
    """
    # HF 토크나이저 준비(필요 시)
    tokenizer = None
    if chunker == "hf":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model)

    count_files = 0
    items = 0

    with zipfile.ZipFile(zip_path, "r") as z, open(out_jsonl, "w", encoding="utf-8") as out:
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

            url = name if ignore_canonical else (canonical_url_from_html(html) or name)
            version = force_version or default_version
            if not force_version:
                mver = re.search(r"/docs/([0-9]+\.[0-9]+|stable|nightly)/", url)
                if mver: version = mver.group(1)

            doc_type = guess_doc_type(url)
            sections = sections_from_html(html)

            for sec in sections:
                base_title = sec["title"]
                base_anchor = sec["anchor"] or slugify(base_title)
                base_text = sec["text"] or ""
                base_code = sec.get("code_blocks", [])
                breadcrumbs = []

                # ---- 2차 청킹 ----
                if chunker == "hf":
                    pieces = split_by_hf_tokenizer_with_context(
                        title=base_title,
                        breadcrumbs=breadcrumbs,
                        content=base_text,
                        code_blocks=base_code,
                        tokenizer=tokenizer,
                        max_len=min(hf_max_len, max_tokens or hf_max_len),  # 사용자가 --max_tokens 510 주면 그 값도 반영
                        overlap=overlap,
                        max_code_lines=max_code_lines,
                        reserve_special=hf_reserve
                    )
                else:
                    # 단어 기반(기존)
                    chunks = split_by_words(base_text, max_tokens=max_tokens, overlap=overlap)
                    pieces = []
                    for idx2, txt in enumerate(chunks):
                        code_for_chunk = base_code if idx2 == 0 else []
                        tfe = build_embed_text(base_title, breadcrumbs, txt, code_for_chunk, max_code_lines)
                        pieces.append({
                            "content": txt,
                            "text_for_embedding": tfe,
                            "code_blocks": code_for_chunk,
                            "embed_num_tokens": len((txt or "").split())  # 근사값
                        })

                # ---- 출력 ----
                for idx, p in enumerate(pieces):
                    content = (p["content"] or "").strip()
                    code_blocks = p.get("code_blocks", []) or []

                    # 완전 빈 청크(본문도 코드도 없음) 스킵
                    if not content and not code_blocks:
                        continue

                    metadata = {
                        "url": url,
                        "path": name,
                        "version": version,
                        "lang": "en",
                        "doc_type": doc_type,
                        "title": base_title,
                        "breadcrumbs": breadcrumbs,
                        "section_anchor": base_anchor,
                        "chunk_index": idx,
                        "split_policy": {
                            "by": "hf" if chunker == "hf" else "words",
                            "max": min(hf_max_len, max_tokens or hf_max_len) if chunker == "hf" else max_tokens,
                            "overlap": overlap
                        },
                        "num_tokens": len(content.split()),  # 기존 QA 스크립트 호환 위해 단어 수 유지
                        "embed_num_tokens": p.get("embed_num_tokens", None),  # HF 토큰 수(실사용 기준)
                        "symbol_tags": extract_symbols(content),
                        "created_at": datetime.utcnow().isoformat() + "Z"
                    }

                    _id = sha1(f"{url}#{base_anchor}|{idx}|{version}")
                    out_obj = {
                        "id": _id,
                        "content": content,
                        "code_blocks": code_blocks,
                        "text_for_embedding": p["text_for_embedding"],
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

    # 공통 청킹 파라미터
    ap.add_argument("--max_tokens", type=int, default=510, help="max per chunk (words for 'words', tokens for 'hf')")
    ap.add_argument("--overlap", type=int, default=60, help="overlap (words for 'words', tokens for 'hf')")

    # 문서 메타
    ap.add_argument("--default_version", type=str, default="2.8")
    ap.add_argument("--force_version", type=str, default=None)
    ap.add_argument("--ignore_canonical", action="store_true")
    ap.add_argument("--max_code_lines", type=int, default=20)

    # 청킹 방식 선택
    ap.add_argument("--chunker", type=str, default="hf", choices=["hf","words"],
                    help="hf: HF tokenizer-based (recommended), words: word-based")

    # HF 청커 옵션
    ap.add_argument("--hf_model", type=str, default="intfloat/e5-large-v2",
                    help="HF tokenizer to use for 'hf' chunker")
    ap.add_argument("--hf_max_len", type=int, default=512,
                    help="Model max length (with special tokens)")
    ap.add_argument("--hf_reserve", type=int, default=10,
                    help="Reserved tokens for special tokens (CLS/SEP, etc.)")

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
        max_code_lines=args.max_code_lines,
        chunker=args.chunker,
        hf_model=args.hf_model,
        hf_max_len=args.hf_max_len,
        hf_reserve=args.hf_reserve
    )
    print(f"Processed files: {n_files}, chunks written: {n_items}")

if __name__ == "__main__":
    main()
