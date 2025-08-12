# Command format : python qa_chunks.py torchdocs_2.8_chunks.jsonl(json 파일 이름)

import json, sys
from collections import Counter, defaultdict
from statistics import mean, median

path = sys.argv[1] if len(sys.argv)>1 else "torchdocs_2.8_chunks.jsonl"

required_top = ["id","content","code_blocks","text_for_embedding","metadata"]
required_meta = ["url","path","version","lang","doc_type","title","section_anchor","chunk_index","num_tokens"]

key_issues = defaultdict(int)
ids=set(); dups=set()
len_content=[]; len_embed=[]
chunk_idx=Counter(); versions=Counter(); doctypes=Counter(); langs=Counter()
code_blocks_present=0; code_blocks_bad=0
empty_content=0; empty_embed=0; num_tokens_mismatch=0

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            r=json.loads(line)
        except Exception as e:
            key_issues["json_parse_error"]+=1
            continue
        for k in required_top:
            if k not in r: key_issues[f"missing:{k}"]+=1
        rid=r.get("id")
        if rid in ids: dups.add(rid)
        ids.add(rid)
        content=r.get("content","")
        embed=r.get("text_for_embedding","")
        meta=r.get("metadata",{})
        if not isinstance(meta, dict): key_issues["metadata_not_dict"]+=1; meta={}
        for mk in required_meta:
            if mk not in meta: key_issues[f"missing_meta:{mk}"]+=1

        if isinstance(content,str):
            wc=len(content.split()); len_content.append(wc)
            if wc==0: empty_content+=1
        else: key_issues["content_not_str"]+=1
        if isinstance(embed,str):
            we=len(embed.split()); len_embed.append(we)
            if we==0: empty_embed+=1
        else: key_issues["embed_not_str"]+=1

        ci=meta.get("chunk_index")
        if isinstance(ci,int): chunk_idx[ci]+=1
        else: key_issues["chunk_index_not_int"]+=1

        cb=r.get("code_blocks",[])
        if isinstance(cb,list) and len(cb)>0:
            code_blocks_present+=1
            if isinstance(ci,int) and ci>0: code_blocks_bad+=1
        elif not isinstance(cb,list):
            key_issues["code_blocks_not_list"]+=1

        nt=meta.get("num_tokens")
        if isinstance(nt,int):
            # allow ±5 words drift
            if abs(nt - (len(content.split()) if isinstance(content,str) else 0))>5:
                num_tokens_mismatch+=1

        versions[meta.get("version","unknown")]+=1
        doctypes[meta.get("doc_type","unknown")]+=1
        langs[meta.get("lang","unknown")]+=1

def pct(x, n): 
    return f"{(100.0*x/n):.2f}%" if n else "0%"

n = len(ids)
print("Total chunks:", n)
print("Duplicate ids:", len(dups))
print("Empty content:", empty_content, f"({pct(empty_content,n)})")
print("Empty text_for_embedding:", empty_embed, f"({pct(empty_embed,n)})")
print("Code blocks present:", code_blocks_present)
print("Code blocks in chunk_index>0 (should be 0):", code_blocks_bad)
print("num_tokens mismatch (tolerance±5):", num_tokens_mismatch, f"({pct(num_tokens_mismatch,n)})")
if len(len_content):
    print("Content words: min/median/mean/max =", min(len_content), median(len_content), round(mean(len_content),1), max(len_content))
if len(len_embed):
    print("Embed words  : min/median/mean/max =", min(len_embed), median(len_embed), round(mean(len_embed),1), max(len_embed))
print("Top chunk_index:", chunk_idx.most_common(6))
print("Versions:", versions.most_common())
print("Doc types:", doctypes.most_common())
print("Langs:", langs.most_common())
print("Key issues:", dict(key_issues))
