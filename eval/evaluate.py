"""تقييم الاسترجاع (Recall@k) والجواب الكامل مقابل أسئلة ذهبية.
صيغة السؤال: {"id","question","expected":[{"book_id"?, "contains"}],"expect_not_found":bool}
"""
import json
import time
from pathlib import Path

from maliki_rag.normalize import normalize

GOLD = Path(__file__).parent / "gold_questions.jsonl"


def _match(text: str, exp: dict, book_id: str) -> bool:
    if exp.get("book_id") and exp["book_id"] != book_id:
        return False
    return normalize(exp["contains"]) in normalize(text)


def run(cfg: dict, full: bool = False, ks=(1, 5, 8)):
    from maliki_rag.pipeline import Pipeline
    pl = Pipeline(cfg)
    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    rows, rec = [], {k: [0, 0] for k in ks}
    nf_ok = nf_tot = ans_ok = ans_tot = 0
    for g in gold:
        _, hits = pl.retrieve(g["question"])
        row = {"id": g["id"]}
        if g.get("expected"):
            ranks = [i for i, h in enumerate(hits, 1)
                     if any(_match(h["chunk"]["text"] + " " + h["matn_text"], e, h["chunk"]["book_id"])
                            for e in g["expected"])]
            first = ranks[0] if ranks else None
            row["first_rank"] = first
            for k in ks:
                rec[k][1] += 1
                rec[k][0] += bool(first and first <= k)
        if full:
            res = pl.ask(g["question"], log=False)
            row["status"] = res["status"]
            row["rejected_quotes"] = len(res["warnings"])
            if g.get("expect_not_found"):
                nf_tot += 1
                nf_ok += res["status"] == "not_found"
            elif g.get("expected"):
                ans_tot += 1
                cited = " ".join(c["quote"] for c in res["citations"] if c["match_type"] == "exact")
                ans_ok += any(_match(cited, e, e.get("book_id") or "") if not e.get("book_id")
                              else _match(cited, {"contains": e["contains"]}, "") for e in g["expected"])
        rows.append(row)
    summary = {f"recall@{k}": (rec[k][0] / rec[k][1] if rec[k][1] else None) for k in ks}
    if full:
        summary["correct_refusal"] = nf_ok / nf_tot if nf_tot else None
        summary["answer_hit"] = ans_ok / ans_tot if ans_tot else None
    out = {"summary": summary, "rows": rows}
    d = Path(__file__).parent / "results"
    d.mkdir(exist_ok=True)
    (d / f"{int(time.time())}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out
