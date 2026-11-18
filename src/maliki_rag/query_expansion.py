"""تحويل سؤال العامة إلى مصطلحات الفقهاء + توجيه الباب."""
import csv

from .normalize import normalize

SYSTEM = """أنت مساعد بحث في كتب الفقه المالكي. وظيفتك تحويل سؤال عامي إلى استعلامات بحث بمصطلحات الفقهاء، دون الإجابة عن السؤال.
أخرج JSON فقط بهذا الشكل:
{"queries": ["..."], "topics": ["..."]}
- queries: من 3 إلى 5 صيغ بحث بألفاظ الفقهاء ومرادفاتها (مثل: الهميان، المنطقة، المخيط، الطيب)، وتتضمن صيغة تذكر عين الصورة المسؤول عنها وصيغة أعم منها.
- topics: من 1 إلى 3 أبواب فقط، تُختار حرفياً من القائمة المعطاة، أو [] إن لم تتأكد."""


def load_glossary(path: str) -> list[dict]:
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    except FileNotFoundError:
        return []
    for r in rows:
        r["terms"] = [t.strip() for t in r["fiqh_terms"].split("|") if t.strip()]
        r["key"] = normalize(r["colloquial"])
    return rows


def glossary_hits(question: str, glossary: list[dict]) -> list[dict]:
    q = normalize(question)
    return [g for g in glossary if g["key"] and g["key"] in q]


def expand(question: str, glossary: list[dict], topics: list[str], llm=None) -> dict:
    hits = glossary_hits(question, glossary)
    queries = [f"{question} {t}" for g in hits for t in g["terms"]]
    chosen = [g["topic_hint"] for g in hits if g.get("topic_hint") in topics]
    if llm is not None:
        hint = "\n".join(f"- {g['colloquial']} = {' / '.join(g['terms'])}" for g in hits) or "لا يوجد"
        user = (f"السؤال: {question}\n\nمرادفات من المعجم:\n{hint}\n\n"
                "قائمة الأبواب المتاحة:\n" + "\n".join(f"- {t}" for t in topics[:200]))
        try:
            r = llm.json(SYSTEM, user, 800)
            queries += [q for q in r.get("queries", []) if isinstance(q, str)]
            chosen += [t for t in r.get("topics", []) if t in topics]
        except Exception as e:
            print("تحذير: فشل توسيع الاستعلام:", e)
    return {"queries": list(dict.fromkeys(queries))[:8], "topics": list(dict.fromkeys(chosen))[:3],
            "glossary_hits": [g["colloquial"] for g in hits]}
