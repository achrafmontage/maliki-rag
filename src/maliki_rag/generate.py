"""القاضي + التوليد المقيَّد + التحقق الحرفي من الاقتباس."""
from .normalize import quote_in

JUDGE_SYSTEM = """أنت مدقق فقهي مالكي صارم. تُعطى سؤالاً ومقاطع من كتب المذهب. لكل مقطع حدد أولاً المسألة التي يعالجها المقطع فعلاً، ثم قارنها بعين السؤال:
- exact: المقطع ينص على حكم نفس الصورة المسؤول عنها بقيودها.
- related: المقطع في مسألة مجاورة أو أصل عام لا ينص على صورة السؤال.
- none: لا صلة له.
كن متشدداً: الاشتراك في الباب أو في لفظ واحد (كالحزام والمنطقة) لا يجعل المقطع exact. المقطع الذي يعالج استعمالاً مختلفاً للشيء نفسه هو related على الأكثر.
أخرج JSON فقط:
{"judgments":[{"id":"P1","issue_in_passage":"المسألة التي يعالجها المقطع فعلاً","match":"exact|related|none","reason":"سبب قصير"}]}"""

ANSWER_SYSTEM = """أنت ناقل أمين لنصوص الفقه المالكي، ولست مفتياً. قواعدك:
1. لا تستعمل أي معلومة خارج المقاطع المعطاة، ولا تقس ولا تستنبط ولا ترجّح.
2. كل حكم في الملخص يجب أن يُسند باقتباس حرفي من المقطع (انسخه كما هو دون تغيير حرف).
3. أبقِ القيود والاستثناءات كما هي (إلا، ما لم، على المشهور، وقيل، خلافاً لـ…).
4. عند اختلاف المقاطع اعرض الأقوال بقائليها دون ترجيح.
5. لا تُضف تعليلاً ولا تفريعاً من عندك. اجعل الصياغة قريبة من لفظ النص.
6. المقاطع الموسومة related لا تُعرض على أنها جواب عن السؤال، بل تُذكر بعبارة: «نص في مسألة مجاورة».
7. «السياق» للفهم فقط ولا يجوز الاقتباس منه.
أخرج JSON فقط:
{"summary":"ملخص قصير بصياغة قريبة من النص، مع أرقام المقاطع مثل [P1]",
 "citations":[{"id":"P1","quote":"اقتباس حرفي من نص المقطع أو من المتن المشروح","note":"ما يفيده الاقتباس"}],
 "disagreements":"وجوه الخلاف إن وُجدت أو فارغ"}"""


def source_label(c: dict) -> str:
    loc = " > ".join(c["path"][-2:]) if c["path"] else ""
    pg = f"ج{c['vol']} ص{c['page']}" if c.get("page") else ""
    return f"{c['title']} — {c['author']} (ت {c['death_hijri']}هـ) {pg} {('| ' + loc) if loc else ''}".strip()


def _fmt(p: str, h: dict, extra: str = "") -> str:
    c = h["chunk"]
    s = f"[{p}] المصدر: {source_label(c)}\n"
    if h["matn_text"]:
        s += f"المتن المشروح: {h['matn_text']}\n"
    s += f"النص: {c['text']}\n"
    if extra:
        s += extra + "\n"
    return s


def judge(llm, question: str, hits: list[dict]) -> dict[str, dict]:
    body = "\n".join(_fmt(f"P{i+1}", h) for i, h in enumerate(hits))
    r = llm.json(JUDGE_SYSTEM, f"السؤال: {question}\n\nالمقاطع:\n{body}", 2500)
    out = {j["id"]: j for j in r.get("judgments", []) if "id" in j}
    return out


def answer(llm, question: str, hits: list[dict], max_passages: int = 6) -> dict:
    labels = {f"P{i+1}": h for i, h in enumerate(hits)}
    verdicts = judge(llm, question, hits)
    kept = {p: h for p, h in labels.items()
            if verdicts.get(p, {}).get("match") in ("exact", "related")}
    base = {"verdicts": verdicts, "labels": labels, "warnings": []}
    if not kept:
        return {**base, "status": "not_found", "summary": "", "citations": [], "disagreements": ""}

    order = sorted(kept, key=lambda p: verdicts[p]["match"] != "exact")[:max_passages]
    body = "\n".join(
        _fmt(p, kept[p], f"التصنيف: {'يعالج عين السؤال' if verdicts[p]['match']=='exact' else 'related (مسألة مجاورة)'}"
             f" — المسألة في المقطع: {verdicts[p].get('issue_in_passage','')}\n"
             f"السياق (غير قابل للاقتباس): {kept[p]['context_prev'][-300:]} ... {kept[p]['context_next'][:300]}")
        for p in order)
    r = llm.json(ANSWER_SYSTEM, f"السؤال: {question}\n\nالمقاطع:\n{body}", 3000)

    valid, warnings = [], []
    for c in r.get("citations", []):
        p = c.get("id")
        if p not in order:
            warnings.append(f"اقتباس برقم مقطع غير معطى: {p}")
            continue
        h = kept[p]
        corpus = h["chunk"]["text"] + " " + h["matn_text"]
        if not quote_in(c.get("quote", ""), corpus):
            warnings.append(f"رُفض اقتباس غير حرفي من {p}")
            continue
        valid.append({**c, "match_type": verdicts[p]["match"]})  # التصنيف من القاضي لا من النموذج

    if not valid:
        status = "not_found"
    elif any(c["match_type"] == "exact" for c in valid):
        status = "found"
    else:
        status = "related_only"
    return {**base, "status": status, "summary": r.get("summary", "") if valid else "",
            "citations": valid, "disagreements": r.get("disagreements", ""), "warnings": warnings}


STATUS_AR = {
    "found": "✅ وُجد نص يعالج المسألة",
    "related_only": "⚠️ لم أجد نصاً في عين المسألة، وهذه نصوص في مسائل مجاورة",
    "not_found": "❌ لم أجد نصاً مطابقاً في الكتب المدخلة",
}


def render(res: dict) -> str:
    L = [f"## {STATUS_AR[res['status']]}", ""]
    if res.get("summary"):
        L += [res["summary"], ""]
    if res.get("disagreements"):
        L += [f"**الخلاف:** {res['disagreements']}", ""]
    for i, c in enumerate(res["citations"], 1):
        h = res["labels"][c["id"]]
        tag = "نص في عين المسألة" if c["match_type"] == "exact" else "نص في مسألة مجاورة وليس في عين سؤالك"
        L += [f"### {i}. {tag}", f"> {c['quote']}", "", f"📖 {source_label(h['chunk'])}", ""]
        if c.get("note"):
            L += [f"_{c['note']}_", ""]
        full = (f"**المتن:** {h['matn_text']}\n\n" if h["matn_text"] else "") + h["chunk"]["text"]
        L += [f"<details><summary>النص الكامل للمقطع</summary>\n\n{full}\n\n</details>", ""]
    if res["warnings"]:
        L += ["**تنبيهات آلية:** " + " | ".join(res["warnings"]), ""]
    L += ["---", "هذا استخراج من النصوص للمراجعة، وليس فتوى. للفتوى يُرجع إلى الجهة المختصة، ويُراجع النص الأصلي قبل الاعتماد."]
    return "\n".join(L)
