from maliki_rag.generate import answer
from maliki_rag.ingest import chunk_matn, chunk_sharh, link_to_matn, pack, parse_book
from maliki_rag.normalize import normalize, quote_in, tokenize
from maliki_rag.query_expansion import glossary_hits


def test_normalize():
    assert normalize("الْمِنْطَقَةُ، وَالإِحْرَامُ") == "المنطقه والاحرام"
    assert normalize("أُنْثَى") == normalize("انثى")


def test_tokenize_prefix():
    assert tokenize("المنطقه والمنطقه") == ["منطقه", "منطقه"]


def test_quote_verification():
    corpus = "قال المصنف: لا يجوز  للمحرم لبس المخيط، إلا لضرورة."
    assert quote_in("لا يجوز للمحرم لبس المَخيط", corpus)
    assert not quote_in("يجوز للمحرم لبس المخيط", corpus)
    assert not quote_in("قصير", "قصير")  # أقل من 8 أحرف مرفوض


SAMPLE = """#META#Header#End#
### | كتاب الأول
### || باب الثاني
# جملة أولى في المتن طويلة بما يكفي للاختبار هنا. جملة ثانية أيضا طويلة بما يكفي للاختبار. PageV01P005
# فقرة أخرى طويلة بما يكفي لتكون مقطعا مستقلا في الاختبار الحالي.
~~ وتكملة للفقرة نفسها.
"""


def test_parse_structure():
    ps = parse_book(SAMPLE)
    assert ps[0]["path"] == ["كتاب الأول", "باب الثاني"]
    assert ps[0]["page"] == 5
    assert "وتكملة" in ps[1]["text"]


def test_sharh_split_and_link():
    paras = [{"path": ["ك"], "vol": 1, "page": 1,
              "text": "(قوله: جملة المتن الأولى هنا) شرحها كذا وكذا. (قوله: عبارة أخرى من المتن) شرح آخر."}]
    ch = list(chunk_sharh(paras))
    assert len(ch) == 2 and ch[0]["matn_ref"].startswith("جمله") is False
    assert "جملة المتن الأولى" in ch[0]["matn_ref"]
    matn = [{"id": "m:1", "kind": "matn", "path": ["كتاب الحج"], "norm": normalize("وهذه جملة المتن الأولى هنا بتمامها")}]
    sh = [{"id": "s:1", "kind": "sharh", "path": ["x"], "norm": "x", "matn_ref": ch[0]["matn_ref"]}]
    assert link_to_matn(matn + sh, 85) == 1
    assert sh[0]["topic"] == "كتاب الحج"


def test_glossary():
    g = [{"colloquial": "حزام", "key": normalize("حزام"), "terms": ["المنطقة"], "topic_hint": "الحج"}]
    assert glossary_hits("حكم وضع حزام للمحرم", g)


class FakeLLM:
    def __init__(self):
        self.n = 0

    def json(self, system, user, max_tokens=0):
        self.n += 1
        if self.n == 1:  # القاضي
            return {"judgments": [{"id": "P1", "match": "related", "issue_in_passage": "نفقة"}]}
        return {"summary": "[P1]", "citations": [
            {"id": "P1", "quote": "نص مختلق غير موجود في المقطع أبدا", "note": ""},
            {"id": "P1", "quote": "يشد المحرم الهميان على وسطه", "note": ""}]}


def test_answer_rejects_fabricated_and_keeps_judge_label():
    chunk = {"id": "a:1", "book_id": "a", "title": "ت", "author": "م", "death_hijri": 1, "path": [],
             "vol": 1, "page": 2, "text": "يشد المحرم الهميان على وسطه لنفقته"}
    hit = {"id": "a:1", "chunk": chunk, "matn_text": "", "context_prev": "", "context_next": ""}
    r = answer(FakeLLM(), "سؤال", [hit])
    assert r["status"] == "related_only"          # لا يُرقّى إلى found
    assert len(r["citations"]) == 1 and len(r["warnings"]) == 1
