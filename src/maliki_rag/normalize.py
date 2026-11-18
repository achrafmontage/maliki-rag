"""تطبيع النص العربي للفهرسة والمطابقة فقط. النص الأصلي يُحفظ للعرض."""
import re

_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
                      "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه"})
_NONWORD = re.compile(r"[^\w\s]|_")
_WS = re.compile(r"\s+")
_PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال")


def strip_diacritics(t: str) -> str:
    return _DIAC.sub("", t)


def normalize(t: str) -> str:
    t = strip_diacritics(t).translate(_MAP)
    t = _NONWORD.sub(" ", t)
    return _WS.sub(" ", t).strip()


def light_stem(tok: str) -> str:
    for p in _PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            return tok[len(p):]
    return tok


def tokenize(t: str) -> list[str]:
    """t يجب أن يكون مطبَّعاً مسبقاً."""
    return [light_stem(w) for w in t.split() if w]


_NEG = {"لا", "لم", "لن", "ليس", "غير", "ما", "بلا", "دون", "الا", "ولا", "فلا", "وليس", "ولم"}


def quote_in(quote: str, corpus: str) -> bool:
    """الاقتباس موجود حرفياً (بعد التطبيع) ولم يُبتر منه أداة نفي أو استثناء قبله مباشرة.
    البتر قبل «لا/ليس/غير/إلا» يقلب الحكم، فنرفضه."""
    q, c = normalize(quote), normalize(corpus)
    if len(q) < 8:
        return False
    start = 0
    while True:
        i = c.find(q, start)
        if i < 0:
            return False
        boundary_ok = (i == 0 or c[i - 1] == " ") and (i + len(q) == len(c) or c[i + len(q)] == " ")
        prev = c[:i].split()[-1:] if i else []
        if boundary_ok and not (prev and prev[0] in _NEG):
            return True
        start = i + 1
