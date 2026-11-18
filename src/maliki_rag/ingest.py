"""الاستيعاب والتقطيع وربط المتن بالشرح."""
import csv
import json
import re
from pathlib import Path

from .normalize import normalize

META_END = "#META#Header#End#"
RE_OPENITI_H = re.compile(r"^###\s*(\|+)\s*(.*)$")
RE_MD_H = re.compile(r"^(#{2,6})\s+(.*)$")
RE_PAGE = re.compile(r"PageV(\d+)P(\d+)")
RE_PAGE2 = re.compile(r"\[(?:ص|صفحة)\s*:?\s*(\d+)\]")
SENT_SPLIT = re.compile(r"(?<=[.؛!؟?])\s+")
D = r"[\u064B-\u0652\u0670]*"  # تشكيل اختياري
RE_QAWLUH = re.compile(rf"(?<!\w)\(?\s*ق{D}و{D}ل{D}ه{D}\s*[:：]?")


# ---------- 1) قراءة الكتاب مع الحفاظ على الهيكل ----------
def parse_book(text: str) -> list[dict]:
    """يعيد فقرات: {path, vol, page, text}"""
    if META_END in text:
        text = text.split(META_END, 1)[1]
    path: list[str] = []
    vol, page = 1, None
    out: list[dict] = []
    buf: list[str] = []
    meta = {}

    def flush():
        nonlocal buf
        if buf:
            t = " ".join(buf).strip()
            if t:
                out.append({"path": list(meta["path"]), "vol": meta["vol"],
                            "page": meta["page"], "text": t})
        buf = []

    for line in text.splitlines():
        line = line.strip()
        m = RE_PAGE.search(line)
        if m:
            vol, page = int(m.group(1)), int(m.group(2))
            line = RE_PAGE.sub("", line).strip()
        else:
            m2 = RE_PAGE2.search(line)
            if m2:
                page = int(m2.group(1))
                line = RE_PAGE2.sub("", line).strip()
        if not line:
            continue
        h = RE_OPENITI_H.match(line)
        lvl, title = None, None
        if h:
            lvl, title = len(h.group(1)), h.group(2).strip()
        else:
            h = RE_MD_H.match(line)
            if h:
                lvl, title = len(h.group(1)) - 1, h.group(2).strip()
        if lvl is not None:
            flush()
            path = path[:lvl - 1] + [title]
            continue
        if line.startswith("~~"):
            buf.append(line.lstrip("~ ").strip())
            continue
        flush()
        meta = {"path": path, "vol": vol, "page": page}
        buf.append(re.sub(r"^#\s*", "", line))
    flush()
    return out


# ---------- 2) التقطيع ----------
def pack(parts: list[str], min_chars: int, max_chars: int) -> list[str]:
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = ""
        cur = (cur + " " + p).strip()
        if len(cur) >= min_chars:
            out.append(cur)
            cur = ""
    if cur:
        if out and len(cur) < min_chars / 2 and len(out[-1]) + len(cur) <= max_chars * 1.3:
            out[-1] += " " + cur
        else:
            out.append(cur)
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def chunk_matn(paras):
    for p in paras:
        for t in pack(sentences(p["text"]), 40, 400):
            yield {**p, "text": t, "matn_ref": None}


def extract_ref(seg: str) -> str:
    """عبارة المتن المشروحة: بعد «قوله» إلى القوس المغلق أو النقطتين أو «ش»."""
    m = RE_QAWLUH.match(seg)
    rest = seg[m.end():] if m else seg
    cut = re.split(r"\)|[:：]|\sش\s", rest, maxsplit=1)[0]
    if len(cut) > 160 or len(cut) < 3:
        cut = rest[:100]
    return re.sub(r"[()\[\]]", " ", cut).strip()


def chunk_sharh(paras):
    cur_ref, cur_path = None, None
    for p in paras:
        if p["path"] != cur_path:
            cur_ref, cur_path = None, p["path"]
        starts = [m.start() for m in RE_QAWLUH.finditer(p["text"])]
        cuts = sorted(set([0] + starts))
        for a, b in zip(cuts, cuts[1:] + [len(p["text"])]):
            seg = p["text"][a:b].strip()
            if not seg:
                continue
            if a in starts:
                cur_ref = extract_ref(seg)
            for t in pack(sentences(seg), 900, 1300):
                yield {**p, "text": t, "matn_ref": cur_ref}


def chunk_asl(paras):
    for p in paras:
        for t in pack(sentences(p["text"]), 700, 1300):
            yield {**p, "text": t, "matn_ref": None}


CHUNKERS = {"matn": chunk_matn, "sharh": chunk_sharh, "hashiya": chunk_sharh, "asl": chunk_asl}


# ---------- 3) ربط الشروح بالمتن ----------
def link_to_matn(chunks: list[dict], threshold: int = 85) -> int:
    from rapidfuzz import fuzz, process
    matn = [c for c in chunks if c["kind"] == "matn"]
    if not matn:
        return 0
    norms = [c["norm"] for c in matn]
    linked = 0
    for c in chunks:
        c["matn_id"], c["topic"] = None, None
        if c["kind"] == "matn":
            c["topic"] = c["path"][0] if c["path"] else None
            continue
        ref = normalize(c.get("matn_ref") or "")
        if len(ref) < 12:
            continue
        best = process.extractOne(ref, norms, scorer=fuzz.partial_ratio, score_cutoff=threshold)
        if best:
            m = matn[best[2]]
            c["matn_id"], c["topic"] = m["id"], m["topic"]
            linked += 1
    return linked


# ---------- 4) التشغيل ----------
def build_chunks(cfg: dict) -> int:
    books = list(csv.DictReader(open(cfg["paths"]["books_csv"], encoding="utf-8")))
    chunks: list[dict] = []
    for b in books:
        f = Path(cfg["paths"]["raw_dir"]) / b["file"]
        if not f.exists():
            print(f"تخطي {b['book_id']}: الملف غير موجود {f}")
            continue
        paras = parse_book(f.read_text(encoding="utf-8"))
        n0 = len(chunks)
        for i, c in enumerate(CHUNKERS[b["kind"]](paras)):
            chunks.append({
                "id": f"{b['book_id']}:{i:06d}", "book_id": b["book_id"], "title": b["title"],
                "author": b["author"], "death_hijri": b["death_hijri"], "kind": b["kind"],
                "path": c["path"], "vol": c["vol"], "page": c["page"], "text": c["text"],
                "norm": normalize(c["text"]), "matn_ref": c.get("matn_ref"),
            })
        print(f"{b['book_id']}: {len(chunks) - n0} مقطع")
    n = link_to_matn(chunks, cfg["chunking"]["link_threshold"])
    print(f"رُبط {n} مقطعاً من الشروح بالمتن")
    out = Path(cfg["paths"]["chunks"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(chunks)
