"""بحث هجين: BM25 + متجهات + دمج RRF + إعادة ترتيب + سياق."""
import json
import pickle
from pathlib import Path

import numpy as np

from .index import read_chunks
from .normalize import normalize, strip_diacritics, tokenize


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.chunks = read_chunks(cfg["paths"]["chunks"])
        self.by_id = {c["id"]: c for c in self.chunks}
        self.pos = {c["id"]: i for i, c in enumerate(self.chunks)}
        idx = Path(cfg["paths"]["index_dir"])
        self.bm25 = pickle.load(open(idx / "bm25.pkl", "rb"))
        self.emb = np.load(idx / "emb.npy").astype(np.float32)
        assert json.loads((idx / "ids.json").read_text()) == [c["id"] for c in self.chunks], \
            "الفهرس لا يطابق المقاطع: أعد بناء الفهرس"
        self.topics = sorted({c["topic"] for c in self.chunks if c["kind"] == "matn" and c["topic"]})
        self.topic_arr = np.array([c.get("topic") or "" for c in self.chunks])
        self._embedder = self._reranker = None

    # ---- تحميل كسول ----
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.cfg["embedding"]["model"])
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.cfg["reranker"]["model"], max_length=512)
        return self._reranker

    # ---- قنوات البحث ----
    def _rank(self, scores: np.ndarray, mask, k: int) -> list[int]:
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if np.isfinite(scores[i])]

    def _channels(self, query: str, mask, k: int) -> list[list[int]]:
        bm = self.bm25.get_scores(tokenize(normalize(query)))
        q = self.embedder.encode([strip_diacritics(query)], normalize_embeddings=True)[0]
        return [self._rank(np.asarray(bm), mask, k), self._rank(self.emb @ q, mask, k)]

    def search(self, question: str, queries: list[str], topics: list[str] | None = None):
        r = self.cfg["retrieval"]
        masks = [None]
        if topics:
            masks.insert(0, np.isin(self.topic_arr, topics))
        fused: dict[int, float] = {}
        for q in dict.fromkeys([question] + list(queries)):
            for m in masks:
                for ranking in self._channels(q, m, r["k_each"]):
                    for rank, i in enumerate(ranking):
                        fused[i] = fused.get(i, 0.0) + 1.0 / (r["rrf_k"] + rank + 1)
        pool = sorted(fused, key=fused.get, reverse=True)[: r["rerank_pool"]]
        scores = {i: fused[i] for i in pool}
        if self.cfg["reranker"]["enabled"] and pool:
            pairs = [(question, self._rerank_text(self.chunks[i])) for i in pool]
            rs = self.reranker.predict(pairs)
            scores = {i: float(s) for i, s in zip(pool, rs)}
            pool = sorted(pool, key=scores.get, reverse=True)
        return [self._hit(i, scores[i]) for i in pool[: r["final_k"]]]

    def _rerank_text(self, c: dict) -> str:
        m = self.by_id.get(c.get("matn_id") or "")
        return (f"المتن: {m['text']}\nالشرح: " if m else "") + c["text"][:1200]

    def _hit(self, i: int, score: float) -> dict:
        c = self.chunks[i]
        m = self.by_id.get(c.get("matn_id") or "")
        prev = self.chunks[i - 1] if i > 0 and self.chunks[i - 1]["book_id"] == c["book_id"] else None
        nxt = self.chunks[i + 1] if i + 1 < len(self.chunks) and self.chunks[i + 1]["book_id"] == c["book_id"] else None
        return {"id": c["id"], "score": score, "chunk": c,
                "matn_text": m["text"] if m else "",
                "context_prev": prev["text"] if prev else "",
                "context_next": nxt["text"] if nxt else ""}
