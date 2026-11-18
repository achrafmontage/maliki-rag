"""بناء فهرس BM25 وفهرس المتجهات."""
import json
import pickle
from pathlib import Path

import numpy as np

from .normalize import strip_diacritics, tokenize


def read_chunks(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build_indexes(cfg: dict, device: str | None = None):
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    chunks = read_chunks(cfg["paths"]["chunks"])
    idx = Path(cfg["paths"]["index_dir"])
    idx.mkdir(parents=True, exist_ok=True)

    bm25 = BM25Okapi([tokenize(c["norm"]) for c in chunks])
    with open(idx / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    model = SentenceTransformer(cfg["embedding"]["model"], device=device)
    model.max_seq_length = cfg["embedding"]["max_seq_length"]
    emb = model.encode([strip_diacritics(c["text"]) for c in chunks],
                       batch_size=cfg["embedding"]["batch_size"],
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(idx / "emb.npy", emb.astype(np.float16))
    (idx / "ids.json").write_text(json.dumps([c["id"] for c in chunks]), encoding="utf-8")
    print(f"تم: {len(chunks)} مقطع في {idx}")
