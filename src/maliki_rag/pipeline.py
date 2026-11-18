import json
import time
from pathlib import Path

from .config import load_config
from .generate import answer, render
from .llm import LLM
from .query_expansion import expand, load_glossary
from .retrieve import Retriever


class Pipeline:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.llm = LLM(self.cfg["llm"]["model"])
        self.ret = Retriever(self.cfg)
        self.glossary = load_glossary(self.cfg["paths"]["glossary"])

    def retrieve(self, question: str):
        exp = expand(question, self.glossary, self.ret.topics, self.llm)
        hits = self.ret.search(question, exp["queries"], exp["topics"])
        return exp, hits

    def ask(self, question: str, log: bool = True) -> dict:
        exp, hits = self.retrieve(question)
        res = answer(self.llm, question, hits)
        res["expansion"] = exp
        res["markdown"] = render(res)
        if log:
            p = Path(self.cfg["paths"]["query_log"])
            p.parent.mkdir(parents=True, exist_ok=True)
            rec = {"t": time.time(), "q": question, "status": res["status"], "expansion": exp,
                   "hits": [h["id"] for h in hits], "citations": res["citations"],
                   "warnings": res["warnings"], "reviewer_correction": None}
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return res
