import json
import os
import re


def parse_json(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("لا يوجد JSON في رد النموذج")
    return json.loads(raw[a:b + 1])


class LLM:
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, max_tokens: int = 3000) -> str:
        kw = dict(model=self.model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
        try:
            r = self.client.messages.create(temperature=0, **kw)
        except Exception as e:  # بعض النماذج لا تقبل temperature
            if "temperature" not in str(e):
                raise
            r = self.client.messages.create(**kw)
        return "".join(b.text for b in r.content if b.type == "text")

    def json(self, system: str, user: str, max_tokens: int = 3000) -> dict:
        for _ in range(2):
            try:
                return parse_json(self.complete(system, user, max_tokens))
            except (ValueError, json.JSONDecodeError):
                continue
        raise ValueError("تعذر الحصول على JSON صالح من النموذج")
