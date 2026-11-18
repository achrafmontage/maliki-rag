import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or ROOT / "config.yaml").read_text(encoding="utf-8"))
    if os.environ.get("MALIKI_INDEX_DIR"):
        cfg["paths"]["index_dir"] = os.environ["MALIKI_INDEX_DIR"]
    # تحويل المسارات إلى مطلقة بالنسبة لجذر المستودع
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg
