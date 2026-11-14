"""Rebuild the standalone candidate dashboard from its editable JSON data."""

import json
from pathlib import Path

root = Path(__file__).parent
data = json.loads((root / "data.json").read_text(encoding="utf-8"))
template = (root / "index.template.html").read_text(encoding="utf-8")
(root / "index.html").write_text(
    template.replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    ),
    encoding="utf-8",
)
