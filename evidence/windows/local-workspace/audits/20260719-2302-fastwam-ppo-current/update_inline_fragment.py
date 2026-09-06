from __future__ import annotations

import json
import re
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
SOURCE = Path(
    r"C:\Users\86136\.codex\visualizations\2026\07\16\019f69a4-b29a-77f1-8ee5-76ebd7e4aa49"
) / "fastwam-ppo-step14-20260719.html"

fragment = SOURCE.read_text(encoding="utf-8")
data = json.loads((AUDIT / "analysis_detailed.json").read_text(encoding="utf-8"))
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

fragment = fragment.replace("fastwam-ppo-step8-1839", "fastwam-ppo-step14-2302")
fragment = fragment.replace("fw1839", "fw2302")
fragment = fragment.replace("global step 1 to 8", "global step 1 to 14")
fragment = re.sub(
    r"const DATA = \{.*?\};\n    const root =",
    "const DATA = " + payload + ";\n    const root =",
    fragment,
    count=1,
    flags=re.DOTALL,
)

SOURCE.write_text(fragment, encoding="utf-8")
print(SOURCE)
