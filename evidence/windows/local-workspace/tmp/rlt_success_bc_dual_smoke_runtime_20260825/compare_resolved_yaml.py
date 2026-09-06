from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, child))
        return out
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


left_path, right_path = map(Path, sys.argv[1:3])
left = flatten(yaml.safe_load(left_path.read_text(encoding="utf-8")))
right = flatten(yaml.safe_load(right_path.read_text(encoding="utf-8")))
diffs = []
for key in sorted(set(left) | set(right)):
    if left.get(key) != right.get(key):
        diffs.append({"path": key, "historical": left.get(key), "single": right.get(key)})
print(json.dumps(diffs, ensure_ascii=False, indent=2))
