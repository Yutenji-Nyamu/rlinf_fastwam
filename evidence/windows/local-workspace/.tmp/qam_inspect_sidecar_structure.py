from __future__ import annotations

import pathlib
import sys

import torch


def walk(value: object, prefix: str = "", depth: int = 0) -> None:
    if depth > 3:
        return
    if isinstance(value, dict):
        print(f"{prefix or '<root>'}: dict[{len(value)}]")
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            walk(child, child_prefix, depth + 1)
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}: {type(value).__name__}[{len(value)}]")
        for index, child in enumerate(value[:8]):
            walk(child, f"{prefix}[{index}]", depth + 1)
    elif torch.is_tensor(value):
        finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
        print(
            f"{prefix}: tensor shape={tuple(value.shape)} "
            f"dtype={value.dtype} finite={finite}"
        )
    elif isinstance(value, (str, int, float, bool, type(None))):
        print(f"{prefix}: {value!r}")
    else:
        print(f"{prefix}: {type(value).__name__}")


path = pathlib.Path(sys.argv[1])
payload = torch.load(path, map_location="cpu", weights_only=False)
walk(payload)

