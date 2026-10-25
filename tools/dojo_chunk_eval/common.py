# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Small CPU-only helpers for the opt-in Dojo recording run."""

import json
import os
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temp.replace(path)


def serial(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(serial(value), allow_nan=False) + "\n")


def episode_dir(meta):
    return (
        Path(os.environ["DOJO_U_OUTPUT"]) / "episodes" / f"layout_{meta['layout']:05d}"
    )


def progress(phase, **kwargs):
    save(
        Path(os.environ["DOJO_U_OUTPUT"]) / "progress.json",
        dict(time=time.time(), phase=phase, **kwargs),
    )
