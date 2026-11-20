"""Rotate full training checkpoints only after their replacement is complete."""
import json
import os
import shutil
import tempfile
from pathlib import Path

def rotate_latest(current):
    assert (current / "dojo-complete.json").is_file()
    for old in current.parent.glob("global_step_*"):
        if (old != current and not old.is_symlink() and old.name[12:].isdigit()
                and (old / "dojo-complete.json").is_file()):
            shutil.rmtree(old)

def archive_best(current, target, assets, score, step):
    if not (current / "dojo-complete.json").is_file():
        raise ValueError("checkpoint incomplete")
    target.mkdir(parents=True, exist_ok=True)
    pointer = target / "best.json"
    previous = json.loads(pointer.read_text()) if pointer.is_file() else {}
    if previous and previous["success"] >= score:
        return previous
    stage = Path(tempfile.mkdtemp(prefix=".copy-", dir=target))
    destination = target / current.name
    try:
        shutil.copytree(current, stage, dirs_exist_ok=True)
        shutil.copytree(assets, stage / "base_assets", dirs_exist_ok=True)
        if not destination.exists():
            stage.rename(destination)
        entry = dict(round=step, success=score, checkpoint=str(destination))
        tmp = target / ".best.next"
        tmp.write_text(json.dumps(entry, indent=2))
        os.replace(tmp, pointer)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    if previous:
        old = Path(previous["checkpoint"])
        if (old.parent == target and old != destination and not old.is_symlink()
                and old.name.startswith("global_step_") and old.name[12:].isdigit()
                and (old / "dojo-complete.json").is_file()):
            shutil.rmtree(old)
    return entry

