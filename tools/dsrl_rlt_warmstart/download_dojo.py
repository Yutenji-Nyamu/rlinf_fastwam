# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Download a task-specific training subset from the official RoboDojo data."""

import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path("/data/chenyiteng/datasets/robodojo-rlt-dustbin-20261106")
REPO = "RoboDojo-Benchmark/RoboDojo"
PREFIX = "data/RoboDojo_lerobot_v21_video"


def fetch(url):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(
            {"http": "http://127.0.0.1:17890", "https": "http://127.0.0.1:17890"}
        )
    )
    with opener.open(url, timeout=90) as r:
        return r.read()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    revision = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPO}"))["sha"]
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}/"
    metadata = {}
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl"):
        data = fetch(url + "meta/" + name)
        p = ROOT / "meta" / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        metadata[name] = data.decode()
    task = next(
        x
        for x in map(json.loads, metadata["tasks.jsonl"].splitlines())
        if "bottles" in x["task"] and "dustbin" in x["task"]
    )
    episodes = [
        x
        for x in map(json.loads, metadata["episodes.jsonl"].splitlines())
        if task["task"] in x["tasks"]
    ]
    assert len(episodes) >= 60
    # Stage1 trains on official training demonstrations; held-out episodes are
    # used only for reconstruction validation, not simulator evaluation seeds.
    selected = episodes[:60]
    info = json.loads(metadata["info.json"])
    paths = []
    for e in selected:
        kw = {
            "episode_index": e["episode_index"],
            "episode_chunk": e["episode_index"] // info["chunks_size"],
        }
        paths.append(info["data_path"].format(**kw))
        for key, feat in info["features"].items():
            if feat["dtype"] == "video":
                paths.append(info["video_path"].format(video_key=key, **kw))

    def one(rel):
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            for attempt in range(3):
                try:
                    data = fetch(url + rel)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            t = p.with_suffix(p.suffix + ".part")
            t.write_bytes(data)
            t.replace(p)
        return {
            "path": rel,
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(one, paths):
            records.append(record)
            if len(records) % 20 == 0:
                print(
                    json.dumps({"downloaded": len(records), "total": len(paths)}),
                    flush=True,
                )
    manifest = {
        "repository": REPO,
        "revision": revision,
        "prefix": PREFIX,
        "task": task,
        "train": selected[:50],
        "validation": selected[50:],
        "files": records,
    }
    (ROOT / "subset-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "complete": True,
                "files": len(records),
                "bytes": sum(x["bytes"] for x in records),
                "root": str(ROOT),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
