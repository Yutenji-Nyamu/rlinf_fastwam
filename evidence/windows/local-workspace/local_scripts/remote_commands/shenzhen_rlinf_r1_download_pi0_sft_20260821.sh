#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
REV=92684e50dca1a5f75adc8d332046c4cf4fa7a3d0
MODEL_ROOT=/data/chenyiteng/models/rlinf
PARTIAL="$MODEL_ROOT/.partial-RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50-20260821"
FINAL="$MODEL_ROOT/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50"
RUN=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
HF="$VENV/bin/hf"
PYTHON="$VENV/bin/python"

test -x "$HF"
test -x "$PYTHON"
test -d "$RUN"
test ! -e "$PARTIAL"
test ! -e "$FINAL"
mkdir -p "$MODEL_ROOT" "$PARTIAL"

exec > >(tee "$RUN/model_download.log") 2>&1

printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
timeout --signal=INT --kill-after=60s 7200s \
  "$HF" download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
    --revision "$REV" \
    --local-dir "$PARTIAL" \
    --max-workers 1

"$PYTHON" - "$REV" "$PARTIAL" "$RUN/model_manifest.json" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import requests

revision, root_arg, manifest_arg = sys.argv[1:]
root = Path(root_arg)
url = (
    "https://huggingface.co/api/models/"
    "RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/"
    f"revision/{revision}?blobs=true"
)
response = requests.get(url, timeout=60)
response.raise_for_status()
metadata = response.json()
assert metadata["sha"] == revision, metadata["sha"]
siblings = metadata["siblings"]
assert len(siblings) == 18, len(siblings)

records = []
total = 0
for sibling in siblings:
    name = sibling["rfilename"]
    expected_size = sibling.get("size")
    if expected_size is None and sibling.get("lfs"):
        expected_size = sibling["lfs"]["size"]
    assert expected_size is not None, sibling
    path = root / name
    assert path.is_file(), path
    actual_size = path.stat().st_size
    assert actual_size == expected_size, (name, actual_size, expected_size)
    total += actual_size
    records.append({"path": name, "size": actual_size})

assert total == 8_067_741_880, total

expected_shards = {
    "model-00001-of-00002.safetensors": (
        4_284_226_576,
        "a6b42e854e78dd59311d7d5121682b5af993778b1868f3210952494e7bab6ab1",
    ),
    "model-00002-of-00002.safetensors": (
        3_783_369_156,
        "3aea4cd15ae930c25d5ad87912f90bd61cc3f61748fee9e6e75fb01610209446",
    ),
}
for name, (expected_size, expected_sha) in expected_shards.items():
    path = root / name
    assert path.stat().st_size == expected_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == expected_sha, (name, digest.hexdigest())

norm_path = root / "physical-intelligence/robotwin/norm_stats.json"
assert norm_path.stat().st_size == 5_149
norm = json.loads(norm_path.read_text())
assert "norm_stats" in norm
assert "state" in norm["norm_stats"] and "actions" in norm["norm_stats"]

index = json.loads((root / "model.safetensors.index.json").read_text())
assert set(index["weight_map"].values()) == set(expected_shards)

manifest = {
    "repo": "RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle",
    "revision": revision,
    "remote_files": len(records),
    "payload_bytes": total,
    "files": records,
    "shard_sha256": {name: sha for name, (_, sha) in expected_shards.items()},
}
Path(manifest_arg).write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: manifest[k] for k in ("repo", "revision", "remote_files", "payload_bytes")}, indent=2))
PY

mv "$PARTIAL" "$FINAL"
test -s "$FINAL/physical-intelligence/robotwin/norm_stats.json"
printf 'end_time=%s\n' "$(date --iso-8601=seconds)"
du -sh "$FINAL"
df -h /data
printf 'model_path=%s\n' "$FINAL"
printf '%s\n' 'R1_PI0_SFT_DOWNLOAD_OK'
