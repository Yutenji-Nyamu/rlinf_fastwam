#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
PID=$(cat "$RUN/driver.pid")

! kill -0 "$PID" 2>/dev/null
test "$(pgrep -u "$(id -u)" -x raylet 2>/dev/null | wc -l || true)" -eq 0
test "$(pgrep -u "$(id -u)" -x gcs_server 2>/dev/null | wc -l || true)" -eq 0
for gpu in 0 1 2 3; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
    printf 'physical GPU %s still has a compute process\n' "$gpu" >&2
    exit 1
  fi
done
test "$(grep -Eic 'Traceback|CUDA out of memory|illegal instruction|SIGSEGV|worker.*died|RayActorError' "$RUN/driver.log" || true)" -eq 0

"$VENV/bin/python" - "$RUN" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
episode_files = sorted(root.rglob("episode_index_env_rank*.csv"))
query_files = sorted(root.rglob("query_index_rollout_rank*.csv"))
npz_files = sorted(root.rglob("trace_rollout_rank*.npz"))

assert len(episode_files) == len(query_files) == len(npz_files) == 4
episodes = []
queries = []
for p in episode_files:
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16
    episodes.extend(rows)
for p in query_files:
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 64
    queries.extend(rows)

assert len(episodes) == 64
assert len(queries) == 256
assert len({row["episode_uid"] for row in episodes}) == 64
assert len({row["reset_id"] for row in episodes}) == 64
assert len({row["query_uid"] for row in queries}) == 256
success = sum(row["success"].strip().lower() in {"1", "true"} for row in episodes)
success_end = sum(row["success_at_end"].strip().lower() in {"1", "true"} for row in episodes)
assert success == success_end == 42

expected_t = np.asarray([1.0, 0.75, 0.5, 0.25], dtype=np.float32)
for p in npz_files:
    with np.load(p, allow_pickle=False) as data:
        assert data["x_chain"].shape == (64, 5, 50, 14)
        assert data["z_endpoint"].shape == (64, 4, 50, 14)
        assert data["final_model_action"].shape == (64, 50, 14)
        assert data["env_action"].shape == (64, 50, 14)
        assert data["robot_state"].shape == (64, 14)
        assert np.array_equal(data["timesteps"], expected_t)
        assert all(np.isfinite(data[key]).all() for key in data.files)

manifests = list(root.rglob("*.json"))
text = "\n".join(p.read_text(encoding="utf-8") for p in manifests)
assert "800baf80d6eab64169cf0e691eb04a681a093ee9" in text
assert "194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f" in text

print(f"episodes={len(episodes)} success={success} failure={len(episodes)-success}")
print(f"queries={len(queries)} unique_reset_ids={len({row['reset_id'] for row in episodes})}")
print(f"rank_npz={len(npz_files)} arrays_finite=1 schedule_exact=1")
PY

printf 'run_bytes=%s\n' "$(du -sb "$RUN" | awk '{print $1}')"
printf 'png_count=%s\n' "$(find "$RUN" -type f -name '*.png' | wc -l)"
printf 'mp4_count=%s\n' "$(find "$RUN" -type f -name '*.mp4' | wc -l)"
printf 'mp4_bytes=%s\n' "$(find "$RUN" -type f -name '*.mp4' -printf '%s\n' | awk '{n+=$1} END {print n+0}')"
printf 'driver_log_last_write=%s\n' "$(date --iso-8601=seconds -d "@$(stat -c %Y "$RUN/driver.log")")"
printf 'raylet_count=0\ngcs_server_count=0\ngpu0_3_compute_count=0\nfatal_count=0\n'
printf '%s\n' 'SZ_PI0_DVAC_FIXED64_TERMINAL_VERIFY_OK'
