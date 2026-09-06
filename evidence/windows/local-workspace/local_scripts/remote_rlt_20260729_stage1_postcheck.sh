set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1
RUN_ROOT=/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1
S1A_NAME=robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1
S1B_NAME=robotwin_adjust_bottle_rlt_stage1_s1b_formal_batch_1step_v1
CHECKPOINT_ROOT="$RUN_ROOT/s1a/$S1A_NAME/checkpoints/global_step_2"
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python

test "$(cat "$EVIDENCE_ROOT/s1a_runtime/exit_code.txt")" = 0
test "$(cat "$EVIDENCE_ROOT/s1a_reload_runtime/exit_code.txt")" = 0
test "$(cat "$EVIDENCE_ROOT/s1b_runtime/exit_code.txt")" = 0
test -f "$CHECKPOINT_ROOT/actor/dcp_checkpoint/.metadata"
test -f "$CHECKPOINT_ROOT/actor/model_state_dict/full_weights.pt"
test "$(find "$RUN_ROOT/s1b/$S1B_NAME" -path '*/checkpoints/*' -type f 2>/dev/null | wc -l)" = 0

"$PYTHON_BIN" -B - "$EVIDENCE_ROOT" "$RUN_ROOT" <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

evidence_root = Path(sys.argv[1])
run_root = Path(sys.argv[2])


def summarize_resources(runtime_name: str) -> dict:
    runtime = evidence_root / runtime_name
    with (runtime / "resources.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{runtime_name}: empty resource log")

    def values(key: str) -> list[float]:
        result = []
        for row in rows:
            value = float(row[key])
            if value >= 0 and math.isfinite(value):
                result.append(value)
        return result

    started = datetime.fromisoformat((runtime / "started_at.txt").read_text().strip())
    finished = datetime.fromisoformat((runtime / "finished_at.txt").read_text().strip())
    available = values("host_available_bytes")
    cgroup = values("cgroup_current_bytes")
    return {
        "samples": len(rows),
        "wall_seconds": (finished - started).total_seconds(),
        "gpu0_peak_mib": max(values("gpu0_used_mib")),
        "gpu1_peak_mib": max(values("gpu1_used_mib")),
        "gpu0_peak_util_pct": max(values("gpu0_util_pct")),
        "gpu1_peak_util_pct": max(values("gpu1_util_pct")),
        "matched_rss_peak_gib": max(values("matched_rss_kib")) / 1024 / 1024,
        "host_available_min_gib": min(available) / 1024**3,
        "host_available_drop_from_first_gib": (available[0] - min(available)) / 1024**3,
        "cgroup_current_peak_gib": max(cgroup) / 1024**3,
        "compute_process_count_peak": max(values("compute_process_count")),
    }


ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
metric_pattern = re.compile(
    r"train/grad_norm=(?P<grad>[0-9.eE+-]+).*?"
    r"train/learning_rate=(?P<lr>[0-9.eE+-]+).*?"
    r"train/loss=(?P<loss>[0-9.eE+-]+).*?"
    r"train/rlt_loss=(?P<rlt>[0-9.eE+-]+).*?"
    r"train/vla_loss=(?P<vla>[0-9.eE+-]+)"
)


def metrics(runtime_name: str) -> list[dict]:
    text = ansi.sub("", (evidence_root / runtime_name / "driver.log").read_text(errors="replace"))
    found = []
    for match in metric_pattern.finditer(text):
        item = {key: float(value) for key, value in match.groupdict().items()}
        if not all(math.isfinite(value) for value in item.values()):
            raise RuntimeError(f"{runtime_name}: non-finite metric {item}")
        if not found or item != found[-1]:
            found.append(item)
    return found


logs = {
    name: (evidence_root / name / "driver.log").read_text(errors="replace")
    for name in ("s1a_runtime", "s1a_reload_runtime", "s1b_runtime")
}
fatal_patterns = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "RayTaskError",
)
for name, text in logs.items():
    hits = [pattern for pattern in fatal_patterns if pattern in text]
    if hits:
        raise RuntimeError(f"{name}: fatal log markers {hits}")

reload_markers = [
    line
    for line in logs["s1a_reload_runtime"].splitlines()
    if '"resume_dir":' in line or "Global Step:" in line
]
expected_resume = (
    "/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/"
    "s1a/robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1/"
    "checkpoints/global_step_2"
)
if expected_resume not in logs["s1a_reload_runtime"]:
    raise RuntimeError("reload log lacks exact resume_dir")
if not any("2/2" in line for line in reload_markers):
    raise RuntimeError("reload log lacks 2/2 no-op progress marker")

checkpoint = (
    run_root
    / "s1a"
    / "robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1"
    / "checkpoints"
    / "global_step_2"
)
checkpoint_files = {
    str(path.relative_to(checkpoint)): path.stat().st_size
    for path in sorted(checkpoint.rglob("*"))
    if path.is_file()
}

summary = {
    "resource": {
        "s1a": summarize_resources("s1a_runtime"),
        "s1a_reload": summarize_resources("s1a_reload_runtime"),
        "s1b": summarize_resources("s1b_runtime"),
    },
    "metrics": {
        "s1a": metrics("s1a_runtime"),
        "s1b": metrics("s1b_runtime"),
    },
    "reload_marker_count": len(reload_markers),
    "checkpoint_files": checkpoint_files,
    "checkpoint_total_gib": sum(checkpoint_files.values()) / 1024**3,
    "s1b_checkpoint_file_count": 0,
}
(evidence_root / "stage1_postcheck.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

printf '%s\n' 'FINAL_GPU'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
printf '%s\n' 'FINAL_RAM'
free -h
printf '%s\n' 'FINAL_DISK'
df -hT /root/autodl-tmp
printf '%s\n' 'RELEVANT_PROCESSES'
pgrep -af 'train_vla_sft|ray::|raylet|gcs_server' || true
printf '%s\n' 'RLT_GIT'
git -C "$RLT_ROOT" branch --show-current
git -C "$RLT_ROOT" rev-parse HEAD
git -C "$RLT_ROOT" status --short
git -C "$RLT_ROOT" rev-list --left-right --count HEAD...@{upstream}
printf '%s\n' 'EVIDENCE_HASHES'
sha256sum \
  "$EVIDENCE_ROOT/s1a_resolved.yaml" \
  "$EVIDENCE_ROOT/s1b_resolved.yaml" \
  "$EVIDENCE_ROOT/stage1_postcheck.json" \
  "$CHECKPOINT_ROOT/actor/dcp_checkpoint/.metadata"
printf '%s\n' 'CHECKPOINT_USAGE'
du -sh "$CHECKPOINT_ROOT"
date -Is
