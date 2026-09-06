set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN_ROOT="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1"
FAILED="$RUN_ROOT/fixed_latent_parity"
PASSED="$RUN_ROOT/fixed_latent_parity_v2"
OUT="$PASSED/validation.txt"

test "$(git -C "$REPO" rev-parse HEAD)" = 3c7d35d716cfd6964b15249e548a0aab99d4cb27
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -s "$FAILED/driver.log"
test -s "$PASSED/driver.log"
grep -F 'assert stored_latent.shape == (1, 50 * 32)' "$FAILED/driver.log"
grep -F 'FIXED_OBSERVATION_LATENT_PARITY_OK=1' "$PASSED/driver.log"
if grep -F 'Traceback (most recent call last)' "$PASSED/driver.log"; then
  echo "PARITY_V2_TRACEBACK=1"
  exit 61
fi
test ! -e "$OUT"

export PASSED
/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY' | tee "$OUT"
import csv
import json
import os
from pathlib import Path

passed = Path(os.environ["PASSED"])
lines = passed.joinpath("driver.log").read_text(encoding="utf-8").splitlines()
records = [line for line in lines if line.startswith("PARITY_RESULT=")]
assert len(records) == 1
result = json.loads(records[0].split("=", 1)[1])
assert result["all_exact"] is True
assert result["input_contract"]["states"] == [1, 14]
assert result["input_contract"]["main_images"] == [1, 480, 640, 3]
assert result["input_contract"]["wrist_images"] == [1, 2, 480, 640, 3]
assert result["latent_contract"]["shape"] == [1, 50, 32]
assert result["latent_contract"]["repeated_horizon"] is True
assert result["forward_input_latent_shape"] == [1, 50, 32]
by_name = {item["name"]: item for item in result["comparisons"]}
assert by_name["env_actions"]["shape"] == [1, 20, 14]
assert by_name["model_actions"]["shape"] == [1, 1600]
assert by_name["denoise_chains"]["shape"] == [1, 5, 50, 32]
assert all(item["exact"] for item in by_name.values())
assert all(item["max_abs_delta"] == 0.0 for item in by_name.values())

with passed.joinpath("resource_monitor", "resources.csv").open(
    newline="", encoding="utf-8"
) as handle:
    rows = list(csv.DictReader(handle))
assert rows
assert all(int(row["cgroup_oom"]) == 0 for row in rows)
assert all(int(row["cgroup_oom_kill"]) == 0 for row in rows)
peak_gpu0 = max(float(row["gpu0_memory_mb"]) for row in rows)
peak_ram = max(float(row["cgroup_ram_mb"]) for row in rows)
print(
    "FIXED_PARITY_VALIDATION",
    json.dumps(
        {
            "all_exact": True,
            "comparison_count": len(by_name),
            "peak_gpu0_mb": peak_gpu0,
            "peak_cgroup_ram_mb": peak_ram,
            "oom": 0,
            "oom_kill": 0,
        },
        sort_keys=True,
    ),
)
print("FIXED_PARITY_ARTIFACT_VALIDATION_OK=1")
PY

if pgrep -x raylet || pgrep -x gcs_server; then
  echo "RAY_REMAINS=1"
  exit 72
fi
if pgrep -af '^/root/autodl-tmp/RLinf/.venv/bin/python .*fixed_latent_parity|^/root/autodl-tmp/RLinf/.venv/bin/python .*monitor_resources.py|^ray::'; then
  echo "PROBE_PROCESSES_REMAIN=1"
  exit 73
fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "FIXED_PARITY_FINAL_STATE_OK=1"
