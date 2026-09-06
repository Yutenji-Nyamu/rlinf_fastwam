#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs
CONTROL="$ROOT/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1"
DVAC="$ROOT/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1"
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
RAY_ADDRESS=172.17.0.1:6389

date --iso-8601=seconds
for item in "control:$CONTROL" "dvac:$DVAC"; do
  label=${item%%:*}
  run=${item#*:}
  echo "=== $label process ==="
  pid=$(<"$run/runtime/wrapper.pid")
  printf 'run=%s\npid=%s alive=%s\n' "$run" "$pid" "$([[ -d /proc/$pid ]] && echo 1 || echo 0)"
  for marker in started_at.txt finished_at.txt exit_code.txt; do
    if [[ -f "$run/runtime/$marker" ]]; then
      printf '%s=' "$marker"
      tr '\n' ' ' < "$run/runtime/$marker"
      echo
    fi
  done
  stat -c 'driver_bytes=%s driver_mtime=%y' "$run/runtime/driver.log"
  "$PY" - "$run/runtime/driver.log" <<'PY'
import pathlib, re, sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "\n")
steps = [(int(a), int(b), m.start()) for m in re.finditer(r"Global Step:\s*(\d+)\s*/\s*(\d+)", text) for a, b in [m.groups()]]
print(f"global_step_entries={len(steps)} latest_complete={steps[-1][0] if steps else 0}/{steps[-1][1] if steps else 100}")
rollouts = re.findall(r"Generating Rollout Epochs:\s*[^\n]+", text)
print("latest_rollout=" + (rollouts[-1].strip() if rollouts else "none"))
if steps:
    start = steps[-1][2]
    end = steps[-2][2] if len(steps) > 1 else max(0, start - 1)
    block = text[start : start + 9000]
    interesting = []
    for line in block.splitlines():
        low = line.lower()
        if any(key in low for key in (
            "global step:", "success", "approx_kl", "clip_fraction", "grad_norm",
            "dvac_", "weight_", "filtered", "group", "evaluation", "time/", "time cost",
        )):
            cleaned = line.strip(" │├─┤")
            if cleaned and cleaned not in interesting:
                interesting.append(cleaned)
    print("--- latest step interesting lines ---")
    print("\n".join(interesting[:45]))
fatal_patterns = {
    "traceback": r"Traceback \(most recent call last\)",
    "cuda_oom": r"CUDA out of memory|torch\.OutOfMemoryError",
    "ray_oom": r"OutOfMemoryError: Task was killed due to the node running low on memory|node was running low on memory",
    "worker_died": r"worker died|ActorDiedError|WorkerCrashedError",
    "nonfinite": r"\bnan\b|\binf\b",
    "vulkan": r"ErrorInitializationFailed|vk::",
}
print("fatal_counts=" + ",".join(f"{k}:{len(re.findall(p, text, flags=re.I))}" for k,p in fatal_patterns.items()))
print("--- log tail ---")
print("\n".join(text.splitlines()[-35:]))
PY
  echo "=== $label checkpoints ==="
  find "$run" -type d -name 'global_step_*' -print0 | while IFS= read -r -d '' ckpt; do
    files=$(find "$ckpt" -type f | wc -l)
    bytes=$(du -sb "$ckpt" | awk '{print $1}')
    printf '%s files=%s bytes=%s\n' "$ckpt" "$files" "$bytes"
  done | sort -V | tail -n 12
done

echo '=== namespaces ==='
RAY_ADDRESS="$RAY_ADDRESS" "$PY" - <<'PY'
import collections, ray
ray.init(address="172.17.0.1:6389", namespace="codex_pi05_live_metrics_readonly", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
print(dict(sorted(collections.Counter(row.get("namespace", "") for row in rows).items())))
ray.shutdown()
PY

echo '=== gpu ==='
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '=== host memory and pressure ==='
awk '/^MemTotal:|^MemAvailable:|^SwapTotal:|^SwapFree:/' /proc/meminfo
cat /proc/pressure/memory
echo '=== filesystems ==='
df -h / /home /data
