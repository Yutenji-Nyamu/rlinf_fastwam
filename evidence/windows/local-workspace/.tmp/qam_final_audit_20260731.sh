set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
run_root=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1
runtime_root=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime
source_cfg="$repo/examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml"
model_dir=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
norm_file="$model_dir/physical-intelligence/robotwin/norm_stats.json"
monitor=/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh
launcher=/root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh
source_resolved=/root/autodl-tmp/qam_source_resolved_20260731_v1.yaml
smoke_resolved=/root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml
resolved_diff=/root/autodl-tmp/qam_source_to_qonly_smoke_20260731_v1.diff

date '+TIME=%F %T %Z'
printf 'HOST=%s\nPWD=%s\nUID=%s\n' "$(hostname)" "$PWD" "$(id -u)"

git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" rev-parse HEAD^{tree}
git -C "$repo" status --porcelain=v1
git -C "$repo" log -1 --format='COMMIT=%H%nSUBJECT=%s'

printf 'PROCESSES_BEGIN\n'
pgrep -x raylet || true
pgrep -x gcs_server || true
pgrep -af '[t]rain_embodied_agent.py|[t]orch.distributed.run' || true
printf 'PROCESSES_END\n'

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
df -h /root/autodl-tmp
df -i /root/autodl-tmp

if test -e "$run_root"; then
  printf 'RUN_ROOT=EXISTS\n'
else
  printf 'RUN_ROOT=ABSENT\n'
fi
if test -e "$runtime_root"; then
  printf 'RUNTIME_ROOT=EXISTS\n'
else
  printf 'RUNTIME_ROOT=ABSENT\n'
fi

sha256sum \
  "$source_cfg" \
  "$source_resolved" \
  "$smoke_resolved" \
  "$resolved_diff" \
  "$norm_file" \
  "$model_dir/model.safetensors.index.json" \
  "$monitor" \
  "$launcher"
bash -n "$launcher"
du -sh "$repo" /root/autodl-tmp/RLinf/.venv /root/autodl-tmp/venvs/qam-oracle-2726d767

/root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
import importlib.metadata as md
import sys

print("PYTHON=" + sys.version.replace("\n", " "))
for name in ("torch", "ray", "hydra-core", "omegaconf"):
    try:
        print(f"PKG {name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"PKG {name}=MISSING")
PY
