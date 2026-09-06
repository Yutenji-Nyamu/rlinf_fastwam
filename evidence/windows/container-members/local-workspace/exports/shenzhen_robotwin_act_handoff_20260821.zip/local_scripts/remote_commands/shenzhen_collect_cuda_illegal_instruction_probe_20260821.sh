set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
PARTIAL="$ROBOTWIN/data/sz_collect_smoke_1ep_20260821"

printf '%s\n' '=== FAILURE LOG ==='
grep -n -m 5 -B 8 -A 8 'illegal instruction' "$RUN/02_collect_1ep.log" || true
printf 'illegal_instruction_count=%s\n' "$(grep -c 'illegal instruction' "$RUN/02_collect_1ep.log" || true)"

printf '%s\n' '=== PARTIAL OUTPUT ==='
if [ -e "$PARTIAL" ]; then
  du -sh "$PARTIAL"
  find "$PARTIAL" -maxdepth 6 -printf '%y\t%s\t%p\n' | sort
else
  printf 'partial=absent\n'
fi

printf '%s\n' '=== SOURCE CALL SITES ==='
grep -Rns --exclude-dir=.git --exclude='*.pyc' \
  -E 'Start Seed and Pre Motion|simulate data episode|Pre Motion Data' \
  "$ROBOTWIN" | head -n 40 || true
sed -n '145,215p' "$ROBOTWIN/scripts/collect_data.py"
sed -n '235,310p' "$ROBOTWIN/code_gen/test_gen_code.py"

printf '%s\n' '=== RUNTIME VERSIONS ==='
python3 - <<'PY'
import importlib.metadata as md

import torch
import warp

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("device_capability", torch.cuda.get_device_capability(0))
print("device_name", torch.cuda.get_device_name(0))
print("warp", getattr(warp, "__version__", md.version("warp-lang")))
for package in ("nvidia-cuda-runtime-cu12", "nvidia-cusparse-cu12", "nvidia-cudnn-cu12"):
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        print(package, "not-installed")
PY
python3 -m pip show nvidia-cuda-runtime-cu12 nvidia-cusparse-cu12 warp-lang | \
  grep -E '^(Name|Version|Location):' || true

printf '%s\n' '=== CUDA CACHES ==='
for path in /home/chenyiteng/.cache/warp /home/chenyiteng/.cache/torch_extensions /home/chenyiteng/.cache/curobo; do
  if [ -e "$path" ]; then
    du -sh "$path"
    find "$path" -maxdepth 3 -type f -printf '%s\t%p\n' | sort | head -n 80
  else
    printf 'absent\t%s\n' "$path"
  fi
done

printf '%s\n' '=== GPU AFTER STOP ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
