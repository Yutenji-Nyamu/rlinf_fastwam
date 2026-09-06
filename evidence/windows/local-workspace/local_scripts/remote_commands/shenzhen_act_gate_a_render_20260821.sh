set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1

mkdir -p "$RUN/configs"
cd "$ROBOTWIN"
nvidia-smi --id=0 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

timeout --signal=INT --kill-after=30s 180s \
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  python scripts/test_render.py 2>&1 | tee "$RUN/01_test_render.log"

grep -F 'Render Well' "$RUN/01_test_render.log"
if grep -F 'Render Error' "$RUN/01_test_render.log"; then
  exit 31
fi
