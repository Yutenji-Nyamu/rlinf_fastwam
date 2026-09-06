set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
CFG="$ROBOTWIN/env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml"
OUT="$ROBOTWIN/data/sz_collect_smoke_1ep_20260821"

test -f "$CFG"
test ! -e "$OUT"
sha256sum "$CFG"

cd "$ROBOTWIN"
timeout --signal=INT --kill-after=60s 1800s \
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  bash collect_data.sh adjust_bottle sz_collect_smoke_1ep_20260821 0 2>&1 | \
  tee "$RUN/02_collect_1ep.log"
