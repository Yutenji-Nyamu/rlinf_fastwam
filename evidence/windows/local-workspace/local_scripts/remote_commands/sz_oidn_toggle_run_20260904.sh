#!/usr/bin/env bash
set -euo pipefail
trial=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904
rl=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
fw=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
out=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/oidn-toggle-20260904
source_config=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2/runtime/resolved.yaml
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$trial:$rl:$fw/src" FASTWAM_CONFIG_DIR="$fw/configs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export ROBOTWIN_PATH="$trial" ROBOT_PLATFORM=ALOHA
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
test -d "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.1-T2V-1.3B"
test -d "$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
cd "$fw"
for denoiser in oidn none; do
  test -f "$out/$denoiser/resolved.yaml"
  test ! -e "$out/$denoiser/trial.log"
  test -z "$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits)"
  date -Is > "$out/$denoiser/started_at.txt"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
  set +e
  timeout --signal=TERM --kill-after=30s 1200s /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python "$trial/oidn_toggle_trial.py" --source-config "$source_config" --output "$out" --denoiser "$denoiser" 2>&1 | tee "$out/$denoiser/trial.log"
  status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" > "$out/$denoiser/exit_code.txt"
  date -Is > "$out/$denoiser/finished_at.txt"
  test "$status" -eq 0
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
