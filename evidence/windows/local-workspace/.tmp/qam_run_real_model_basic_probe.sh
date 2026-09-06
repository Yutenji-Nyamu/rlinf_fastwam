set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv

cd "$repo"
test "$(git rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
test -z "$(
  ps -eo args |
    grep -E 'train_embodied_agent|ray::|raylet|gcs_server' |
    grep -v grep || true
)"
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
"$venv/bin/python" -u /root/autodl-tmp/qam_real_model_basic_probe.py
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
