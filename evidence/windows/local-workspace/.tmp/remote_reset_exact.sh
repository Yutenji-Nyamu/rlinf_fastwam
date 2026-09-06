set -eu
cd /root/autodl-tmp/RLinf_fastwam_rlinf
sed -n '1,140p' rlinf/envs/robotwin/robotwin_env.py
sed -n '400,515p' rlinf/envs/robotwin/robotwin_env.py
sed -n '680,760p' rlinf/workers/env/env_worker.py
sed -n '70,120p' examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline.yaml 2>/dev/null || true
sed -n '70,120p' examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml 2>/dev/null || true
