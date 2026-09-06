set -euo pipefail

RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1

test -s "$RUN_ROOT/post_smoke_live_audit.txt"
if pgrep -af '[t]rain_embodied_agent.py|[m]onitor_resources.py|[r]aylet|[g]cs_server|[E]mbodiedSACFSDPPolicy|[M]ultiStepRolloutWorker|[E]nvWorker'; then
  echo "TARGET_PROCESSES_REMAIN=1"
  exit 71
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "TARGET_PROCESSES_REMAIN=0"
