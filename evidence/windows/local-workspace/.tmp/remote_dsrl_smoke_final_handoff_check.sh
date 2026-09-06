set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
EXPECTED=d664bf349b63b75f41d51c8295cb0a330780d783

test "$(git -C "$REPO" branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED"
test "$(git -C "$REPO" rev-parse '@{upstream}')" = "$EXPECTED"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
if pgrep -af '[t]rain_embodied_agent.py|[m]onitor_resources.py|[r]aylet|[g]cs_server|[E]mbodiedSACFSDPPolicy|[M]ultiStepRolloutWorker|[E]nvWorker'; then
  echo "OWN_PROCESSES_REMAIN=1"
  exit 72
fi
if pgrep -x git || pgrep -af '[/]usr/lib/git-core/git-remote-https'; then
  echo "GIT_PROCESSES_REMAIN=1"
  exit 73
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "FINAL_HEAD=$EXPECTED"
echo "FINAL_HANDOFF_CHECK_OK=1"
