set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
MAIN_ROOT=/root/autodl-tmp/RLinf
DSRL_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_pre_smoke_20260729

date -Is
hostname
printf '%s\n' '--- GPU ---'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
printf '%s\n' '--- RAM ---'
free -h
printf '%s\n' '--- DISK ---'
df -hT /root/autodl-tmp
printf '%s\n' '--- RELEVANT PROCESSES ---'
pgrep -af \
  'ray::|raylet|gcs_server|train_embodied_agent|train_vla_sft|RoboTwin|robotwin|probe_robotwin_rlt_prefix_contract' \
  || true
printf '%s\n' '--- WORKTREES ---'
git -C "$MAIN_ROOT" worktree list --porcelain
printf '%s\n' '--- MAIN STATUS ---'
git -C "$MAIN_ROOT" status --short
printf '%s\n' '--- DSRL STATUS ---'
git -C "$DSRL_ROOT" status --short
printf '%s\n' '--- RLT STATUS ---'
git -C "$RLT_ROOT" status --short
printf '%s\n' '--- RLT HEAD ---'
git -C "$RLT_ROOT" branch --show-current
git -C "$RLT_ROOT" rev-parse HEAD
printf '%s\n' '--- EVIDENCE ---'
find "$EVIDENCE_ROOT" -maxdepth 1 -type f \
  -printf '%f\t%s bytes\n' | sort
