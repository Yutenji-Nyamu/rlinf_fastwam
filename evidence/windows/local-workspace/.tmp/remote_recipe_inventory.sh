set -eu

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN="$REPO/logs/20260718_100910-robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu"
PI0=/root/autodl-tmp/RLinf
PI0RUN="$PI0/logs/20260715_132507-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline-env16-rollout16-g8-step0-to-100"

echo '=== REPO ==='
git -C "$REPO" rev-parse HEAD
git -C "$REPO" branch --show-current
git -C "$REPO" status --short
git -C "$REPO" diff --name-status

echo '=== CURRENT RUN FILES ==='
find "$RUN" -maxdepth 3 -type f -printf '%P\t%s\n' | sort

echo '=== CURRENT SOURCE CONFIG ==='
sed -n '1,280p' "$REPO/examples/embodiment/config/robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu.yaml"

echo '=== CURRENT ENV CONFIG ==='
sed -n '1,240p' "$REPO/examples/embodiment/config/env/robotwin_move_stapler_pad.yaml"

echo '=== CURRENT MODEL CONFIG ==='
sed -n '1,260p' "$REPO/examples/embodiment/config/model/fastwam_robotwin.yaml"

echo '=== RESOLVED CANDIDATES ==='
find "$RUN" -maxdepth 3 -type f \( -iname '*config*' -o -iname '*.yaml' \) -printf '%p\t%s\n' | sort

echo '=== PI0 SOURCE CONFIG CANDIDATES ==='
find "$PI0/examples/embodiment/config" -maxdepth 1 -type f -iname '*adjust_bottle*grpo*openpi*' -printf '%p\n' | sort

echo '=== PI0 RUN CONFIG FILES ==='
find "$PI0RUN" -maxdepth 3 -type f \( -iname '*config*' -o -iname '*.yaml' \) -printf '%p\t%s\n' | sort

echo '=== COMMUNITY REPO CANDIDATES ==='
find /root/autodl-tmp -maxdepth 3 -type d \( -iname '*wzii*' -o -iname '*fastwam*rlinf*' -o -iname '*rlinf*fastwam*' \) -printf '%p\n' 2>/dev/null | sort

echo '=== CRITICAL SYMBOLS ==='
grep -RnsE 'def (flow_sde_rollout|recompute_logprob|predict_action_batch|default_forward|freeze_action_only|calculate_advantage|compute_policy_loss)|filter_rewards|grpo_norm_by_std|normalize_advantages|noise_level|trainable' \
  "$REPO/rlinf/models/embodiment/fastwam" \
  "$REPO/rlinf/algorithms" \
  "$REPO/rlinf/workers" 2>/dev/null | head -n 300 || true

echo '=== ERROR SCAN COUNTS ==='
for pat in 'Traceback' 'CUDA out of memory' 'OutOfMemory' 'NCCL' 'ActorDied' 'WorkerCrashed' 'nan' 'NaN' 'inf' 'ERROR'; do
  printf '%s\t' "$pat"
  grep -aic "$pat" "$RUN/run_embodiment.log" || true
done
