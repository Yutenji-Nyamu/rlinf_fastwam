#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_pre_stage1_cleanup_20260729
PPO_SMOKE=/root/autodl-tmp/RLinf/logs/20260714_170304-robotwin_adjust_bottle_ppo_openpi_a800_2gpu_smoke-cpu-saving/robotwin_ppo_openpi_2gpu_smoke_cpu_saving/checkpoints
PPO_FORMAL=/root/autodl-tmp/RLinf/logs/20260714_181545-robotwin_adjust_bottle_ppo_openpi_a800_2gpu_baseline-env32-rollout8-step0-to-100/robotwin_ppo_openpi_2gpu_env32_rollout8_cpu_saving/checkpoints
GRPO_SMOKE=/root/autodl-tmp/RLinf/logs/20260715_113256-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_smoke-env16-rollout16-g8-offload-on/robotwin_grpo_openpi_2gpu_env16_rollout16_g8_smoke/checkpoints
GRPO_FORMAL=/root/autodl-tmp/RLinf/logs/20260715_132507-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline-env16-rollout16-g8-step0-to-100/robotwin_grpo_openpi_2gpu_env16_rollout16_g8_baseline/checkpoints
MOTUS_RUN_A=/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133/logs_single_20260602_170538/opd_checkpoints/turn_switch
MOTUS_RUN_B=/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133/logs_single_20260601_082941/opd_checkpoints/turn_switch

rlinf_delete=(
  "$PPO_SMOKE/global_step_1"
  "$PPO_FORMAL/global_step_10"
  "$GRPO_SMOKE/global_step_1"
  "$GRPO_FORMAL/global_step_10"
  "$GRPO_FORMAL/global_step_20"
  "$GRPO_FORMAL/global_step_30"
  "$GRPO_FORMAL/global_step_40"
  "$GRPO_FORMAL/global_step_50"
  "$GRPO_FORMAL/global_step_60"
  "$GRPO_FORMAL/global_step_70"
  "$GRPO_FORMAL/global_step_80"
  "$GRPO_FORMAL/global_step_90"
)
rlinf_keep=(
  "$PPO_FORMAL/global_step_20"
  "$GRPO_FORMAL/global_step_100"
)

test ! -e "$EVIDENCE_ROOT"
test "${#rlinf_delete[@]}" -eq 12
for path in "${rlinf_delete[@]}" "${rlinf_keep[@]}"; do
  test -d "$path"
  real=$(realpath -e -- "$path")
  case "$real" in
    /root/autodl-tmp/RLinf/logs/*/checkpoints/global_step_*) ;;
    *) printf 'UNSAFE_RLINF_PATH\t%s\n' "$real"; exit 20 ;;
  esac
done

mapfile -d '' -t motus_files < <(
  find "$MOTUS_RUN_A" "$MOTUS_RUN_B" -xdev -type f -name '*.pt' -print0 | sort -z
)
test "${#motus_files[@]}" -eq 51
for path in "${motus_files[@]}"; do
  real=$(realpath -e -- "$path")
  case "$real" in
    "$MOTUS_RUN_A"/*.pt|"$MOTUS_RUN_B"/*.pt) ;;
    *) printf 'UNSAFE_MOTUS_PATH\t%s\n' "$real"; exit 21 ;;
  esac
done

mkdir -p "$EVIDENCE_ROOT"
printf 'started_at\t%s\n' "$(date --iso-8601=seconds)" > "$EVIDENCE_ROOT/summary.tsv"
df -B1 /root/autodl-tmp > "$EVIDENCE_ROOT/df_before.txt"

: > "$EVIDENCE_ROOT/deleted_rlt_dcp.tsv"
for path in "${rlinf_delete[@]}"; do
  bytes=$(du -sx --block-size=1 -- "$path" | awk '{print $1}')
  mtime=$(stat -c '%y' -- "$path")
  printf '%s\t%s\t%s\n' "$bytes" "$mtime" "$path" >> "$EVIDENCE_ROOT/deleted_rlt_dcp.tsv"
done

: > "$EVIDENCE_ROOT/deleted_motus_checkpoints.tsv"
for path in "${motus_files[@]}"; do
  stat -c '%s\t%y\t%n' -- "$path" >> "$EVIDENCE_ROOT/deleted_motus_checkpoints.tsv"
done

rlinf_bytes=$(awk '{sum += $1} END {printf "%.0f", sum}' "$EVIDENCE_ROOT/deleted_rlt_dcp.tsv")
motus_bytes=$(awk '{sum += $1} END {printf "%.0f", sum}' "$EVIDENCE_ROOT/deleted_motus_checkpoints.tsv")
printf 'rlt_dcp_count\t%s\n' "${#rlinf_delete[@]}" >> "$EVIDENCE_ROOT/summary.tsv"
printf 'rlt_dcp_bytes\t%s\n' "$rlinf_bytes" >> "$EVIDENCE_ROOT/summary.tsv"
printf 'motus_checkpoint_count\t%s\n' "${#motus_files[@]}" >> "$EVIDENCE_ROOT/summary.tsv"
printf 'motus_checkpoint_bytes\t%s\n' "$motus_bytes" >> "$EVIDENCE_ROOT/summary.tsv"

rm -rf -- "${rlinf_delete[@]}"
rm -f -- "${motus_files[@]}"

for path in "${rlinf_delete[@]}"; do
  test ! -e "$path"
done
for path in "${motus_files[@]}"; do
  test ! -e "$path"
done
for path in "${rlinf_keep[@]}"; do
  test -d "$path"
done
test "$(find "$MOTUS_RUN_A" "$MOTUS_RUN_B" -xdev -type f -name '*.pt' | wc -l)" -eq 0

df -B1 /root/autodl-tmp > "$EVIDENCE_ROOT/df_after.txt"
printf 'finished_at\t%s\n' "$(date --iso-8601=seconds)" >> "$EVIDENCE_ROOT/summary.tsv"
sha256sum \
  "$EVIDENCE_ROOT/deleted_rlt_dcp.tsv" \
  "$EVIDENCE_ROOT/deleted_motus_checkpoints.tsv" \
  "$EVIDENCE_ROOT/df_before.txt" \
  "$EVIDENCE_ROOT/df_after.txt" \
  "$EVIDENCE_ROOT/summary.tsv" > "$EVIDENCE_ROOT/SHA256SUMS"

printf 'CLEANUP_SUMMARY\n'
cat "$EVIDENCE_ROOT/summary.tsv"
printf 'DISK_AFTER\n'
df -hT /root/autodl-tmp
printf 'KEPT_RLINF_ENDPOINTS\n'
du -sh -- "${rlinf_keep[@]}"
printf 'MOTUS_REMAINING_PT\n'
find "$MOTUS_RUN_A" "$MOTUS_RUN_B" -xdev -type f -name '*.pt' -print
printf 'EVIDENCE\n'
cat "$EVIDENCE_ROOT/SHA256SUMS"
