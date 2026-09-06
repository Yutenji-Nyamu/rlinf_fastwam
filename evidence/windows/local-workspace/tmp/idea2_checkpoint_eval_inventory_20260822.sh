#!/usr/bin/env bash
set -u

BASE=/root/autodl-tmp/RLinf/logs/20260715_132507-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline-env16-rollout16-g8-step0-to-100/robotwin_grpo_openpi_2gpu_env16_rollout16_g8_baseline
V1=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821/idea2_dvac_apply_formal_100step_2gpu16env_20260821
V2=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
SRC=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
SFT=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/RLinf-Pi0-RoboTwin-SFT-adjust_bottle

date --iso-8601=seconds
hostname
printf '\n=== RUNNING IDEA2 ===\n'
ps -eo pid,etimes,args | grep -E 'idea2_dvac_r_only_downweight_formal_100step|train_embodied_agent.py' | grep -v grep || true

for label in baseline v1 v2; do
  case "$label" in
    baseline) root="$BASE" ;;
    v1) root="$V1" ;;
    v2) root="$V2" ;;
  esac
  printf '\n=== %s ROOT ===\n%s\n' "$label" "$root"
  if [ ! -d "$root" ]; then
    echo MISSING
    continue
  fi
  printf '%s\n' '-- checkpoints --'
  find "$root/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\t%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort -V
  printf '%s\n' '-- checkpoint sizes --'
  du -sh "$root"/checkpoints/global_step_* 2>/dev/null | sort -V
  latest=$(find "$root/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n1)
  if [ -n "$latest" ]; then
    printf '%s\n' "-- $latest top files --"
    find "$root/checkpoints/$latest" -maxdepth 4 -type f -printf '%P\t%s\n' 2>/dev/null | sort | head -n 80
  fi
  printf '%s\n' '-- log latest global step --'
  grep -aE 'Global Step: ' "$root/metrics.log" 2>/dev/null | tail -n 2 || true
done

printf '\n=== SFT CANDIDATES ===\n'
for p in \
  "$SFT" \
  /root/autodl-tmp/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
  /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/RLinf-Pi0-RoboTwin-SFT-adjust_bottle; do
  if [ -e "$p" ]; then
    printf 'FOUND\t%s\t' "$p"
    if [ -L "$p" ]; then readlink -f "$p"; else echo regular; fi
    du -shL "$p" 2>/dev/null || true
  fi
done

printf '\n=== EVAL CONFIG CANDIDATES ===\n'
for p in \
  /root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml \
  /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml; do
  if [ -f "$p" ]; then
    echo "--- $p"
    grep -nE 'component_placement|only_eval|total_num_envs|rollout_epoch|max_episode_steps|max_steps_per_rollout_epoch|use_fixed_reset_state_ids|is_eval|seeds_path|save_video|video_base_dir|model_path|num_action_chunks|ckpt_path|resume_dir|enabled:' "$p" || true
  fi
done

printf '\n=== CHECKPOINT LOAD SOURCE ===\n'
for p in \
  "$SRC/rlinf/workers/rollout/hf/huggingface_worker.py" \
  "$SRC/rlinf/workers/actor/fsdp_actor_worker.py" \
  "$SRC/rlinf/runners/embodied_runner.py"; do
  if [ -f "$p" ]; then
    echo "--- $p"
    grep -nE 'ckpt_path|resume_dir|load_checkpoint|load_model|DCP|checkpoint' "$p" | head -n 120 || true
  fi
done

printf '\n=== SPACE ===\n'
df -h /root/autodl-tmp
