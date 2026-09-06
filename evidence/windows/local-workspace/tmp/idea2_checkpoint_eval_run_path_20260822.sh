#!/usr/bin/env bash
set -u
SRC=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
printf '%s\n' '=== RUNNER 420-500 ==='
sed -n '420,500p' "$SRC/rlinf/runners/embodied_runner.py"
printf '%s\n' '=== ACTOR CLASS SAVE/LOAD SYMBOLS ==='
grep -nE 'def (save|load)_checkpoint|save_full_model_weights|full_weights' "$SRC/rlinf/workers/actor/fsdp_actor_worker.py" || true
printf '%s\n' '=== BASE ACTOR CHECKPOINT SYMBOLS ==='
grep -R -nE 'def (save|load)_checkpoint|save_full_model_weights|full_weights' "$SRC/rlinf/workers" | head -n 100 || true
printf '%s\n' '=== ENTRYPOINT RUNNER CONSTRUCTION ==='
grep -R -nE 'EmbodiedRunner\(|only_eval|init_workers\(|runner.run' "$SRC/examples/embodiment/train_embodied_agent.py" "$SRC/evaluations/eval_embodied_agent.py" "$SRC/rlinf" | head -n 160 || true
printf '%s\n' '=== DISK ACTUAL BY RUN ROOT ==='
du -sh \
 /root/autodl-tmp/RLinf/logs/20260715_132507-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline-env16-rollout16-g8-step0-to-100/robotwin_grpo_openpi_2gpu_env16_rollout16_g8_baseline \
 /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821/idea2_dvac_apply_formal_100step_2gpu16env_20260821 \
 /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
