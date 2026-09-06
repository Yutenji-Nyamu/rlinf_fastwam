#!/usr/bin/env bash
set -u
SRC=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
printf '%s\n' '=== EVAL ENTRY ==='
sed -n '1,120p' "$SRC/evaluations/eval_embodied_agent.py"
printf '%s\n' '=== TRAIN ENTRY ==='
sed -n '1,170p' "$SRC/examples/embodiment/train_embodied_agent.py"
printf '%s\n' '=== EVAL RUNNER ==='
sed -n '1,180p' "$SRC/rlinf/runners/embodied_eval_runner.py"
printf '%s\n' '=== BASE FSDP CHECKPOINT ==='
grep -R -nE 'class.*FSDP|def save_checkpoint|def load_checkpoint|save_full_model_weights' "$SRC/rlinf/workers/actor" "$SRC/rlinf/hybrid_engines" | head -n 120 || true
grep -R -l 'class FSDP' "$SRC/rlinf" | head -n 20 || true
printf '%s\n' '=== DCP CHECKPOINT UTILS ==='
grep -R -nE 'dcp_checkpoint|StateDictOptions|save_full_model_weights|full_weights.pt' "$SRC/rlinf" | head -n 200 || true
