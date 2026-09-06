#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
MODEL="$WT/rlinf/models/embodiment/openpi/openpi_action_model.py"
YAML="$WT/evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml"

printf '%s\n' '=== SAMPLE_METHOD_SIGNATURE_AND_BODY ==='
LINE=$(grep -n '^    def sample_mean_var_val' "$MODEL" | cut -d: -f1)
test -n "$LINE"
START=$((LINE-5)); END=$((LINE+125))
sed -n "${START},${END}p" "$MODEL"
printf '%s\n' '=== SAMPLE_NOISE ==='
grep -n -A25 -B5 'def sample_noise' "$MODEL" || true
printf '%s\n' '=== PARITY_RESOLVED_CONFIG_SOURCE ==='
sed -n '1,300p' "$YAML"
printf '%s\n' '=== SEED_MEMBERSHIP ==='
grep -R -n '100100052' /data/chenyiteng/projects/rlinf-shenzhen /data/chenyiteng/results/rlinf-shenzhen 2>/dev/null | sed -n '1,40p' || true
