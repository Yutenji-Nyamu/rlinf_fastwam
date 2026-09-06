#!/usr/bin/env bash
set -u

python=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
formal=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3
smoke=/data/chenyiteng/results/rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823

"$python" -c 'import torch, ray; print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda)); print("ray="+ray.__version__)'
printf '%s\n' '--- formal checkpoint/config fields ---'
grep -nE 'save_interval|eval_interval|save_full_model_weights|use_orig_params|min_buffer_size|warmup|update_epoch|num_train_envs|num_eval_envs|max_steps|total_num_steps|checkpoint' \
  "$formal/runtime/resolved.yaml" 2>/dev/null || true

printf '%s\n' '--- smoke resolved files and fields ---'
while IFS= read -r file; do
  printf '### %s\n' "$file"
  grep -nE 'save_interval|eval_interval|save_full_model_weights|use_orig_params|min_buffer_size|warmup|update_epoch|num_train_envs|num_eval_envs|max_steps|total_num_steps|checkpoint' "$file" || true
done < <(find "$smoke" -type f -name 'resolved.yaml' 2>/dev/null | sort)

printf '%s\n' '--- smoke checkpoint exact files ---'
find "$smoke" -type f -path '*/checkpoints/*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 120 || true

printf '%s\n' '--- smoke save log context ---'
while IFS= read -r file; do
  printf '### %s\n' "$file"
  tr '\r' '\n' < "$file" \
    | sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' \
    | grep -E 'Saving checkpoint|Global Step:|update_step|exit|checkpoint' \
    | tail -n 100 || true
done < <(find "$smoke" -type f -name 'driver.log' 2>/dev/null | sort)

