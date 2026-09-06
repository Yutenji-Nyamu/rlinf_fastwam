#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
V4=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
V5=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30
LOG="$V5/runtime/driver.log"

echo '=== live ==='
TZ=Asia/Shanghai date --iso-8601=seconds
pid=$(cat "$V5/runtime/wrapper.pid")
kill -0 "$pid"
echo "wrapper_alive=yes pid=$pid"
if [[ -f "$V5/runtime/exit_code.txt" ]]; then printf 'exit_code='; cat "$V5/runtime/exit_code.txt"; else echo 'exit_code=pending'; fi
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$LOG" || true
grep -aE 'Resuming training|Global Step:|Generating Rollout Epochs:' "$LOG" | tail -n 18 || true

echo '=== resolved_leaf_diff ==='
"$VENV/bin/python" - "$V4/runtime/resolved.yaml" "$V5/runtime/resolved.yaml" <<'PY'
import sys, yaml
old = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
new = yaml.safe_load(open(sys.argv[2], encoding='utf-8'))

def flat(x, prefix=''):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            key = f'{prefix}.{k}' if prefix else str(k)
            out.update(flat(v, key))
    elif isinstance(x, list):
        out[prefix] = x
    else:
        out[prefix] = x
    return out

a, b = flat(old), flat(new)
missing = object()
diffs = [k for k in sorted(set(a) | set(b)) if a.get(k, missing) != b.get(k, missing)]
allowed = {
    'runner.resume_dir', 'runner.logger.log_path',
    'env.train.video_cfg.video_base_dir', 'env.train.task_config.save_path',
    'env.eval.video_cfg.video_base_dir', 'env.eval.task_config.save_path',
    'actor.model.output_dir', 'algorithm.dvac_gradient_weighting.output_dir',
}
unexpected = sorted(set(diffs) - allowed)
print('leaf_count_v4=', len(a), 'leaf_count_v5=', len(b))
print('diff_count=', len(diffs), 'unexpected_count=', len(unexpected))
for k in diffs:
    print('DIFF', k, '::', a.get(k), '=>', b.get(k))
assert not unexpected, unexpected

checks = {
  'physical_gpus': new['cluster']['component_placement']['actor, env, rollout'],
  'max_steps': new['runner']['max_steps'],
  'resume_dir': new['runner']['resume_dir'],
  'train_envs': new['env']['train']['total_num_envs'],
  'rollout_epochs': new['env']['train']['rollout_epoch'],
  'eval_envs': new['env']['eval']['total_num_envs'],
  'eval_fixed_seeds': new['env']['eval']['use_fixed_reset_state_ids'],
  'global_batch': new['actor']['global_batch_size'],
  'micro_batch': new['actor']['micro_batch_size'],
  'update_epoch': new['algorithm']['update_epoch'],
  'group_size': new['algorithm']['group_size'],
  'dvac_mode': new['algorithm']['dvac_gradient_weighting']['mode'],
  'dvac_L': new['algorithm']['dvac_gradient_weighting']['selected_l'],
  'dvac_recent': new['algorithm']['dvac_gradient_weighting']['window_steps'],
  'dvac_warmup': new['algorithm']['dvac_gradient_weighting']['warmup_steps'],
  'dvac_z_clip': new['algorithm']['dvac_gradient_weighting']['z_clip'],
  'dvac_strength': new['algorithm']['dvac_gradient_weighting']['strength'],
  'eval_interval': new['runner']['val_check_interval'],
  'save_interval': new['runner']['save_interval'],
  'train_video': new['env']['train']['video_cfg']['save_video'],
  'eval_video': new['env']['eval']['video_cfg']['save_video'],
}
for k, v in checks.items(): print(k, '=', v)
PY

echo '=== physical_gpu4567 ==='
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4,5,6,7 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
free -h | sed -n '1,3p'
echo SZ_DVAC_GRPO_V5_PARAMETER_AND_LIVE_AUDIT_OK
