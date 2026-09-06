#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
MODEL=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys23-localshard-v4/resolved.yaml
printf 'HEAD='; git -C "$WT" rev-parse HEAD
printf 'STATUS='; test -z "$(git -C "$WT" status --porcelain)" && echo clean || echo dirty
python3 - "$MODEL/conversion_manifest.json" "$PACKET" <<'PY'
import json, sys, yaml
m=json.load(open(sys.argv[1]))
keys=['format','source','source_revision','source_keys','target_keys','missing_keys','unexpected_keys','shape_mismatches','dtype_mismatch_count','dtype_note','state_contract_digest','norm_stats']
print('MANIFEST='+json.dumps({k:m.get(k) for k in keys}, sort_keys=True))
c=yaml.safe_load(open(sys.argv[2]))
o={
 'placement':c['cluster']['component_placement'],
 'task_train':c['env']['train']['task_config']['task_name'],
 'step_lim_train':c['env']['train']['task_config']['step_lim'],
 'task_eval':c['env']['eval']['task_config']['task_name'],
 'step_lim_eval':c['env']['eval']['task_config']['step_lim'],
 'train_envs':c['env']['train']['total_num_envs'],
 'train_rollout':c['env']['train']['rollout_epoch'],
 'eval_envs':c['env']['eval']['total_num_envs'],
 'outer_train':c['env']['train']['max_episode_steps'],
 'outer_eval':c['env']['eval']['max_episode_steps'],
 'model_path':c['actor']['model']['model_path'],
 'H':c['actor']['model']['num_action_chunks'],
 'action_dim':c['actor']['model']['action_dim'],
 'M':c['actor']['model']['num_steps'],
 'openpi_config':c['actor']['model']['openpi']['config_name'],
 'images':c['actor']['model']['openpi']['num_images_in_input'],
 'noise_level':c['actor']['model']['openpi']['noise_level'],
 'GB':c['actor']['global_batch_size'],
 'MB':c['actor']['micro_batch_size'],
 'update':c['algorithm']['update_epoch'],
}
print('OLD_PACKET_RESOLVED='+json.dumps(o, sort_keys=True))
PY
