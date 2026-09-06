#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
out=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_config_check_20260829_v1
base_head=a2ae5cbe81049fb7027c43ea85483ed3ffc3ce2f
control=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_mb256_warm20k_replay80k_control
pure02=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s0p5_mb256_warm20k_replay80k_fresh480
pure03=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s1p0_mb256_warm20k_replay80k_fresh480
pure04=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s1p5_mb256_warm20k_replay80k_fresh480
pure04_gpu1=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s1p5_mb256_warm20k_replay80k_gpu1_fresh480
pure05=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s2p0_mb256_warm20k_replay80k_fresh480

cd "$repo"
test "$(git rev-parse HEAD)" = "$base_head"
test "$(git branch --show-current)" = codex/rlt-dvac-pure-reference-bc
status=$(git status --short)
printf '%s\n' "$status"
test "$(printf '%s\n' "$status" | sed '/^$/d' | wc -l)" -eq 3
printf '%s\n' "$status" | grep -F "$pure03.yaml"
printf '%s\n' "$status" | grep -F "$pure04.yaml"
printf '%s\n' "$status" | grep -F "$pure04_gpu1.yaml"

mkdir -p "$out"
export PYTHONPATH="$repo:$assets" EMBODIED_PATH="$repo/examples/embodiment" REPO_PATH="$repo"
export RLINF_CODE_WORKING_DIR="$repo" ROBOTWIN_PATH="$assets" ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_pure03_pure04_compose_dummy
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

for cfg in "$control" "$pure02" "$pure03" "$pure04" "$pure04_gpu1" "$pure05"; do
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
    --config-path "$repo/examples/embodiment/config" --config-name "$cfg" \
    --cfg job --resolve >"$out/${cfg}.yaml"
done

"$venv/bin/python" - "$out" "$control" "$pure02" "$pure03" "$pure04" "$pure04_gpu1" "$pure05" <<'PY'
import sys, yaml
from pathlib import Path

root=Path(sys.argv[1]); names=sys.argv[2:]
def flat(x,p=''):
    out={}
    if isinstance(x,dict):
        for k,v in x.items(): out.update(flat(v,f'{p}.{k}' if p else str(k)))
    elif isinstance(x,list): out[p]=x
    else: out[p]=x
    return out
cfg={n:flat(yaml.safe_load((root/f'{n}.yaml').read_text())) for n in names}
allowed_method={'algorithm.rlt_dvac.strength','algorithm.rlt_dvac.output_dir','runner.logger.experiment_name'}
for left,right in [(names[1],names[2]),(names[2],names[3]),(names[3],names[5])]:
    keys=sorted(k for k in set(cfg[left])|set(cfg[right]) if cfg[left].get(k)!=cfg[right].get(k))
    print(f'{right}_MINUS_{left}')
    for k in keys: print(f'{k}: {cfg[left].get(k)!r} -> {cfg[right].get(k)!r}')
    unexpected=set(keys)-allowed_method
    if unexpected: raise SystemExit(f'unexpected method diff: {sorted(unexpected)}')
left,right=names[3],names[4]
keys=sorted(k for k in set(cfg[left])|set(cfg[right]) if cfg[left].get(k)!=cfg[right].get(k))
print(f'{right}_MINUS_{left}')
for k in keys: print(f'{k}: {cfg[left].get(k)!r} -> {cfg[right].get(k)!r}')
if set(keys)!={'cluster.component_placement.actor, env, rollout'}:
    raise SystemExit(f'unexpected gpu1 diff: {keys}')
base,method=names[0],names[2]
keys=sorted(k for k in set(cfg[base])|set(cfg[method]) if cfg[base].get(k)!=cfg[method].get(k))
print('PURE03_MINUS_CONTROL_COUNT',len(keys))
for k in keys: print(f'{k}: {cfg[base].get(k)!r} -> {cfg[method].get(k)!r}')
PY

git diff --check
git add \
  "examples/embodiment/config/${pure03}.yaml" \
  "examples/embodiment/config/${pure04}.yaml" \
  "examples/embodiment/config/${pure04_gpu1}.yaml"
git commit -m 'config(rlt): add Pure03 and Pure04 formal variants'
git rev-parse HEAD | tee "$out/source_head.txt"
git status --short

if ! timeout 90 git push personal HEAD:codex/rlt-dvac-pure-reference-bc; then
  timeout 120 bash -lc 'source /etc/network_turbo; git push personal HEAD:codex/rlt-dvac-pure-reference-bc'
fi
git ls-remote personal refs/heads/codex/rlt-dvac-pure-reference-bc | tee "$out/remote_head.txt"
