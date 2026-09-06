#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
out=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_prelaunch_20260828_v1
control=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_mb256_warm20k_replay80k_control
s05=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s0p5_mb256_warm20k_replay80k_fresh480
s20=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s2p0_mb256_warm20k_replay80k_fresh480

cd "$repo"
test "$(git rev-parse HEAD)" = cb88e9c5d817a248fef6d6ee02127b874ab851a8
test "$(git status --short | wc -l)" -eq 1
git status --short | grep -F 'robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s2p0_mb256_warm20k_replay80k_fresh480.yaml'
git diff --check

mkdir -p "$out"
export PYTHONPATH="$repo:$assets"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export RLINF_CODE_WORKING_DIR="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_dvac_pure_dual_compose_dummy
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

for cfg in "$control" "$s05" "$s20"; do
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
    --config-path "$repo/examples/embodiment/config" --config-name "$cfg" \
    --cfg job --resolve >"$out/${cfg}.yaml"
done

"$venv/bin/python" - "$out/${control}.yaml" "$out/${s05}.yaml" "$out/${s20}.yaml" <<'PY'
import sys, yaml
def flat(x, p=''):
    out={}
    if isinstance(x, dict):
        for k,v in x.items(): out.update(flat(v, f'{p}.{k}' if p else str(k)))
    elif isinstance(x, list): out[p]=x
    else: out[p]=x
    return out
paths=sys.argv[1:]
c,a,b=[flat(yaml.safe_load(open(p, encoding='utf-8'))) for p in paths]
for label,left,right in [('PURE_S05_MINUS_CONTROL',c,a),('PURE_S20_MINUS_S05',a,b)]:
    keys=sorted(k for k in set(left)|set(right) if left.get(k)!=right.get(k))
    print(label)
    for k in keys: print(f'{k}: {left.get(k)!r} -> {right.get(k)!r}')
    if label=='PURE_S20_MINUS_S05':
        allowed={'algorithm.rlt_dvac.strength','algorithm.rlt_dvac.output_dir','runner.logger.experiment_name'}
        unexpected=set(keys)-allowed
        if unexpected: raise SystemExit(f'unexpected s20 diff: {sorted(unexpected)}')
PY

git add examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s2p0_mb256_warm20k_replay80k_fresh480.yaml
git commit -m 'config(rlt): add strong pure DVAC formal variant'
git rev-parse HEAD
git status --short
