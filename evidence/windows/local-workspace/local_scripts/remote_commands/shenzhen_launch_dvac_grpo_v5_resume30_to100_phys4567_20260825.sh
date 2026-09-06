#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=66c863bc5a45e90cb5161b30af54355b1104c810
BRANCH=codex/sz-current-pi0-dvac-grpo
RAY_ADDRESS=172.17.0.1:6389
OLD_NAME=dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
OLD_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/$OLD_NAME
CHECKPOINT=$OLD_RUN/robotwin_grpo_openpi_dvac_global_z_matched/checkpoints/global_step_30
NAME=dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/$NAME
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/$NAME
EXPERIMENT=robotwin_grpo_openpi_dvac_global_z_matched

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" rev-parse "personal/$BRANCH")" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test -d "$ROBOTWIN"
test -s "$OLD_RUN/runtime/resolved.yaml"
test -s "$CHECKPOINT/actor/dcp_checkpoint/.metadata"
test "$(find "$CHECKPOINT/actor" -maxdepth 1 -type f -name 'dvac_state_rank*.json' | wc -l)" = 4
test ! -e "$RUN"
test ! -e "$PACKET"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPUs 4,5,6,7 are not idle' >&2
  exit 20
fi

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

ARGS=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_grpo_openpi
  'cluster.component_placement={actor\, env\, rollout:"4,5,6,7"}'
  "runner.logger.log_path=$RUN"
  "runner.logger.experiment_name=$EXPERIMENT"
  runner.max_epochs=1000
  runner.max_steps=100
  runner.val_check_interval=10
  runner.save_interval=10
  "runner.resume_dir=$CHECKPOINT"
  algorithm.update_epoch=2
  algorithm.dvac_gradient_weighting.mode=apply
  env.train.total_num_envs=128
  env.train.rollout_epoch=4
  env.train.max_episode_steps=200
  env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN"
  env.train.video_cfg.save_video=true
  "env.train.video_cfg.video_base_dir=$RUN/video/train"
  "env.train.task_config.save_path=$RUN/robotwin_data/train"
  env.eval.total_num_envs=64
  env.eval.rollout_epoch=1
  env.eval.max_episode_steps=200
  env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true
  "env.eval.assets_path=$ROBOTWIN"
  env.eval.video_cfg.save_video=true
  "env.eval.video_cfg.video_base_dir=$RUN/video/eval"
  "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
  actor.micro_batch_size=32
  actor.global_batch_size=2048
  "actor.model.model_path=$MODEL"
)

install -d -m 755 "$PACKET"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" --cfg job --resolve > "$PACKET/resolved.yaml"

"$VENV/bin/python" - "$OLD_RUN/runtime/resolved.yaml" "$PACKET/resolved.yaml" "$PACKET/resume_parity.json" <<'PY'
from __future__ import annotations
import json, sys, yaml

old_path, new_path, out_path = sys.argv[1:]
old = yaml.safe_load(open(old_path, encoding='utf-8'))
new = yaml.safe_load(open(new_path, encoding='utf-8'))

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
diffs = {k: {'v4': a.get(k, '<MISSING>'), 'resume': b.get(k, '<MISSING>')}
         for k in sorted(set(a) | set(b)) if a.get(k, missing) != b.get(k, missing)}
allowed = {
    'runner.resume_dir',
    'runner.logger.log_path',
    'env.train.video_cfg.video_base_dir',
    'env.train.task_config.save_path',
    'env.eval.video_cfg.video_base_dir',
    'env.eval.task_config.save_path',
    'actor.model.output_dir',
    'algorithm.dvac_gradient_weighting.output_dir',
}
unexpected = sorted(set(diffs) - allowed)
assert not unexpected, unexpected
assert new['runner']['resume_dir'].endswith('/global_step_30')
assert new['runner']['max_steps'] == old['runner']['max_steps'] == 100
assert new['cluster']['component_placement'] == old['cluster']['component_placement'] == {'actor, env, rollout': '4,5,6,7'}
for path in (
    ('env','train','total_num_envs'), ('env','train','rollout_epoch'),
    ('actor','global_batch_size'), ('actor','micro_batch_size'),
    ('algorithm','update_epoch'),
):
    x, y = old, new
    for key in path:
        x, y = x[key], y[key]
    assert x == y
payload = {
    'source': old_path, 'candidate': new_path,
    'differences': diffs, 'unexpected_differences': unexpected,
    'conclusion': 'all scientific, sampling, batching, evaluation, and checkpoint cadence fields equal v4; only output paths and resume_dir differ',
}
json.dump(payload, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
open(out_path, 'a', encoding='utf-8').write('\n')
print('RESUME_PARITY_OK differences=', len(diffs), 'unexpected=0')
PY

printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$PACKET/command.txt"
printf '\n' >> "$PACKET/command.txt"
cat > "$PACKET/contract.json" <<EOF
{
  "source_head": "$HEAD",
  "physical_gpus": [4, 5, 6, 7],
  "resume_from": "$CHECKPOINT",
  "resume_step": 30,
  "target_total_step": 100,
  "remaining_steps": 70,
  "train": "128 env x 4 rollout epochs = 512 trajectories/step; G8; max 2048 chunk records/step",
  "actor": "GB2048/MB32/update2; max 2 optimizer calls/step",
  "dvac": "global-z; L3; recent5 restored exactly; warmup1; weights[0,2]",
  "eval": "fixed64 every10",
  "checkpoint": "every10"
}
EOF

mkdir -p "$RUN/runtime"
cp "$PACKET/resolved.yaml" "$PACKET/resume_parity.json" "$PACKET/contract.json" "$PACKET/command.txt" "$RUN/runtime/"
printf '%s\n' \
  "started_at=$(date --iso-8601=seconds)" \
  "source_head=$HEAD" \
  'physical_gpus=4,5,6,7' \
  "resume_from=$CHECKPOINT" \
  'resume_step=30; target_total_step=100; remaining_steps=70' \
  'train=128 env x 4 epochs = 512 trajectories/step; G8; max 2048 chunk records/step' \
  'actor=GB2048/MB32/update2; max 2 optimizer calls/step' \
  'dvac=global-z L3; exact recent5 sidecars; warmup1; weights[0,2]' \
  'eval=fixed64 every10; checkpoint every10' \
  'parity=all scientific fields equal v4; only output paths and resume_dir differ' \
  'normal_stop=total step100' \
  'hard_timeout=129600s' > "$RUN/runtime/launch_manifest.txt"

cat > "$RUN/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 129600s "$@" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
WRAP
chmod 700 "$RUN/runtime/wrapper.sh"

nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" \
  > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/runtime/wrapper.pid"
printf '%s\n' "$pid" > "$RUN/runtime/owned.pgid"

cat > "$RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct,gpu6_used_mib,gpu6_util_pct,gpu7_used_mib,gpu7_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
chmod 700 "$RUN/runtime/observer.sh"
nohup setsid bash "$RUN/runtime/observer.sh" "$pid" "$RUN/runtime/resource.csv" > "$RUN/runtime/observer.log" 2>&1 < /dev/null &
observer=$!
printf '%s\n' "$observer" > "$RUN/runtime/observer.pid"

sleep 15
kill -0 "$pid"
kill -0 "$observer"
printf 'run=%s\nwrapper_pid=%s\nobserver_pid=%s\ncheckpoint=%s\n' "$RUN" "$pid" "$observer" "$CHECKPOINT"
tail -n 40 "$RUN/runtime/driver.log" || true
echo SZ_DVAC_GRPO_V5_RESUME30_TO100_LAUNCHED
