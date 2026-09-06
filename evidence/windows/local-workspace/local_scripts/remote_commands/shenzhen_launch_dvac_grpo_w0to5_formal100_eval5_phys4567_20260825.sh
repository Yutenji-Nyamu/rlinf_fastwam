#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
BRANCH=codex/sz-current-pi0-dvac-grpo-w0to5
RAY_ADDRESS=172.17.0.1:6389
NAME=dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/$NAME
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/$NAME
EXPERIMENT=robotwin_grpo_openpi_dvac_global_z_w0to5_matched_eval5
BASELINE=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2/resolved.yaml
W02=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4/runtime/resolved.yaml

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" rev-parse "personal/$BRANCH")" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test -s "$BASELINE"
test -s "$W02"
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test -d "$ROBOTWIN"
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
  runner.val_check_interval=5
  runner.save_interval=10
  runner.resume_dir=null
  algorithm.update_epoch=2
  algorithm.dvac_gradient_weighting.mode=apply
  algorithm.dvac_gradient_weighting.weight_min=0.0
  algorithm.dvac_gradient_weighting.weight_max=5.0
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

"$VENV/bin/python" - "$BASELINE" "$W02" "$PACKET/resolved.yaml" "$PACKET/parity.json" <<'PY'
from __future__ import annotations
import json, sys, yaml

baseline_path, w02_path, candidate_path, output_path = sys.argv[1:]
docs = [yaml.safe_load(open(path, encoding="utf-8")) for path in (baseline_path, w02_path, candidate_path)]

def flat(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flat(child, name))
    elif isinstance(value, list):
        out[prefix] = value
    else:
        out[prefix] = value
    return out

baseline, w02, new = map(flat, docs)
missing = object()
def differences(a, b):
    return {key: {"old": a.get(key, "<MISSING>"), "new": b.get(key, "<MISSING>")}
            for key in sorted(set(a) | set(b)) if a.get(key, missing) != b.get(key, missing)}

path_leaves = {
    "runner.logger.experiment_name", "runner.logger.log_path",
    "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
    "actor.model.output_dir", "algorithm.dvac_gradient_weighting.output_dir",
}
w02_allowed = path_leaves | {
    "runner.val_check_interval",
    "algorithm.dvac_gradient_weighting.weight_min",
    "algorithm.dvac_gradient_weighting.weight_max",
}
w02_diff = differences(w02, new)
w02_unexpected = sorted(set(w02_diff) - w02_allowed)
assert not w02_unexpected, w02_unexpected

baseline_diff = differences(baseline, new)
baseline_allowed_exact = path_leaves | {
    "cluster.component_placement.actor, env, rollout",
    "env.eval.seeds_path", "env.train.seeds_path",
    "runner.val_check_interval",
}
baseline_unexpected = sorted(
    key for key in baseline_diff
    if not key.startswith("algorithm.dvac_gradient_weighting.")
    and key not in baseline_allowed_exact
)
assert not baseline_unexpected, baseline_unexpected

cfg = docs[2]
dvac = cfg["algorithm"]["dvac_gradient_weighting"]
assert cfg["runner"]["resume_dir"] is None
assert cfg["runner"]["max_steps"] == 100
assert cfg["runner"]["val_check_interval"] == 5
assert cfg["runner"]["save_interval"] == 10
assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": "4,5,6,7"}
assert cfg["env"]["train"]["total_num_envs"] == 128
assert cfg["env"]["train"]["rollout_epoch"] == 4
assert cfg["algorithm"]["group_size"] == 8
assert cfg["actor"]["global_batch_size"] == 2048
assert cfg["actor"]["micro_batch_size"] == 32
assert cfg["algorithm"]["update_epoch"] == 2
assert dvac["mode"] == "apply" and dvac["selected_l"] == 3
assert dvac["window_steps"] == 5 and dvac["warmup_steps"] == 1
assert dvac["z_clip"] == 2.0 and dvac["strength"] == 0.5
assert dvac["weight_min"] == 0.0 and dvac["weight_max"] == 5.0
assert cfg["env"]["eval"]["total_num_envs"] == 64
assert cfg["env"]["eval"]["use_fixed_reset_state_ids"] is True

payload = {
    "baseline": baseline_path,
    "w0to2": w02_path,
    "candidate": candidate_path,
    "w0to2_differences": w02_diff,
    "w0to2_unexpected_differences": w02_unexpected,
    "baseline_unexpected_non_dvac_differences": baseline_unexpected,
    "conclusion": "all unapproved scientific, sampling, batching, optimizer, model and seed leaves match both controls",
}
json.dump(payload, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(output_path, "a", encoding="utf-8").write("\n")
print("W0TO5_PARITY_OK w02_unexpected=0 baseline_unexpected_non_dvac=0")
PY

printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$PACKET/command.txt"
printf '\n' >> "$PACKET/command.txt"
cat > "$PACKET/contract.json" <<EOF
{
  "source_head": "$HEAD",
  "physical_gpus": [4, 5, 6, 7],
  "fresh_start": true,
  "outer_steps": 100,
  "train": "128 env x 4 rollout epochs = 512 trajectories/step; G8; max 2048 chunk records/step",
  "actor": "GB2048/MB32/update2; max 2 optimizer calls/step",
  "dvac": "global-z; L3; recent5; warmup1; piecewise continuous weights[0,5]",
  "eval": "fixed64 every5",
  "checkpoint": "every10"
}
EOF

mkdir -p "$RUN/runtime"
cp "$PACKET/resolved.yaml" "$PACKET/parity.json" "$PACKET/contract.json" "$PACKET/command.txt" "$RUN/runtime/"
printf '%s\n' \
  "started_at=$(date --iso-8601=seconds)" \
  "source_head=$HEAD" \
  'physical_gpus=4,5,6,7' \
  'fresh_start=true; target_step=100' \
  'train=128 env x 4 epochs = 512 trajectories/step; G8; max 2048 chunk records/step' \
  'actor=GB2048/MB32/update2; max 2 optimizer calls/step' \
  'dvac=global-z L3; recent5; warmup1; continuous weights[0,5]' \
  'eval=fixed64 every5; checkpoint every10' \
  'parity=unexpected differences versus matched GRPO and weights[0,2] controls are zero' \
  'normal_stop=complete step100' \
  'hard_timeout=216000s' > "$RUN/runtime/launch_manifest.txt"

cat > "$RUN/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 216000s "$@" > "$runtime/driver.log" 2>&1
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
cat "$PACKET/contract.json"
cat "$PACKET/parity.json"
printf 'run=%s\nwrapper_pid=%s\nobserver_pid=%s\n' "$RUN" "$pid" "$observer"
tail -n 50 "$RUN/runtime/driver.log" || true
echo SZ_DVAC_GRPO_W0TO5_FORMAL100_EVAL5_LAUNCHED
