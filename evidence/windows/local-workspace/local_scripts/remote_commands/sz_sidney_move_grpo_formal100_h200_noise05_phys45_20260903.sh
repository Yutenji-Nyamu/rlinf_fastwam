#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
HEAD=f50e235c5ab1f4390f0ba92bfb13390ed0a86810
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
NAME=move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
PACKET="$ROOT/packets/$NAME"
RUN="$ROOT/runs/$NAME"
EXPERIMENT=pi05_sidney_move_grpo_formal100_2gpu64x4_g8_b1024_u2_m10_noise0p5_h200_fixed32_phys45_v1

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$MODEL/model.safetensors"
test -s "$MODEL/conversion_manifest.json"
test ! -e "$PACKET"
test ! -e "$RUN"
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status >/dev/null
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPU4/5 is occupied; refusing launch' >&2
  exit 20
fi

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

ARGS=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_move_stapler_pad_grpo_openpi_pi05_sidney
  'cluster.component_placement={actor\, env\, rollout:"4,5"}'
  "runner.logger.log_path=$RUN"
  "runner.logger.experiment_name=$EXPERIMENT"
  runner.only_eval=false runner.max_epochs=1000 runner.max_steps=100
  runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
  algorithm.group_size=8 algorithm.update_epoch=2 algorithm.adv_type=grpo
  algorithm.logprob_type=chunk_level algorithm.filter_rewards=true
  algorithm.dvac_gradient_weighting.mode=off
  env.train.total_num_envs=64 env.train.rollout_epoch=4
  env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=false
  "env.train.task_config.save_path=$RUN/robotwin_data/train"
  env.train.task_config.task_name=move_stapler_pad env.train.task_config.step_lim=200
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN"
  env.eval.video_cfg.save_video=true
  "env.eval.video_cfg.video_base_dir=$RUN/video/eval"
  "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
  env.eval.task_config.task_name=move_stapler_pad env.eval.task_config.step_lim=200
  actor.global_batch_size=1024 actor.micro_batch_size=32
  "actor.model.model_path=$MODEL"
  actor.model.num_steps=10 actor.model.openpi.num_steps=10
  actor.model.openpi.noise_level=0.5
  ++actor.fsdp_config.checkpoint_format=local_shard
)

mkdir -p "$PACKET" "$RUN/runtime"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" --cfg job --resolve > "$PACKET/resolved.yaml"
"$VENV/bin/python" - "$PACKET/resolved.yaml" "$MODEL/conversion_manifest.json" "$PACKET/contract.json" <<'PY'
import json, sys, yaml
c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
m = json.load(open(sys.argv[2], encoding="utf-8"))
assert m["source_keys"] == m["target_keys"] == 813
assert not m["missing_keys"] and not m["unexpected_keys"] and not m["shape_mismatches"]
assert c["cluster"]["component_placement"] == {"actor, env, rollout": "4,5"}
assert c["runner"]["max_steps"] == 100
assert c["runner"]["val_check_interval"] == 5 and c["runner"]["save_interval"] == 10
assert c["algorithm"]["group_size"] == 8 and c["algorithm"]["update_epoch"] == 2
assert c["algorithm"]["dvac_gradient_weighting"]["mode"] == "off"
assert c["actor"]["global_batch_size"] == 1024 and c["actor"]["micro_batch_size"] == 32
assert c["actor"]["model"]["num_action_chunks"] == 50
assert c["actor"]["model"]["openpi"]["config_name"] == "pi05_sidney_robotwin"
assert c["actor"]["model"]["openpi"]["num_steps"] == 10
assert c["actor"]["model"]["openpi"]["noise_level"] == 0.5
assert c["actor"]["fsdp_config"]["checkpoint_format"] == "local_shard"
for split, n in (("train", 64), ("eval", 32)):
    e = c["env"][split]
    assert e["total_num_envs"] == n
    assert e["max_episode_steps"] == 200
    assert e["max_steps_per_rollout_epoch"] == 200
    assert e["task_config"]["task_name"] == "move_stapler_pad"
    assert e["task_config"]["step_lim"] == 200
assert c["env"]["train"]["rollout_epoch"] == 4
payload = {
    "task": "move_stapler_pad", "physical_gpus": [4, 5], "fresh": True,
    "target_steps": 100, "train_envs": 64, "rollout_epochs": 4,
    "trajectories_per_step": 256, "group_size": 8, "groups_per_step": 32,
    "episode_action_limit": 200, "H": 50, "C": 50, "M": 10,
    "max_query_records_per_step": 1024, "global_batch": 1024,
    "micro_batch": 32, "update_epochs": 2, "optimizer_calls_per_step": 2,
    "noise_level": 0.5, "fixed_eval_episodes": 32,
    "eval_interval": 5, "save_interval": 10, "checkpoint_format": "local_shard",
}
json.dump(payload, open(sys.argv[3], "w", encoding="utf-8"), indent=2)
open(sys.argv[3], "a", encoding="utf-8").write("\n")
print(json.dumps(payload, indent=2))
PY

printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$PACKET/command.txt"
printf '\n' >> "$PACKET/command.txt"
printf '%s\n' "$HEAD" > "$PACKET/source-head.txt"
sha256sum "$PACKET/resolved.yaml" "$MODEL/conversion_manifest.json" > "$PACKET/sha256.txt"
cp "$PACKET/resolved.yaml" "$PACKET/contract.json" "$PACKET/command.txt" "$PACKET/source-head.txt" "$PACKET/sha256.txt" "$RUN/runtime/"

cat > "$RUN/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 345600s "$@" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
WRAP
chmod 700 "$RUN/runtime/wrapper.sh"

nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" \
  > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN/runtime/wrapper.pid"
printf '%s\n' "$PID" > "$RUN/runtime/owned.pgid"

cat > "$RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
echo 'timestamp,driver_alive,host_mem_available_kib,gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 4,5 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
chmod 700 "$RUN/runtime/observer.sh"
nohup setsid bash "$RUN/runtime/observer.sh" "$PID" "$RUN/runtime/resource.csv" > "$RUN/runtime/observer.log" 2>&1 < /dev/null &
OBSERVER=$!
printf '%s\n' "$OBSERVER" > "$RUN/runtime/observer.pid"
printf '%s\n' \
  "started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" \
  "source_head=$HEAD" \
  'physical_gpus=4,5' \
  'fresh=true target_step=100' \
  'move_stapler_pad H50 C50 M10 horizon200 noise0.5' \
  '64 env x rollout4 = 256 trajectories; G8; max1024 records' \
  'GB1024 MB32 update2; fixed32/eval5; local-shard/save10' > "$RUN/runtime/launch_manifest.txt"

sleep 15
kill -0 "$PID"
printf 'run=%s\npacket=%s\nwrapper_pid=%s\nobserver_pid=%s\n' "$RUN" "$PACKET" "$PID" "$OBSERVER"
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SIDNEY_PI05_MOVE_GRPO_FORMAL100_H200_NOISE05_PHYS45_LAUNCHED
