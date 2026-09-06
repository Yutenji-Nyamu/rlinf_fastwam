#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=74617ced87d64045ab6850d0efd90956a494af66
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo
NAME=ppo-control-smoke1-2gpu64x4-b1024-noeval-localshard-phys23-v1
RUN="$ROOT/runs/$NAME"
EXPERIMENT=robotwin_ppo_control_smoke1_2gpu64x4_b1024_noeval_localshard_phys23_v1

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test ! -e "$RUN"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
if nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPUs 2,3 are not idle' >&2
  exit 20
fi

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

args=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_ppo_openpi_dvac_action_adv
  'cluster.component_placement={actor\, env\, rollout:"2,3"}'
  "runner.logger.log_path=$RUN"
  "runner.logger.experiment_name=$EXPERIMENT"
  runner.max_epochs=1000 runner.max_steps=1 runner.val_check_interval=-1 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=2 algorithm.adv_type=gae algorithm.loss_type=actor_critic algorithm.filter_rewards=false
  algorithm.logprob_type=chunk_level
  algorithm.dvac_gradient_weighting.mode=off
  algorithm.dvac_gradient_weighting.application=logprob_st
  algorithm.dvac_gradient_weighting.weight_min=null
  algorithm.dvac_gradient_weighting.weight_max=null
  env.train.total_num_envs=64 env.train.rollout_epoch=4
  env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN"
  env.train.video_cfg.save_video=false
  "env.train.video_cfg.video_base_dir=$RUN/video/train"
  "env.train.task_config.save_path=$RUN/robotwin_data/train"
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true
  "env.eval.assets_path=$ROBOTWIN"
  env.eval.video_cfg.save_video=false
  "env.eval.video_cfg.video_base_dir=$RUN/video/eval"
  "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
  actor.micro_batch_size=32 actor.global_batch_size=1024
  "actor.model.model_path=$MODEL"
  actor.fsdp_config.checkpoint_format=local_shard
)

mkdir -p "$RUN/runtime"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY' > "$RUN/runtime/preexisting_namespaces.json"
import json, os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_ppo_smoke_preflight", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
groups = {}
for row in rows:
    ns = str(row.get("namespace", ""))
    if ns.startswith("RLinf"):
        groups.setdefault(ns, []).append(row["name"])
print(json.dumps({k: sorted(v) for k, v in sorted(groups.items())}, indent=2))
ray.shutdown()
PY
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits > "$RUN/runtime/gpu4_7_before.txt"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" --cfg job --resolve > "$RUN/runtime/resolved.yaml"
"$VENV/bin/python" - "$RUN/runtime/resolved.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": "2,3"}
assert cfg["runner"]["max_steps"] == 1 and cfg["runner"]["val_check_interval"] == -1
assert cfg["algorithm"]["adv_type"] == "gae" and cfg["algorithm"]["loss_type"] == "actor_critic"
assert cfg["algorithm"]["logprob_type"] == "chunk_level"
dvac = cfg["algorithm"]["dvac_gradient_weighting"]
assert dvac["mode"] == "off" and dvac["application"] == "logprob_st"
assert dvac["weight_min"] is None and dvac["weight_max"] is None
assert cfg["env"]["train"]["total_num_envs"] == 64 and cfg["env"]["train"]["rollout_epoch"] == 4
assert cfg["env"]["eval"]["total_num_envs"] == 32
assert cfg["actor"]["global_batch_size"] == 1024 and cfg["actor"]["micro_batch_size"] == 32
assert cfg["actor"]["model"]["add_value_head"] is True
assert cfg["actor"]["fsdp_config"]["checkpoint_format"] == "local_shard"
print("PPO_CONTROL_SMOKE_RESOLVED_OK")
PY
printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" > "$RUN/runtime/command.txt"
printf '\n' >> "$RUN/runtime/command.txt"
printf '%s\n' \
  "started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" \
  "source_head=$HEAD" \
  'physical_gpus=2,3' \
  'fresh_start=true; target_step=1' \
  'train=64 env x 4 rollout epochs = 256 trajectories/step' \
  'actor=GB1024/MB32/update2; PPO GAE+critic' \
  'method=Control chunk-level; DVAC off' \
  'eval=disabled only for smoke; videos=false; terminal checkpoint local_shard global_step_1' \
  'normal_stop=step1/checkpoint/exit0; hard_timeout=7200s' > "$RUN/runtime/launch_manifest.txt"

cat > "$RUN/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 7200s "$@" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
WRAP
chmod 700 "$RUN/runtime/wrapper.sh"
nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" \
  > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/runtime/wrapper.pid"
printf '%s\n' "$pid" > "$RUN/runtime/owned.pgid"

sleep 20
kill -0 "$pid"
printf 'run=%s\nwrapper_pid=%s\n' "$RUN" "$pid"
nvidia-smi -i 2,3,4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_PPO_CONTROL_SMOKE1_LAUNCHED
