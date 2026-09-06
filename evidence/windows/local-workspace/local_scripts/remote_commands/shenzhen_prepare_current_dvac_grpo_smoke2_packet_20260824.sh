#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=66c863bc5a45e90cb5161b30af54355b1104c810
BRANCH=codex/sz-current-pi0-dvac-grpo
RAY_ADDRESS=172.17.0.1:6389
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-smoke2-4gpu128train64eval-v1
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-smoke2-4gpu128train64eval-v1
EXPERIMENT=robotwin_adjust_bottle_dvac_global_z_smoke2

printf 'MARKER=SZ_PREPARE_CURRENT_DVAC_GRPO_SMOKE2_PACKET_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" rev-parse "personal/$BRANCH")" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test ! -e "$RUN"
test ! -e "$PACKET"
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test -d "$ROBOTWIN"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
if nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 0-3 are not idle' >&2
  exit 1
fi
install -d -m 755 "$PACKET"

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES
export RAY_ADDRESS
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_CODE_WORKING_DIR="$WT"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

ARGS=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_grpo_openpi
  'cluster.component_placement={actor\, env\, rollout:0-3}'
  "runner.logger.log_path=$RUN"
  "runner.logger.experiment_name=$EXPERIMENT"
  'runner.max_epochs=1000'
  'runner.max_steps=2'
  'runner.val_check_interval=2'
  'runner.save_interval=2'
  'runner.resume_dir=null'
  'algorithm.update_epoch=2'
  'algorithm.dvac_gradient_weighting.mode=apply'
  'env.train.total_num_envs=128'
  'env.train.rollout_epoch=4'
  'env.train.max_episode_steps=200'
  'env.train.max_steps_per_rollout_epoch=200'
  "env.train.assets_path=$ROBOTWIN"
  'env.train.video_cfg.save_video=false'
  "env.train.video_cfg.video_base_dir=$RUN/video/train"
  "env.train.task_config.save_path=$RUN/robotwin_data/train"
  'env.eval.total_num_envs=64'
  'env.eval.rollout_epoch=1'
  'env.eval.max_episode_steps=200'
  'env.eval.max_steps_per_rollout_epoch=200'
  'env.eval.use_fixed_reset_state_ids=true'
  "env.eval.assets_path=$ROBOTWIN"
  'env.eval.video_cfg.save_video=false'
  "env.eval.video_cfg.video_base_dir=$RUN/video/eval"
  "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
  'actor.micro_batch_size=32'
  'actor.global_batch_size=2048'
  "actor.model.model_path=$MODEL"
)

"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${ARGS[@]}" --cfg job --resolve > "$PACKET/resolved.yaml"

"$VENV/bin/python" - "$PACKET/resolved.yaml" "$HEAD" <<'PY' > "$PACKET/contract.json"
from __future__ import annotations
import json
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
dvac = cfg.algorithm.dvac_gradient_weighting
trajectories = cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch
records = trajectories * (
    cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
)
assert OmegaConf.to_container(cfg.cluster.component_placement) == {"actor, env, rollout": "0-3"}
assert cfg.runner.max_steps == 2
assert cfg.runner.val_check_interval == cfg.runner.save_interval == 2
assert trajectories == 512 and trajectories // cfg.algorithm.group_size == 64
assert records == 2048
assert cfg.actor.global_batch_size == 2048 and cfg.actor.micro_batch_size == 32
assert cfg.algorithm.update_epoch == 2
assert dvac.mode == "apply" and dvac.selected_l == 3
assert dvac.window_steps == 5 and dvac.warmup_steps == 1
assert dvac.z_clip == 2.0 and dvac.strength == 0.5
assert 1.0 - dvac.z_clip * dvac.strength == 0.0
assert 1.0 + dvac.z_clip * dvac.strength == 2.0
assert cfg.env.eval.total_num_envs == 64 and cfg.env.eval.use_fixed_reset_state_ids
print(json.dumps({
    "source_head": sys.argv[2],
    "physical_gpus": [0, 1, 2, 3],
    "outer_steps": 2,
    "train_envs": 128,
    "rollout_epochs": 4,
    "trajectories_per_step": trajectories,
    "groups_per_step": trajectories // cfg.algorithm.group_size,
    "max_chunk_records_per_step": records,
    "global_batch": 2048,
    "micro_batch": 32,
    "update_epoch": 2,
    "optimizer_calls_per_step": 2,
    "dvac": {"selected_l": 3, "recent_steps": 5, "warmup_steps": 1, "weight_range": [0, 2]},
    "fixed_eval_episodes": 64,
    "eval_after_step": 2,
    "checkpoint_after_step": 2,
}, indent=2))
PY

sha256sum "$PACKET/resolved.yaml" > "$PACKET/resolved.yaml.sha256"
printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$PACKET/command.txt"
printf '\n' >> "$PACKET/command.txt"

cat > "$PACKET/launch.sh" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=66c863bc5a45e90cb5161b30af54355b1104c810
BRANCH=codex/sz-current-pi0-dvac-grpo
RAY_ADDRESS=172.17.0.1:6389
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-smoke2-4gpu128train64eval-v1
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-smoke2-4gpu128train64eval-v1
EXPERIMENT=robotwin_adjust_bottle_dvac_global_z_smoke2
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" rev-parse "personal/$BRANCH")" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test ! -e "$RUN"
test "$(sha256sum "$PACKET/resolved.yaml" | awk '{print $1}')" = "$(awk '{print $1}' "$PACKET/resolved.yaml.sha256")"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
if nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 0-3 are not idle' >&2
  exit 1
fi
source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ARGS=(
  --config-path "$WT/examples/embodiment/config" --config-name robotwin_adjust_bottle_grpo_openpi
  'cluster.component_placement={actor\, env\, rollout:0-3}'
  "runner.logger.log_path=$RUN" "runner.logger.experiment_name=$EXPERIMENT"
  runner.max_epochs=1000 runner.max_steps=2 runner.val_check_interval=2 runner.save_interval=2 runner.resume_dir=null
  algorithm.update_epoch=2 algorithm.dvac_gradient_weighting.mode=apply
  env.train.total_num_envs=128 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=false "env.train.video_cfg.video_base_dir=$RUN/video/train" "env.train.task_config.save_path=$RUN/robotwin_data/train"
  env.eval.total_num_envs=64 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200 env.eval.use_fixed_reset_state_ids=true
  "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=false "env.eval.video_cfg.video_base_dir=$RUN/video/eval" "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
  actor.micro_batch_size=32 actor.global_batch_size=2048 "actor.model.model_path=$MODEL"
)
mkdir -p "$RUN/runtime"
cp "$PACKET/resolved.yaml" "$PACKET/contract.json" "$PACKET/command.txt" "$RUN/runtime/"
printf '%s\n' "started_at=$(date --iso-8601=seconds)" "source_head=$HEAD" 'physical_gpus=0,1,2,3' \
  'train=128 env x 4 epochs = 512 trajectories/step; G8; max 2048 chunk records/step' \
  'actor=GB2048/MB32/update2; 2 optimizer calls/step' 'dvac=L3; recent5; warmup1; weights[0,2]' \
  'eval=fixed64 after step2; save=global_step_2' 'normal_stop=step2' 'hard_timeout=7200s' > "$RUN/runtime/launch_manifest.txt"
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
nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/runtime/wrapper.pid"
printf '%s\n' "$pid" > "$RUN/runtime/owned.pgid"
cat > "$RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,gpu0_used_mib,gpu0_util_pct,gpu1_used_mib,gpu1_util_pct,gpu2_used_mib,gpu2_util_pct,gpu3_used_mib,gpu3_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
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
printf 'run=%s\nwrapper_pid=%s\nobserver_pid=%s\n' "$RUN" "$pid" "$observer"
tail -n 30 "$RUN/runtime/driver.log" || true
printf 'MARKER=SZ_CURRENT_DVAC_GRPO_SMOKE2_LAUNCHED\n'
LAUNCH
chmod 700 "$PACKET/launch.sh"
bash -n "$PACKET/launch.sh"

printf '%s\n' '=== CONTRACT ==='
cat "$PACKET/contract.json"
printf '%s\n' '=== COMMAND ==='
cat "$PACKET/command.txt"
printf '%s\n' '=== LIVE RESOURCE ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
printf 'packet=%s\nrun=%s\nlauncher=%s\nresolved_sha256=%s\n' "$PACKET" "$RUN" "$PACKET/launch.sh" "$(awk '{print $1}' "$PACKET/resolved.yaml.sha256")"
printf 'MARKER=SZ_PREPARE_CURRENT_DVAC_GRPO_SMOKE2_PACKET_OK\n'
