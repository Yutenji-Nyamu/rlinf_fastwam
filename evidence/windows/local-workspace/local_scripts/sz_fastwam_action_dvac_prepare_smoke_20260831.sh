#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
HEAD=a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv
NAME=fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2
RUN="$ROOT/runs/$NAME"
RELOAD="$ROOT/runs/${NAME}-reload-step2"
PACKET="$ROOT/packets/$NAME"

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$RUN"
test ! -e "$RELOAD"
test ! -e "$PACKET"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
if nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPUs 2,3 are not idle' >&2
  exit 20
fi

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FW:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

mkdir -p "$PACKET"
common=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_move_stapler_pad_grpo_fastwam_action_dvac_adv
  'cluster.component_placement={actor\, env\, rollout:"2,3"}'
  runner.max_epochs=1000 runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1
  env.train.total_num_envs=32 env.train.rollout_epoch=1
  env.train.max_episode_steps=192 env.train.max_steps_per_rollout_epoch=192 env.train.video_cfg.save_video=false
  "env.train.assets_path=$ROBOTWIN"
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=192 env.eval.max_steps_per_rollout_epoch=192
  env.eval.use_fixed_reset_state_ids=true env.eval.video_cfg.save_video=false
  "env.eval.assets_path=$ROBOTWIN"
  actor.micro_batch_size=2 actor.global_batch_size=256
  algorithm.group_size=8 algorithm.update_epoch=1
  actor.fsdp_config.checkpoint_format=dcp actor.fsdp_config.cast_forward_inputs=false
)

"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${common[@]}" \
  runner.resume_dir=null \
  "runner.logger.log_path=$RUN" runner.logger.experiment_name=fastwam_action_dvac_adv_smoke_fresh \
  "env.train.task_config.save_path=$RUN/robotwin_data/train" \
  "env.eval.task_config.save_path=$RUN/robotwin_data/eval" \
  "env.eval.video_cfg.video_base_dir=$RUN/video/eval" \
  --cfg job --resolve \
  > "$PACKET/resolved.yaml"

"$VENV/bin/python" - "$PACKET/resolved.yaml" "$PACKET/contract.json" <<'PY'
import json, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": "2,3"}
assert cfg["env"]["train"]["total_num_envs"] == 32
assert cfg["env"]["train"]["rollout_epoch"] == 1
assert cfg["env"]["eval"]["total_num_envs"] == 32
assert cfg["algorithm"]["group_size"] == 8
assert cfg["algorithm"]["update_epoch"] == 1
assert cfg["algorithm"]["logprob_type"] == "action_level"
dvac = cfg["algorithm"]["dvac_gradient_weighting"]
assert (dvac["mode"], dvac["application"], dvac["selected_l"]) == ("apply", "action_advantage", 5)
assert (dvac["window_steps"], dvac["warmup_steps"]) == (5, 1)
assert (dvac["weight_min"], dvac["weight_max"]) == (0.5, 1.5)
assert cfg["actor"]["global_batch_size"] == 256
assert cfg["actor"]["micro_batch_size"] == 2
for owner in ("actor", "rollout"):
    model = cfg[owner]["model"]
    assert (model["model_type"], model["action_horizon"], model["num_action_chunks"], model["num_inference_steps"], model["action_dim"]) == ("fastwam", 32, 24, 10, 14)
assert cfg["actor"]["fsdp_config"]["checkpoint_format"] == "dcp"
contract = {
    "head": "a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7",
    "physical_gpus": [2, 3],
    "train_envs": 32,
    "eval_envs": 32,
    "rollout_epochs": 1,
    "trajectories": 32,
    "group_size": 8,
    "groups": 4,
    "max_query_records": 256,
    "global_batch": 256,
    "micro_batch": 2,
    "update_epochs": 1,
    "H_C_M_D": [32, 24, 10, 14],
    "dvac": {"L": 5, "history": 5, "warmup": 1, "weights": [0.5, 1.5]},
    "checkpoint": "DCP; fresh Step1 then fresh-process resume Step2",
}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(contract, indent=2) + "\n")
print(json.dumps(contract, indent=2))
PY

cat > "$PACKET/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv
NAME=fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2
RUN="$ROOT/runs/$NAME"
RELOAD="$ROOT/runs/${NAME}-reload-step2"
PACKET="$ROOT/packets/$NAME"

finish() {
  rc=$?
  printf '%s\n' "$rc" > "$PACKET/worker_exit_code.txt"
  date --iso-8601=seconds > "$PACKET/worker_finished_at.txt"
  if [ "$rc" -ne 0 ]; then date --iso-8601=seconds > "$PACKET/SMOKE_FAILED"; fi
}
trap finish EXIT

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FW:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

common=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_move_stapler_pad_grpo_fastwam_action_dvac_adv
  'cluster.component_placement={actor\, env\, rollout:"2,3"}'
  runner.max_epochs=1000 runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1
  env.train.total_num_envs=32 env.train.rollout_epoch=1
  env.train.max_episode_steps=192 env.train.max_steps_per_rollout_epoch=192 env.train.video_cfg.save_video=false
  "env.train.assets_path=$ROBOTWIN"
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=192 env.eval.max_steps_per_rollout_epoch=192
  env.eval.use_fixed_reset_state_ids=true env.eval.video_cfg.save_video=false
  "env.eval.assets_path=$ROBOTWIN"
  actor.micro_batch_size=2 actor.global_batch_size=256
  algorithm.group_size=8 algorithm.update_epoch=1
  actor.fsdp_config.checkpoint_format=dcp actor.fsdp_config.cast_forward_inputs=false
)

mkdir -p "$RUN/runtime"
cp "$PACKET/resolved.yaml" "$RUN/runtime/resolved.yaml"
cp "$PACKET/contract.json" "$RUN/runtime/contract.json"
date --iso-8601=seconds > "$RUN/runtime/started_at.txt"
set +e
timeout --signal=TERM --kill-after=180s 21600s \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${common[@]}" runner.resume_dir=null \
  "runner.logger.log_path=$RUN" runner.logger.experiment_name=fastwam_action_dvac_adv_smoke_fresh \
  "env.train.task_config.save_path=$RUN/robotwin_data/train" \
  "env.eval.task_config.save_path=$RUN/robotwin_data/eval" \
  "env.eval.video_cfg.video_base_dir=$RUN/video/eval" \
  > "$RUN/runtime/driver.log" 2>&1
fresh_rc=$?
set -e
printf '%s\n' "$fresh_rc" > "$RUN/runtime/exit_code.txt"
date --iso-8601=seconds > "$RUN/runtime/finished_at.txt"
test "$fresh_rc" -eq 0
CKPT="$RUN/fastwam_action_dvac_adv_smoke_fresh/checkpoints/global_step_1"
test -s "$CKPT/actor/dcp_checkpoint/.metadata"
test "$(find "$CKPT/actor/dcp_checkpoint" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -eq 2
test -s "$CKPT/actor/dvac_state_rank0000.json"
test -s "$CKPT/actor/dvac_state_rank0001.json"
"$VENV/bin/python" - "$RUN" "$CKPT" <<'PY'
import glob, json, sys, torch
run, ckpt = sys.argv[1:]
files = sorted(glob.glob(run + "/dvac_train/actor_rank*/runner_step_*.pt"))
assert len(files) == 2, files
for path in files:
    item = torch.load(path, map_location="cpu", weights_only=True)
    assert item["warmup"] is True
    assert tuple(item["variance"].shape)[-1] == 24
    assert torch.isfinite(item["variance"]).all()
    assert torch.equal(item["weights"], torch.ones_like(item["weights"]))
for rank in range(2):
    side = json.load(open(f"{ckpt}/actor/dvac_state_rank{rank:04d}.json", encoding="utf-8"))
    assert side["application"] == "action_advantage"
    assert len(side["recent_stats"]["steps"]) == 1
print("FASTWAM_ACTION_DVAC_STEP1_OK")
PY

sleep 20
mkdir -p "$RELOAD/runtime"
date --iso-8601=seconds > "$RELOAD/runtime/started_at.txt"
set +e
timeout --signal=TERM --kill-after=180s 21600s \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${common[@]}" runner.max_steps=2 "runner.resume_dir=$CKPT" \
  "runner.logger.log_path=$RELOAD" runner.logger.experiment_name=fastwam_action_dvac_adv_smoke_reload_step2 \
  "env.train.task_config.save_path=$RELOAD/robotwin_data/train" \
  "env.eval.task_config.save_path=$RELOAD/robotwin_data/eval" \
  "env.eval.video_cfg.video_base_dir=$RELOAD/video/eval" \
  > "$RELOAD/runtime/driver.log" 2>&1
reload_rc=$?
set -e
printf '%s\n' "$reload_rc" > "$RELOAD/runtime/exit_code.txt"
date --iso-8601=seconds > "$RELOAD/runtime/finished_at.txt"
test "$reload_rc" -eq 0
grep -F "Resuming training from checkpoint directory $CKPT" "$RELOAD/runtime/driver.log" >/dev/null
RELOAD_CKPT="$RELOAD/fastwam_action_dvac_adv_smoke_reload_step2/checkpoints/global_step_2"
test -s "$RELOAD_CKPT/actor/dcp_checkpoint/.metadata"
test "$(find "$RELOAD_CKPT/actor/dcp_checkpoint" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -eq 2
test -s "$RELOAD_CKPT/actor/dvac_state_rank0000.json"
test -s "$RELOAD_CKPT/actor/dvac_state_rank0001.json"
"$VENV/bin/python" - "$RELOAD" "$RELOAD_CKPT" <<'PY'
import glob, json, sys, torch
run, ckpt = sys.argv[1:]
files = sorted(glob.glob(run + "/dvac_train/actor_rank*/runner_step_*.pt"))
assert len(files) == 2, files
all_weights = []
for path in files:
    item = torch.load(path, map_location="cpu", weights_only=True)
    assert item["warmup"] is False
    weights = item["weights"].float()
    assert torch.isfinite(weights).all()
    assert float(weights.min()) >= 0.5 - 1e-6
    assert float(weights.max()) <= 1.5 + 1e-6
    all_weights.append(weights.reshape(-1))
weights = torch.cat(all_weights)
assert float(weights.std(unbiased=False)) > 0.0
for rank in range(2):
    side = json.load(open(f"{ckpt}/actor/dvac_state_rank{rank:04d}.json", encoding="utf-8"))
    assert side["application"] == "action_advantage"
    assert len(side["recent_stats"]["steps"]) == 2
summary = {"min": float(weights.min()), "max": float(weights.max()), "mean": float(weights.mean()), "std": float(weights.std(unbiased=False))}
open(run + "/runtime/weight_summary.json", "w", encoding="utf-8").write(json.dumps(summary, indent=2) + "\n")
print("FASTWAM_ACTION_DVAC_STEP2_RESUME_OK", json.dumps(summary, sort_keys=True))
PY

date --iso-8601=seconds > "$PACKET/SMOKE_OK"
printf 'FASTWAM_ACTION_DVAC_SMOKE_OK\n'
WORKER
chmod 700 "$PACKET/worker.sh"
bash -n "$PACKET/worker.sh"
date --iso-8601=seconds > "$PACKET/prepared_at.txt"
nohup setsid bash "$PACKET/worker.sh" > "$PACKET/worker_outer.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$PACKET/worker.pid"
printf 'PACKET=%s\nRUN=%s\nRELOAD=%s\nWORKER_PID=%s\nFASTWAM_ACTION_DVAC_SMOKE_LAUNCHED\n' "$PACKET" "$RUN" "$RELOAD" "$!"
