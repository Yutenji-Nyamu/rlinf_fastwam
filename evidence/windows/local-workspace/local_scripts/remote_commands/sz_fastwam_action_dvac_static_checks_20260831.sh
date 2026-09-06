set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
HEAD=a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7
PROBE=/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/probes/static-a6ad77ea

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FW:$ROBOTWIN"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA

git -C "$WT" diff --check
"$VENV/bin/python" -m py_compile \
  "$WT/rlinf/algorithms/dvac_train_weighting.py" \
  "$WT/rlinf/algorithms/utils.py" \
  "$WT/rlinf/algorithms/losses.py" \
  "$WT/rlinf/models/embodiment/fastwam/fastwam_rl.py" \
  "$WT/rlinf/models/embodiment/fastwam/fastwam_policy.py" \
  "$WT/rlinf/workers/rollout/hf/huggingface_worker.py" \
  "$WT/rlinf/workers/actor/embodied_fsdp_actor_worker.py"
cd "$WT"
"$VENV/bin/python" -m pytest tests/unit_tests/test_dvac_train_weighting.py -q

mkdir -p "$PROBE"
"$VENV/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_move_stapler_pad_grpo_fastwam_action_dvac_adv \
  --cfg job --resolve \
  runner.logger.log_path="$PROBE/run" \
  runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1 \
  env.train.rollout_epoch=1 actor.global_batch_size=256 algorithm.update_epoch=1 \
  > "$PROBE/resolved.yaml"

"$VENV/bin/python" - "$PROBE/resolved.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
assert cfg["algorithm"]["logprob_type"] == "action_level"
dvac = cfg["algorithm"]["dvac_gradient_weighting"]
assert dvac["mode"] == "apply"
assert dvac["application"] == "action_advantage"
assert dvac["selected_l"] == 5
assert dvac["window_steps"] == 5 and dvac["warmup_steps"] == 1
assert dvac["weight_min"] == 0.5 and dvac["weight_max"] == 1.5
assert cfg["env"]["train"]["total_num_envs"] == 32
assert cfg["env"]["train"]["rollout_epoch"] == 1
assert cfg["env"]["eval"]["total_num_envs"] == 32
assert cfg["actor"]["global_batch_size"] == 256
assert cfg["actor"]["micro_batch_size"] == 2
assert cfg["algorithm"]["group_size"] == 8
assert cfg["algorithm"]["update_epoch"] == 1
for owner in ("actor", "rollout"):
    model = cfg[owner]["model"]
    assert (model["model_type"], model["action_horizon"], model["num_action_chunks"], model["num_inference_steps"], model["action_dim"]) == ("fastwam", 32, 24, 10, 14)
assert cfg["actor"]["fsdp_config"]["checkpoint_format"] == "dcp"
print("FASTWAM_ACTION_DVAC_STATIC_COMPOSE_OK")
PY
