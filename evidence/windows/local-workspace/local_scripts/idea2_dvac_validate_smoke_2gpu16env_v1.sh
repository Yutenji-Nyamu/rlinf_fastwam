set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
robotwin=/root/autodl-tmp/RoboTwin_RLinf
runtime=/root/autodl-tmp/RLinf/.venv/bin/python
run_id=idea2_dvac_sft_smoke_2gpu_16env_v1
config_dir=/root/autodl-tmp/idea2_dvac_run_configs
config="$config_dir/$run_id.yaml"
runtime_dir=/root/autodl-tmp/idea2_dvac_runtime/$run_id
output="$target/outputs/$run_id"
expected_config_sha=d4b7393e1a02f118f6ea5ab2d303f92e83779d3e2a968a7920aebadbd35bc018

test "$(git -C "$target" rev-parse HEAD)" = 61996e15cc7f5a32bd6012b61b20893d94636c82
test -z "$(git -C "$target" status --porcelain)"
test "$(sha256sum "$config" | cut -d' ' -f1)" = "$expected_config_sha"
test ! -e "$output"
bash -n "$runtime_dir/observe_resources.sh"
bash -n "$runtime_dir/launch.sh"
bash -n "$runtime_dir/start.sh"
if grep -E '(^|[^[:alnum:]_])(kill|pkill|timeout|TERM|INT)([^[:alnum:]_]|$)' \
  "$runtime_dir/observe_resources.sh" "$runtime_dir/launch.sh" "$runtime_dir/start.sh"; then
  printf 'FORBIDDEN_BEHAVIOR_CONTROL_FOUND=1\n' >&2
  exit 1
fi

cd "$target"
REPO_PATH="$target" \
EMBODIED_PATH="$target/examples/embodiment" \
PYTHONPATH="$target:$robotwin" \
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES= \
  "$runtime" evaluations/eval_embodied_agent.py \
    --config-path "$config_dir" \
    --config-name "$run_id" \
    --cfg job \
    --resolve >/dev/null

PYTHONDONTWRITEBYTECODE=1 "$runtime" - "$config" <<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
assert cfg.runner.task_type == "embodied_eval"
assert cfg.runner.only_eval is True
assert cfg.runner.ckpt_path is None
assert cfg.cluster.component_placement == {"env, rollout": "0-1"}
assert cfg.env.eval.total_num_envs == 16
assert cfg.env.eval.rollout_epoch == 1
assert cfg.env.eval.max_episode_steps == 200
assert cfg.env.eval.max_steps_per_rollout_epoch == 200
assert cfg.env.eval.use_fixed_reset_state_ids is True
assert cfg.env.eval.video_cfg.save_video is True
assert cfg.rollout.model.num_action_chunks == 50
assert cfg.rollout.model.num_steps == 4
assert cfg.rollout.model.action_dim == 14
assert cfg.rollout.model.openpi.action_chunk == 50
assert cfg.rollout.model.openpi.action_env_dim == 14
assert cfg.rollout.model.openpi.num_images_in_input == 3
assert cfg.rollout.dvac_telemetry.enabled is True
assert cfg.rollout.dvac_telemetry.run_id == "idea2_dvac_sft_smoke_2gpu_16env_v1"
assert "timeout" not in cfg.rollout.dvac_telemetry.launch_command
assert "kill" not in cfg.rollout.dvac_telemetry.launch_command
assert "--config-name idea2_dvac_sft_smoke_2gpu_16env_v1" in (
    cfg.rollout.dvac_telemetry.launch_command
)
assert cfg.runner.logger.log_path.endswith("idea2_dvac_sft_smoke_2gpu_16env_v1")
print("SMOKE16_CONFIG_CONTRACT_PASS=1")
print("EXPECTED_EPISODES=16")
print("EXPECTED_POLICY_QUERIES=64")
print("RESOURCE_MONITOR_MODE=OBSERVE_ONLY")
PY

test ! -e "$output"
printf 'CONFIG_SHA256=%s\n' "$expected_config_sha"
printf 'ENTRYPOINT_CFG_RESOLVE_PASS=1\nNO_MODEL_OR_ENV_RUN=1\n'
