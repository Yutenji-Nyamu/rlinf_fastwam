set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
robotwin=/root/autodl-tmp/RoboTwin_RLinf
runtime=/root/autodl-tmp/RLinf/.venv/bin/python
config_dir=/root/autodl-tmp/idea2_dvac_run_configs
config_name=idea2_dvac_sft_smoke_2gpu_2env_v1
config_path="$config_dir/$config_name.yaml"
output="$target/outputs/$config_name"
expected_hash=62207051678d2ad515c74aa574282b9bc04592f3b97ea10258d43263483b4f71

cd "$target"
test "$(git rev-parse HEAD)" = 61996e15cc7f5a32bd6012b61b20893d94636c82
test -z "$(git status --short)"
test -f "$config_path"
test "$(sha256sum "$config_path" | awk '{print $1}')" = "$expected_hash"
test ! -e "$output"

REPO_PATH="$target" \
EMBODIED_PATH="$target/examples/embodiment" \
PYTHONPATH="$target:$robotwin" \
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES= \
  "$runtime" evaluations/eval_embodied_agent.py \
    --config-path "$config_dir" \
    --config-name "$config_name" \
    --cfg job \
    --resolve >/dev/null

PYTHONDONTWRITEBYTECODE=1 "$runtime" - "$config_path" <<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
assert cfg.cluster.component_placement == {"env, rollout": "0-1"}
assert cfg.env.eval.total_num_envs == 2
assert cfg.env.eval.rollout_epoch == 1
assert cfg.env.eval.max_episode_steps == 200
assert cfg.rollout.model.num_action_chunks == 50
assert cfg.rollout.model.num_steps == 4
assert cfg.rollout.model.action_dim == 14
assert cfg.rollout.dvac_telemetry.enabled is True
assert cfg.rollout.dvac_telemetry.source_commit == (
    "61996e15cc7f5a32bd6012b61b20893d94636c82"
)
assert cfg.rollout.dvac_telemetry.robotwin_commit == (
    "481380fbd97cbf9ff830aedfb2279851e1e58969"
)
assert cfg.rollout.dvac_telemetry.seed_file_sha256 == (
    "194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f"
)
assert cfg.rollout.dvac_telemetry.renderer == "sapien"
assert "CUDA_VISIBLE_DEVICES=0,1" in cfg.rollout.dvac_telemetry.launch_command
assert "--config-name idea2_dvac_sft_smoke_2gpu_2env_v1" in (
    cfg.rollout.dvac_telemetry.launch_command
)
assert cfg.runner.ckpt_path is None
print("EXTERNAL_SMOKE_CONFIG_CONTRACT_PASS=1")
PY

test ! -e "$output"
printf 'CONFIG_SHA256=%s\n' "$expected_hash"
printf 'ENTRYPOINT_CFG_RESOLVE_PASS=1\n'
printf 'NO_MODEL_OR_ENV_RUN=1\n'
