set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
robotwin=/root/autodl-tmp/RoboTwin_RLinf
runtime=/root/autodl-tmp/RLinf/.venv/bin/python

cd "$target"
test "$(git rev-parse HEAD)" = 73da63f01d52290c3537f12fb4bfb55cfd82f94c
test -z "$(git status --short)"

EMBODIED_PATH="$target/examples/embodiment" \
REPO_PATH="$target" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$target:$robotwin" \
  "$runtime" - "$target" <<'PY'
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

root = Path(sys.argv[1])
config_dir = root / "evaluations" / "robotwin"
with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
    cfg = compose(config_name="robotwin_adjust_bottle_openpi_dvac_eval")

run_id = "idea2_dvac_sft_smoke_2gpu_2env_v1"
output = root / "outputs" / run_id
cfg.cluster.component_placement = {"env, rollout": "0-1"}
cfg.runner.logger.log_path = str(output)
cfg.runner.logger.experiment_name = run_id
cfg.env.eval.total_num_envs = 2
cfg.env.eval.assets_path = "/root/autodl-tmp/RoboTwin_RLinf"
cfg.rollout.model.model_path = (
    "/root/autodl-tmp/models/rlinf/"
    "RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
)
cfg.rollout.dvac_telemetry.enabled = True
cfg.rollout.dvac_telemetry.run_id = run_id
cfg.rollout.dvac_telemetry.source_commit = (
    "73da63f01d52290c3537f12fb4bfb55cfd82f94c"
)
OmegaConf.resolve(cfg)

assert cfg.cluster.component_placement == {"env, rollout": "0-1"}
assert cfg.env.eval.total_num_envs == 2
assert cfg.env.eval.rollout_epoch == 1
assert cfg.env.eval.max_episode_steps == 200
assert cfg.rollout.model.num_action_chunks == 50
assert cfg.rollout.model.num_steps == 4
assert cfg.rollout.model.action_dim == 14
assert cfg.rollout.dvac_telemetry.enabled is True
assert cfg.rollout.dvac_telemetry.output_dir == f"{output}/dvac_telemetry"
assert cfg.rollout.dvac_telemetry.source_commit == (
    "73da63f01d52290c3537f12fb4bfb55cfd82f94c"
)
assert not output.exists()

print("SMOKE_CANDIDATE_CONTRACT_PASS=1")
print("EXPECTED_EPISODES=2")
print("EXPECTED_POLICY_QUERIES=8")
print("--- RESOLVED YAML BEGIN ---")
print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False), end="")
print("--- RESOLVED YAML END ---")
PY
