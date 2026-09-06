set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
runtime=/root/autodl-tmp/RLinf/.venv/bin/python
robotwin=/root/autodl-tmp/RoboTwin_RLinf
expected_head=73da63f01d52290c3537f12fb4bfb55cfd82f94c

date '+SERVER_TIME=%Y-%m-%d %H:%M:%S %:z'
test "$(git -C "$target" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$target" branch --show-current)" = \
  codex/idea2-dvac-pi0-robotwin
test "$(git -C "$target" diff --name-only --diff-filter=D | wc -l)" -eq 0
test -x "$runtime"
test -d "$robotwin"

printf '%s\n' '--- PRETEST-1: source/status/diff ---'
git -C "$target" diff --check
git -C "$target" status --short
git -C "$target" diff --stat

printf '%s\n' '--- PRETEST-2: syntax compile without bytecode writes ---'
PYTHONDONTWRITEBYTECODE=1 "$runtime" - "$target" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = [
    "evaluations/eval_embodied_agent.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/utils/dvac_telemetry.py",
    "rlinf/workers/env/env_worker.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "tests/unit_tests/test_dvac_telemetry.py",
]
for relative_path in paths:
    path = root / relative_path
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"COMPILE_PASS={relative_path}")
PY

printf '%s\n' '--- PRETEST-3: two targeted synthetic tests ---'
cd "$target"
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$target:$robotwin" \
  "$runtime" -m pytest -q tests/unit_tests/test_dvac_telemetry.py

printf '%s\n' '--- PRETEST-4: dedicated Hydra config compose/contract ---'
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
resolved = OmegaConf.to_container(cfg, resolve=True)

assert resolved["runner"]["task_type"] == "embodied_eval"
assert resolved["runner"]["only_eval"] is True
assert resolved["cluster"]["component_placement"] == {"env, rollout": "0-7"}
assert resolved["env"]["eval"]["total_num_envs"] == 128
assert resolved["env"]["eval"]["rollout_epoch"] == 1
assert resolved["env"]["eval"]["max_episode_steps"] == 200
assert resolved["env"]["eval"]["use_fixed_reset_state_ids"] is True
assert resolved["env"]["eval"]["task_config"]["task_name"] == "adjust_bottle"
assert resolved["env"]["eval"]["task_config"]["camera"]["collect_wrist_camera"] is True
assert resolved["rollout"]["model"]["num_action_chunks"] == 50
assert resolved["rollout"]["model"]["action_dim"] == 14
assert resolved["rollout"]["model"]["num_steps"] == 4
assert resolved["rollout"]["model"]["openpi"]["action_env_dim"] == 14
assert resolved["rollout"]["model"]["openpi"]["num_images_in_input"] == 3
assert resolved["rollout"]["dvac_telemetry"]["enabled"] is False
assert resolved["rollout"]["dvac_telemetry"]["save_query_inputs"] is True
assert resolved["rollout"]["dvac_telemetry"]["source_commit"] is None
assert resolved["rollout"]["dvac_telemetry"]["robotwin_commit"] is None
assert resolved["rollout"]["dvac_telemetry"]["seed_file_sha256"] is None
assert resolved["rollout"]["dvac_telemetry"]["renderer"] == "sapien"
assert resolved["rollout"]["dvac_telemetry"]["launch_command"] is None
assert len(resolved["rollout"]["dvac_telemetry"]["checkpoint_revision"]) == 40
assert len(resolved["rollout"]["dvac_telemetry"]["norm_stats_sha256"]) == 64
print("HYDRA_COMPOSE_PASS=1")
print("DEFAULT_TELEMETRY_ENABLED=0")
print("CONTRACT=adjust_bottle,H50,C50,M4,D14,three_cameras,fixed_ids,200_slots")
PY

printf '%s\n' '--- PRETEST-5: post-test tree ---'
test "$(git -C "$target" diff --name-only --diff-filter=D | wc -l)" -eq 0
git -C "$target" status --short
printf 'PRETEST_SCRIPT_COMPLETE=1\n'
