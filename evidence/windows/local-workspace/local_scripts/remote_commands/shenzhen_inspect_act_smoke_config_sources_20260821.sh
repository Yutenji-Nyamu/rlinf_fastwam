set -euo pipefail

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
XPL="$ROBOTWIN/XPolicyLab"

test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C "$XPL" rev-parse HEAD)" = c07a09614dd44cc4a67483bcb9a82e7439d99926

printf '%s\n' '=== demo_clean.yml ==='
sha256sum "$ROBOTWIN/env_cfg/task_config/demo_clean.yml"
sed -n '1,220p' "$ROBOTWIN/env_cfg/task_config/demo_clean.yml"

printf '%s\n' '=== eval config directory ==='
find "$ROBOTWIN/env_cfg/eval" -maxdepth 1 -type f -printf '%f\n' | sort

printf '%s\n' '=== all_tasks.yml ==='
sha256sum "$ROBOTWIN/env_cfg/eval/all_tasks.yml"
sed -n '1,220p' "$ROBOTWIN/env_cfg/eval/all_tasks.yml"

printf '%s\n' '=== eval_policy multitask help ==='
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin
cd "$ROBOTWIN"
bash scripts/eval_policy.sh multitask --help
