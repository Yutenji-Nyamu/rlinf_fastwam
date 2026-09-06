set -euo pipefail
v6=/data/chenyiteng/projects/lerobot-sidney/lerobot-30da8e687a6d
v4=/data/chenyiteng/projects/lerobot-sidney/lerobot-0b067df57d21
echo '=== v6 robotwin imports/top ==='
sed -n '1,120p' "$v6/src/lerobot/envs/robotwin.py"
echo '=== v6 class and factory signatures ==='
grep -n '^class RoboTwinEnv\|^def create_robotwin_envs\|^    def __init__\|^    def reset\|^    def step\|^    def close' "$v6/src/lerobot/envs/robotwin.py"
sed -n '360,660p' "$v6/src/lerobot/envs/robotwin.py"
echo '=== v4 env utils/factory ==='
ls "$v4/src/lerobot/envs"
grep -R "def preprocess_observation\|def make_env\|class EnvConfig" -n "$v4/src/lerobot/envs" | head -60 || true
echo '=== v6 config block ==='
sed -n '730,850p' "$v6/src/lerobot/envs/configs.py"
