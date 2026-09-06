set -euo pipefail
native=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin
rlrt=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
lerobot=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat

echo '=== native vendored identity ==='
git -C /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711 ls-files -s third_party/RoboTwin | head -20 || true
test -e "$native/.git" && ls -ld "$native/.git" || true

echo '=== base task stability test ==='
nl -ba "$native/envs/_base_task.py" | sed -n '120,180p'

echo '=== lerobot direct reset ==='
nl -ba "$lerobot/src/lerobot/envs/robotwin.py" | sed -n '45,65p;350,390p;455,525p'

echo '=== rlinf retry ==='
nl -ba "$rlrt/robotwin/envs/vector_env.py" | sed -n '135,182p'

echo '=== rlinf seed paths and config ==='
cfg=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rl-current/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml
test -f "$cfg" && grep -nE 'seeds_path|max_episode_steps|step_lim|task_name|group_size|use_fixed_reset' "$cfg" || true

echo '=== source hashes ==='
sha256sum "$native/envs/_base_task.py" "$lerobot/src/lerobot/envs/robotwin.py" "$rlrt/robotwin/envs/vector_env.py"
