set -eu
ROOT=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
grep -R -n -A120 -B10 '^def preprocess_observation' "$ROOT/src/lerobot/envs" | head -220
