set -eu
LOG=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/adjust_bottle-5eps-20260902-154311/eval.log
echo '=== log head ==='
sed -n '1,180p' "$LOG"
echo '=== recent bash history ==='
tail -120 /home/chenyiteng/.bash_history 2>/dev/null | grep -E 'lerobot|sidney|robotwin' || true
echo '=== source git status ==='
cd /data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
git status --short
git diff --stat
echo '=== env entrypoints ==='
find src/lerobot -type f -iname '*robotwin*' -o -iname '*eval*.py' | sort
grep -R -n "class.*RoboTwin\|RoboTwin" src/lerobot/envs 2>/dev/null | head -80 || true
