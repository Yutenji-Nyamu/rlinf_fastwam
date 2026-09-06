set -u
RT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

echo '=== robotwin source identity ==='
git -C "$RT" status --short --branch || true
git -C "$RT" log -3 --format='%H %s' || true
git -C "$RT" remote -v || true

echo '=== vector env imports/classes ==='
sed -n '1,245p' "$RT/robotwin/envs/vector_env.py"
echo '=== vector env pool/reset/close ==='
sed -n '320,475p' "$RT/robotwin/envs/vector_env.py"
echo '=== base task close ==='
sed -n '1,45p' "$RT/envs/_base_task.py"
sed -n '600,680p' "$RT/envs/_base_task.py"

echo '=== references to clear cache ==='
grep -RIn --include='*.py' -E 'clear_cache_freq|sapien_clear_cache|clear_cache\(' "$RT/robotwin" "$RT/envs" | head -n 100 || true

echo '=== upstream cleanup commit presence ==='
git -C "$RT" branch -a --contains 48e76afa8172ec2b806a690a0c1c9cf26ec045ed 2>/dev/null || true
git -C "$RT" show --stat --oneline 48e76afa8172ec2b806a690a0c1c9cf26ec045ed 2>/dev/null || true
