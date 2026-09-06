set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
git -C "$root" rev-parse HEAD
git -C "$root" branch --show-current
git -C "$root" status --porcelain
sha256sum "$root/examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml" "$root/tests/unit_tests/test_online_bc.py" "$root/rlinf/envs/robotwin/robotwin_env.py"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
free -h
df -B1 /data /home
cat /proc/pressure/memory
test ! -e /data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8
test ! -e /home/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-bc32x1-b1024-u10-eval8x4-gpu6-formal100-20260905-v2
test -z "$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits)"
