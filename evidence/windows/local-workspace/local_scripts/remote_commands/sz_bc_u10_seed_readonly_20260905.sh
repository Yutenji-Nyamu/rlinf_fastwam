set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
cat "$root/rlinf/envs/robotwin/seed_utils.py"
cat "$root/examples/embodiment/config/env/robotwin_adjust_bottle.yaml"
find /data/chenyiteng/results/rlinf-shenzhen/online-bc/sft-leaf-wrap-nativeopt-local-orig-false-20260905 -type f -name '*.pt' -printf '%f %s bytes\n'
