set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
sed -n '184,342p' "$root/rlinf/hybrid_engines/fsdp/strategy/base.py"
find /data/chenyiteng/results/rlinf-shenzhen/online-bc/sft-leaf-wrap-nativeopt-local-orig-false-20260905 /data/chenyiteng/results/rlinf-shenzhen/online-bc/sync-probe-orig-false-20260905 -xdev -type f -size +1G -printf '%s\t%p\n'
