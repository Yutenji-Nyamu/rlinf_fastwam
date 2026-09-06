set -u
F=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/rlinf/workers/env/env_worker.py
date '+TIME %Y-%m-%d %H:%M:%S %Z'
for range in 105,175 229,278 420,470 1325,1390 1420,1475; do
  s=${range%,*}; e=${range#*,}
  echo "SECTION $s-$e"
  nl -ba "$F" | sed -n "${s},${e}p"
done
