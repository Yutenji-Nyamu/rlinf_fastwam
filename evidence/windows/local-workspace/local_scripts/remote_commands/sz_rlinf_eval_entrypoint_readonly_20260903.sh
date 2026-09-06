set -eu
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
F=$(find "$WT" -type f -name 'eval_embodied_agent.py' -print -quit)
echo "F=$F"
test -s "$F"
nl -ba "$F" | sed -n '1,240p'
echo '=== Eval runner ==='
grep -R -n -C 5 'class EmbodiedEvalRunner' "$WT/rlinf" | head -200
