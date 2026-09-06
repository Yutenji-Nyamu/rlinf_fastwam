set -eu
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
grep -R -l -E 'only_eval:[[:space:]]*(True|true)' "$WT/examples" | head -30 | while read -r f; do
  echo "=== $f ==="
  grep -n -C 4 -E 'component_placement|only_eval|weight_sync|rollout:|actor:' "$f" | head -100
done
echo '=== runner only_eval sync calls ==='
grep -R -n -C 4 -E 'sync_model_(from_actor|to_rollout)|only_eval' "$WT/rlinf/runners" "$WT/rlinf/workers" | head -240
