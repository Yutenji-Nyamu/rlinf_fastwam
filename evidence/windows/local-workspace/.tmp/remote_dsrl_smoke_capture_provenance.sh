set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
EVIDENCE="$REPO/docs/rlinf-robotwin-pi0-traditional-rl/evidence"

test -d "$RUN_ROOT"
cp -f \
  "$EVIDENCE/FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml" \
  "$RUN_ROOT/FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml"
cp -f \
  "$EVIDENCE/RESUME_SMOKE_VALIDATED_RESOLVED_20260728.yaml" \
  "$RUN_ROOT/RESUME_SMOKE_VALIDATED_RESOLVED_20260728.yaml"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "repo=$REPO"
  echo "branch=$(git -C "$REPO" branch --show-current)"
  echo "head=$(git -C "$REPO" rev-parse HEAD)"
  echo "upstream=$(git -C "$REPO" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')"
  echo "worktree_status_begin"
  git -C "$REPO" status --short --branch
  echo "worktree_status_end"
  echo "driver_cwd=$(readlink "/proc/$(cat "$RUN_ROOT/fresh.pid")/cwd" 2>/dev/null || true)"
  "$PY" -B -c 'import sys, torch, ray, hydra; print(f"python={sys.version.split()[0]}"); print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"ray={ray.__version__}"); print(f"hydra={hydra.__version__}")'
  sha256sum \
    "$RUN_ROOT/FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml" \
    "$RUN_ROOT/RESUME_SMOKE_VALIDATED_RESOLVED_20260728.yaml"
} > "$RUN_ROOT/run_provenance.txt"

cat /sys/fs/cgroup/memory.events > "$RUN_ROOT/resource_monitor/fresh/memory.events.early.txt"
awk '$1 ~ /^(anon|file|shmem|file_mapped|inactive_file|active_file|kernel_stack|pagetables|slab)$/ {print}' \
  /sys/fs/cgroup/memory.stat > "$RUN_ROOT/resource_monitor/fresh/memory.stat.early.txt"

echo "PROVENANCE_CAPTURED=1"
cat "$RUN_ROOT/run_provenance.txt"
