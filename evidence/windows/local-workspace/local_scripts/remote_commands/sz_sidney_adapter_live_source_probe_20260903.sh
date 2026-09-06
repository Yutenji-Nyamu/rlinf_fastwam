set -eu

ROOT=/data/chenyiteng/projects/rlinf-shenzhen
BASE=$ROOT/worktrees/pi05-robotwin-rl
CANON=$ROOT/RLinf
SIDNEY=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab

echo '=== identity/time ==='
date -Is
id

echo '=== base worktree ==='
git -C "$BASE" status --short --branch
git -C "$BASE" rev-parse HEAD
git -C "$BASE" rev-parse personal/codex/sz-pi05-robotwin-rl
git -C "$BASE" remote -v

echo '=== worktree inventory ==='
git -C "$CANON" worktree list --porcelain

echo '=== target paths ==='
for p in \
  rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi_rlinf.py \
  rlinf/utils/ckpt_convertor/openpi/convert.py \
  rlinf/models/embodiment/openpi/dataconfig/__init__.py \
  rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml; do
  printf '%s\t' "$p"
  git -C "$BASE" ls-files "$p"
done

echo '=== converter tree ==='
find "$BASE/rlinf/utils/ckpt_convertor/openpi" -maxdepth 2 -type f -printf '%P\n' | sort

echo '=== pi05 dataconfig registrations ==='
grep -RInE 'pi05|robotwin|DataConfig|CONFIG' \
  "$BASE/rlinf/models/embodiment/openpi/dataconfig/__init__.py" \
  "$BASE/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py" | head -n 160 || true

echo '=== sidney checkpoint top files ==='
find "$SIDNEY" -maxdepth 2 -type f -printf '%P\t%s\n' | sort | head -n 120

echo '=== relevant tests ==='
find "$BASE/tests" -path '*openpi*' -type f -printf '%P\n' | sort | head -n 160 || true

echo '=== current GPU jobs (read-only) ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true
