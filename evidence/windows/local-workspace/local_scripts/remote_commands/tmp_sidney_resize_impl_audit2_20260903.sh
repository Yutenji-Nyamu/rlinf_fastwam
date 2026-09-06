set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
RLPY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
OPENPI=$($RLPY - <<'PY'
import openpi, pathlib
print(pathlib.Path(openpi.__file__).parent)
PY
)
echo "OPENPI=$OPENPI"
grep -R -n -E "class ModelTransformFactory|class ResizeImages|def resize_with_pad" "$OPENPI" rlinf/models/embodiment/openpi | head -120 || true
for hit in $(grep -R -l -E "class ModelTransformFactory|class ResizeImages|def resize_with_pad" "$OPENPI" | head -20); do
  echo "=== $hit ==="
  grep -n -A90 -B15 -E "class ModelTransformFactory|class ResizeImages|def resize_with_pad" "$hit" | head -220
done
echo '=== P-v6 compare ==='
find /data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v6 -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | sort || true
cat /data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v6/compare.log 2>/dev/null || true
