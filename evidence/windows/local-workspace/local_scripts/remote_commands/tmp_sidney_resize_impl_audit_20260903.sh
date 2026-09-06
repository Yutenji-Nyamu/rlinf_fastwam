set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
LR=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
RLPY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$LR"
echo '=== LeRobot resize definitions ==='
grep -R -n "def resize_with_pad_torch" src/lerobot | head -20
file=$(grep -R -l "def resize_with_pad_torch" src/lerobot | head -1)
line=$(grep -n "def resize_with_pad_torch" "$file" | head -1 | cut -d: -f1)
start=$((line-20)); end=$((line+100)); sed -n "${start},${end}p" "$file"
echo '=== RLinf/OpenPI model transforms ==='
cd "$WT"
grep -R -n -E "class ModelTransformFactory|ResizeImages|resize_with_pad" rlinf/models/embodiment/openpi /data/chenyiteng/projects/rlinf-shenzhen/openpi 2>/dev/null | head -100 || true
$RLPY - <<'PY'
import inspect
import openpi.transforms as transforms
import openpi.models.model as model
print('ModelTransformFactory')
print(inspect.getsource(model.ModelTransformFactory))
for name in ('ResizeImages','resize_with_pad'):
    obj=getattr(transforms,name,None)
    if obj is not None:
        print(name,inspect.getsource(obj))
PY
echo '=== P-v6 reports ==='
find /data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v6 -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | sort || true
tail -120 /data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v6/compare.log 2>/dev/null || true
