set -euo pipefail
dst=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
for f in \
  src/lerobot/policies/factory.py \
  src/lerobot/policies/pi0_fast/modeling_pi0_fast.py \
  src/lerobot/policies/pretrained.py \
  src/lerobot/policies/pi0/modeling_pi0.py \
  src/lerobot/policies/pi05/modeling_pi05.py \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  src/lerobot/policies/evo1/modeling_evo1.py
do
  echo "=== $f ==="
  grep -n -E '^from typing import' "$dst/$f" | head -3 | cat -A
done
echo '=== patch bytes/head ==='
file /data/chenyiteng/projects/lerobot-sidney/lerobot_v060_py310_unpack_compat.patch
sed -n '1,18p' /data/chenyiteng/projects/lerobot-sidney/lerobot_v060_py310_unpack_compat.patch | cat -A
