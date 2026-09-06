set -euo pipefail
LR=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
cd "$LR"
echo '=== factory pretrained load ==='
sed -n '260,360p' src/lerobot/policies/factory.py
echo '=== pi05 policy inference ==='
sed -n '900,1010p' src/lerobot/policies/pi05/modeling_pi05.py
sed -n '1150,1295p' src/lerobot/policies/pi05/modeling_pi05.py
echo '=== pi05 processor ==='
sed -n '1,300p' src/lerobot/policies/pi05/processor_pi05.py
echo '=== input adapter test ==='
sed -n '1,170p' tests/processor/test_pi05_processor.py
echo '=== source config/process files ==='
find /data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab -maxdepth 1 -type f -printf '%f\n' | sort
