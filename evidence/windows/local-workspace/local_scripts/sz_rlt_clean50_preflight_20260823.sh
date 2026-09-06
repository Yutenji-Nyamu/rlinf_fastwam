#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
RLT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

echo '=== CONVERTERS ==='
for file in \
  "$ROBOTWIN/policy/pi0/scripts/process_data.py" \
  "$ROBOTWIN/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py" \
  "$RLT/examples/sft/config/robotwin_rlt_stage1_sft_openpi_current_ar.yaml" \
  "$RLT/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current.yaml"
do
  if [ -f "$file" ]; then
    echo "FILE $file"
  else
    echo "MISSING $file"
  fi
done

echo '=== LOCAL_PARENT_PIN_OBJECT ==='
if git -C "$ROBOTWIN" cat-file -e c3ddfa8b97d5519efa828b075999bd0006778e5e^{commit} 2>/dev/null; then
  echo 'PIN_COMMIT_PRESENT'
  for rel in \
    policy/pi0/scripts/process_data.py \
    policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
  do
    printf '%s ' "$rel"
    git -C "$ROBOTWIN" show "c3ddfa8b97d5519efa828b075999bd0006778e5e:$rel" | sha256sum
  done
else
  echo 'PIN_COMMIT_MISSING'
fi

echo '=== PYTHON_DEPS ==='
"$PY" -B - <<'PY'
import cv2, h5py, huggingface_hub, pyarrow
print('cv2', cv2.__version__)
print('h5py', h5py.__version__)
print('huggingface_hub', huggingface_hub.__version__)
print('pyarrow', pyarrow.__version__)
try:
    import lerobot
    print('lerobot', getattr(lerobot, '__version__', 'import-ok'))
except Exception as exc:
    print('lerobot_import_error', type(exc).__name__, str(exc))
    raise
PY

echo '=== SOURCE_SIGNATURES ==='
grep -nE 'def data_transform|def create_empty_dataset|def populate_dataset|DEFAULT_DATASET_CONFIG' \
  "$ROBOTWIN/policy/pi0/scripts/process_data.py" \
  "$ROBOTWIN/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py" || true

echo '=== CONVERTER_SEARCH ==='
find /data/chenyiteng/projects -type f \( -name 'process_data.py' -o -name 'convert_aloha_data_to_lerobot_robotwin.py' \) -print 2>/dev/null | sort
find /data/chenyiteng/projects/robotwin-native/RoboTwin/XPolicyLab/policy -type f -iname '*aloha*lerobot*.py' -print 2>/dev/null | sort

echo '=== CURRENT_XPOLICY_PI0_PROCESS ==='
sha256sum /data/chenyiteng/projects/robotwin-native/RoboTwin/XPolicyLab/policy/Pi_0/openpi/scripts/process_data.py

echo '=== PINNED_OFFICIAL_CONVERTERS ==='
for rel in \
  policy/pi0/scripts/process_data.py \
  policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
do
  url="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/$rel"
  printf '%s ' "$rel"
  curl --proxy http://127.0.0.1:7890 -fsSL --connect-timeout 10 --max-time 30 "$url" | sha256sum
done
