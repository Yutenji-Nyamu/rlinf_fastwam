set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate act

root=/data/chenyiteng/projects/robotwin-native/RoboTwin
act_root="$root/XPolicyLab/policy/ACT"
source_data="$root/data/demo_clean/adjust_bottle/aloha_agilex/data"
processed="$act_root/processed_data/demo_clean/adjust_bottle/aloha_agilex-joint"
run_root=/data/chenyiteng/runs/robotwin-native/adjust_bottle/official_act_adjust_bottle_20260821_v1

cd "$root"
test "$(git rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C XPolicyLab rev-parse HEAD)" = c07a09614dd44cc4a67483bcb9a82e7439d99926
test "$(find "$source_data" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)" -eq 50
if [ -e "$processed" ]; then
  printf 'REFUSE_EXISTING_PROCESSED_TARGET=%s\n' "$processed" >&2
  exit 40
fi

mkdir -p "$run_root"
if [ -f "$act_root/TASK_CONFIGS.json" ]; then
  cp "$act_root/TASK_CONFIGS.json" "$run_root/TASK_CONFIGS.before_preprocess.json"
fi

printf '%s\n' '=== ACT OFFICIAL PREPROCESS ==='
cd "$act_root"
set -o pipefail
timeout --signal=INT --kill-after=60s 7200s \
  bash process_data.sh demo_clean adjust_bottle aloha_agilex joint 2>&1 | \
  tee "$run_root/04_act_preprocess.log"

printf '%s\n' '=== VERIFY ACT PROCESSED DATA ==='
test -d "$processed"
test "$(find "$processed" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)" -eq 50
python - <<'PY'
import json
from pathlib import Path

path = Path("TASK_CONFIGS.json")
data = json.loads(path.read_text())
key = "demo_clean-adjust_bottle-aloha_agilex-joint"
assert key in data, (key, sorted(data))
print("task_config_key=", key)
print("task_config_value=", data[key])
PY
cp TASK_CONFIGS.json "$run_root/TASK_CONFIGS.after_preprocess.json"
du -sh "$processed"
df -h "$root"
git -C "$root" status --short --branch
