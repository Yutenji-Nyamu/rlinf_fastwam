set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
NATIVE_VENV=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
NATIVE_SRC=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
NATIVE_MODEL=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
MODEL=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
PACKET="$ROOT/packets/parity-m10-phys4-v5-hfhome"
RUN="$ROOT/smokes/parity-m10-phys4-v5-hfhome"
HEAD=1f4d35c1335d759b9a2e8264e40c136ca08f9ce8

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$PACKET"
test ! -e "$RUN"
mkdir -p "$PACKET" "$RUN/runtime"
"$VENV/bin/python" "$WT/toolkits/lerobot/sidney_pi05_parity.py" prepare \
  --output "$RUN/input.npz" --seed 1234 --prompt 'move the stapler onto the pad'
cat > "$PACKET/commands.txt" <<EOF
env -u http_proxy -u HTTP_PROXY -u https_proxy -u HTTPS_PROXY -u all_proxy -u ALL_PROXY CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$NATIVE_SRC/src HF_HOME=/data/chenyiteng/cache/huggingface-sidney HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $NATIVE_VENV/bin/python $WT/toolkits/lerobot/sidney_pi05_parity.py export-native --input $RUN/input.npz --model $NATIVE_MODEL --source-revision e49e2ab6c11f07511573b67261bd129e88d0a416 --output $RUN/native.pt --device cuda:0 > $RUN/native.log 2>&1
env -u http_proxy -u HTTP_PROXY -u https_proxy -u HTTPS_PROXY -u all_proxy -u ALL_PROXY CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$WT:$ROBOTWIN HF_HOME=/data/chenyiteng/cache/huggingface-sidney HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi $VENV/bin/python $WT/toolkits/lerobot/sidney_pi05_parity.py export-rlinf --input $RUN/input.npz --model $MODEL --output $RUN/rlinf.pt --device cuda:0 > $RUN/rlinf.log 2>&1
$VENV/bin/python $WT/toolkits/lerobot/sidney_pi05_parity.py compare --native $RUN/native.pt --rlinf $RUN/rlinf.pt --rtol 1e-2 --atol 5e-3 > $RUN/report.json
EOF
cat > "$PACKET/contract.json" <<'EOF'
{
  "kind": "native LeRobot versus current RLinf semantic action parity",
  "physical_gpus": [4],
  "execution": "sequential; models never co-resident",
  "prompt": "move the stapler onto the pad",
  "input_seed": 1234,
  "noise_shape": [1, 50, 32],
  "H": 50,
  "M": 10,
  "compared": ["three_images", "image_masks", "normalized_state14", "padded_state32", "tokens", "token_mask", "model_actions32", "final_actions14"],
  "sample_rtol": 0.01,
  "sample_atol": 0.005,
  "native_tokenizer_cache": "/data/chenyiteng/cache/huggingface-sidney"
}
EOF
printf '%s\n' "$HEAD" > "$PACKET/source-head.txt"
printf '%s\n' prepared > "$PACKET/packet-complete.txt"
cat > "$PACKET/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/parity-m10-phys4-v5-hfhome
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v5-hfhome
test "$(git -C "$WT" rev-parse HEAD)" = 1f4d35c1335d759b9a2e8264e40c136ca08f9ce8
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPU4 is occupied; refusing Sidney parity' >&2
  exit 20
fi
monitor() {
  while true; do
    date --iso-8601=seconds
    nvidia-smi -i 4 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    awk '/MemAvailable:/ {print}' /proc/meminfo
    sleep 30
  done
}
monitor > "$RUN/runtime/resources.log" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
set +e
bash -e "$PACKET/commands.txt" > "$RUN/runtime/wrapper.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/runtime/exit_code.txt"
exit "$RC"
EOF
chmod 0755 "$PACKET/run.sh"
sha256sum "$PACKET/run.sh" > "$PACKET/run-sha256.txt"
cat "$PACKET/run-sha256.txt"
