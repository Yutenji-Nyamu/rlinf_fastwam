set -euo pipefail

RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
OLD_SLUG=fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
NEW_SLUG=fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v1
PACKET_ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets
RUN_ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs
OLD_PACKET="$PACKET_ROOT/$OLD_SLUG"
OLD_RUN="$RUN_ROOT/$OLD_SLUG"
NEW_PACKET="$PACKET_ROOT/$NEW_SLUG"
NEW_RUN="$RUN_ROOT/$NEW_SLUG"

test "$(git -C "$WT" rev-parse HEAD)" = 7b2331c55d14397cfb4cb16181470ddc8afae44a
test -z "$(git -C "$WT" status --porcelain)"
test -f "$OLD_PACKET/command.txt"
test -f "$OLD_PACKET/resolved.yaml"
test -f "$OLD_PACKET/contract.json"
if test -e "$NEW_PACKET" || test -e "$NEW_RUN"; then
  test ! -f "$NEW_RUN/runtime/wrapper.pid"
  test ! -f "$NEW_RUN/runtime/exit_code.txt"
  test ! -f "$NEW_RUN/runtime/driver.log"
fi
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

for gpu_index in 6 7; do
  gpu_used=$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "$gpu_used" -lt 256
done

mkdir -p "$NEW_PACKET" "$NEW_RUN/runtime"
cp -a "$OLD_PACKET/." "$NEW_PACKET/"
cp "$OLD_RUN/runtime/wrapper.sh" "$NEW_RUN/runtime/wrapper.sh"
cp "$OLD_RUN/runtime/observer.sh" "$NEW_RUN/runtime/observer.sh"

sed -i "s/$OLD_SLUG/$NEW_SLUG/g" \
  "$NEW_PACKET/command.txt" "$NEW_PACKET/resolved.yaml"
sed -i \
  -e 's/env\.train\.rollout_epoch=4/env.train.rollout_epoch=8/' \
  -e 's/actor\.global_batch_size=1024/actor.global_batch_size=2048/' \
  "$NEW_PACKET/command.txt"
sed -i \
  -e '/^    rollout_epoch: 4$/s/4/8/' \
  -e '/^  global_batch_size: 1024$/s/1024/2048/' \
  "$NEW_PACKET/resolved.yaml"

sed -i \
  -e 's/"rollout_epochs": 4/"rollout_epochs": 8/' \
  -e 's/"trajectories_per_step": 128/"trajectories_per_step": 256/' \
  -e 's/"groups_per_step": 16/"groups_per_step": 32/' \
  -e 's/"max_query_records": 1024/"max_query_records": 2048/' \
  -e 's/"global_batch": 1024/"global_batch": 2048/' \
  -e 's/"presentations_per_step": 2048/"presentations_per_step": 4096/' \
  -e 's/"trajectories": 12800/"trajectories": 25600/' \
  -e 's/"max_query_records": 102400/"max_query_records": 204800/' \
  -e 's/"presentations": 204800/"presentations": 409600/' \
  "$NEW_PACKET/contract.json"

sha256sum "$NEW_PACKET/command.txt" > "$NEW_PACKET/command.sha256"
sha256sum "$NEW_PACKET/resolved.yaml" > "$NEW_PACKET/resolved.sha256"
cp "$NEW_PACKET/command.txt" "$NEW_RUN/runtime/command.txt"
cp "$NEW_PACKET/contract.json" "$NEW_RUN/runtime/contract.json"
cp "$NEW_PACKET/resolved.yaml" "$NEW_RUN/runtime/resolved.yaml"
cp "$NEW_PACKET/source_head.txt" "$NEW_RUN/runtime/source_head.txt"
cp "$NEW_PACKET/packet_complete.txt" "$NEW_RUN/runtime/packet_complete.txt"

cp "$OLD_RUN/runtime/launch_manifest.txt" "$NEW_RUN/runtime/launch_manifest.txt"
launch_time=$(date --iso-8601=seconds)
sed -i \
  -e "s/^started_at=.*/started_at=$launch_time/" \
  -e 's/^sampling=.*/sampling=32 env x rollout8 = 256 trajectories\/step; G8; 32 groups; max2048 query records/' \
  -e 's/^optimization=.*/optimization=GB2048\/MB2\/update2; 2 optimizer calls; 4096 presentations\/step; lr5e-6/' \
  "$NEW_RUN/runtime/launch_manifest.txt"

test "$(grep -o 'env.train.rollout_epoch=8' "$NEW_PACKET/command.txt" | wc -l)" -eq 1
test "$(grep -o 'actor.global_batch_size=2048' "$NEW_PACKET/command.txt" | wc -l)" -eq 1
test "$(grep -c '^    rollout_epoch: 8$' "$NEW_PACKET/resolved.yaml")" -eq 1
test "$(grep -c '^  global_batch_size: 2048$' "$NEW_PACKET/resolved.yaml")" -eq 1
test "$(grep -c 'rollout_epoch=4\|global_batch_size=1024' "$NEW_PACKET/command.txt" || true)" -eq 0
if grep -q "$OLD_SLUG" "$NEW_PACKET/command.txt" "$NEW_PACKET/resolved.yaml"; then
  exit 1
fi

printf 'COMMAND_DIFF\n'
diff -u "$OLD_PACKET/command.txt" "$NEW_PACKET/command.txt" || true
printf 'RESOLVED_RELEVANT\n'
grep -nE 'rollout_epoch:|global_batch_size:|log_path:|video_base_dir:|save_path:' "$NEW_PACKET/resolved.yaml"
printf 'CONTRACT\n'
cat "$NEW_PACKET/contract.json"

export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN:${PYTHONPATH:-}"
export HF_HOME=/data/chenyiteng/cache/huggingface
export TRANSFORMERS_CACHE=/data/chenyiteng/cache/huggingface/hub
export XDG_CACHE_HOME=/data/chenyiteng/cache
cd "$WT"

nohup setsid bash "$NEW_RUN/runtime/wrapper.sh" "$NEW_RUN/runtime" \
  > "$NEW_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$NEW_RUN/runtime/wrapper.pid"
sleep 2
test -d "/proc/$wrapper_pid"
wrapper_pgid=$(ps -o pgid= -p "$wrapper_pid" | tr -d ' ')
printf '%s\n' "$wrapper_pgid" > "$NEW_RUN/runtime/owned.pgid"

nohup setsid bash "$NEW_RUN/runtime/observer.sh" "$wrapper_pid" "$NEW_RUN/runtime/resource.csv" \
  > "$NEW_RUN/runtime/observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$NEW_RUN/runtime/observer.pid"

printf 'LAUNCHED wrapper_pid=%s wrapper_pgid=%s observer_pid=%s\n' "$wrapper_pid" "$wrapper_pgid" "$observer_pid"
sleep 8
ps -o user,pid,ppid,pgid,etimes,rss,stat,args -p "$wrapper_pid" "$observer_pid"
tail -n 50 "$NEW_RUN/runtime/driver.log" 2>/dev/null || true
