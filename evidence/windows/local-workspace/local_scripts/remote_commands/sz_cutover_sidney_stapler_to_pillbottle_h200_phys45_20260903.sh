#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
HEAD=f50e235c5ab1f4390f0ba92bfb13390ed0a86810
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney

OLD_NAME=move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
NEW_NAME=move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
OLD_EXPERIMENT=pi05_sidney_move_grpo_formal100_2gpu64x4_g8_b1024_u2_m10_noise0p5_h200_fixed32_phys45_v1
NEW_EXPERIMENT=pi05_sidney_move_pillbottle_pad_grpo_formal100_2gpu64x4_g8_b1024_u2_m10_noise0p5_h200_fixed32_phys45_v1
OLD_RUN="$ROOT/runs/$OLD_NAME"
OLD_PACKET="$ROOT/packets/$OLD_NAME"
NEW_RUN="$ROOT/runs/$NEW_NAME"
NEW_PACKET="$ROOT/packets/$NEW_NAME"

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$MODEL/model.safetensors"
test -s "$OLD_PACKET/command.txt"
test -s "$OLD_RUN/runtime/resolved.yaml"
test ! -e "$NEW_RUN"
if [[ -d "$NEW_PACKET" && ! -e "$NEW_PACKET/packet_complete.txt" ]]; then
  rm -f -- "$NEW_PACKET/command.txt" "$NEW_PACKET/resolved.yaml" "$NEW_PACKET/contract.json" \
    "$NEW_PACKET/source-head.txt" "$NEW_PACKET/sha256.txt"
  rmdir -- "$NEW_PACKET"
fi
test ! -e "$NEW_PACKET"
test -f "$ROBOTWIN/envs/move_pillbottle_pad.py"
test -f "$ROBOTWIN/description/task_instruction/move_pillbottle_pad.json"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

setup_env() {
  source "$VENV/bin/activate"
  unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
  export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
  export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
  export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
  export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
}

namespace_count() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_sidney_pill_count", logging_level="ERROR")
print(sum(1 for row in ray.util.list_named_actors(all_namespaces=True)
          if row.get("namespace") == os.environ["TARGET_NAMESPACE"]))
ray.shutdown()
PY
}

cleanup_namespace() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 "$VENV/bin/python" - <<'PY'
import os, time, ray
target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_sidney_pill_cleanup", logging_level="ERROR")
def rows():
    return sorted(r["name"] for r in ray.util.list_named_actors(all_namespaces=True)
                  if r.get("namespace") == target)
names = rows()
assert len(names) in (0, 15), (target, names)
managers = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
for name in sorted(names, key=lambda value: (value in managers, value)):
    try:
        ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
    except ValueError:
        pass
for _ in range(120):
    if not rows():
        break
    time.sleep(1)
else:
    raise RuntimeError(rows())
print(f"cleaned_namespace={target}; actors={len(names)}")
ray.shutdown()
PY
}

stop_tree() {
  local root=$1
  mapfile -t pids < <("$VENV/bin/python" - "$root" <<'PY'
import os, sys
root = int(sys.argv[1]); children = {}
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    try:
        fields = open(f"/proc/{name}/stat").read().split()
        children.setdefault(int(fields[3]), []).append(int(name))
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
out = []
def visit(pid):
    for child in children.get(pid, []):
        visit(child)
    out.append(pid)
visit(root)
print("\n".join(map(str, out)))
PY
  )
  for pid in "${pids[@]}"; do
    [[ ! -d "/proc/$pid" ]] || test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
  done
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 90); do
    local alive=0
    for pid in "${pids[@]}"; do [[ -d "/proc/$pid" ]] && alive=$((alive + 1)); done
    (( alive == 0 )) && return 0
    sleep 1
  done
  for pid in "${pids[@]}"; do [[ ! -d "/proc/$pid" ]] || kill -KILL "$pid"; done
}

# Prepare the complete packet while the old run is still using GPU 4/5.
mkdir -p "$NEW_PACKET"
"$VENV/bin/python" - "$OLD_PACKET/command.txt" "$NEW_PACKET/command.txt" \
  "$OLD_RUN" "$NEW_RUN" "$OLD_EXPERIMENT" "$NEW_EXPERIMENT" <<'PY'
import sys
src, dst, old_run, new_run, old_exp, new_exp = sys.argv[1:]
s = open(src, encoding="utf-8").read()
replacements = (
    (old_run, new_run, None),
    (old_exp, new_exp, 1),
    ("env.train.task_config.task_name=move_stapler_pad", "env.train.task_config.task_name=move_pillbottle_pad", 1),
    ("env.eval.task_config.task_name=move_stapler_pad", "env.eval.task_config.task_name=move_pillbottle_pad", 1),
)
counts = {}
for old, new, expected in replacements:
    count = s.count(old)
    if expected is None:
        assert count >= 3, (old, count)
    else:
        assert count == expected, (old, count, expected)
    counts[old] = count
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8").write(s)
print(counts)
PY

setup_env
eval "$(cat "$NEW_PACKET/command.txt") --cfg job --resolve" > "$NEW_PACKET/resolved.yaml"
"$VENV/bin/python" - "$OLD_RUN/runtime/resolved.yaml" "$NEW_PACKET/resolved.yaml" \
  "$NEW_PACKET/contract.json" <<'PY'
import copy, json, sys, yaml
old = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
new = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))

assert new["cluster"]["component_placement"] == {"actor, env, rollout": "4,5"}
assert new["runner"]["max_steps"] == 100
assert new["runner"]["val_check_interval"] == 5 and new["runner"]["save_interval"] == 10
assert new["algorithm"]["group_size"] == 8 and new["algorithm"]["update_epoch"] == 2
assert new["algorithm"]["logprob_type"] == "chunk_level"
assert new["algorithm"]["dvac_gradient_weighting"]["mode"] == "off"
assert new["actor"]["global_batch_size"] == 1024 and new["actor"]["micro_batch_size"] == 32
assert new["actor"]["model"]["num_action_chunks"] == 50
assert new["actor"]["model"]["openpi"]["config_name"] == "pi05_sidney_robotwin"
assert new["actor"]["model"]["openpi"]["num_steps"] == 10
assert new["actor"]["model"]["openpi"]["noise_level"] == 0.5
assert new["actor"]["fsdp_config"]["checkpoint_format"] == "local_shard"
for split, count in (("train", 64), ("eval", 32)):
    env = new["env"][split]
    assert env["total_num_envs"] == count
    assert env["max_episode_steps"] == 200
    assert env["max_steps_per_rollout_epoch"] == 200
    assert env["task_config"]["task_name"] == "move_pillbottle_pad"
    assert env["task_config"]["step_lim"] == 200
assert new["env"]["train"]["rollout_epoch"] == 4

def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        return (value
                .replace("move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1",
                         "move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1")
                .replace("pi05_sidney_move_pillbottle_pad_grpo_formal100_2gpu64x4_g8_b1024_u2_m10_noise0p5_h200_fixed32_phys45_v1",
                         "pi05_sidney_move_grpo_formal100_2gpu64x4_g8_b1024_u2_m10_noise0p5_h200_fixed32_phys45_v1")
                .replace("move_pillbottle_pad", "move_stapler_pad"))
    return value
assert normalize(new) == old, "resolved configs differ beyond task/run naming"

payload = {
    "task": "move_pillbottle_pad", "physical_gpus": [4, 5], "fresh": True,
    "target_steps": 100, "train_envs": 64, "rollout_epochs": 4,
    "trajectories_per_step": 256, "group_size": 8, "groups_per_step": 32,
    "episode_action_limit": 200, "H": 50, "C": 50, "M": 10,
    "max_query_records_per_step": 1024, "global_batch": 1024,
    "micro_batch": 32, "update_epochs": 2, "optimizer_calls_per_step": 2,
    "noise_level": 0.5, "fixed_eval_episodes": 32,
    "eval_interval": 5, "save_interval": 10, "checkpoint_format": "local_shard",
    "only_semantic_change_from_source_run": "task_name: move_stapler_pad -> move_pillbottle_pad",
}
json.dump(payload, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(sys.argv[3], "a", encoding="utf-8").write("\n")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
printf '%s\n' "$HEAD" > "$NEW_PACKET/source-head.txt"
sha256sum "$NEW_PACKET/resolved.yaml" "$NEW_PACKET/command.txt" "$NEW_PACKET/contract.json" > "$NEW_PACKET/sha256.txt"
printf '%s\n' complete > "$NEW_PACKET/packet_complete.txt"

# Exact cutover: terminate only this run and its Ray namespace. Preserve shared Ray and RLinf_1.
test "$(namespace_count RLinf)" = 15
test "$(namespace_count RLinf_1)" = 15
OLD_PID=$(<"$OLD_RUN/runtime/wrapper.pid")
test "$(ps -o user= -p "$OLD_PID" | xargs)" = chenyiteng
ps -o args= -p "$OLD_PID" | grep -F "$OLD_RUN/runtime/wrapper.sh" >/dev/null
test "$(<"$OLD_RUN/runtime/owned.pgid")" = "$OLD_PID"
OLD_OBSERVER=$(cat "$OLD_RUN/runtime/observer.pid" 2>/dev/null || true)
OLD_STEP=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$OLD_RUN/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
stop_tree "$OLD_PID"
if [[ -n "$OLD_OBSERVER" && -d "/proc/$OLD_OBSERVER" ]]; then stop_tree "$OLD_OBSERVER"; fi
cleanup_namespace RLinf
for _ in $(seq 1 120); do
  [[ -z "$(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]] && break
  sleep 1
done
test -z "$(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
test "$(namespace_count RLinf_1)" = 15
cat > "$OLD_RUN/runtime/stopped_by_user_for_move_pillbottle_20260903.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested fresh task replacement: move_stapler_pad -> move_pillbottle_pad
last_complete_step=$OLD_STEP
namespace=RLinf
cleanup=owned process tree and exact namespace only; shared Ray and RLinf_1 unchanged
EOF

# Launch the fresh task immediately on the released pair.
mkdir -p "$NEW_RUN/runtime"
cp "$NEW_PACKET"/{resolved.yaml,contract.json,command.txt,source-head.txt,sha256.txt,packet_complete.txt} "$NEW_RUN/runtime/"
cat > "$NEW_RUN/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; command_file=$2
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 345600s bash -lc "$(cat "$command_file")" > "$runtime/driver.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$runtime/exit_code.txt"; date --iso-8601=seconds > "$runtime/finished_at.txt"; exit "$rc"
WRAP
chmod 700 "$NEW_RUN/runtime/wrapper.sh"
cat > "$NEW_RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
echo 'timestamp,driver_alive,host_mem_available_kib,gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 4,5 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"; sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
chmod 700 "$NEW_RUN/runtime/observer.sh"
printf '%s\n' \
  "started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" "source_head=$HEAD" 'physical_gpus=4,5' \
  'fresh=true target_step=100' 'move_pillbottle_pad H50 C50 M10 horizon200 noise0.5' \
  '64 env x rollout4 = 256 trajectories; G8; max1024 records' \
  'GB1024 MB32 update2; fixed32/eval5; local-shard/save10' > "$NEW_RUN/runtime/launch_manifest.txt"

setup_env
nohup setsid bash "$NEW_RUN/runtime/wrapper.sh" "$NEW_RUN/runtime" "$NEW_RUN/runtime/command.txt" \
  > "$NEW_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
NEW_PID=$!
printf '%s\n' "$NEW_PID" > "$NEW_RUN/runtime/wrapper.pid"
printf '%s\n' "$NEW_PID" > "$NEW_RUN/runtime/owned.pgid"
nohup setsid bash "$NEW_RUN/runtime/observer.sh" "$NEW_PID" "$NEW_RUN/runtime/resource.csv" \
  > "$NEW_RUN/runtime/observer.log" 2>&1 < /dev/null &
NEW_OBSERVER=$!
printf '%s\n' "$NEW_OBSERVER" > "$NEW_RUN/runtime/observer.pid"

COUNT=0
for _ in $(seq 1 360); do
  kill -0 "$NEW_PID"
  COUNT=$(namespace_count RLinf)
  [[ "$COUNT" == 15 ]] && break
  sleep 1
done
test "$COUNT" = 15
test "$(namespace_count RLinf_1)" = 15
printf 'old_last_complete_step=%s\nnew_run=%s\nnew_packet=%s\nnew_wrapper_pid=%s\nnew_observer_pid=%s\n' \
  "$OLD_STEP" "$NEW_RUN" "$NEW_PACKET" "$NEW_PID" "$NEW_OBSERVER"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
tail -n 20 "$NEW_RUN/runtime/driver.log" || true
echo SIDNEY_PI05_PILLBOTTLE_GRPO_CUTOVER_OK
