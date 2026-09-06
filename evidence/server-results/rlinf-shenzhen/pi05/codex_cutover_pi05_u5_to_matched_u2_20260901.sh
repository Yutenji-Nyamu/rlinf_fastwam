#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
HEAD=256eeeb4459b4bd5db85bfc6a0eb315771e8c38c
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05

OLD_CONTROL_NAME=pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1
OLD_DVAC_NAME=pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1
NEW_CONTROL_NAME=pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2
NEW_DVAC_NAME=pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys67-localshard-v2

OLD_CONTROL="$ROOT/runs/$OLD_CONTROL_NAME"
OLD_DVAC="$ROOT/runs/$OLD_DVAC_NAME"
NEW_CONTROL="$ROOT/runs/$NEW_CONTROL_NAME"
NEW_DVAC="$ROOT/runs/$NEW_DVAC_NAME"
OLD_CONTROL_PACKET="$ROOT/packets/$OLD_CONTROL_NAME"
OLD_DVAC_PACKET="$ROOT/packets/$OLD_DVAC_NAME"
NEW_CONTROL_PACKET="$ROOT/packets/$NEW_CONTROL_NAME"
NEW_DVAC_PACKET="$ROOT/packets/$NEW_DVAC_NAME"

OLD_CONTROL_EXPERIMENT=robotwin_pi05_grpo_control_formal100_2gpu64x4_g8_b512_u5_m5_fixed32_eval5_phys45_localshard_v1
OLD_DVAC_EXPERIMENT=robotwin_pi05_grpo_dvac_action_adv_w0p5to1p5_formal100_2gpu64x4_g8_b512_u5_m5_fixed32_eval5_phys67_localshard_v1
NEW_CONTROL_EXPERIMENT=robotwin_pi05_grpo_control_formal100_2gpu64x4_g8_b1024_u2_m5_fixed32_eval5_phys45_localshard_v2
NEW_DVAC_EXPERIMENT=robotwin_pi05_grpo_dvac_action_adv_w0p5to1p5_formal100_2gpu64x4_g8_b1024_u2_m5_fixed32_eval5_phys67_localshard_v2

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$NEW_CONTROL"; test ! -e "$NEW_DVAC"
for packet in "$NEW_CONTROL_PACKET" "$NEW_DVAC_PACKET"; do
  if [[ -d "$packet" && ! -e "$packet/packet_complete.txt" ]]; then rm -rf -- "$packet"; fi
  [[ ! -e "$packet" || -s "$packet/packet_complete.txt" ]]
done
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

setup_env() {
  source "$VENV/bin/activate"
  unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
  export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
  export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
  export PYTHONPATH="$WT:$ROBOTWIN" OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
}

prepare_packet() {
  local old_packet=$1 new_packet=$2 old_run=$3 new_run=$4 old_exp=$5 new_exp=$6 kind=$7
  if [[ -s "$new_packet/packet_complete.txt" ]]; then
    echo "reuse_complete_packet=$new_packet"
    return 0
  fi
  mkdir -p "$new_packet"
  "$VENV/bin/python" - "$old_packet/command.txt" "$new_packet/command.txt" "$old_run" "$new_run" "$old_exp" "$new_exp" <<'PY'
import sys
src, dst, old_run, new_run, old_exp, new_exp = sys.argv[1:]
s = open(src, encoding="utf-8").read()
for old, new, expected in (
    (old_run, new_run, 5),
    (old_exp, new_exp, 1),
    ("algorithm.update_epoch=5", "algorithm.update_epoch=2", 1),
    ("actor.global_batch_size=512", "actor.global_batch_size=1024", 1),
):
    count = s.count(old)
    assert count == expected, (old, count, expected)
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8").write(s)
PY
  setup_env
  eval "$(cat "$new_packet/command.txt") --cfg job --resolve" > "$new_packet/resolved.yaml"
  "$VENV/bin/python" - "$old_packet/resolved.yaml" "$new_packet/resolved.yaml" "$new_packet/contract.json" "$kind" <<'PY'
import json, sys, yaml
old = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
new = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
assert new["algorithm"]["update_epoch"] == 2
assert new["actor"]["global_batch_size"] == 1024
assert new["actor"]["micro_batch_size"] == 32
assert new["env"]["train"]["total_num_envs"] == 64
assert new["env"]["train"]["rollout_epoch"] == 4
assert new["env"]["eval"]["total_num_envs"] == 32
assert new["algorithm"]["group_size"] == 8
assert new["actor"]["model"]["num_steps"] == 5
assert new["actor"]["optim"]["lr"] == 5e-6
assert new["runner"]["val_check_interval"] == 5
assert new["runner"]["save_interval"] == 10
assert new["runner"]["resume_dir"] is None
assert new["actor"]["fsdp_config"]["checkpoint_format"] == "local_shard"

def flatten(v, p=()):
    out = {}
    if isinstance(v, dict):
        for k, x in v.items(): out.update(flatten(x, p + (str(k),)))
    elif isinstance(v, list):
        for i, x in enumerate(v): out.update(flatten(x, p + (str(i),)))
    else: out[".".join(p)] = v
    return out
a, b = flatten(old), flatten(new)
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
allowed = (
    "algorithm.update_epoch", "actor.global_batch_size",
    "runner.logger.log_path", "runner.logger.experiment_name", "runner.output_dir",
    "algorithm.dvac_gradient_weighting.output_dir",
    "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
)
unexpected = [k for k in diff if not k.startswith(allowed)]
assert not unexpected, unexpected
assert "algorithm.update_epoch" in diff and "actor.global_batch_size" in diff
payload = {
    "kind": sys.argv[4], "fresh_start": True, "source_head": "256eeeb4459b4bd5db85bfc6a0eb315771e8c38c",
    "common": {"train_envs": 64, "rollout_epochs": 4, "trajectories_per_step": 256,
               "max_query_records": 1024, "eval_envs": 32, "global_batch": 1024,
               "micro_batch": 32, "update_epochs": 2, "optimizer_steps_per_outer": 2,
               "group_size": 8, "denoise_steps": 5, "eval_every": 5,
               "save_every": 10, "checkpoint_format": "local_shard"},
    "old_to_new_diff": diff, "unexpected": unexpected,
}
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2); f.write("\n")
print(json.dumps({"kind": sys.argv[4], "diff": diff, "unexpected": unexpected}, ensure_ascii=False))
PY
  printf '%s\n' "$HEAD" > "$new_packet/source_head.txt"
  printf '%s\n' prepared > "$new_packet/packet_complete.txt"
}

# Prepare and resolve both packets while the old jobs are still using the GPUs.
prepare_packet "$OLD_CONTROL_PACKET" "$NEW_CONTROL_PACKET" "$OLD_CONTROL" "$NEW_CONTROL" "$OLD_CONTROL_EXPERIMENT" "$NEW_CONTROL_EXPERIMENT" 'pi0.5 GRPO Control matched GB1024/update2'
prepare_packet "$OLD_DVAC_PACKET" "$NEW_DVAC_PACKET" "$OLD_DVAC" "$NEW_DVAC" "$OLD_DVAC_EXPERIMENT" "$NEW_DVAC_EXPERIMENT" 'pi0.5 GRPO-DVAC Action-Adv [0.5,1.5] matched GB1024/update2'

"$VENV/bin/python" - "$NEW_CONTROL_PACKET/resolved.yaml" "$NEW_DVAC_PACKET/resolved.yaml" <<'PY'
import sys, yaml
a = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
b = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
for cfg in (a, b):
    assert cfg["algorithm"]["update_epoch"] == 2
    assert cfg["actor"]["global_batch_size"] == 1024
assert a["algorithm"]["logprob_type"] == "chunk_level"
assert a["algorithm"]["dvac_gradient_weighting"]["mode"] == "off"
w = b["algorithm"]["dvac_gradient_weighting"]
assert b["algorithm"]["logprob_type"] == "action_level"
assert (w["mode"], w["application"], w["weight_min"], w["weight_max"]) == ("apply", "action_advantage", 0.5, 1.5)
print("NEW_PAIR_RESOLVED_OK")
PY

namespace_count() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_pi05_u2_count", logging_level="ERROR")
print(sum(1 for r in ray.util.list_named_actors(all_namespaces=True) if r.get("namespace") == os.environ["TARGET_NAMESPACE"]))
ray.shutdown()
PY
}

cleanup_namespace() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 "$VENV/bin/python" - <<'PY'
import os, time, ray
target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_pi05_u2_cleanup", logging_level="ERROR")
def rows(): return sorted(r["name"] for r in ray.util.list_named_actors(all_namespaces=True) if r.get("namespace") == target)
names = rows(); assert len(names) in (0, 15), (target, names)
managers = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
for name in sorted(names, key=lambda value: (value in managers, value)):
    try: ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
    except ValueError: pass
for _ in range(120):
    if not rows(): break
    time.sleep(1)
else: raise RuntimeError(rows())
print(f"cleaned_namespace={target}; actors={len(names)}")
ray.shutdown()
PY
}

wrapper_pid_exact() {
  local run=$1 pid
  pid=$(<"$run/runtime/wrapper.pid")
  test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
  ps -o args= -p "$pid" | grep -F "$run/runtime/wrapper.sh" >/dev/null
  test "$(<"$run/runtime/owned.pgid")" = "$pid"
  printf '%s\n' "$pid"
}

gpu_unique_job() {
  local devices=$1 pid job count=0
  local -a jobs=()
  while read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/environ" ]] || continue
    test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
    job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    test -n "$job"; jobs+=("$job"); count=$((count + 1))
  done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  test "$count" -ge 6
  mapfile -t unique < <(printf '%s\n' "${jobs[@]}" | sort -u)
  test "${#unique[@]}" -eq 1
  printf '%s\n' "${unique[0]}"
}

stop_process_tree() {
  local root=$1
  mapfile -t pids < <("$VENV/bin/python" - "$root" <<'PY'
import os, sys
root = int(sys.argv[1]); children = {}
for name in os.listdir('/proc'):
    if not name.isdigit(): continue
    try:
        fields = open(f'/proc/{name}/stat').read().split()
        children.setdefault(int(fields[3]), []).append(int(name))
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError): pass
out=[]
def visit(pid):
    for child in children.get(pid, []): visit(child)
    out.append(pid)
visit(root)
print('\n'.join(map(str, out)))
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

wait_job_gone() {
  local devices=$1 old_job=$2 pid job remaining
  for _ in $(seq 1 120); do
    remaining=0
    while read -r pid; do
      [[ -n "$pid" && -r "/proc/$pid/environ" ]] || continue
      job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
      [[ "$job" == "$old_job" ]] && remaining=$((remaining + 1))
    done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
    (( remaining == 0 )) && return 0
    sleep 1
  done
  return 1
}

stop_old() {
  local run=$1 namespace=$2 devices=$3 label=$4 wrapper observer old_job step
  step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$run/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
  wrapper=$(<"$run/runtime/wrapper.pid")
  if ! kill -0 "$wrapper" 2>/dev/null; then
    test "$(namespace_count "$namespace")" = 0
    test -z "$(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    observer=$(cat "$run/runtime/observer.pid" 2>/dev/null || true)
    if [[ -n "$observer" && -d "/proc/$observer" ]]; then stop_process_tree "$observer"; fi
    cat > "$run/runtime/stopped_by_user_for_matched_update2_20260901.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested fresh replacement with pi0-matched GB1024/update2
last_complete_step=$step
ray_job_id=already_gone_before_retry
namespace=$namespace
cleanup=owned process tree exited and namespace auto-cleared; shared Ray unchanged
EOF
    printf '%s_old_step=%s old_job=already_gone_before_retry\n' "$label" "$step"
    return 0
  fi
  wrapper=$(wrapper_pid_exact "$run")
  test "$(namespace_count "$namespace")" = 15
  old_job=$(gpu_unique_job "$devices")
  observer=$(cat "$run/runtime/observer.pid" 2>/dev/null || true)
  stop_process_tree "$wrapper"
  if [[ -n "$observer" && -d "/proc/$observer" ]]; then stop_process_tree "$observer"; fi
  cleanup_namespace "$namespace"
  wait_job_gone "$devices" "$old_job"
  cat > "$run/runtime/stopped_by_user_for_matched_update2_20260901.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested fresh replacement with pi0-matched GB1024/update2
last_complete_step=$step
ray_job_id=$old_job
namespace=$namespace
cleanup=owned process tree and exact namespace only; shared Ray unchanged
EOF
  printf '%s_old_step=%s old_job=%s\n' "$label" "$step" "$old_job"
}

start_runtime() {
  local run=$1 packet=$2 placement=$3 method=$4
  mkdir -p "$run/runtime"
  cp "$packet"/{resolved.yaml,contract.json,command.txt,packet_complete.txt,source_head.txt} "$run/runtime/"
  printf '%s\n' \
    "started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" "source_head=$HEAD" "physical_gpus=$placement" \
    'fresh_start=true; target_step=100' 'train=64 env x 4 rollout epochs = 256 trajectories/step; G8; max1024 query records' \
    'actor=GB1024/MB32/update2; 2 optimizer calls/outer; pi0.5 M5' "method=$method" \
    'eval=fixed32 every5; videos=true; checkpoint=local_shard every10' > "$run/runtime/launch_manifest.txt"
  cat > "$run/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; command_file=$2
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 259200s bash -lc "$(cat "$command_file")" > "$runtime/driver.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$runtime/exit_code.txt"; date --iso-8601=seconds > "$runtime/finished_at.txt"; exit "$rc"
WRAP
  chmod 700 "$run/runtime/wrapper.sh"
  cat > "$run/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2; a=$3; b=$4
printf '%s\n' "timestamp,driver_alive,host_mem_available_kib,gpu${a}_used_mib,gpu${a}_util_pct,gpu${b}_used_mib,gpu${b}_util_pct" > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i "$a,$b" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"; sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
  chmod 700 "$run/runtime/observer.sh"
}

wait_namespace_ready() {
  local namespace=$1 pid=$2 count=0
  for _ in $(seq 1 360); do
    kill -0 "$pid"
    count=$(namespace_count "$namespace")
    [[ "$count" == 15 ]] && return 0
    sleep 1
  done
  return 1
}

launch_new() {
  local run=$1 packet=$2 namespace=$3 a=$4 b=$5 placement=$6 method=$7 pid observer job
  setup_env
  start_runtime "$run" "$packet" "$placement" "$method"
  nohup setsid bash "$run/runtime/wrapper.sh" "$run/runtime" "$run/runtime/command.txt" > "$run/runtime/wrapper.log" 2>&1 < /dev/null &
  pid=$!; printf '%s\n' "$pid" > "$run/runtime/wrapper.pid"; printf '%s\n' "$pid" > "$run/runtime/owned.pgid"
  nohup setsid bash "$run/runtime/observer.sh" "$pid" "$run/runtime/resource.csv" "$a" "$b" > "$run/runtime/observer.log" 2>&1 < /dev/null &
  observer=$!; printf '%s\n' "$observer" > "$run/runtime/observer.pid"
  wait_namespace_ready "$namespace" "$pid"
  job=$(gpu_unique_job "$a,$b")
  printf 'new_run=%s\nwrapper_pid=%s\nray_job=%s\nnamespace=%s\n' "$run" "$pid" "$job" "$namespace"
}

# Cut over one GPU pair at a time to minimize idle time and preserve namespace order.
stop_old "$OLD_CONTROL" RLinf 4,5 control
launch_new "$NEW_CONTROL" "$NEW_CONTROL_PACKET" RLinf 4 5 4,5 'pi0.5 GRPO Control; pi0-matched GB1024/update2'
stop_old "$OLD_DVAC" RLinf_1 6,7 dvac
launch_new "$NEW_DVAC" "$NEW_DVAC_PACKET" RLinf_1 6 7 6,7 'pi0.5 GRPO-DVAC Action-Adv [0.5,1.5]; pi0-matched GB1024/update2'

test "$(namespace_count RLinf)" = 15
test "$(namespace_count RLinf_1)" = 15
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_PI05_MATCHED_UPDATE2_DUAL_CUTOVER_OK
