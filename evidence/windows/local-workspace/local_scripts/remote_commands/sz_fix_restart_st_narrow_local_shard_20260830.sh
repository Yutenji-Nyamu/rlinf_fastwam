#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
OLD_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
OLD_HEAD=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
FIX_COMMIT=306ce2e98a06b6f439a1070d8942e20132e48d49
BRANCH=codex/sz-st-dvac-local-shard
NEW_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/st-dvac-local-shard
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
OLD_NAME=dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
NEW_NAME=dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2
OLD_RUN="$ROOT/runs/$OLD_NAME"
NEW_RUN="$ROOT/runs/$NEW_NAME"
PACKET="$ROOT/packets/$NEW_NAME"
ACTION_RUN="$ROOT/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"
ACTION_NAMESPACE=RLinf
ST_NAMESPACE=RLinf_1

test "$(git -C "$OLD_WT" rev-parse HEAD)" = "$OLD_HEAD"
test -z "$(git -C "$OLD_WT" status --short)"
test ! -e "$NEW_WT"
test ! -e "$NEW_RUN"
test ! -e "$PACKET"
! git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"
git -C "$MAIN" diff "$FIX_COMMIT^" "$FIX_COMMIT" -- rlinf/hybrid_engines/fsdp/fsdp_model_manager.py \
  | git -C "$OLD_WT" apply --check -

# Isolated ST-only source: exactly the historically proven one-file manager patch.
git -C "$MAIN" worktree add -b "$BRANCH" "$NEW_WT" "$OLD_HEAD"
git -C "$NEW_WT" cherry-pick "$FIX_COMMIT"
NEW_HEAD=$(git -C "$NEW_WT" rev-parse HEAD)
test -z "$(git -C "$NEW_WT" status --short)"
test "$(git -C "$NEW_WT" diff --numstat "$OLD_HEAD"..HEAD | wc -l)" = 1
test "$(git -C "$NEW_WT" diff --numstat "$OLD_HEAD"..HEAD | awk '{print $1":"$2":"$3}')" = '14:1:rlinf/hybrid_engines/fsdp/fsdp_model_manager.py'
git -C "$NEW_WT" push personal "HEAD:refs/heads/$BRANCH"

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA
export REPO_PATH="$NEW_WT" EMBODIED_PATH="$NEW_WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$NEW_WT"
export PYTHONPATH="$NEW_WT:$ROBOTWIN" OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

install -d -m 755 "$PACKET"
ARGS=(
  --config-path "$NEW_WT/examples/embodiment/config" --config-name robotwin_adjust_bottle_grpo_openpi
  'cluster.component_placement={actor\, env\, rollout:"6,7"}'
  "runner.logger.log_path=$NEW_RUN"
  runner.logger.experiment_name=robotwin_dvac_st_global_z_w0p8to1p2_formal100_2gpu64x4_b1024_fixed32_eval5_phys67_localshard_v2
  runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=2 algorithm.adv_type=grpo algorithm.filter_rewards=true algorithm.logprob_type=chunk_level
  algorithm.dvac_gradient_weighting.mode=apply algorithm.dvac_gradient_weighting.selected_l=3
  algorithm.dvac_gradient_weighting.warmup_steps=1 algorithm.dvac_gradient_weighting.window_steps=5
  algorithm.dvac_gradient_weighting.weight_min=0.8 algorithm.dvac_gradient_weighting.weight_max=1.2
  env.train.total_num_envs=64 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true
  "env.train.video_cfg.video_base_dir=$NEW_RUN/video/train" "env.train.task_config.save_path=$NEW_RUN/robotwin_data/train"
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true
  "env.eval.video_cfg.video_base_dir=$NEW_RUN/video/eval" "env.eval.task_config.save_path=$NEW_RUN/robotwin_data/eval"
  actor.micro_batch_size=32 actor.global_batch_size=1024 "actor.model.model_path=$MODEL"
  +actor.fsdp_config.checkpoint_format=local_shard
)
"$VENV/bin/python" "$NEW_WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" --cfg job --resolve > "$PACKET/resolved.yaml"
printf '%q ' "$VENV/bin/python" "$NEW_WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$PACKET/command.txt"
printf '\n' >> "$PACKET/command.txt"

"$VENV/bin/python" - "$OLD_RUN/runtime/resolved.yaml" "$PACKET/resolved.yaml" "$PACKET/parity.json" <<'PY'
import json, sys, yaml
old_path, new_path, out_path = sys.argv[1:]
def load(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
def flat(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(flat(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list): out[prefix] = value
    else: out[prefix] = value
    return out
left, right = flat(load(old_path)), flat(load(new_path))
missing = object()
diff = {key: {"old": left.get(key, "<MISSING>"), "new": right.get(key, "<MISSING>")}
        for key in sorted(set(left) | set(right)) if left.get(key, missing) != right.get(key, missing)}
run_scoped = {
    "runner.logger.log_path", "runner.logger.experiment_name",
    "env.train.seeds_path", "env.eval.seeds_path",
    "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
    "actor.model.output_dir", "algorithm.dvac_gradient_weighting.output_dir",
}
allowed = run_scoped | {"actor.fsdp_config.checkpoint_format"}
unexpected = sorted(set(diff) - allowed)
assert not unexpected, unexpected
assert right["actor.fsdp_config.checkpoint_format"] == "local_shard"
for key in (
    "cluster.component_placement.actor, env, rollout", "runner.max_steps",
    "runner.val_check_interval", "runner.save_interval", "algorithm.update_epoch",
    "algorithm.adv_type", "algorithm.logprob_type", "algorithm.group_size",
    "algorithm.dvac_gradient_weighting.mode", "algorithm.dvac_gradient_weighting.selected_l",
    "algorithm.dvac_gradient_weighting.weight_min", "algorithm.dvac_gradient_weighting.weight_max",
    "env.train.total_num_envs", "env.train.rollout_epoch", "env.eval.total_num_envs",
    "actor.global_batch_size", "actor.micro_batch_size",
):
    assert left[key] == right[key], (key, left[key], right[key])
payload = {"unexpected": unexpected, "diff": diff, "scientific_config_unchanged": True,
           "only_runtime_change": "FSDP checkpoint_format dcp -> local_shard"}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False); handle.write("\n")
print("ST_LOCAL_SHARD_PACKET_PARITY_OK unexpected=0")
PY

cat > "$PACKET/contract.json" <<EOF
{
  "source_base": "$OLD_HEAD",
  "source_head": "$NEW_HEAD",
  "source_delta": "306ce2e one-file FSDP checkpoint-format propagation only",
  "run": "$NEW_RUN",
  "physical_gpus": [6, 7],
  "train": "64 env x 4 rollout epochs = 256 trajectories/step; G8; max1024 records",
  "actor": "GB1024/MB32/update2",
  "method": "ST-DVAC global-z [0.8,1.2], unchanged",
  "eval": "fixed32 every5",
  "checkpoint": "local_shard every10",
  "normal_stop": "complete global_step_100 and exit0",
  "hard_timeout_seconds": 216000
}
EOF
sha256sum "$PACKET/resolved.yaml" "$PACKET/command.txt" > "$PACKET/sha256.txt"
printf '%s\n' "completed_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" > "$PACKET/packet_complete.txt"

wrapper_contract() {
  local run=$1 pid pgid
  pid=$(<"$run/runtime/wrapper.pid"); pgid=$(<"$run/runtime/owned.pgid")
  test "$pid" = "$pgid"; kill -0 "$pid"
  test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
  ps -o args= -p "$pid" | grep -F "$run/runtime/wrapper.sh" >/dev/null
  printf '%s\n' "$pid"
}
gpu_unique_job() {
  local devices=$1 minimum=$2 process_pid job count=0
  local -a jobs=()
  while read -r process_pid; do
    [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
    test "$(ps -o user= -p "$process_pid" | xargs)" = chenyiteng
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    test -n "$job"; jobs+=("$job"); count=$((count + 1))
  done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  test "$count" -ge "$minimum"
  mapfile -t unique < <(printf '%s\n' "${jobs[@]}" | sort -u)
  test "${#unique[@]}" -eq 1
  printf '%s\n' "${unique[0]}"
}
namespace_names() {
  local namespace=$1
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$namespace" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_st_local_shard_names", logging_level="ERROR")
print("\n".join(sorted(row["name"] for row in ray.util.list_named_actors(all_namespaces=True)
                       if row.get("namespace") == os.environ["TARGET_NAMESPACE"])))
ray.shutdown()
PY
}
cleanup_st_namespace_preserve_action() {
  local expected_action_file=$1
  RAY_ADDRESS="$RAY_ADDRESS" ACTION_EXPECTED_FILE="$expected_action_file" "$VENV/bin/python" - <<'PY'
import os, time, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_st_local_shard_cutover", logging_level="ERROR")
def names(ns):
    return sorted(row["name"] for row in ray.util.list_named_actors(all_namespaces=True)
                  if row.get("namespace") == ns)
with open(os.environ["ACTION_EXPECTED_FILE"], encoding="utf-8") as handle:
    expected = [line.rstrip("\n") for line in handle if line.rstrip("\n")]
assert names("RLinf") == expected
targets = names("RLinf_1")
assert len(targets) == 15, targets
managers = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
for name in sorted(targets, key=lambda item: (item in managers, item)):
    try: ray.kill(ray.get_actor(name, namespace="RLinf_1"), no_restart=True)
    except ValueError: pass
for _ in range(120):
    if not names("RLinf_1"): break
    time.sleep(1)
else: raise SystemExit(f"ST namespace cleanup incomplete: {names('RLinf_1')}")
assert names("RLinf") == expected
print(f"ST_NAMESPACE_CLEAN action_actors={len(expected)}")
ray.shutdown()
PY
}
wait_old_job_gone() {
  local old_job=$1 remaining=0 process_pid job
  for _ in $(seq 1 120); do
    remaining=0
    while read -r process_pid; do
      [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
      job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
      [[ "$job" == "$old_job" ]] && remaining=$((remaining + 1))
    done < <(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
    (( remaining == 0 )) && break; sleep 1
  done
  test "$remaining" -eq 0
}

RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
ACTION_PID=$(wrapper_contract "$ACTION_RUN")
OLD_PID=$(wrapper_contract "$OLD_RUN")
ACTION_JOB=$(gpu_unique_job 4,5 6)
OLD_JOB=$(gpu_unique_job 6,7 6)
test -n "$ACTION_JOB"
test -n "$OLD_JOB"
test "$ACTION_JOB" != "$OLD_JOB"
namespace_names "$ACTION_NAMESPACE" > "$PACKET/action_actor_names_before.txt"
test "$(wc -l < "$PACKET/action_actor_names_before.txt")" = 15
test "$(namespace_names "$ST_NAMESPACE" | wc -l)" = 15
ACTION_LOG_SIZE_BEFORE=$(stat -c %s "$ACTION_RUN/runtime/driver.log")

# Stop only the unrecoverable stuck ST v1 and its exact namespace.
kill -TERM -- "-$OLD_PID"
for _ in $(seq 1 60); do ! kill -0 "$OLD_PID" 2>/dev/null && break; sleep 1; done
if kill -0 "$OLD_PID" 2>/dev/null; then kill -KILL -- "-$OLD_PID"; sleep 2; fi
! kill -0 "$OLD_PID" 2>/dev/null
OLD_OBSERVER=$(cat "$OLD_RUN/runtime/observer.pid" 2>/dev/null || true)
if [[ -n "$OLD_OBSERVER" ]] && kill -0 "$OLD_OBSERVER" 2>/dev/null; then
  kill -TERM -- "-$OLD_OBSERVER" 2>/dev/null || kill -TERM "$OLD_OBSERVER" 2>/dev/null || true
fi
cleanup_st_namespace_preserve_action "$PACKET/action_actor_names_before.txt"
wait_old_job_gone "$OLD_JOB"
kill -0 "$ACTION_PID"
test "$(gpu_unique_job 4,5 6)" = "$ACTION_JOB"

cat > "$OLD_RUN/runtime/stopped_for_local_shard_repair.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=Step10 default-DCP coordination hang; checkpoint empty and unrecoverable
replacement=$NEW_RUN
cleanup=owned PGID $OLD_PID and exact namespace $ST_NAMESPACE only; Action/shared Ray preserved
EOF

mkdir -p "$NEW_RUN/runtime"
cp "$PACKET"/{resolved.yaml,parity.json,contract.json,command.txt,sha256.txt,packet_complete.txt} "$NEW_RUN/runtime/"
cat > "$NEW_RUN/runtime/launch_manifest.txt" <<EOF
started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
source_base=$OLD_HEAD
source_head=$NEW_HEAD
physical_gpus=6,7
fresh_start=true; target_step=100
train=64 env x 4 epochs = 256 trajectories/step; G8; max1024 records
actor=GB1024/MB32/update2
method=ST-DVAC global-z [0.8,1.2], unchanged
eval=fixed32 every5; checkpoint=local_shard every10
normal_stop=complete step100/checkpoint/exit0; hard_timeout=216000s
EOF
cat > "$NEW_RUN/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 216000s "$@" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
WRAP
chmod 700 "$NEW_RUN/runtime/wrapper.sh"

nohup setsid bash "$NEW_RUN/runtime/wrapper.sh" "$NEW_RUN/runtime" \
  "$VENV/bin/python" "$NEW_WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" \
  > "$NEW_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
NEW_PID=$!
printf '%s\n' "$NEW_PID" > "$NEW_RUN/runtime/wrapper.pid"
printf '%s\n' "$NEW_PID" > "$NEW_RUN/runtime/owned.pgid"

cat > "$NEW_RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,gpu6_used_mib,gpu6_util_pct,gpu7_used_mib,gpu7_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"; sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
chmod 700 "$NEW_RUN/runtime/observer.sh"
nohup setsid bash "$NEW_RUN/runtime/observer.sh" "$NEW_PID" "$NEW_RUN/runtime/resource.csv" > "$NEW_RUN/runtime/observer.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$NEW_RUN/runtime/observer.pid"

NEW_COUNT=0
for _ in $(seq 1 300); do
  kill -0 "$NEW_PID"
  NEW_COUNT=$(namespace_names "$ST_NAMESPACE" | wc -l)
  [[ "$NEW_COUNT" = 15 ]] && break
  sleep 1
done
test "$NEW_COUNT" = 15
NEW_JOB=$(gpu_unique_job 6,7 6)
test -n "$NEW_JOB"
test "$NEW_JOB" != "$OLD_JOB"
kill -0 "$ACTION_PID"
test "$(gpu_unique_job 4,5 6)" = "$ACTION_JOB"
test "$(namespace_names "$ACTION_NAMESPACE" | diff -u "$PACKET/action_actor_names_before.txt" -)" = ''
sleep 5
test "$(stat -c %s "$ACTION_RUN/runtime/driver.log")" -ge "$ACTION_LOG_SIZE_BEFORE"
test "$(grep -aEic 'Traceback|OutOfMemory|CUDA out of memory|nonfinite|nan loss' "$NEW_RUN/runtime/driver.log" 2>/dev/null || true)" = 0

printf 'new_head=%s\nnew_branch=%s\nnew_run=%s\nnew_pid=%s\nold_job=%s\nnew_job=%s\n' \
  "$NEW_HEAD" "$BRANCH" "$NEW_RUN" "$NEW_PID" "$OLD_JOB" "$NEW_JOB"
printf 'action_pid=%s\naction_job=%s\naction_preserved=true\n' "$ACTION_PID" "$ACTION_JOB"
printf 'resolved_checkpoint_format='; "$VENV/bin/python" - "$PACKET/resolved.yaml" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding='utf-8') as f: cfg=yaml.safe_load(f)
print(cfg['actor']['fsdp_config']['checkpoint_format'])
PY
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_ST_LOCAL_SHARD_V2_RESTART_OK
