#!/usr/bin/env bash
set -euo pipefail

RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
FASTWAM_SRC=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN_SHARED=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROBOTWIN_FIX=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
OLD_SLUG=fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
NEW_SLUG=fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2
OLD_EXP=fastwam_grpo_control_formal100_2gpu32x4_g8_b1024_u2_m10_fixed32_eval5_phys67_dcp_v1
NEW_EXP=fastwam_grpo_control_resume10_to100_2gpu32x4_g8_b1024_u2_m10_fixed32_eval5_phys67_dcp_renderlife_v2
OLD_PACKET="$ROOT/packets/$OLD_SLUG"
OLD_RUN="$ROOT/runs/$OLD_SLUG"
NEW_PACKET="$ROOT/packets/$NEW_SLUG"
NEW_RUN="$ROOT/runs/$NEW_SLUG"
STEP10="$OLD_RUN/$OLD_EXP/checkpoints/global_step_10"

namespace_count() {
  local target=$1
  RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - "$target" <<'PY'
import os, ray, sys
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_fastwam_renderlife_preflight", logging_level="ERROR")
target = sys.argv[1]
print(sum(1 for row in ray.util.list_named_actors(all_namespaces=True) if row.get("namespace") == target))
ray.shutdown()
PY
}

test "$(git -C "$WT" rev-parse HEAD)" = 7b2331c55d14397cfb4cb16181470ddc8afae44a
test -z "$(git -C "$WT" status --porcelain)"
test "$(git -C "$ROBOTWIN_FIX" rev-parse HEAD)" = 8c7380c118ce7ca8a4ea4df53d753adc8fab0df2
test -z "$(git -C "$ROBOTWIN_FIX" status --porcelain)"
test -d "$FASTWAM_SRC/fastwam"
test -d "$ROBOTWIN_SHARED/assets/objects"
test "$(readlink -f "$ROBOTWIN_FIX/assets/objects")" = "$ROBOTWIN_SHARED/assets/objects"
test -s "$OLD_PACKET/command.txt"
test -s "$STEP10/actor/dcp_checkpoint/.metadata"
test "$(find "$STEP10/actor/dcp_checkpoint" -name '*.distcp' -type f | wc -l)" -eq 2
test "$(cat "$OLD_RUN/runtime/exit_code.txt")" = 255
test ! -e "$NEW_PACKET"
test ! -e "$NEW_RUN"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
test "$(namespace_count RLinf)" = 15
test "$(namespace_count RLinf_1)" = 0
test -z "$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"

mkdir -p "$NEW_PACKET" "$NEW_RUN/runtime"
"$VENV/bin/python" - "$OLD_PACKET/command.txt" "$NEW_PACKET/command.txt" "$OLD_RUN" "$NEW_RUN" "$OLD_EXP" "$NEW_EXP" "$STEP10" <<'PY'
import sys
src, dst, old_run, new_run, old_exp, new_exp, step10 = sys.argv[1:]
s = open(src, encoding="utf-8").read()
for old, new, expected in (
    (old_run, new_run, 4),
    (old_exp, new_exp, 1),
    ("runner.resume_dir=null", f"runner.resume_dir={step10}", 1),
):
    count = s.count(old)
    assert count == expected, (old, count, expected)
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8").write(s)
PY

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN_FIX" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FASTWAM_SRC:$ROBOTWIN_FIX"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export HF_HOME=/data/chenyiteng/cache/huggingface
export XDG_CACHE_HOME=/data/chenyiteng/cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$WT"

bash -lc "$(cat "$NEW_PACKET/command.txt") --cfg job --resolve" > "$NEW_PACKET/resolved.yaml"
"$VENV/bin/python" - "$OLD_PACKET/resolved.yaml" "$NEW_PACKET/resolved.yaml" "$NEW_PACKET/contract.json" "$STEP10" <<'PY'
import json, sys, yaml
old = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
new = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
step10 = sys.argv[4]
assert new["runner"]["resume_dir"] == step10
assert new["runner"]["max_steps"] == 100
assert new["runner"]["val_check_interval"] == 5
assert new["runner"]["save_interval"] == 10
assert new["env"]["train"]["total_num_envs"] == 32
assert new["env"]["train"]["rollout_epoch"] == 4
assert new["env"]["train"]["enable_offload"] is True
assert new["env"]["eval"]["total_num_envs"] == 32
assert new["env"]["eval"]["enable_offload"] is True
assert new["algorithm"]["group_size"] == 8
assert new["algorithm"]["update_epoch"] == 2
assert new["actor"]["global_batch_size"] == 1024
assert new["actor"]["micro_batch_size"] == 2
assert new["actor"]["fsdp_config"]["checkpoint_format"] == "dcp"

def flat(value, path=()):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(flat(child, path + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            out.update(flat(child, path + (str(index),)))
    else:
        out[".".join(path)] = value
    return out

a, b = flat(old), flat(new)
diff = {key: (a.get(key), b.get(key)) for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)}
allowed_prefixes = (
    "runner.resume_dir",
    "runner.logger.log_path",
    "runner.logger.experiment_name",
    "env.train.task_config.save_path",
    "env.train.video_cfg.video_base_dir",
    "env.eval.task_config.save_path",
    "env.eval.video_cfg.video_base_dir",
)
unexpected = [key for key in diff if not key.startswith(allowed_prefixes)]
assert not unexpected, unexpected
payload = {
    "scientific_config": "identical to original v1",
    "allowed_resolved_differences": diff,
    "rlinf_head": "7b2331c55d14397cfb4cb16181470ddc8afae44a",
    "robotwin_base": "0008ae6800df9f75fc8de7098bacb01735fd8fd2",
    "robotwin_fix": "8c7380c118ce7ca8a4ea4df53d753adc8fab0df2",
    "resume_checkpoint": step10,
    "acceptance_boundary": "complete Step-15 fixed evaluation without OIDN/Python/Ray fatal",
}
with open(sys.argv[3], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

printf '%s\n' \
  'packet_complete=true' \
  'scientific_config=original v1 unchanged' \
  'only_runtime_change=RoboTwin vector-scope renderer cache lifecycle' \
  'resume=complete global_step_10 DCP' > "$NEW_PACKET/packet_complete.txt"
printf '%s\n' \
  'rlinf=7b2331c55d14397cfb4cb16181470ddc8afae44a' \
  'fastwam=7faa71108368fbb3b6885649f112af607427a2d4' \
  'robotwin_base=0008ae6800df9f75fc8de7098bacb01735fd8fd2' \
  'robotwin_fix=8c7380c118ce7ca8a4ea4df53d753adc8fab0df2' > "$NEW_PACKET/source_head.txt"
sha256sum "$NEW_PACKET"/{command.txt,resolved.yaml,contract.json,source_head.txt} > "$NEW_PACKET/sha256.txt"
cp "$NEW_PACKET"/{command.txt,resolved.yaml,contract.json,packet_complete.txt,source_head.txt,sha256.txt} "$NEW_RUN/runtime/"

cat > "$NEW_RUN/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 432000s bash -lc "$(cat "$runtime/command.txt")" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
WRAP
chmod 700 "$NEW_RUN/runtime/wrapper.sh"

cat > "$NEW_RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,gpu6_used_mib,gpu6_util_pct,gpu7_used_mib,gpu7_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
chmod 700 "$NEW_RUN/runtime/observer.sh"

cat > "$NEW_RUN/runtime/launch_manifest.txt" <<EOF
started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
physical_gpus=6,7
source_rlinf=7b2331c55d14397cfb4cb16181470ddc8afae44a
source_fastwam=7faa71108368fbb3b6885649f112af607427a2d4
source_robotwin_fix=8c7380c118ce7ca8a4ea4df53d753adc8fab0df2
resume_from=$STEP10
target_step=100
train=32 env x 4 rollout epochs = 128 trajectories/step; G8; max1024 query records
actor=GB1024/MB2/update2; 2 optimizer calls/outer; Fast-WAM M10 H32 C24
eval=fixed32 every5; videos=true; checkpoint=DCP every10
offload=train_env true; eval_env true; actor false; rollout true
acceptance=must complete Step15 fixed evaluation without former OIDN/Python/Ray fatal
EOF

nohup setsid bash "$NEW_RUN/runtime/wrapper.sh" "$NEW_RUN/runtime" > "$NEW_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$NEW_RUN/runtime/wrapper.pid"
printf '%s\n' "$pid" > "$NEW_RUN/runtime/owned.pgid"
nohup setsid bash "$NEW_RUN/runtime/observer.sh" "$pid" "$NEW_RUN/runtime/resource.csv" > "$NEW_RUN/runtime/observer.log" 2>&1 < /dev/null &
observer=$!
printf '%s\n' "$observer" > "$NEW_RUN/runtime/observer.pid"
sleep 5
kill -0 "$pid"
printf 'NEW_RUN=%s\nWRAPPER_PID=%s\nOBSERVER_PID=%s\n' "$NEW_RUN" "$pid" "$observer"
cat "$NEW_PACKET/contract.json"
