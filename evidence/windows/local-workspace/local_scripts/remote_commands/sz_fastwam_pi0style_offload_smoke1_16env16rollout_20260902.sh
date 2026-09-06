set -euo pipefail

RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
FASTWAM_SRC=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
BASE_SLUG=fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2
FAILED_SLUG=fastwam-grpo-pi0style-offload-smoke1-2gpu32x8-g8-b2048-u2-m10-phys67-v1
SLUG=fastwam-grpo-pi0style-offload-smoke1-2gpu16x16-g8-b2048-u2-m10-phys67-v1
BASE_PACKET="$ROOT/packets/$BASE_SLUG"
FAILED_RUN="$ROOT/runs/$FAILED_SLUG"
PACKET="$ROOT/packets/$SLUG"
RUN="$ROOT/runs/$SLUG"
BASE_EXP=fastwam_grpo_control_formal100_2gpu32x8_g8_b2048_u2_m10_fixed32_eval5_phys67_dcp_v2
EXP=fastwam_grpo_pi0style_offload_smoke1_2gpu16x16_g8_b2048_u2_m10_phys67_v1

test "$(git -C "$WT" rev-parse HEAD)" = 7b2331c55d14397cfb4cb16181470ddc8afae44a
test -z "$(git -C "$WT" status --porcelain)"
test -d "$FASTWAM_SRC/fastwam"
test -s "$BASE_PACKET/command.txt"
test -s "$BASE_PACKET/resolved.yaml"
test -s "$FAILED_RUN/runtime/wrapper.sh"
test -s "$FAILED_RUN/runtime/observer.sh"
test "$(cat "$FAILED_RUN/runtime/exit_code.txt")" = 255
test ! -e "$PACKET"
test ! -e "$RUN"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
test -z "$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os,ray
ray.init(address=os.environ['RAY_ADDRESS'], namespace='codex_fastwam_fallback_preflight', logging_level='ERROR')
assert not ray.util.list_named_actors(all_namespaces=True)
ray.shutdown()
PY

mkdir -p "$PACKET" "$RUN/runtime"
cp "$BASE_PACKET/command.txt" "$PACKET/command.txt"
sed -i \
  -e "s/$BASE_SLUG/$SLUG/g" \
  -e "s/$BASE_EXP/$EXP/g" \
  -e 's/runner.max_steps=100/runner.max_steps=1/' \
  -e 's/env.train.total_num_envs=32/env.train.total_num_envs=16/' \
  -e 's/env.train.rollout_epoch=8/env.train.rollout_epoch=16/' \
  -e 's/env.train.enable_offload=true/env.train.enable_offload=false/' \
  -e 's/env.eval.enable_offload=true/env.eval.enable_offload=false/' \
  "$PACKET/command.txt"
sed -i '$ s/$/ actor.enable_offload=true rollout.enable_offload=true/' "$PACKET/command.txt"

export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FASTWAM_SRC:$ROBOTWIN"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export HF_HOME=/data/chenyiteng/cache/huggingface
export XDG_CACHE_HOME=/data/chenyiteng/cache
cd "$WT"

bash -lc "$(cat "$PACKET/command.txt") --cfg job --resolve" > "$PACKET/resolved.yaml"
"$VENV/bin/python" - "$BASE_PACKET/resolved.yaml" "$PACKET/resolved.yaml" <<'PY'
import sys,yaml
old=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
new=yaml.safe_load(open(sys.argv[2],encoding='utf-8'))
def flat(obj,p=()):
    if isinstance(obj,dict):
        out={}
        for k,v in obj.items(): out.update(flat(v,p+(str(k),)))
        return out
    if isinstance(obj,list): return {'.'.join(p):obj}
    return {'.'.join(p):obj}
a,b=flat(old),flat(new)
diff={k:(a.get(k),b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
allowed={
 'runner.max_steps','runner.logger.log_path','runner.logger.experiment_name',
 'env.train.total_num_envs','env.train.rollout_epoch',
 'env.train.enable_offload','env.eval.enable_offload','actor.enable_offload',
 'env.train.video_cfg.video_base_dir','env.train.task_config.save_path',
 'env.eval.video_cfg.video_base_dir','env.eval.task_config.save_path',
}
unexpected=set(diff)-allowed
assert not unexpected,(unexpected,diff)
expected={
 'runner.max_steps':(100,1),
 'env.train.total_num_envs':(32,16),
 'env.train.rollout_epoch':(8,16),
 'env.train.enable_offload':(True,False),
 'env.eval.enable_offload':(True,False),
 'actor.enable_offload':(False,True),
}
for k,v in expected.items(): assert diff.get(k)==v,(k,diff.get(k),v)
assert b['env.train.total_num_envs']==16
assert b['env.train.rollout_epoch']==16
assert b['algorithm.group_size']==8
assert b['actor.global_batch_size']==2048
assert b['actor.micro_batch_size']==2
assert b['algorithm.update_epoch']==2
assert b['rollout.enable_offload'] is True
assert b['actor.fsdp_config.checkpoint_format']=='dcp'
print('RESOLVED_DIFF_OK')
for k,v in diff.items(): print(f'{k}: {v[0]!r} -> {v[1]!r}')
PY

cat > "$PACKET/contract.json" <<EOF
{
  "kind": "resource_smoke_fallback_only",
  "fallback_trigger": "32-env smoke OOM at update with GPU peaks 81017/81011 MiB",
  "max_steps": 1,
  "physical_gpus": [6, 7],
  "train_envs": 16,
  "rollout_epochs": 16,
  "trajectories": 256,
  "group_size": 8,
  "groups": 32,
  "max_query_records": 2048,
  "global_batch": 2048,
  "micro_batch": 2,
  "update_epoch": 2,
  "optimizer_calls": 2,
  "train_env_offload": false,
  "eval_env_offload": false,
  "actor_offload": true,
  "rollout_offload": true
}
EOF
git -C "$WT" rev-parse HEAD > "$PACKET/source_head.txt"
sha256sum "$PACKET/command.txt" "$PACKET/resolved.yaml" > "$PACKET/sha256.txt"
TZ=Asia/Shanghai date --iso-8601=seconds > "$PACKET/packet_complete.txt"
cp "$PACKET"/{command.txt,resolved.yaml,contract.json,source_head.txt,sha256.txt,packet_complete.txt} "$RUN/runtime/"
cp "$FAILED_RUN/runtime/wrapper.sh" "$FAILED_RUN/runtime/observer.sh" "$RUN/runtime/"
chmod +x "$RUN/runtime/wrapper.sh" "$RUN/runtime/observer.sh"

cat > "$RUN/runtime/launch_manifest.txt" <<EOF
started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
source_head=7b2331c55d14397cfb4cb16181470ddc8afae44a
purpose=one-step 16-env fallback resource smoke; not formal
physical_gpus=6,7
sampling=16 env x rollout16 = 256 trajectories; G8; max2048 query records
optimization=GB2048/MB2/update2; Fast-WAM H32/C24/M10
only_method_delta=train/eval env offload true-to-false; actor offload false-to-true; rollout offload stays true
fallback_reason=32 env x rollout8 completed rollout then OOM at update; peak 81017/81011 MiB
EOF

nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/runtime/wrapper.pid"
sleep 2
test -d "/proc/$pid"
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
test "$pgid" = "$pid"
printf '%s\n' "$pgid" > "$RUN/runtime/owned.pgid"
nohup setsid bash "$RUN/runtime/observer.sh" "$pid" "$RUN/runtime/resource_2s.csv" > "$RUN/runtime/observer.log" 2>&1 < /dev/null &
observer=$!
printf '%s\n' "$observer" > "$RUN/runtime/observer.pid"

echo "LAUNCHED run=$RUN packet=$PACKET wrapper=$pid observer=$observer"
echo 'COMMAND:'
cat "$PACKET/command.txt"
