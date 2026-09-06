set -euo pipefail

RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
FASTWAM_SRC=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
BASE_SLUG=fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2
SMOKE_SLUG=fastwam-grpo-pi0style-offload-smoke1-2gpu16x16-g8-b2048-u2-m10-phys67-v1
SLUG=fastwam-grpo-control-formal100-2gpu16x16-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-pi0style-v3
BASE_PACKET="$ROOT/packets/$BASE_SLUG"
BASE_RUN="$ROOT/runs/$BASE_SLUG"
SMOKE_PACKET="$ROOT/packets/$SMOKE_SLUG"
PACKET="$ROOT/packets/$SLUG"
RUN="$ROOT/runs/$SLUG"
SMOKE_EXP=fastwam_grpo_pi0style_offload_smoke1_2gpu16x16_g8_b2048_u2_m10_phys67_v1
EXP=fastwam_grpo_control_formal100_2gpu16x16_g8_b2048_u2_m10_fixed32_eval5_phys67_dcp_pi0style_v3

test "$(git -C "$WT" rev-parse HEAD)" = 7b2331c55d14397cfb4cb16181470ddc8afae44a
test -z "$(git -C "$WT" status --porcelain)"
test "$(git -C /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711 rev-parse HEAD)" = 7faa71108368fbb3b6885649f112af607427a2d4
test -d "$FASTWAM_SRC/fastwam"
test -s "$BASE_PACKET/resolved.yaml"
test -s "$SMOKE_PACKET/command.txt"
test -s "$SMOKE_PACKET/resolved.yaml"
test -s "$BASE_RUN/runtime/wrapper.sh"
test -s "$BASE_RUN/runtime/observer.sh"
test ! -e "$PACKET"
test ! -e "$RUN"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
test -z "$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"

mkdir -p "$PACKET" "$RUN/runtime"
cp "$SMOKE_PACKET/command.txt" "$PACKET/command.txt"
sed -i \
  -e "s/$SMOKE_SLUG/$SLUG/g" \
  -e "s/$SMOKE_EXP/$EXP/g" \
  -e 's/runner.max_steps=1/runner.max_steps=100/' \
  "$PACKET/command.txt"

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
"$VENV/bin/python" - "$BASE_PACKET/resolved.yaml" "$SMOKE_PACKET/resolved.yaml" "$PACKET/resolved.yaml" <<'PY'
import sys,yaml

def flat(obj,p=()):
    if isinstance(obj,dict):
        out={}
        for k,v in obj.items(): out.update(flat(v,p+(str(k),)))
        return out
    if isinstance(obj,list): return {'.'.join(p):obj}
    return {'.'.join(p):obj}

base,smoke,new=[flat(yaml.safe_load(open(p,encoding='utf-8'))) for p in sys.argv[1:]]

def diff(a,b):
    return {k:(a.get(k),b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}

run_paths={
 'runner.logger.log_path','runner.logger.experiment_name',
 'env.train.video_cfg.video_base_dir','env.train.task_config.save_path',
 'env.eval.video_cfg.video_base_dir','env.eval.task_config.save_path',
}
formal_expected={
 'env.train.total_num_envs':(32,16),
 'env.train.rollout_epoch':(8,16),
 'env.train.enable_offload':(True,False),
 'env.eval.enable_offload':(True,False),
 'actor.enable_offload':(False,True),
}
d_formal=diff(base,new)
assert set(d_formal)==set(formal_expected)|run_paths,(set(d_formal)-set(formal_expected)-run_paths,d_formal)
for k,v in formal_expected.items(): assert d_formal[k]==v,(k,d_formal[k],v)

d_smoke=diff(smoke,new)
assert set(d_smoke)=={'runner.max_steps'}|run_paths,(set(d_smoke)-{'runner.max_steps'}-run_paths,d_smoke)
assert d_smoke['runner.max_steps']==(1,100)

assert new['env.train.total_num_envs']==16
assert new['env.train.rollout_epoch']==16
assert new['algorithm.group_size']==8
assert new['actor.global_batch_size']==2048
assert new['actor.micro_batch_size']==2
assert new['algorithm.update_epoch']==2
assert new['env.eval.total_num_envs']==32
assert new['runner.val_check_interval']==5
assert new['runner.save_interval']==10
assert new['actor.fsdp_config.checkpoint_format']=='dcp'
assert new['actor.model.action_horizon']==32
assert new['actor.model.num_action_chunks']==24
assert new['actor.model.num_inference_steps']==10
assert new['actor.optim.lr']==5e-6
assert new['env.train.seed']==0 and new['actor.seed']==1234
print('FORMAL_RESOLVED_DIFF_OK unexpected=0')
print('DIFF_FROM_PRIOR_32x8_FORMAL')
for k,v in d_formal.items(): print(f'{k}: {v[0]!r} -> {v[1]!r}')
print('DIFF_FROM_SUCCESSFUL_SMOKE')
for k,v in d_smoke.items(): print(f'{k}: {v[0]!r} -> {v[1]!r}')
PY

cat > "$PACKET/contract.json" <<EOF
{
  "kind": "formal100_fresh",
  "source_head": "7b2331c55d14397cfb4cb16181470ddc8afae44a",
  "official_fastwam_head": "7faa71108368fbb3b6885649f112af607427a2d4",
  "physical_gpus": [6, 7],
  "steps": 100,
  "train_envs": 16,
  "rollout_epochs": 16,
  "trajectories_per_step": 256,
  "trajectories_total": 25600,
  "max_action_slots_total": 4915200,
  "group_size": 8,
  "groups_per_step": 32,
  "max_query_records_per_step": 2048,
  "max_query_records_total": 204800,
  "global_batch": 2048,
  "micro_batch": 2,
  "update_epoch": 2,
  "optimizer_calls_per_step": 2,
  "optimizer_calls_total": 200,
  "presentations_per_step": 4096,
  "presentations_total": 409600,
  "fixed_eval_episodes": 640,
  "checkpoint": "dcp every10"
}
EOF
git -C "$WT" rev-parse HEAD > "$PACKET/source_head.txt"
sha256sum "$PACKET/command.txt" "$PACKET/resolved.yaml" > "$PACKET/sha256.txt"
TZ=Asia/Shanghai date --iso-8601=seconds > "$PACKET/packet_complete.txt"
cp "$PACKET"/{command.txt,resolved.yaml,contract.json,source_head.txt,sha256.txt,packet_complete.txt} "$RUN/runtime/"
cp "$BASE_RUN/runtime/wrapper.sh" "$BASE_RUN/runtime/observer.sh" "$RUN/runtime/"
chmod +x "$RUN/runtime/wrapper.sh" "$RUN/runtime/observer.sh"

cat > "$RUN/runtime/launch_manifest.txt" <<EOF
started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
source_head=7b2331c55d14397cfb4cb16181470ddc8afae44a
official_fastwam_head=7faa71108368fbb3b6885649f112af607427a2d4
physical_gpus=6,7
fresh_start=true
target_steps=100
sampling=16 env x rollout16 = 256 trajectories/step; G8; 32 groups; max2048 query records
model=Fast-WAM H32/C24/M10/D14; episode192; action-expert only
optimization=GB2048/MB2/update2; 2 optimizer calls; 4096 presentations/step; lr5e-6
evaluation=fixed32 every5; eval videos true
checkpoint=DCP every10
offload=train_env=false; eval_env=false; actor=true; rollout=true
provenance=successful one-step resource smoke; prior formal scientific budget preserved
EOF

nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
wrapper=$!
printf '%s\n' "$wrapper" > "$RUN/runtime/wrapper.pid"
sleep 2
test -d "/proc/$wrapper"
pgid=$(ps -o pgid= -p "$wrapper" | tr -d ' ')
test "$pgid" = "$wrapper"
printf '%s\n' "$pgid" > "$RUN/runtime/owned.pgid"
nohup setsid bash "$RUN/runtime/observer.sh" "$wrapper" "$RUN/runtime/resource.csv" > "$RUN/runtime/observer.log" 2>&1 < /dev/null &
observer=$!
printf '%s\n' "$observer" > "$RUN/runtime/observer.pid"

echo "LAUNCHED run=$RUN packet=$PACKET wrapper=$wrapper pgid=$pgid observer=$observer"
sleep 8
ps -o user,pid,ppid,pgid,etimes,rss,stat,args -p "$wrapper" "$observer"
tail -n 80 "$RUN/runtime/driver.log" 2>/dev/null || true
