#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
dataset=/data/chenyiteng/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
dataset_validation=$dataset/rlt_canonical_validation.json
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
norm_stats=$model/physical-intelligence/robotwin/norm_stats.json
chain_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260823-v1
runtime=$chain_root/runtime
stage1=$chain_root/stage1
stage1_runtime=$stage1/runtime
stage1_experiment=robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1
stage1_endpoint=$stage1/$stage1_experiment/checkpoints/global_step_2000
artifact_dir=$stage1/artifacts
artifact_manifest=$artifact_dir/stage1_artifact_manifest.json
manifest_id=sz-rlt-stage1-current-ar-clean50-2k-v1
stage2=$chain_root/stage2
stage2_runtime=$stage2/runtime
stage2_experiment=robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_v1
observer=/data/chenyiteng/results/rlinf-rlt/launchers/shenzhen_rlt_current_smoke_observer_gpu4_5_20260823.sh
ray_address=127.0.0.1:6389
expected_head=f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4
expected_branch=codex/sz-rlt-pi0-robotwin-ar
stage1_config=$worktree/examples/sft/config/robotwin_rlt_stage1_sft_openpi_current_ar.yaml
stage2_config=$worktree/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250.yaml
eval_seeds=$worktree/rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json
expected_stage1_config_sha=63b934af03a0e649256104e986fa1524391d4a6cb319187849f0ad39f1a88329
expected_stage2_config_sha=92e4a212b78148fd5af65fdb82fc49359b9c5321c3531e0d667bde4bed613bf6
expected_seeds_sha=fb9c3353e27b83aad6fe7ff778437d960b084de9d981c2af68615d52769952a7
expected_norm_sha=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$worktree" branch --show-current)" = "$expected_branch"
test "$(git -C "$worktree" rev-parse "personal/$expected_branch")" = "$expected_head"
test -z "$(git -C "$worktree" status --porcelain)"
test "$(sha256sum "$stage1_config" | awk '{print $1}')" = "$expected_stage1_config_sha"
test "$(sha256sum "$stage2_config" | awk '{print $1}')" = "$expected_stage2_config_sha"
test "$(sha256sum "$eval_seeds" | awk '{print $1}')" = "$expected_seeds_sha"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$expected_norm_sha"
test -x "$venv/bin/python"
test -x "$venv/bin/ray"
test -d "$robotwin"
test -s "$dataset/meta/info.json"
test -s "$dataset_validation"
test -s "$model/model-00001-of-00002.safetensors"
test -s "$model/model-00002-of-00002.safetensors"
test -r "$observer"
test ! -e "$chain_root"
RAY_ADDRESS="$ray_address" "$venv/bin/ray" status >/dev/null
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 4-5 are not idle' >&2
  exit 1
fi

source "$venv/bin/activate"
unset CUDA_VISIBLE_DEVICES
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS="$ray_address"
export ROBOTWIN_RLT_CLEAN50_PATH="$dataset"
export ROBOTWIN_PI0_BASE_PATH="$model"
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export ROBOTWIN_PATH="$robotwin"
export ROBOTWIN_ASSETS_PATH="$robotwin"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$worktree"
export EMBODIED_PATH="$worktree/examples/embodiment"
export PYTHONPATH="$worktree:$robotwin${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$runtime" "$stage1_runtime" "$artifact_dir" "$stage2_runtime"

stage1_args=(
  --config-path "$worktree/examples/sft/config"
  --config-name robotwin_rlt_stage1_sft_openpi_current_ar
  'cluster.component_placement={actor\,env\,rollout:4-5}'
  "runner.logger.log_path=$stage1"
  "runner.logger.experiment_name=$stage1_experiment"
  'runner.max_steps=2000'
  'runner.save_interval=2000'
  'runner.val_check_interval=-1'
  'actor.micro_batch_size=16'
  'actor.global_batch_size=32'
  'actor.optim.total_training_steps=2000'
  'actor.optim.lr_warmup_steps=100'
)

"$venv/bin/python" "$worktree/examples/sft/train_vla_sft.py" \
  "${stage1_args[@]}" --cfg job --resolve > "$stage1_runtime/resolved.yaml"
sha256sum "$stage1_runtime/resolved.yaml" > "$stage1_runtime/resolved.yaml.sha256"
printf '%q ' "$venv/bin/python" "$worktree/examples/sft/train_vla_sft.py" "${stage1_args[@]}" > "$stage1_runtime/command.txt"
printf '\n' >> "$stage1_runtime/command.txt"

printf '%s\n' \
  'name=RLT current formal chain v1' \
  "source_head=$expected_head" \
  'physical_gpus=4,5' \
  'shared_ray_address=127.0.0.1:6389' \
  'stage1=2000 optimizer steps; MB16/rank; GB32; 64000 sample presentations' \
  'stage2=250 cycles; 8 train env; 2000 train episodes' \
  'stage2=max 400000 primitive slots; max 40000 macro transitions' \
  'stage2=UTD5; critic:actor=2; GB512; MB128' \
  'periodic_eval=4 env x 5 waves every 25 cycles; 200 episodes total' \
  'checkpoints=stage1 endpoint only; stage2 every 25 cycles (10)' \
  'normal_stop=stage1 step2000 then stage2 cycle250' \
  'failure_stop=nonzero stage exit, incomplete stage1 artifact, or hard timeout' \
  > "$runtime/launch_manifest.txt"

nohup setsid bash -c '
  set -u
  runtime=$1; stage1_runtime=$2; stage1_endpoint=$3; artifact_manifest=$4
  manifest_id=$5; dataset=$6; dataset_validation=$7; norm_stats=$8
  expected_norm_sha=$9; expected_head=${10}; stage1_config_sha=${11}
  stage2_config_sha=${12}; stage2_runtime=${13}; stage2=${14}
  stage2_experiment=${15}; worktree=${16}; venv=${17}; shift 17
  date --iso-8601=seconds > "$runtime/started_at.txt"
  date --iso-8601=seconds > "$stage1_runtime/started_at.txt"
  timeout --signal=TERM --kill-after=180s 10800s "$@" > "$stage1_runtime/driver.log" 2>&1
  s1_rc=$?
  printf "%s\n" "$s1_rc" > "$stage1_runtime/exit_code.txt"
  date --iso-8601=seconds > "$stage1_runtime/finished_at.txt"
  if [ "$s1_rc" -ne 0 ]; then
    printf "stage1_failed=%s\n" "$s1_rc" > "$runtime/chain_exit.txt"
    exit "$s1_rc"
  fi

  full_weights=$stage1_endpoint/actor/model_state_dict/full_weights.pt
  dcp_metadata=$stage1_endpoint/actor/dcp_checkpoint/.metadata
  if [ ! -s "$full_weights" ] || [ ! -s "$dcp_metadata" ]; then
    printf "%s\n" "Stage1 exited zero but endpoint is incomplete" >> "$stage1_runtime/driver.log"
    printf "%s\n" 91 > "$runtime/chain_exit.txt"
    exit 91
  fi
  dataset_validation_sha=$(sha256sum "$dataset_validation" | awk "{print \$1}")
  full_weights_size=$(stat -c %s "$full_weights")
  full_weights_sha=$(sha256sum "$full_weights" | awk "{print \$1}")
  stage1_resolved_sha=$(awk "{print \$1}" "$stage1_runtime/resolved.yaml.sha256")
  "$venv/bin/python" -B -c '\''import json,sys; from pathlib import Path; out=Path(sys.argv[1]); payload={"id":sys.argv[2],"source_head":sys.argv[3],"current_base_commit":"7d07a4212ee6858cc333e1d4fab7a37256d1f839","reconstruction":"current_causal_ar","model_identity":"exact_pi0","train_vla":False,"stage1_checkpoint":sys.argv[4],"full_weights":{"path":sys.argv[5],"size_bytes":int(sys.argv[6]),"sha256":sys.argv[7]},"dataset":{"path":sys.argv[8],"episodes":50,"frames":7188,"fps":50,"state_dim":14,"action_dim":14,"validation_sha256":sys.argv[9]},"source_config_sha256":sys.argv[10],"resolved_config_sha256":sys.argv[11],"norm_stats":{"path":sys.argv[12],"sha256":sys.argv[13]},"model_contract":{"action_horizon":50,"action_chunk":10,"action_dim":14,"z_rl_dim":2048,"prefix_seq_len":768,"image_only":True,"use_mask":True,"canonical_adapter_version":"robotwin_aloha_canonical_v1"}}; tmp=out.with_suffix(out.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); tmp.replace(out)'\'' \
    "$artifact_manifest" "$manifest_id" "$expected_head" "$stage1_endpoint" \
    "$full_weights" "$full_weights_size" "$full_weights_sha" "$dataset" \
    "$dataset_validation_sha" "$stage1_config_sha" "$stage1_resolved_sha" \
    "$norm_stats" "$expected_norm_sha"
  sha256sum "$artifact_manifest" > "$artifact_manifest.sha256"

  export RLT_LOG_ROOT="$stage2"
  export RLT_STAGE1_MODEL_PATH="$stage1_endpoint"
  export RLT_STAGE1_MANIFEST_PATH="$artifact_manifest"
  export RLT_STAGE1_MANIFEST_ID="$manifest_id"
  export RLT_STAGE1_MANIFEST_SHA256
  RLT_STAGE1_MANIFEST_SHA256=$(sha256sum "$artifact_manifest" | awk "{print \$1}")
  export RLT_NORM_STATS_SHA256="$expected_norm_sha"

  stage2_args=(
    --config-path "$worktree/examples/embodiment/config"
    --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250
    "cluster.component_placement={actor\\,env\\,rollout:4-5}"
    "runner.logger.log_path=$stage2"
    "runner.logger.experiment_name=$stage2_experiment"
    runner.resume_dir=null
    runner.ckpt_path=null
  )
  "$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" \
    "${stage2_args[@]}" --cfg job --resolve > "$stage2_runtime/resolved.yaml"
  sha256sum "$stage2_runtime/resolved.yaml" > "$stage2_runtime/resolved.yaml.sha256"
  printf "%q " "$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" "${stage2_args[@]}" > "$stage2_runtime/command.txt"
  printf "\n" >> "$stage2_runtime/command.txt"
  printf "%s\n" "$RLT_STAGE1_MANIFEST_SHA256" > "$stage2_runtime/stage1_manifest_sha256.txt"
  date --iso-8601=seconds > "$stage2_runtime/started_at.txt"
  timeout --signal=TERM --kill-after=180s 72000s \
    "$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" \
    "${stage2_args[@]}" > "$stage2_runtime/driver.log" 2>&1
  s2_rc=$?
  printf "%s\n" "$s2_rc" > "$stage2_runtime/exit_code.txt"
  date --iso-8601=seconds > "$stage2_runtime/finished_at.txt"
  printf "stage1=0 stage2=%s\n" "$s2_rc" > "$runtime/chain_exit.txt"
  date --iso-8601=seconds > "$runtime/finished_at.txt"
  exit "$s2_rc"
' _ \
  "$runtime" "$stage1_runtime" "$stage1_endpoint" "$artifact_manifest" \
  "$manifest_id" "$dataset" "$dataset_validation" "$norm_stats" \
  "$expected_norm_sha" "$expected_head" "$expected_stage1_config_sha" \
  "$expected_stage2_config_sha" "$stage2_runtime" "$stage2" \
  "$stage2_experiment" "$worktree" "$venv" \
  "$venv/bin/python" "$worktree/examples/sft/train_vla_sft.py" "${stage1_args[@]}" \
  > "$runtime/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$runtime/wrapper.pid"
printf '%s\n' "$wrapper_pid" > "$runtime/owned.pgid"

nohup setsid bash "$observer" "$wrapper_pid" "$runtime/resource.csv" \
  > "$runtime/resource_observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$runtime/resource_observer.pid"

sleep 12
if ! kill -0 "$wrapper_pid" 2>/dev/null; then
  tail -n 100 "$stage1_runtime/driver.log" >&2 || true
  exit 1
fi
kill -0 "$observer_pid"

printf 'chain_root=%s\nwrapper_pid=%s\nobserver_pid=%s\nstage1_endpoint=%s\nstage2_root=%s\n' \
  "$chain_root" "$wrapper_pid" "$observer_pid" "$stage1_endpoint" "$stage2"
tail -n 40 "$stage1_runtime/driver.log" || true
printf '%s\n' 'SZ_RLT_CURRENT_FORMAL_CHAIN_LAUNCHED'
