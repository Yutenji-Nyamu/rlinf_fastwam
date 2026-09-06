#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
dataset=/data/chenyiteng/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
dataset_validation=$dataset/rlt_canonical_validation.json
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
norm_stats=$model/physical-intelligence/robotwin/norm_stats.json
run_root=/data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823
runtime=$run_root/runtime
experiment=robotwin_adjust_bottle_rlt_stage1_current_ar_smoke2_v1
endpoint=$run_root/$experiment/checkpoints/global_step_2
artifact_dir=$run_root/artifacts
artifact_manifest=$artifact_dir/stage1_artifact_manifest.json
manifest_id=sz-rlt-stage1-current-ar-smoke2-v1
observer=/data/chenyiteng/results/rlinf-rlt/launchers/shenzhen_rlt_current_smoke_observer_gpu4_5_20260823.sh
expected_head=bdd875283b3f3516c439e5c79c902cf5c2da58b6
expected_branch=codex/sz-rlt-pi0-robotwin-ar
expected_config_sha=63b934af03a0e649256104e986fa1524391d4a6cb319187849f0ad39f1a88329
expected_norm_sha=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
config=$worktree/examples/sft/config/robotwin_rlt_stage1_sft_openpi_current_ar.yaml

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$worktree" branch --show-current)" = "$expected_branch"
test "$(git -C "$worktree" rev-parse "personal/$expected_branch")" = "$expected_head"
test -z "$(git -C "$worktree" status --porcelain)"
test "$(sha256sum "$config" | awk '{print $1}')" = "$expected_config_sha"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$expected_norm_sha"
test -x "$venv/bin/python"
test -d "$dataset"
test -s "$dataset/meta/info.json"
test -s "$dataset_validation"
test -s "$model/model-00001-of-00002.safetensors"
test -s "$model/model-00002-of-00002.safetensors"
test -r "$observer"
test ! -e "$run_root"

if pgrep -u "$(id -u)" -af 'train_vla_sft.py.*robotwin_rlt_stage1' \
  | grep -v -E 'pgrep -u|stage1_smoke2_gpu4_5' >/dev/null; then
  printf '%s\n' 'an RLT Stage1 process already exists' >&2
  exit 1
fi
# Current RLinf SFT also constructs Cluster and launches its actor group via Ray.
# Do not attach it to an unrelated live DSRL/RLT control plane.
if pgrep -u "$(id -u)" -x raylet >/dev/null || \
  pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
  printf '%s\n' 'chenyiteng already has a live Ray cluster; wait for DSRL to exit' >&2
  exit 1
fi
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 4-5 are not idle' >&2
  exit 1
fi

source "$venv/bin/activate"
unset CUDA_VISIBLE_DEVICES
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_RLT_CLEAN50_PATH="$dataset"
export ROBOTWIN_PI0_BASE_PATH="$model"
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export RLT_LOG_ROOT="$run_root"
export REPO_PATH="$worktree"
export EMBODIED_PATH="$worktree/examples/embodiment"
export PYTHONPATH="$worktree${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$runtime" "$artifact_dir"

args=(
  --config-path "$worktree/examples/sft/config"
  --config-name robotwin_rlt_stage1_sft_openpi_current_ar
  'cluster.component_placement={actor\,env\,rollout:4-5}'
  "runner.logger.log_path=$run_root"
  "runner.logger.experiment_name=$experiment"
  'runner.max_steps=2'
  'runner.save_interval=2'
  'runner.val_check_interval=-1'
  'actor.micro_batch_size=16'
  'actor.global_batch_size=32'
  'actor.optim.total_training_steps=2'
  'actor.optim.lr_warmup_steps=1'
)

"$venv/bin/python" "$worktree/examples/sft/train_vla_sft.py" \
  "${args[@]}" --cfg job --resolve > "$runtime/resolved.yaml"
sha256sum "$runtime/resolved.yaml" > "$runtime/resolved.yaml.sha256"
printf '%q ' \
  "$venv/bin/python" \
  "$worktree/examples/sft/train_vla_sft.py" \
  "${args[@]}" \
  > "$runtime/command.txt"
printf '\n' >> "$runtime/command.txt"

dataset_validation_sha=$(sha256sum "$dataset_validation" | awk '{print $1}')
resolved_sha=$(awk '{print $1}' "$runtime/resolved.yaml.sha256")
printf '%s\n' \
  "prepared_at=$(date --iso-8601=seconds)" \
  "source_head=$expected_head" \
  'physical_gpus=4,5' \
  'actor_world_size=2' \
  "dataset=$dataset" \
  'dataset_episodes=50' \
  'dataset_frames=7188' \
  'dataset_fps=50' \
  'state_dim=14' \
  'action_dim=14' \
  'micro_batch_per_rank=16' \
  'global_batch=32' \
  'optimizer_steps=2' \
  'sample_presentations=64' \
  'checkpoint=global_step_2' \
  "manifest_id=$manifest_id" \
  > "$runtime/launch_manifest.txt"

nohup setsid bash -c '
  runtime=$1
  endpoint=$2
  artifact_manifest=$3
  manifest_id=$4
  dataset=$5
  dataset_validation_sha=$6
  norm_stats=$7
  norm_sha=$8
  source_head=$9
  config_sha=${10}
  resolved_sha=${11}
  py=${12}
  shift 12
  date --iso-8601=seconds > "$runtime/started_at.txt"
  timeout --signal=TERM --kill-after=120s 1800s "$@" > "$runtime/driver.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    full_weights=$endpoint/actor/model_state_dict/full_weights.pt
    dcp_metadata=$endpoint/actor/dcp_checkpoint/.metadata
    if [ ! -s "$full_weights" ] || [ ! -s "$dcp_metadata" ]; then
      printf "%s\n" "Stage1 exited zero but endpoint is incomplete" >> "$runtime/driver.log"
      rc=91
    else
      full_weights_size=$(stat -c %s "$full_weights")
      full_weights_sha=$(sha256sum "$full_weights" | cut -d" " -f1)
      "$py" -B -c '\''import json,sys; from pathlib import Path; out=Path(sys.argv[1]); payload={"id":sys.argv[2],"source_head":sys.argv[3],"current_base_commit":"7d07a4212ee6858cc333e1d4fab7a37256d1f839","reconstruction":"current_causal_ar","model_identity":"exact_pi0","train_vla":False,"stage1_checkpoint":sys.argv[4],"full_weights":{"path":sys.argv[5],"size_bytes":int(sys.argv[6]),"sha256":sys.argv[7]},"dataset":{"path":sys.argv[8],"episodes":50,"frames":7188,"fps":50,"state_dim":14,"action_dim":14,"validation_sha256":sys.argv[9]},"source_config_sha256":sys.argv[10],"resolved_config_sha256":sys.argv[11],"norm_stats":{"path":sys.argv[12],"sha256":sys.argv[13]},"model_contract":{"action_horizon":50,"action_chunk":10,"action_dim":14,"z_rl_dim":2048,"prefix_seq_len":768,"image_only":True,"use_mask":True,"canonical_adapter_version":"robotwin_aloha_canonical_v1"}}; tmp=out.with_suffix(out.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); tmp.replace(out)'\'' \
        "$artifact_manifest" "$manifest_id" "$source_head" "$endpoint" \
        "$full_weights" "$full_weights_size" "$full_weights_sha" \
        "$dataset" "$dataset_validation_sha" "$config_sha" "$resolved_sha" \
        "$norm_stats" "$norm_sha"
      sha256sum "$artifact_manifest" > "$artifact_manifest.sha256"
    fi
  fi
  printf "%s\n" "$rc" > "$runtime/exit_code.txt"
  date --iso-8601=seconds > "$runtime/finished_at.txt"
  exit "$rc"
' _ \
  "$runtime" \
  "$endpoint" \
  "$artifact_manifest" \
  "$manifest_id" \
  "$dataset" \
  "$dataset_validation_sha" \
  "$norm_stats" \
  "$expected_norm_sha" \
  "$expected_head" \
  "$expected_config_sha" \
  "$resolved_sha" \
  "$venv/bin/python" \
  "$venv/bin/python" \
  "$worktree/examples/sft/train_vla_sft.py" \
  "${args[@]}" \
  > "$runtime/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$runtime/wrapper.pid"
printf '%s\n' "$wrapper_pid" > "$runtime/owned.pgid"

nohup setsid bash "$observer" "$wrapper_pid" "$runtime/resource.csv" \
  > "$runtime/resource_observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$runtime/resource_observer.pid"

sleep 10
if ! kill -0 "$wrapper_pid" 2>/dev/null; then
  tail -n 100 "$runtime/driver.log" >&2 || true
  exit 1
fi
kill -0 "$observer_pid"

printf 'run_root=%s\nruntime=%s\nwrapper_pid=%s\nobserver_pid=%s\nresolved_sha256=%s\nendpoint=%s\nartifact_manifest=%s\n' \
  "$run_root" \
  "$runtime" \
  "$wrapper_pid" \
  "$observer_pid" \
  "$resolved_sha" \
  "$endpoint" \
  "$artifact_manifest"
tail -n 30 "$runtime/driver.log" || true
printf '%s\n' 'SZ_RLT_CURRENT_STAGE1_SMOKE2_LAUNCHED'
