#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_smoke_20260807_v1
experiment_name=robotwin_adjust_bottle_ogpo_ca_smoke_8env_1update_v1
run_dir="${run_root}/${experiment_name}"
evidence_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1
runtime_root="${evidence_root}/runtime"
source_config=examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml
source_config_sha256=f777a0caceacf260c427093fdb0a493f6bd77c3d444cf233a57727ce9027c291
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
norm_sha256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

cd "$repo"
test "$(git branch --show-current)" = codex/ogpo-pi0-robotwin
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --short)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
test "$(sha256sum "$source_config" | awk '{print $1}')" = "$source_config_sha256"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$norm_sha256"
test -d "$assets"
test ! -e "$run_root"
test ! -L "$run_root"
test ! -e "$evidence_root"
test ! -L "$evidence_root"

mapfile -t active_rows < <(
  {
    ps -eo pid=,comm=,args= \
      | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
    pgrep -ax raylet || true
    pgrep -ax gcs_server || true
  }
)
test "${#active_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF'
)
test "${#compute_rows[@]}" = 0

mkdir -p "$runtime_root"
export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

overrides=(
  "runner.logger.log_path=${run_root}"
  "runner.logger.experiment_name=${experiment_name}"
  "runner.max_epochs=2"
  "runner.max_steps=2"
  "runner.resume_dir=null"
  "runner.ckpt_path=null"
  "algorithm.ogpo.start_training_rows=79"
  "algorithm.ogpo.total_online_rows=80"
  "algorithm.ogpo.replay_capacity=80"
  "algorithm.ogpo.utd_q=1.0"
  "algorithm.ogpo.utd_pi=1.0"
  "algorithm.ogpo.baseline_eval=false"
  "algorithm.ogpo.final_eval=true"
  "algorithm.ogpo.eval_interval_rows=80"
  "algorithm.ogpo.checkpoint_interval_rows=80"
  "env.train.rollout_epoch=1"
  "env.train.total_num_envs=8"
  "env.train.max_steps_per_rollout_epoch=10"
  "env.eval.rollout_epoch=1"
  "env.eval.total_num_envs=4"
  "env.eval.max_steps_per_rollout_epoch=10"
  "actor.optim.total_training_steps=1"
)

cp "$source_config" "$runtime_root/source_config.yaml"
"${venv}/bin/python" -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_ogpo_openpi \
  "${overrides[@]}" \
  --cfg job --resolve >"$runtime_root/resolved.yaml"
resolved_sha256="$(sha256sum "$runtime_root/resolved.yaml" | awk '{print $1}')"

RESOLVED="$runtime_root/resolved.yaml" RUN_ROOT="$run_root" \
EXPERIMENT_NAME="$experiment_name" \
  "${venv}/bin/python" -B - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["RESOLVED"])
assert cfg.runner.logger.log_path == os.environ["RUN_ROOT"]
assert cfg.runner.logger.experiment_name == os.environ["EXPERIMENT_NAME"]
assert cfg.runner.max_epochs == 2
assert cfg.runner.max_steps == 2
assert cfg.runner.resume_dir is None
assert cfg.runner.ckpt_path is None
assert cfg.runner.use_training_pipeline is False
assert cfg.algorithm.ogpo.start_training_rows == 79
assert cfg.algorithm.ogpo.total_online_rows == 80
assert cfg.algorithm.ogpo.replay_capacity == 80
assert cfg.algorithm.ogpo.utd_q == 1.0
assert cfg.algorithm.ogpo.utd_pi == 1.0
assert cfg.algorithm.ogpo.state_batch_size == 64
assert cfg.algorithm.ogpo.candidate_group_size == 8
assert cfg.algorithm.ogpo.candidate_microbatch_per_rank == 32
assert cfg.algorithm.ogpo.baseline_eval is False
assert cfg.algorithm.ogpo.final_eval is True
assert cfg.algorithm.ogpo.eval_interval_rows == 80
assert cfg.algorithm.ogpo.checkpoint_interval_rows == 80
assert cfg.env.train.rollout_epoch == 1
assert cfg.env.train.total_num_envs == 8
assert cfg.env.train.max_episode_steps == 200
assert cfg.env.train.max_steps_per_rollout_epoch == 10
assert cfg.env.eval.rollout_epoch == 1
assert cfg.env.eval.total_num_envs == 4
assert cfg.env.eval.max_episode_steps == 200
assert cfg.env.eval.max_steps_per_rollout_epoch == 10
assert cfg.actor.model.num_action_chunks == 10
assert cfg.actor.model.openpi.action_horizon == 50
assert cfg.actor.optim.total_training_steps == 1
assert "UNRESOLVED" not in Path(os.environ["RESOLVED"]).read_text()
PY

smoke_cmd=(
  "${venv}/bin/python" -B
  examples/embodiment/train_embodied_agent.py
  --config-path "${repo}/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_ogpo_openpi
  "${overrides[@]}"
)
printf '%q ' "${smoke_cmd[@]}" >"$runtime_root/exact_command.txt"
printf '\n' >>"$runtime_root/exact_command.txt"

{
  printf 'prepared_at\t%s\n' "$(date --iso-8601=seconds)"
  printf 'branch\t%s\n' "$(git branch --show-current)"
  printf 'head\t%s\n' "$(git rev-parse HEAD)"
  printf 'upstream_left_right\t%s\n' "$(git rev-list --left-right --count HEAD...@{upstream})"
  printf 'source_config_sha256\t%s\n' "$source_config_sha256"
  printf 'norm_sha256\t%s\n' "$norm_sha256"
  printf 'resolved_sha256\t%s\n' "$resolved_sha256"
  printf 'run_root\t%s\n' "$run_root"
  printf 'run_dir\t%s\n' "$run_dir"
  printf 'runtime_root\t%s\n' "$runtime_root"
  printf 'train_envs\t8\n'
  printf 'train_policy_queries_nominal\t8\n'
  printf 'train_primitive_rows_target\t80\n'
  printf 'paired_updates_target\t1\n'
  printf 'imagined_candidate_chains_target\t512\n'
  printf 'eval_envs\t4\n'
  printf 'eval_policy_queries\t4\n'
  printf 'eval_primitive_slots\t40\n'
  printf 'checkpoints_target\t1\n'
  printf 'runner_cycles_nominal\t1\n'
  printf 'runner_cycles_hard_cap\t2\n'
  printf 'wall_timeout_seconds\t1800\n'
} >"$runtime_root/run_provenance.tsv"

{
  printf '%s\n' 'PASS: exit code 0, total_online_rows=80, actor_updates=1, critic_updates=1, one eval and one checkpoint.'
  printf '%s\n' 'FAIL/STOP: two cycles still below 80 rows; update count differs from one; NaN/Inf; shape/channel/Ray error; GPU/cgroup OOM; or 1800 s timeout.'
  printf '%s\n' 'No automatic algorithm, batch, environment-count, or precision fallback.'
} >"$runtime_root/stop_conditions.txt"

{
  date --iso-8601=seconds
  git status --short --branch
  git rev-parse HEAD
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  free -b
  cat /sys/fs/cgroup/memory.current
  cat /sys/fs/cgroup/memory.events
  df -B1 /root/autodl-tmp
} >"$runtime_root/resources_before.txt"

printf 'PREPARED_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'HEAD\t%s\n' "$expected_head"
printf 'RESOLVED_SHA256\t%s\n' "$resolved_sha256"
printf 'RUNTIME_ROOT\t%s\n' "$runtime_root"
printf 'RUN_DIR\t%s\n' "$run_dir"
printf 'EXACT_COMMAND\t%s\n' "$(cat "$runtime_root/exact_command.txt")"
printf '%s\n' OGPO_ROBOTWIN_SMOKE_PREPARED
