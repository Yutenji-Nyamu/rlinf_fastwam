#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1
experiment_name=robotwin_adjust_bottle_ogpo_ca_formal_35k_v1
run_dir="${run_root}/${experiment_name}"
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
resolved="$runtime_root/resolved.yaml"
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
source_config_sha256=f777a0caceacf260c427093fdb0a493f6bd77c3d444cf233a57727ce9027c291
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
norm_sha256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
resolved_sha256=77419258766880fca7b8d15d725dfb9f2fba5b88d2bfe21eb918fd760e9a7bc9

cd "$repo"
test "$(git branch --show-current)" = codex/ogpo-pi0-robotwin
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --short)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
test "$(sha256sum examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml | awk '{print $1}')" = "$source_config_sha256"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$norm_sha256"
test "$(sha256sum "$resolved" | awk '{print $1}')" = "$resolved_sha256"
test "$(sha256sum "$runtime_root/source_config.yaml" | awk '{print $1}')" = "$source_config_sha256"
test ! -e "$run_root"
test ! -e "$runtime_root/started_at.txt"
test ! -e "$runtime_root/exact_command.txt"
test ! -e "$runtime_root/run_provenance.tsv"
test ! -e "$runtime_root/stop_conditions.txt"
test ! -e "$runtime_root/resources_before.txt"

mapfile -t active_rows < <(
  {
    ps -eo pid=,comm=,args= \
      | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
    pgrep -ax raylet || true
    pgrep -ax gcs_server || true
  }
)
test "${#active_rows[@]}" = 0
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF')"

RESOLVED="$resolved" RUN_ROOT="$run_root" EXPERIMENT_NAME="$experiment_name" \
  "${venv}/bin/python" -B - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["RESOLVED"])
assert cfg.runner.logger.log_path == os.environ["RUN_ROOT"]
assert cfg.runner.logger.experiment_name == os.environ["EXPERIMENT_NAME"]
assert cfg.runner.max_epochs == 1_000_000
assert cfg.runner.max_steps == -1
assert cfg.runner.resume_dir is None
assert cfg.runner.ckpt_path is None
assert cfg.runner.use_training_pipeline is False
assert cfg.runner.overlap_env_bootstrap is False

ogpo = cfg.algorithm.ogpo
assert cfg.algorithm.loss_type == "embodied_ogpo"
assert cfg.algorithm.adv_type == "ogpo"
assert cfg.algorithm.gamma == 0.999
assert ogpo.variant == "ogpo_ca"
assert ogpo.model_horizon == 50
assert ogpo.execution_horizon == 10
assert ogpo.model_action_dim == 32
assert ogpo.active_action_dim == 14
assert ogpo.flow_steps == 4
assert ogpo.sigma_init == 0.01
assert ogpo.gaussian_clip == 3.0
assert ogpo.normalize_denoising_horizon is True
assert ogpo.normalize_act_space_dimension is True
assert ogpo.candidate_group_size == 8
assert ogpo.candidate_microbatch_per_rank == 32
assert ogpo.clip_epsilon == 0.01
assert ogpo.bc_coeff == 1.0
assert ogpo.actor_tau == 0.005
assert ogpo.critic_tau == 0.05
assert ogpo.num_q_heads == 10
assert list(ogpo.critic_hidden_dims) == [512, 512, 512, 512, 512]
assert ogpo.critic_lr == 3.0e-4
assert ogpo.state_batch_size == 64
assert ogpo.start_training_rows == 10_000
assert ogpo.total_online_rows == 35_000
assert ogpo.replay_capacity == 40_000
assert ogpo.utd_q == 0.1
assert ogpo.utd_pi == 0.1
assert ogpo.offline_ratio == 0.0
assert ogpo.use_success_buffer_q is False
assert ogpo.best_of_n == 1
assert ogpo.baseline_eval is True
assert ogpo.final_eval is True
assert ogpo.eval_interval_rows == 20_000
assert ogpo.checkpoint_interval_rows == 50_000

assert cfg.cluster.num_nodes == 1
# Hydra's --cfg YAML printer drops quoting around the colon-containing source
# string. Reloading the evidence artifact therefore represents it as this map;
# the live run still composes the original quoted source YAML.
assert OmegaConf.to_container(cfg.cluster.component_placement) == {
    "actor, env, rollout": "0-1"
}
assert cfg.env.train.rollout_epoch == 1
assert cfg.env.train.total_num_envs == 8
assert cfg.env.train.auto_reset is False
assert cfg.env.train.max_episode_steps == 200
assert cfg.env.train.max_steps_per_rollout_epoch == 200
assert cfg.env.eval.rollout_epoch == 5
assert cfg.env.eval.total_num_envs == 4
assert cfg.env.eval.auto_reset is True
assert cfg.env.eval.max_episode_steps == 200
assert cfg.env.eval.max_steps_per_rollout_epoch == 200
assert cfg.actor.micro_batch_size == 32
assert cfg.actor.global_batch_size == 64
assert cfg.actor.model.num_action_chunks == 10
assert cfg.actor.model.openpi.action_horizon == 50
assert cfg.actor.model.openpi.action_chunk == 10
assert cfg.actor.model.openpi.action_env_dim == 14
assert cfg.actor.model.openpi.num_steps == 4
assert cfg.actor.model.openpi.train_expert_only is True
assert cfg.actor.model.openpi.use_ogpo is True
assert cfg.actor.model.openpi.use_dsrl is False
assert cfg.actor.model.openpi.use_rlt is False
assert cfg.actor.optim.lr == 5.6e-6
assert cfg.actor.optim.total_training_steps == 230_000
assert cfg.actor.optim.lr_warmup_steps == 0
assert cfg.actor.optim.lr_scheduler == "constant"
assert cfg.actor.fsdp_config.sharding_strategy == "full_shard"
assert cfg.actor.fsdp_config.use_orig_params is True
assert "UNRESOLVED" not in Path(os.environ["RESOLVED"]).read_text()
PY

formal_cmd=(
  "${venv}/bin/python" -B
  examples/embodiment/train_embodied_agent.py
  --config-path "${repo}/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_ogpo_openpi
  "runner.logger.log_path=${run_root}"
  "runner.logger.experiment_name=${experiment_name}"
  "runner.resume_dir=null"
  "runner.ckpt_path=null"
  "algorithm.ogpo.total_online_rows=35000"
  "algorithm.ogpo.start_training_rows=10000"
  "algorithm.ogpo.utd_q=0.1"
  "algorithm.ogpo.utd_pi=0.1"
  "algorithm.ogpo.replay_capacity=40000"
)
printf '%q ' "${formal_cmd[@]}" >"$runtime_root/exact_command.txt"
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
  printf 'total_primitive_rows\t35000\n'
  printf 'warmup_primitive_rows\t10000\n'
  printf 'learning_primitive_rows\t25000\n'
  printf 'paired_utd\t0.1\n'
  printf 'actor_updates_target\t2500\n'
  printf 'critic_updates_target\t2500\n'
  printf 'imagined_candidate_chains_target\t1280000\n'
  printf 'td_next_action_chains_target\t160000\n'
  printf 'replay_capacity_global\t40000\n'
  printf 'train_envs\t8\n'
  printf 'train_episode_step_cap\t200\n'
  printf 'train_policy_queries_nominal\t3500\n'
  printf 'eval_count_target\t3\n'
  printf 'eval_episodes_per_eval\t20\n'
  printf 'eval_episodes_target\t60\n'
  printf 'checkpoint_interval_rows\t50000\n'
  printf 'checkpoints_target\t1\n'
  printf 'intermediate_recovery_checkpoints\t0\n'
  printf 'wall_time_estimate_hours\t19-21-not-a-hard-bound\n'
} >"$runtime_root/run_provenance.tsv"

{
  printf '%s\n' 'NATURAL COMPLETE: exact total_online_rows=35000, 2500 actor updates, 2500 critic updates, baseline/periodic/final evals, final checkpoint, and driver exit code 0.'
  printf '%s\n' 'STRUCTURAL FAILURE: nonzero driver exit, uncaught exception, NaN/Inf failure, GPU/cgroup OOM, Ray actor death, or missing final artifacts.'
  printf '%s\n' 'No automatic algorithm, batch, environment-count, precision, checkpoint-interval, or budget fallback.'
  printf '%s\n' 'The unchanged 50000-row checkpoint interval means there is no intermediate recovery checkpoint; the runner performs one final save at exact row 35000.'
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
printf '%s\n' OGPO_ROBOTWIN_FORMAL_PREPARED
