#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
experiment_name=robotwin_adjust_bottle_ogpo_ca_formal_90k_utd005_v2
run_dir="${run_root}/${experiment_name}"
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime
resolved="$runtime_root/resolved.yaml"
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
source_config_sha256=f777a0caceacf260c427093fdb0a493f6bd77c3d444cf233a57727ce9027c291
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
norm_sha256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
resolved_sha256=352f8e80752d60624a0c53c62d21dcc10bdc8e6712a433c18d0eec0ee1f56a36

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
expected_ogpo = {
    "variant": "ogpo_ca",
    "model_horizon": 50,
    "execution_horizon": 10,
    "model_action_dim": 32,
    "active_action_dim": 14,
    "flow_steps": 4,
    "sigma_init": 0.01,
    "gaussian_clip": 3.0,
    "normalize_denoising_horizon": True,
    "normalize_act_space_dimension": True,
    "candidate_group_size": 8,
    "candidate_microbatch_per_rank": 32,
    "clip_epsilon": 0.01,
    "bc_coeff": 1.0,
    "actor_tau": 0.005,
    "critic_tau": 0.05,
    "num_q_heads": 10,
    "critic_lr": 3.0e-4,
    "state_batch_size": 64,
    "start_training_rows": 10_000,
    "total_online_rows": 90_000,
    "replay_capacity": 100_000,
    "utd_q": 0.05,
    "utd_pi": 0.05,
    "offline_ratio": 0.0,
    "use_success_buffer_q": False,
    "best_of_n": 1,
    "baseline_eval": True,
    "final_eval": True,
    "eval_interval_rows": 10_000,
    "checkpoint_interval_rows": 30_000,
}
for key, value in expected_ogpo.items():
    assert getattr(ogpo, key) == value, (key, getattr(ogpo, key), value)
assert list(ogpo.critic_hidden_dims) == [512, 512, 512, 512, 512]
assert cfg.algorithm.loss_type == "embodied_ogpo"
assert cfg.algorithm.adv_type == "ogpo"
assert cfg.algorithm.gamma == 0.999

assert cfg.cluster.num_nodes == 1
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
  "algorithm.ogpo.total_online_rows=90000"
  "algorithm.ogpo.start_training_rows=10000"
  "algorithm.ogpo.utd_q=0.05"
  "algorithm.ogpo.utd_pi=0.05"
  "algorithm.ogpo.replay_capacity=100000"
  "algorithm.ogpo.baseline_eval=true"
  "algorithm.ogpo.final_eval=true"
  "algorithm.ogpo.eval_interval_rows=10000"
  "algorithm.ogpo.checkpoint_interval_rows=30000"
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
  printf 'fresh_start\ttrue\n'
  printf 'total_primitive_rows\t90000\n'
  printf 'warmup_primitive_rows\t10000\n'
  printf 'learning_primitive_rows\t80000\n'
  printf 'paired_utd\t0.05\n'
  printf 'actor_updates_target\t4000\n'
  printf 'critic_updates_target\t4000\n'
  printf 'critic_and_bc_batch_samples_target\t256000\n'
  printf 'imagined_candidate_chains_target\t2048000\n'
  printf 'td_next_action_chains_target\t256000\n'
  printf 'replay_capacity_global\t100000\n'
  printf 'train_envs\t8\n'
  printf 'train_episode_step_cap\t200\n'
  printf 'train_episode_equivalents\t450\n'
  printf 'train_policy_queries_nominal\t9000\n'
  printf 'eval_interval_rows\t10000\n'
  printf 'eval_count_target\t10\n'
  printf 'eval_episodes_per_eval\t20\n'
  printf 'eval_episodes_target\t200\n'
  printf 'checkpoint_interval_rows\t30000\n'
  printf 'checkpoints_target\t3\n'
  printf 'intermediate_recovery_checkpoints\t2\n'
  printf 'wall_time_nominal_hours\t23.71\n'
  printf 'wall_time_variation_note\tapproximately_plus_or_minus_10_percent_not_a_hard_deadline\n'
} >"$runtime_root/run_provenance.tsv"

{
  printf '%s\n' 'NATURAL COMPLETE: exact total_online_rows=90000, 4000 actor updates, 4000 critic updates, ten fixed-policy eval points, checkpoints at 30000/60000/90000 rows, and driver exit code 0.'
  printf '%s\n' 'STRUCTURAL FAILURE: nonzero driver exit, uncaught training exception, NaN/Inf failure, GPU/cgroup OOM, Ray actor death, or missing required final artifacts.'
  printf '%s\n' 'No automatic algorithm, batch, environment-count, precision, checkpoint-interval, or budget fallback.'
  printf '%s\n' 'The 23.71-hour estimate is nominal and not an automatic wall-clock stop.'
} >"$runtime_root/stop_conditions.txt"

{
  date --iso-8601=seconds
  git status --short --branch
  git rev-parse HEAD
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
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
printf '%s\n' OGPO_ROBOTWIN_FORMAL_V2_PREPARED_FROM_PARTIAL
