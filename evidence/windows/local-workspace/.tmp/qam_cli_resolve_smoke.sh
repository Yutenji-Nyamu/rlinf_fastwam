set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
expected=/root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml
actual=/root/autodl-tmp/qam_qonly_smoke_cli_resolved_20260731_v1.yaml
cd "$repo"
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin_RLinf"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
"$venv/bin/python" -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_qam_openpi \
  runner.logger.log_path=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1 \
  runner.logger.experiment_name=robotwin_adjust_bottle_qam_qonly_smoke_20260731_v1 \
  runner.max_steps=1 \
  runner.save_interval=1 \
  algorithm.qam.phase=q_only \
  algorithm.qam.warmup_global_inserts=2 \
  algorithm.qam.min_replay_per_rank=1 \
  algorithm.qam.max_updates_per_step=2 \
  actor.global_batch_size=2 \
  actor.micro_batch_size=1 \
  +actor.fsdp_config.save_full_model_weights=false \
  --cfg job \
  --resolve >"$actual"
sha256sum "$expected" "$actual"
cmp "$expected" "$actual"
printf 'QAM_CLI_RESOLVED_EXACT=1\n'
