#!/usr/bin/env bash
set +e
declare -- VENV="/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin"
declare -- WT="/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421"
declare -- RUN="/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2"
declare -a ARGS=([0]="--config-path" [1]="/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421/examples/embodiment/config" [2]="--config-name" [3]="robotwin_adjust_bottle_grpo_openpi" [4]="cluster.component_placement={actor\\, env\\, rollout:4-7}" [5]="runner.logger.log_path=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2" [6]="runner.max_epochs=1000" [7]="runner.max_steps=100" [8]="runner.val_check_interval=10" [9]="runner.save_interval=10" [10]="runner.resume_dir=null" [11]="algorithm.update_epoch=2" [12]="env.train.total_num_envs=128" [13]="env.train.rollout_epoch=4" [14]="env.train.max_episode_steps=200" [15]="env.train.max_steps_per_rollout_epoch=200" [16]="env.train.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support" [17]="env.eval.total_num_envs=64" [18]="env.eval.rollout_epoch=1" [19]="env.eval.max_episode_steps=200" [20]="env.eval.max_steps_per_rollout_epoch=200" [21]="env.eval.use_fixed_reset_state_ids=true" [22]="env.eval.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support" [23]="actor.micro_batch_size=32" [24]="actor.global_batch_size=2048" [25]="actor.model.model_path=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50")
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$RUN/driver.log" 2>&1
rc=$?
printf "exit_time=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$rc" > "$RUN/driver.exit"
exit "$rc"
