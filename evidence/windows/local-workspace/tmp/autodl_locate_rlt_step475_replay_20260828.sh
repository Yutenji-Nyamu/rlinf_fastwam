#!/usr/bin/env bash
set -u
find /root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3 \
  -path '*/checkpoints/global_step_475/actor/sac_components/replay_buffer/rank_0/trajectory_index.json' -print
