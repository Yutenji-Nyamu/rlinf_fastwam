#!/usr/bin/env bash
set -u
metrics=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/metrics.log
tail -70 "$metrics" | grep -E 'Global Step:|Elapsed:|success_once=|global_min_replay_size=|actor_updates_run=|critic_updates_run=|update_step=|ready_for_online=|baseline_count=|baseline_frozen=|baseline_mean=|baseline_std='
