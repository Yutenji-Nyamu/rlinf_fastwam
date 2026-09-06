#!/usr/bin/env bash
set -u
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
v2runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
stat -c 'DRIVER_LOG_SIZE=%s MTIME=%y' "$runtime/driver.log"
printf 'CURRENT_WORKERS\n'
ps -p 74235,74237,74238,74249,74262,74268 -o pid=,ppid=,stat=,etime=,rss=,pcpu=,comm=,args= 2>/dev/null || true
printf 'CURRENT_ERROR_CONTEXT\n'
grep -a -B 5 -A 12 -E "No module named 'curobo.types.math'|Generating Rollout Epochs|ActorDiedError|RayActorError" "$runtime/driver.log" | tail -n 100 || true
printf 'V2_CUROBO_AND_PROGRESS\n'
grep -a -n -E "No module named 'curobo.types.math'|Generating Rollout Epochs" "$v2runtime/driver.log" 2>/dev/null | head -n 20 || true
printf 'RAY_ACTOR_STATES\n'
/root/autodl-tmp/RLinf/.venv/bin/ray list actors --detail 2>/dev/null | grep -E 'actor_id:|class_name:|state:|death_cause:' | tail -n 80 || true
