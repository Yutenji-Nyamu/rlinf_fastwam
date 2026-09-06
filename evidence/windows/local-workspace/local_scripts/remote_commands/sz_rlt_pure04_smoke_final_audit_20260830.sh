#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
control=/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-control-phys2-1cycle-20260830-v2
pure=/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-pure04-phys3-1cycle-20260830-v2
action=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
st=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1

date --iso-8601=seconds
for run in "$control" "$pure"; do
  printf '\nRUN %s\n' "$run"
  test "$(cat "$run/runtime/exit_code.txt")" = 0
  checkpoint=$(find "$run" -type d -name global_step_1 -print -quit)
  test -n "$checkpoint"
  metadata_count=$(find "$checkpoint" -name .metadata -type f | wc -l)
  complete_count=$(find "$checkpoint" -name complete.json -type f | wc -l)
  test "$metadata_count" -ge 1
  printf 'exit=0 checkpoint=%s files=%s metadata=%s complete_markers=%s\n' \
    "$checkpoint" \
    "$(find "$checkpoint" -type f | wc -l)" \
    "$metadata_count" \
    "$complete_count"
  "$venv/bin/python" -c 'from omegaconf import OmegaConf; import sys; c=OmegaConf.load(sys.argv[1]); print("mode=%s max_steps=%s train_env=%s gb=%s mb=%s"%(c.algorithm.rlt_dvac.mode,c.runner.max_steps,c.env.train.total_num_envs,c.actor.global_batch_size,c.actor.micro_batch_size))' "$run/runtime/resolved.yaml"
  "$venv/bin/python" -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print("resource_samples=%d min_available_gib=%.2f gpu2_peak_mib=%d gpu3_peak_mib=%d gpu4_peak_mib=%d gpu5_peak_mib=%d gpu6_peak_mib=%d gpu7_peak_mib=%d"%(len(r),min(int(x["mem_available_kib"]) for x in r)/1024/1024,*[max(int(x[f"gpu{i}_mib"]) for x in r) for i in range(2,8)]))' "$run/runtime/resources.csv"
  grep -E 'rlt_dvac/(enabled|mode_apply|baseline_frozen|baseline_count|success_query_ratio|success_bc_applied|weight_mean|weight_ess_ratio)|rlt/(actor_updates_run|critic_updates_run)' "$run/runtime/driver.log" | tail -n 32 || true
  test "$(grep -Eci 'Traceback|OOM|out of memory|RayActorError|worker.*died' "$run/runtime/driver.log" || true)" = 0
done

printf '\nGRPO_SURVIVAL\n'
for run in "$action" "$st"; do
  pid=$(<"$run/runtime/wrapper.pid")
  kill -0 "$pid"
  before=$(awk -F '\t' -v target="$run" '$1==target {print $3":"$4}' "$control/runtime/grpo_before.tsv")
  printf '%s alive=1 before=%s after=%s:%s fatal=%s\n' \
    "$run" "$before" "$(stat -c %s "$run/runtime/driver.log")" "$(stat -c %Y "$run/runtime/driver.log")" \
    "$(grep -Eci 'Traceback|OOM|out of memory|worker.*died|RayActorError' "$run/runtime/driver.log" || true)"
done

grep '^MemAvailable:' /proc/meminfo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'SMOKE_FINAL_AUDIT_OK\n'
