#!/usr/bin/env bash

# Narrow read-only verification of the two OGPO v2 recovery points and final log.

run=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2/robotwin_adjust_bottle_ogpo_ca_formal_90k_utd005_v2
export_dir=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2

printf '## run_root\n'
stat -c '%n | %F | %y' "$run" "$run/checkpoints" 2>&1
find "$run" -maxdepth 2 -type f \
  -printf '%P | %s bytes | %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort

printf '\n## checkpoint_summary\n'
for step in 22 43; do
  checkpoint="$run/checkpoints/global_step_$step"
  printf '\ncheckpoint=%s\n' "$checkpoint"
  stat -c '%F | %y' "$checkpoint" 2>&1
  printf 'apparent_size='; du -sh --apparent-size "$checkpoint" 2>/dev/null | awk '{print $1}'
  printf 'disk_size='; du -sh "$checkpoint" 2>/dev/null | awk '{print $1}'
  printf 'file_count='; find "$checkpoint" -type f 2>/dev/null | wc -l
  printf 'notable_files:\n'
  find "$checkpoint" -type f \
    \( -iname '*complete*' -o -iname '*success*' -o -iname '*manifest*' -o -iname '*meta*' -o -iname '*state*' -o -name '*.json' -o -name '*.yaml' -o -name '*.txt' \) \
    -printf '%P | %s bytes\n' 2>/dev/null | sort | head -n 120
done

printf '\n## export_file_inventory\n'
find "$export_dir" -maxdepth 3 -type f \
  -printf '%P | %s bytes | %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort

printf '\n## small_export_text\n'
while IFS= read -r file; do
  printf '\nfile=%s\n' "$file"
  sed -n '1,160p' "$file" 2>/dev/null || true
done < <(
  find "$export_dir" -maxdepth 3 -type f -size -1048576c \
    \( -name '*.json' -o -name '*.txt' -o -name '*.yaml' -o -name '*.yml' -o -name '*.status' -o -name '*.exit' \) \
    -print 2>/dev/null | sort
)

printf '\n## final_metric_fields\n'
if [[ -r "$run/metrics.log" ]]; then
  grep -aE 'ogpo/(total_online_rows|policy_version|actor_updates|critic_updates|replay_rows|success_rows)|success_at_end=|num_trajectories=|return=' "$run/metrics.log" | tail -n 40
else
  printf 'metrics.log missing\n'
fi
