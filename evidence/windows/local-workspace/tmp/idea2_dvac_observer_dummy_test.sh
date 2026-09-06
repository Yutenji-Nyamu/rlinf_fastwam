set -euo pipefail

runtime_dir=/root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1
probe_dir="$runtime_dir/observer_dummy_probe"
test ! -e "$probe_dir"
sleep 5 &
probe_pid=$!
bash "$runtime_dir/observe_resources.sh" "$probe_pid" "$probe_dir"
wait "$probe_pid"
test -s "$probe_dir/resources.csv"
test -s "$probe_dir/process_rss.tsv"
test -s "$probe_dir/observer_exit.txt"
sample_rows=$(( $(wc -l < "$probe_dir/resources.csv") - 1 ))
test "$sample_rows" -ge 4
printf 'OBSERVER_DUMMY_PROBE_PASS=1\nRESOURCE_GPU_ROWS=%s\n' "$sample_rows"
cat "$probe_dir/observer_exit.txt"
