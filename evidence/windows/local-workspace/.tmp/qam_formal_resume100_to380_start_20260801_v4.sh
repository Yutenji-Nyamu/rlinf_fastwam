set -euo pipefail
launcher=/root/autodl-tmp/qam_formal_resume100_to380_launch_20260801_v4.sh
supervisor_log=/root/autodl-tmp/qam_formal_resume100_to380_supervisor_20260801_v4.log
bash -n "$launcher"
nohup bash "$launcher" >"$supervisor_log" 2>&1 </dev/null &
supervisor_pid=$!
printf 'supervisor_pid=%s\n' "$supervisor_pid"
sleep 3
ps -p "$supervisor_pid" -o pid=,ppid=,pgid=,stat=,lstart=,cmd=
if test -s "$supervisor_log"; then tail -40 "$supervisor_log"; fi
