set -u
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v1
control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v1/runtime
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v1/runtime

echo '== exit codes and processes =='
for path in "$control/exit_code.txt" "$method/exit_code.txt"; do printf '%s=' "$path"; cat "$path" 2>/dev/null || true; done
ps -eo pid,pgid,etimes,cmd | grep -E '50011|578072|578902|578909|578919' | grep -v grep || true

for spec in "control:$control/foreground.log" "method:$method/foreground.log"; do
  name=${spec%%:*}
  log=${spec#*:}
  echo "== $name first exception =="
  grep -n -B 30 -A 160 "Exception occurred while running EnvWorker" "$log" 2>/dev/null | head -240 || true
  echo "== $name error keywords =="
  grep -n -E 'FileNotFoundError|PermissionError|Address already in use|ModuleNotFoundError|ImportError|ValueError|KeyError|RuntimeError|Exception:' "$log" 2>/dev/null | head -120 || true
done

echo '== env worker ray logs =='
session=$(cat "$pair/ray_temp_dir.txt" 2>/dev/null)/session_latest/logs
find "$session" -maxdepth 1 -type f \( -name '*581219*' -o -name '*583455*' \) -print 2>/dev/null | while IFS= read -r path; do
  echo "===== $path ====="
  tail -160 "$path" 2>/dev/null || true
done

echo '== memory events =='
cat /sys/fs/cgroup/memory.events
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
