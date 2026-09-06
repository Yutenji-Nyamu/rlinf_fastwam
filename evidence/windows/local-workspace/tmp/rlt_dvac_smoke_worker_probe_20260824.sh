printf 'GPU_PROCESSES\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
printf 'WORKERS\n'
ps -eo pid,ppid,etimes,%cpu,%mem,rss,stat,wchan:24,cmd --sort=-%cpu | grep -E '4038|ray::|RLTACFSDP|RolloutGroup|EnvGroup|MultiStep|ActorGroup' | grep -v grep | head -40
printf 'RAY_STATUS\n'
/root/autodl-tmp/RLinf/.venv/bin/ray status 2>/dev/null | head -80 || true
