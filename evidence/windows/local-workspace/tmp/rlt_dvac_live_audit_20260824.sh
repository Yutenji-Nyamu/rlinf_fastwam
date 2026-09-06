set -u

printf 'IDENTITY\n'
hostname
pwd
id -u
date -Is
ps -p 1 -o pid=,lstart=,cmd=

printf 'RESOURCES\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
printf 'cgroup_memory_current='; cat /sys/fs/cgroup/memory.current 2>/dev/null || true
printf 'cgroup_memory_peak='; cat /sys/fs/cgroup/memory.peak 2>/dev/null || true
printf 'cgroup_memory_max='; cat /sys/fs/cgroup/memory.max 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
df -h /root/autodl-tmp
ps -eo pid,ppid,etimes,%cpu,%mem,rss,cmd --sort=-rss | grep -E 'python|ray|torchrun|nvidia|rlt' | grep -v grep | head -30 || true

printf 'RLT_DVAC_WORKTREE\n'
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
git -C "$repo" rev-parse --show-toplevel
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count HEAD...@{upstream}
git -C "$repo" log -1 --oneline --decorate

printf 'RLT_BASE_WORKTREE\n'
base=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
git -C "$base" rev-parse HEAD
git -C "$base" status --short

printf 'DVAC_CONFIG_FILES\n'
find "$repo" -type f \( -name '*teacher*dvac*.yaml' -o -name '*8env250*.yaml' \) -print | sort

printf 'DVAC_CONFIG_CONTENT\n'
cfg=$(find "$repo" -type f -name '8env250_teacher_dvac_w0to2.yaml' -print -quit)
if [ -n "$cfg" ]; then
  printf 'config=%s\n' "$cfg"
  sed -n '1,260p' "$cfg"
fi

printf 'BASE_CONFIG_CONTENT\n'
basecfg=$(find "$repo" -type f -name '*8env250*.yaml' ! -name '*teacher*dvac*' -print | sort | head -1)
if [ -n "$basecfg" ]; then
  printf 'config=%s\n' "$basecfg"
  sed -n '1,260p' "$basecfg"
fi

printf 'RLT_LAUNCH_REFERENCES\n'
find /root/autodl-tmp -maxdepth 4 -type f \( -name '*rlt*launch*' -o -name '*rlt*smoke*' -o -name '*stage2*.sh' -o -name 'launch_command.txt' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -r | head -80

printf 'RLT_HISTORICAL_OUTPUTS\n'
find /root/autodl-tmp -maxdepth 5 -type d \( -iname '*rlt*250*' -o -iname '*rlt*480*' -o -iname '*rlt*formal*' -o -iname '*rlt*smoke*' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -r | head -100

printf 'STAGE1_AND_CHECKPOINT_CANDIDATES\n'
find /root/autodl-tmp -maxdepth 6 \( -type f -o -type d \) \( -name 'global_step_2000' -o -name 'global_step_250' -o -name 'global_step_480' -o -iname '*stage1*2000*' -o -iname '*resume250*' \) -print 2>/dev/null | head -120
