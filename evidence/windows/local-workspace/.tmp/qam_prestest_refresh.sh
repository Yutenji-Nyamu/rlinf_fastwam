set -eu
echo '=== identity ==='
hostname
pwd
id -u
date '+%F %T %Z'
echo '=== processes ==='
pgrep -af 'train_embodied_agent|ray::|rlt|dsrl|qam' || true
echo '=== gpu ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo '=== disk ==='
df -h /root/autodl-tmp
echo '=== qam git ==='
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin branch --show-current
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin rev-parse HEAD
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin status --short
echo '=== other heads/status counts ==='
for d in \
  /root/autodl-tmp/RLinf_dsrl_pi0_robotwin \
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin
do
  if [ -d "$d" ]; then
    printf '%s\n' "$d"
    git -C "$d" branch --show-current
    git -C "$d" rev-parse --short HEAD
    git -C "$d" status --short | wc -l
  fi
done
