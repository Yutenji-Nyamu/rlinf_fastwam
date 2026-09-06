set -u

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
echo "SNAPSHOT_TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
echo "GIT"
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
echo "PROCESSES"
pgrep -af 'train_embodied_agent.py|ray::|raylet|gcs_server' || true
echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
echo "RUN_ROOTS"
for path in \
  /root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1 \
  /root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1; do
  test -e "$path" && echo "PRESENT $path" || echo "ABSENT $path"
done
echo "DISK"
df -h /root/autodl-tmp

