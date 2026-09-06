set -u

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
launcher=/root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh
run_root=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1
runtime_root=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
hostname
pwd
id -u
echo "GIT"
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" rev-parse 'HEAD^{tree}'
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
echo "RUNTIME_DIFF_FROM_IMPLEMENTATION"
git -C "$repo" diff --name-status \
  7bc5f87086035087adf6d44ddda76eb5a9e54ee8..HEAD
echo "PROCESSES"
pgrep -af '[t]rain_embodied_agent.py|[t]orch.distributed.run' || true
pgrep -x raylet || true
pgrep -x gcs_server || true
echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
echo "MEMORY"
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.stat | awk '$1=="anon" || $1=="file" {print}'
cat /sys/fs/cgroup/memory.events
echo "DISK"
df -B1 /root/autodl-tmp
echo "TARGETS"
for path in "$run_root" "$runtime_root"; do
  test -e "$path" && echo "PRESENT $path" || echo "ABSENT $path"
done
echo "HASHES"
sha256sum \
  "$launcher" \
  /root/autodl-tmp/qam_source_resolved_20260731_v1.yaml \
  /root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml \
  /root/autodl-tmp/qam_source_to_qonly_smoke_20260731_v1.diff \
  /root/autodl-tmp/qam_resource_monitor_20260731_v1.sh

