set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
XPL="$ROBOTWIN/XPolicyLab"
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
COLLECT_CFG="$ROBOTWIN/env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml"
COLLECT_ROOT="$ROBOTWIN/data/sz_collect_smoke_1ep_20260821"
TRAIN_OUT="$RUN/06_act_train_smoke_1epoch_not_for_eval"
HF_LEAF=/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50

printf '%s\n' '=== IDENTITY AND SOURCE ==='
date --iso-8601=seconds
hostname
id
uptime
printf 'robotwin_head=%s\n' "$(git -C "$ROBOTWIN" rev-parse HEAD)"
printf 'xpl_head=%s\n' "$(git -C "$XPL" rev-parse HEAD)"
git -C "$ROBOTWIN" status --short
git -C "$XPL" status --short -- \
  policy/ACT/imitate_episodes.py \
  policy/ACT/utils.py \
  policy/ACT/detr/act_policy.py \
  policy/ACT/detr/main.py
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C "$XPL" rev-parse HEAD)" = c07a09614dd44cc4a67483bcb9a82e7439d99926
test -z "$(git -C "$XPL" status --porcelain -- policy/ACT/imitate_episodes.py policy/ACT/utils.py policy/ACT/detr/act_policy.py policy/ACT/detr/main.py)"

printf '%s\n' '=== TARGET PRECHECK ==='
for target in "$COLLECT_CFG" "$COLLECT_ROOT" "$RUN" "$TRAIN_OUT"; do
  if [ -e "$target" ]; then
    printf 'EXISTS\t%s\n' "$target"
  else
    printf 'ABSENT\t%s\n' "$target"
  fi
done
test -f "$HF_LEAF/policy_last.ckpt"
test -f "$HF_LEAF/dataset_stats.pkl"
stat --printf='%s\t%n\n' "$HF_LEAF/policy_last.ckpt" "$HF_LEAF/dataset_stats.pkl"

printf '%s\n' '=== GPU AND OWNED PROCESSES ==='
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
ps -u "$(id -u)" -o pid,ppid,etimes,rss,stat,cmd --sort=pid | \
  grep -E 'RoboTwin|XPolicyLab|collect_data|test_render|imitate_episodes|eval_policy|policy_server|ray|python' || true

printf '%s\n' '=== FILESYSTEMS AND PLACEMENT ==='
df -hT / /home /data
df -i / /home /data
for path in \
  /home/chenyiteng \
  /home/chenyiteng/miniforge3 \
  /data/chenyiteng \
  "$ROBOTWIN" \
  /data/chenyiteng/models/robotwin2-hf-a967b852; do
  printf 'PATH=%s\n' "$path"
  findmnt -T "$path" -no TARGET,SOURCE,FSTYPE,OPTIONS
  du -sh "$path"
done
for path in \
  /home/chenyiteng/.cache \
  /home/chenyiteng/.cache/huggingface \
  /home/chenyiteng/.cache/torch \
  /home/chenyiteng/miniforge3/pkgs \
  /data/chenyiteng/projects/robotwin-native/RoboTwin/assets \
  /data/chenyiteng/projects/robotwin-native/RoboTwin/data \
  /data/chenyiteng/projects/robotwin-native/RoboTwin/XPolicyLab/policy/ACT/processed_data; do
  if [ -e "$path" ]; then du -sh "$path"; fi
done

printf '%s\n' '=== PROXY SERVICE AND ROUTE ==='
systemctl is-active mihomo.service || true
systemctl is-enabled mihomo.service || true
ss -lnt | grep -E '127\.0\.0\.1:(7890|9090)' || true
printf 'http_proxy=%s\n' "${http_proxy:-unset}"
printf 'https_proxy=%s\n' "${https_proxy:-unset}"
printf 'all_proxy=%s\n' "${all_proxy:-unset}"

printf '%s\n' '=== SMALL CONNECTIVITY ==='
curl --silent --show-error --location --max-time 30 --output /dev/null \
  --write-out 'github_proxy code=%{http_code} time=%{time_total}s speed=%{speed_download}Bps bytes=%{size_download}\n' \
  https://github.com/
curl --silent --show-error --location --max-time 30 --output /dev/null \
  --write-out 'hf_proxy code=%{http_code} time=%{time_total}s speed=%{speed_download}Bps bytes=%{size_download}\n' \
  https://huggingface.co/

printf '%s\n' '=== BOUNDED 10MB PROXY SPEED PROBE ==='
curl --silent --show-error --location --max-time 120 --output /dev/null \
  --write-out 'cloudflare_10mb code=%{http_code} time=%{time_total}s speed=%{speed_download}Bps bytes=%{size_download}\n' \
  'https://speed.cloudflare.com/__down?bytes=10000000'
