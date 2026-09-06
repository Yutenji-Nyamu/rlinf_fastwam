set -eu
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
runs=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab
printf '=== time ===\n'
date '+%F %T %Z'
printf '=== source ===\n'
cd "$src"
printf 'path=%s\n' "$PWD"
printf 'head='; git rev-parse HEAD
printf 'branch='; git branch --show-current
printf 'describe='; git describe --always --dirty --tags
printf '%s\n' '-- status porcelain --'
git status --short
printf '%s\n' '-- unstaged diff stat --'
git diff --stat
printf '%s\n' '-- staged diff stat --'
git diff --cached --stat
printf '%s\n' '-- remotes --'
git remote -v || true
printf '%s\n' '-- branches --'
git branch -vv
printf '=== runs ===\n'
test -d "$runs"
printf 'path=%s\n' "$runs"
printf 'top_level_dirs='; find "$runs" -mindepth 1 -maxdepth 1 -type d | wc -l
printf 'all_files='; find "$runs" -type f | wc -l
printf 'all_bytes='; find "$runs" -type f -printf '%s\n' | awk '{s+=$1} END{printf "%.0f\n",s+0}'
printf 'light_files='; find "$runs" -type f ! -iname '*.mp4' ! -iname '*.avi' ! -iname '*.webm' | wc -l
printf 'light_bytes='; find "$runs" -type f ! -iname '*.mp4' ! -iname '*.avi' ! -iname '*.webm' -printf '%s\n' | awk '{s+=$1} END{printf "%.0f\n",s+0}'
printf 'video_files='; find "$runs" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.webm' \) | wc -l
printf 'video_bytes='; find "$runs" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.webm' \) -printf '%s\n' | awk '{s+=$1} END{printf "%.0f\n",s+0}'
for task in adjust_bottle move_stapler_pad; do
  printf 'task=%s dirs=' "$task"
  find "$runs" -mindepth 1 -maxdepth 1 -type d -name "${task}*" | wc -l
  printf 'task=%s light_files=' "$task"
  find "$runs" -type f -path "*/${task}*/*" ! -iname '*.mp4' ! -iname '*.avi' ! -iname '*.webm' | wc -l
  printf 'task=%s light_bytes=' "$task"
  find "$runs" -type f -path "*/${task}*/*" ! -iname '*.mp4' ! -iname '*.avi' ! -iname '*.webm' -printf '%s\n' | awk '{s+=$1} END{printf "%.0f\n",s+0}'
  printf 'task=%s video_files=' "$task"
  find "$runs" -type f -path "*/${task}*/*" \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.webm' \) | wc -l
  printf 'task=%s video_bytes=' "$task"
  find "$runs" -type f -path "*/${task}*/*" \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.webm' \) -printf '%s\n' | awk '{s+=$1} END{printf "%.0f\n",s+0}'
done
