set -u
date --iso-8601=seconds
repo='/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official'
target='/data/chenyiteng/models/fasterwam/release-6bf9471'

echo '=== ACTIVE ==='
ps -eo user,pid,ppid,etime,%cpu,%mem,rss,stat,cmd \
  | grep -Ei '[s]napshot_download|release-6bf9471|[u]v sync.*environments/robotwin|[i]nstall_robotwin' | sed -n '1,100p'

echo '=== DOWNLOAD ==='
find "$target" -maxdepth 5 -type f \( -name '*.incomplete' -o -name 'step_029355.pt' -o -name 'dataset_stats.json' \) -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3

echo '=== UV ==='
du -sh "$repo/.venvs/robotwin" /data/chenyiteng/cache/uv-fasterwam 2>/dev/null || true
stat -c '%s %y %n' "$repo/environments/robotwin/uv.lock"
sha256sum "$repo/environments/robotwin/uv.lock"
printf 'head_blob_sha256='
git -C "$repo" show HEAD:environments/robotwin/uv.lock | sha256sum | awk '{print $1}'

echo '=== GIT ==='
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short --branch | sed -n '1,160p'
git -C "$repo" diff --numstat -- environments/robotwin/uv.lock
git -C "$repo" diff --stat -- environments/robotwin/uv.lock
git -C "$repo" diff -- environments/robotwin/uv.lock | sed -n '1,100p'

echo '=== GPU3 ==='
nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

