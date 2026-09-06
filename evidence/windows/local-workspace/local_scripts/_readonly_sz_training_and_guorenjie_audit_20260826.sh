set -eu

export LC_ALL=C

sudo -S -p '' true

echo '=== HOST ==='
date -Is
uptime
free -h
df -hT / /home /data 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

echo '=== ACTIVE TRAINING PROCESSES (READ ONLY) ==='
sudo -n ps -eo user:16,pid,ppid,lstart,etime,%cpu,%mem,rss,stat,args --sort=start_time \
  | grep -E 'guorenjie|chenyiteng' \
  | grep -E 'lerobot-train|torchrun|rlinf|main\.py|ray::|python' \
  | tail -n 160 || true

echo '=== CHENYITENG GRPO/DVAC CANDIDATES ==='
sudo -n ps -u chenyiteng -o pid=,ppid=,lstart=,etime=,%cpu=,%mem=,rss=,stat=,args= \
  | grep -Ei 'grpo|dvac|rlinf|ray|robotwin' \
  | tail -n 200 || true

echo '=== RECENT CHENYITENG RESULT DIRECTORIES ==='
sudo -n -u chenyiteng bash -lc '
  for root in /data/chenyiteng/results /data/chenyiteng/RLinf /home/chenyiteng; do
    [ -d "$root" ] || continue
    find "$root" -maxdepth 4 -type d \( -iname "*dvac*" -o -iname "*grpo*" \) -mmin -720 -printf "%TY-%Tm-%Td %TH:%TM:%TS %p\n" 2>/dev/null
  done | sort | tail -n 80
' || true

echo '=== GUORENJIE ID / PRIVILEGE / GPU JOBS ==='
sudo -n id guorenjie
sudo -n -l -U guorenjie 2>&1 || true
sudo -n stat -c '%A %a %U:%G %n' /home/guorenjie
sudo -n ps -u guorenjie -o pid=,ppid=,lstart=,etime=,%cpu=,%mem=,rss=,stat=,tty=,args= \
  | grep -E 'lerobot-train|torchrun' || true

echo '=== GUORENJIE ACTIVE PID SECURITY / NETWORK ==='
for pid in $(sudo -n pgrep -u guorenjie -f 'lerobot-train|torchrun' || true); do
  echo "--- PID $pid"
  sudo -n awk '/^(Name|Uid|Gid|TracerPid|CapInh|CapPrm|CapEff|CapBnd|NoNewPrivs|Seccomp):/{print}' "/proc/$pid/status" 2>/dev/null || true
done
sudo -n ss -tpn 2>/dev/null | grep -E 'pid=(157596|157731|158278|158288|158428|158429)' || true

echo '=== GUORENJIE PROJECT / GIT / LAUNCH CONFIG ==='
sudo -n -u guorenjie bash -lc '
  project=/home/guorenjie/research/smolvla-libero-clp
  repo="$project/lerobot"
  printf "project_size="; du -sh "$project" | awk "{print \\$1}"
  printf "outputs_size="; du -sh "$project/outputs" | awk "{print \\$1}"
  printf "hf_cache_size="; du -sh /home/guorenjie/.cache/huggingface 2>/dev/null | awk "{print \\$1}"
  git -C "$repo" rev-parse HEAD
  git -C "$repo" status --short
  git -C "$repo" diff --stat
  echo "--- run_pi0_fullft_s30k.sh"
  sed -n "1,240p" "$project/run_pi0_fullft_s30k.sh"
  echo "--- current output roots"
  find "$project/outputs/train" -mindepth 1 -maxdepth 1 -type d -printf "%TY-%Tm-%Td %TH:%TM:%TS %k KiB %f\n" 2>/dev/null | sort | tail -n 30
  echo "--- checkpoint training steps"
  find "$project/outputs/train" -path "*/checkpoints/*/training_step.json" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 12 | cut -d" " -f2- | while IFS= read -r f; do printf "%s: " "$f"; tr -d "\n" < "$f"; echo; done
  echo "--- active CLP code hooks"
  grep -RInE "clp_remove_indices|CLP|remove_indices" \
    "$repo/src/lerobot/policies/pi0/clp_prune_pi0.py" \
    "$repo/src/lerobot/policies/pi0/configuration_pi0.py" \
    "$repo/src/lerobot/policies/pi0/modeling_pi0.py" 2>/dev/null | head -n 160
  echo "--- safety-sensitive calls in custom launch/source"
  grep -RInE "sudo|rm[[:space:]]+-rf|pkill|kill[[:space:]]|/etc/|chmod|chown|curl|wget|requests\\.|subprocess|os\\.system" \
    "$project"/*.sh "$project"/scripts \
    "$repo/src/lerobot/policies/pi0/clp_prune_pi0.py" \
    "$repo/src/lerobot/policies/pi0/pi0_vlm_lora.py" 2>/dev/null | head -n 120 || true
' || true

echo '=== LIBERO DATASET METADATA ==='
sudo -n -u guorenjie bash -lc '
python - <<"PYDATA"
import json
from pathlib import Path

roots = [
    Path("/home/guorenjie/.cache/huggingface/lerobot/hub/datasets--lerobot--libero"),
    Path("/home/guorenjie/.cache/huggingface/hub/datasets--lerobot--libero"),
]
for root in roots:
    print("root", root, "exists", root.exists())
    if not root.exists():
        continue
    for p in sorted(root.rglob("info.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print("bad_info", p, e)
            continue
        print("info", p)
        for k in ("codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks", "total_videos", "total_chunks", "fps", "splits", "chunks_size"):
            if k in d:
                print(f"  {k}={d[k]}")
        print("  features=", sorted((d.get("features") or {}).keys()))
    for p in sorted(root.rglob("tasks.parquet"))[:4]:
        print("tasks_file", p, "bytes", p.stat().st_size)
PYDATA
' || true

echo '=== PRIOR EVALUATION SUMMARIES ==='
sudo -n -u guorenjie bash -lc '
python - <<"PYEVAL"
import json
from pathlib import Path

root = Path("/home/guorenjie/research/smolvla-libero-clp/outputs/eval")
keys = {"success_rate", "pc_success", "success", "avg_sum_reward", "mean_reward", "num_episodes", "n_episodes"}
for p in sorted(root.glob("pi0*/eval_info.json"), key=lambda x: x.stat().st_mtime):
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    found = []
    def walk(v, path=()):
        if len(found) >= 20:
            return
        if isinstance(v, dict):
            for k, x in v.items():
                np = path + (str(k),)
                if str(k).lower() in keys and isinstance(x, (int, float, bool, str)):
                    found.append((".".join(np), x))
                elif len(np) <= 5:
                    walk(x, np)
        elif isinstance(v, list) and len(path) <= 3:
            for i, x in enumerate(v[:3]):
                walk(x, path + (str(i),))
    walk(data)
    print(p.parent.name, found)
PYEVAL
' || true

echo '=== PRIOR ANALYSIS REPORT EXCERPTS ==='
sudo -n -u guorenjie bash -lc '
  project=/home/guorenjie/research/smolvla-libero-clp
  for f in \
    "$project/outputs/pi0_redundancy/diagnostic_3m_v2/report.md" \
    "$project/outputs/pi0_redundancy/layer_ablation_full/report.md" \
    "$project/outputs/pi0_redundancy/report.md" \
    "$project/outputs/redundancy/diagnostic_3m_v2/report.md"; do
    [ -f "$f" ] || continue
    echo "--- $f"
    sed -n "1,180p" "$f"
  done
' || true

echo '=== END ==='
