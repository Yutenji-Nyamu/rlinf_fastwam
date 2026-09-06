set -eu
export LC_ALL=C
sudo -S -p '' true

sudo -n -u guorenjie bash -lc '
project=/home/guorenjie/research/smolvla-libero-clp
repo="$project/lerobot"

echo "=== ID / REPO ==="
id
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" diff --stat

echo "=== THREE ACTIVE EXPERIMENTS ==="
ps -u guorenjie -o pid=,ppid=,lstart=,etime=,%cpu=,%mem=,rss=,stat=,tty=,args= \
  | grep -E "lerobot-train|torchrun" || true

echo "=== OUTPUT / CHECKPOINT STATUS ==="
for name in \
  pi0_libero_fullft_rel_vis_3m_pure_s30k \
  pi0_libero_fullft_rel_vis_3m_v2_default_s30k \
  pi0_libero_fullft_rel_vis_paper_s30k; do
  d="$project/outputs/train/$name"
  if [ -d "$d" ]; then
    du -sh "$d"
    find "$d" -path "*/training_step.json" -type f -printf "%T@ %p\n" | sort -n | cut -d" " -f2- | while IFS= read -r f; do printf "%s: " "$f"; tr -d "\n" < "$f"; echo; done
  else
    echo "$name: no output directory yet"
  fi
done

echo "=== DATASET INFO ==="
python - <<"PY"
import json
from pathlib import Path
for root in [
    Path("/home/guorenjie/.cache/huggingface/lerobot/hub/datasets--lerobot--libero"),
    Path("/home/guorenjie/.cache/huggingface/hub/datasets--lerobot--libero"),
]:
    print("cache_root", root, "exists", root.exists())
    for p in sorted(root.rglob("info.json")):
        d=json.loads(p.read_text())
        print("info", p)
        print({k:d.get(k) for k in ["codebase_version","robot_type","total_episodes","total_frames","total_tasks","total_videos","total_chunks","fps","splits","chunks_size"] if k in d})
        print("features", sorted((d.get("features") or {}).keys()))
PY

echo "=== DISK ==="
du -sh "$project" "$project/outputs" /home/guorenjie/.cache/huggingface 2>/dev/null
'

echo "=== SECURITY ==="
sudo -n id guorenjie
sudo -n -l -U guorenjie 2>&1 || true
sudo -n stat -c '%A %a %U:%G %n' /home/guorenjie
for pid in $(sudo -n pgrep -u guorenjie -f 'lerobot-train|torchrun' || true); do
  printf 'pid=%s ' "$pid"
  sudo -n awk '/^(TracerPid|CapEff|NoNewPrivs|Seccomp):/{printf "%s=%s ",$1,$2} END{print ""}' "/proc/$pid/status" 2>/dev/null || true
done
echo "network_connections:"
sudo -n ss -tpn 2>/dev/null | grep -E 'users:\(\("(python|torchrun)"' | grep -E 'pid=(157596|157731|158278|158288|158428|158429)' || true

echo "=== CUSTOM-CODE SAFETY-SENSITIVE MATCHES ==="
sudo -n -u guorenjie bash -lc '
project=/home/guorenjie/research/smolvla-libero-clp
repo="$project/lerobot"
grep -RInE "sudo|rm[[:space:]]+-rf|pkill|kill[[:space:]]|/etc/|chmod|chown|curl|wget|requests\\.|subprocess|os\\.system" \
  "$project"/*.sh "$project"/scripts \
  "$repo/src/lerobot/policies/pi0/clp_prune_pi0.py" \
  "$repo/src/lerobot/policies/pi0/pi0_vlm_lora.py" 2>/dev/null | head -n 120 || true
'
