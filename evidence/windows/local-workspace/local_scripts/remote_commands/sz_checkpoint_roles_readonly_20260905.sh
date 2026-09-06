set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
grep -R -n -E 'def (load_checkpoint|save_checkpoint)|local_shard_checkpoint|full_weights.pt|checkpoint_rank_' "$root/rlinf" --include='*.py' | head -100
du -x -B1 --max-depth=1 /data/chenyiteng/results/rlinf-shenzhen/online-bc
ps -u chenyiteng -o pid,ppid,etime,stat,comm | awk '$NF=="du" || $NF=="python3" {print}'
cat /proc/pressure/io
