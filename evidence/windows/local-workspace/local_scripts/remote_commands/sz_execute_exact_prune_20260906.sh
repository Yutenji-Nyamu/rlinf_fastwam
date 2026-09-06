set -eu
nice -n 19 ionice -c 3 /usr/bin/python3 -B /data/chenyiteng/results/server-maintenance-20260906/sz_exact_checkpoint_prune_20260906.py /data/chenyiteng/results/server-maintenance-20260906/cleanup-allowlist.json --execute
