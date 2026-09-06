set -eu
nice -n 19 ionice -c 3 /usr/bin/python3 -B /data/chenyiteng/results/server-maintenance-20260906/exact-prune-v2.py /data/chenyiteng/results/server-maintenance-20260906/step10-exact-nanoseconds.json --execute
