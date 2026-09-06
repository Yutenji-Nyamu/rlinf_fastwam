set -eu
export PYTHONDONTWRITEBYTECODE=1 GIT_TERMINAL_PROMPT=0
nice -n 19 ionice -c 3 /usr/bin/python3 -B /data/chenyiteng/results/server-maintenance-20260906/sz_publish_archive_receipt.py publication-receipt.tar.gz
