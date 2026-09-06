set -eu
export LC_ALL=C

sudo -S -p '' bash -c '
runuser -u guorenjie -- bash -c "
cd /home/guorenjie
/home/guorenjie/.conda/envs/lerobot-clp/bin/python - <<\"PY\"
import json
from pathlib import Path

root = Path(\"/home/guorenjie/research/smolvla-libero-clp/outputs/eval\")
paths = sorted(root.glob(\"pi0*/eval_info.json\"), key=lambda p: p.stat().st_mtime)
print(\"eval_files\", len(paths))
for p in paths:
    try:
        d = json.loads(p.read_text())
    except Exception as exc:
        print(p.parent.name, \"json_error\", exc)
        continue
    print(\"---\", p.parent.name)
    print(\"top_keys\", list(d) if isinstance(d, dict) else type(d).__name__)
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, (int, float, str, bool)):
                print(key, value)
            elif isinstance(value, dict):
                shallow = {k:v for k,v in value.items() if isinstance(v,(int,float,str,bool))}
                if shallow:
                    print(key, shallow)
PY
"
'
