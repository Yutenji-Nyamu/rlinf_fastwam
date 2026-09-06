set -euo pipefail

run_dir=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1
cd "$run_dir"

find dvac_telemetry -maxdepth 3 -type f -printf '%p\t%s bytes\n' | sort
sha256sum dvac_telemetry/*.npz dvac_telemetry/*.csv

for file in \
    dvac_telemetry/query_index_rollout_rank*.csv \
    dvac_telemetry/episode_index_env_rank*.csv; do
    echo "FILE=$file"
    head -n 3 "$file"
done

/root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np

root = Path("dvac_telemetry")
for path in sorted(root.glob("trace_rollout_rank*.npz")):
    with np.load(path, allow_pickle=False) as data:
        print(f"NPZ={path}")
        for key in data.files:
            arr = data[key]
            finite = bool(np.isfinite(arr).all()) if np.issubdtype(arr.dtype, np.number) else "n/a"
            print(f"  {key}: shape={arr.shape} dtype={arr.dtype} finite={finite}")
PY
