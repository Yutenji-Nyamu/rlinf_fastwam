set -euo pipefail

output_dir=/root/autodl-tmp/idea2_dvac_analysis/first_collection_v3_20260820
cd "$output_dir"

sha256sum *
wc -l query_metrics.csv horizon_metrics.csv

/root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd
from PIL import Image, ImageStat

root = Path(".")
for path in sorted(root.glob("*.png")):
    with Image.open(path) as image:
        stat = ImageStat.Stat(image.convert("RGB"))
        extrema = image.convert("RGB").getextrema()
        print(f"IMAGE={path.name} size={image.size} mode={image.mode} extrema={extrema} mean={stat.mean}")

query = pd.read_csv("query_metrics.csv")
print("SUCCESS_BEFORE_COUNTS")
print(query.groupby(["query_idx", "success_before"]).size().to_string())
print("FAILURE_QUERIES")
print(
    query.loc[~query.success, [
        "query_uid", "query_idx", "success_before", "V_total_L2", "V_total_L3", "V_total_L4"
    ]].to_string(index=False)
)
print("L3_QUANTILES")
print(query.V_total_L3.quantile([0, .25, .5, .75, 1]).to_string())
PY
