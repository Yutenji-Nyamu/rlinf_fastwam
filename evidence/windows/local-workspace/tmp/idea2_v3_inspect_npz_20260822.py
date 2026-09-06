from pathlib import Path

import numpy as np

root = Path(
    r"C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-dvac-telemetry\evidence\v3_formal_live_g1_20260822\raw\run\dvac_train"
)

for path in sorted(root.glob("actor_rank*/rollout_step*.npz")):
    print(f"FILE={path.relative_to(root)}")
    with np.load(path) as data:
        for key in data.files:
            arr = data[key]
            finite = np.isfinite(arr).all() if np.issubdtype(arr.dtype, np.number) else True
            if np.issubdtype(arr.dtype, np.number) and arr.size:
                print(
                    f"  {key}: shape={arr.shape} dtype={arr.dtype} finite={int(finite)} "
                    f"min={np.nanmin(arr):.8g} mean={np.nanmean(arr):.8g} max={np.nanmax(arr):.8g}"
                )
            else:
                print(f"  {key}: shape={arr.shape} dtype={arr.dtype} finite={int(finite)}")
