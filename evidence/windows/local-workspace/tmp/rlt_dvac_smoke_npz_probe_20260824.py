from pathlib import Path

import numpy as np

root = Path(
    "/root/autodl-tmp/experiments/"
    "rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/"
    "robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_smoke_8env1c_v1/"
    "rlt_dvac"
)
for path in sorted(root.glob("actor_rank*/update_*.npz")):
    with np.load(path, allow_pickle=False) as data:
        print(path.relative_to(root))
        for key in data.files:
            value = data[key]
            if np.issubdtype(value.dtype, np.number) and value.size:
                finite = np.isfinite(value).all()
                print(
                    f"  {key}: shape={value.shape} dtype={value.dtype} "
                    f"min={float(np.min(value)):.7g} "
                    f"mean={float(np.mean(value)):.7g} "
                    f"max={float(np.max(value)):.7g} finite={finite}"
                )
            else:
                print(f"  {key}: shape={value.shape} dtype={value.dtype}")

print("INFERRED_BASELINE")
for path in sorted(root.glob("actor_rank*/update_00000000.npz")):
    with np.load(path, allow_pickle=False) as data:
        selected = data["selected_v"].astype(np.float64).reshape(-1)
        z_scores = data["z_scores"].astype(np.float64).reshape(-1)
        unclipped = np.abs(z_scores) < 1.999
        design = np.stack([np.ones(unclipped.sum()), z_scores[unclipped]], axis=1)
        mean, std = np.linalg.lstsq(
            design, np.log(selected[unclipped] + 1e-12), rcond=None
        )[0]
        print(
            f"  {path.parent.name}: mean={mean:.9g} std={std:.9g} "
            f"unclipped={int(unclipped.sum())}"
        )
