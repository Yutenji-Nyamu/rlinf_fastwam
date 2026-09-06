# GRPO-DVAC v5 live snapshot through complete Step 40

- Actual branch: v4 Step 1--30 plus v5 resumed Step 31--40; discarded v4 Step 31--33 is intentionally excluded.
- Capture: next Step 41 was incomplete; no Step 40 checkpoint/fixed-64 existed yet.
- Inputs: small driver log, TensorBoard scalar JSON, resolved YAML, and one-minute resource CSV only.
- Outputs: three phone-readable PNGs, one self-contained interactive HTML dashboard, stitched console CSV, and summary JSON.
- No checkpoint, video, Ray session log, or ZIP is included.

## Key values

- Train success Step 40: 99.02%.
- Paired mean DVAC-GRPO through Step 40: -0.42 percentage points.
- Paired trailing-5 difference: -0.98 percentage points.
- Fixed-64 through Step 40: 242/256 versus 242/256.
- Latest KL/clip/grad: 0.0140 / 0.0339 / 6.406.
- Latest DVAC ESS/weight mean: 0.848 / 1.005.
- Latest host available / max card memory: 971.1 GiB / 65.1 GiB.
