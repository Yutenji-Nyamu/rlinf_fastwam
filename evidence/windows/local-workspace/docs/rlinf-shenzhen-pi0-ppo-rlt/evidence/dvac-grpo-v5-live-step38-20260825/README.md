# GRPO-DVAC v5 live snapshot through complete Step 38

- Actual branch: v4 Step 1--30 plus v5 resumed Step 31--38; discarded v4 Step 31--33 is intentionally excluded.
- Capture: next Step 39 was incomplete; no Step 40 checkpoint/fixed-64 existed yet.
- Inputs: small driver log, TensorBoard scalar JSON, resolved YAML, and one-minute resource CSV only.
- Outputs: three phone-readable PNGs, one self-contained interactive HTML dashboard, stitched console CSV, and summary JSON.
- No checkpoint, video, Ray session log, or ZIP is included.

## Key values

- Train success Step 38: 92.38%.
- Paired mean DVAC-GRPO through Step 38: -0.42 percentage points.
- Paired trailing-5 difference: -0.78 percentage points.
- Fixed-64 through Step 30: 179/192 versus 179/192.
- Latest KL/clip/grad: 0.0163 / 0.0545 / 14.373.
- Latest DVAC ESS/weight mean: 0.842 / 1.037.
- Latest host available / max card memory: 1185.9 GiB / 54.3 GiB.
