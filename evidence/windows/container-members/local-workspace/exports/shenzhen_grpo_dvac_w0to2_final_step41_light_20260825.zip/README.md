# GRPO-DVAC weights [0,2] closeout through complete Step 41

- Actual branch: v4 Step 1--30 plus v5 resumed Step 31--41; discarded v4 Step 31--33 is intentionally excluded.
- Capture: the user-requested stop occurred during incomplete Step 42; only complete Step 1--41 is analyzed.
- Inputs: small driver log, TensorBoard scalar JSON, resolved YAML, and one-minute resource CSV only.
- Outputs: three phone-readable PNGs, one self-contained interactive HTML dashboard, stitched console CSV, and summary JSON.
- Main comparison colors: DVAC orange `#E4572E`, matched GRPO deep teal `#00796B`, and baseline-only light teal `#80CBC4`; markers and line styles remain distinct.
- No checkpoint, video, Ray session log, or ZIP is included.

## Key values

- Train success Step 41: 94.92%.
- Paired mean DVAC-GRPO through Step 41: -0.39 percentage points.
- Paired trailing-5 difference: -0.66 percentage points.
- Fixed-64 through Step 40: 242/256 versus 242/256.
- Latest KL/clip/grad: 0.0226 / 0.0818 / 15.671.
- Latest DVAC ESS/weight mean: 0.839 / 1.002.
- Latest host available / max card memory: 851.7 GiB / 62.0 GiB.
