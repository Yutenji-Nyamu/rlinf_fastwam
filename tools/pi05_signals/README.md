# pi0.5 action signals

Inference-only collection: seven candidate signals plus DV on the original RoboTwin DV50 policy. Native B16/H50/M10 inference is retained. No training, reward changes, or action reweighting occur here.

| Signal | Action-indexed definition |
|---|---|
| DV | Sum of coordinate population variances across last 5 clean estimates |
| Fresco | Vector variance of last 5 input latents; equivalent to sum of coordinate variances |
| GEO | Path length of velocity changes divided by mean velocity norm |
| GeoAAC | Positive relative growth of the stage-weighted prefix geometry; first position invalid |
| SHIFT | log1p distance between first/last denoising all-layer residual means |
| Norm | Mean L2 residual norm across last 5 forwards and deepest 3 layers |
| SR | Stable rank of final-forward last-layer hidden vectors in a fixed 5-action window |
| U-GROW | Relative disagreement of complete 10-step and 5-step solves with shared full initial noise |

The U-GROW branch adds five expert evaluations and reuses the visual/language prefix. Its RNG changes are isolated. Hidden hooks observe complete decoder residuals before the final normalization. All scores preserve the action axis; SR remains a neighborhood statistic and GeoAAC retains prefix dependence.

`patch_model.py` accepts only the locked original wrapper source and applies three sampling-entry changes. The disabled path preserves existing inference. `run_batch.py` verifies full-chain/action/RNG parity with a real B16 model before each episode batch. Raw observations retain native resolution; model preprocessing remains unchanged.

`validate_batch.py CONFIG` recomputes all scores, checks masks/trace alignment, and decodes every preview video. `--self-test NEW_DIRECTORY` also tests rejection of corrupted DV data. A completed file alone is insufficient: acceptance requires `validation.json.passed=true`.

`run_queue.py` claims unique batches under a file lock, binds formal execution to a frozen successful smoke, monitors both compute and graphics processes, and delegates exact process cleanup to the existing ProcessGuard. Native fatal failures propagate as exit 99. Already claimed/interrupted batches are not silently retried.

The external maintenance owner owns resource handoff: signals, saved Dojo work, then the original RLT checkpoint chain. `priority_owner.py` is a site-specific adapter requiring frozen owner/source/config identities. It must be checked on the deployment host before use; idle GPU utilization alone does not establish availability.

Raw hidden storage uses FP16 for all-round snapshots; FP32 final hidden is separately retained for exact SR recomputation. Endpoint layer means and per-layer norm grids are FP32. Successful terminal chunks retain the original limitation: submitted action indices do not prove the exact TOPP-interpolated physical prefix executed.
