# DVAC global-z `[0,2]` stopped-at-g49 evidence package

## What this package is

This is the lightweight, high-information closeout for the π0/GRPO DVAC
**global-z `[0,2]`** run that was interrupted by an AutoDL container restart on
2026-08-24.

- Latest complete training record: **Global Step 49/100**.
- The following rollout reached only `14/16`; it did not form Global Step 50
  and is excluded from all curves and summaries.
- This was not a natural 100-step completion.
- The success curves are **on-policy training-rollout success**, not held-out
  fixed-ID evaluation.
- No CUDA OOM or cgroup OOM/OOM-kill event was recorded before the restart.

## Headline snapshot at g49

- Raw / trailing-5 / trailing-10 success:
  `93.359% / 93.203% / 93.320%`.
- g1--49 mean success: `90.139%`.
- Compared with original GRPO over the common g1--49 interval:
  mean / latest-5 / latest-10 differences were
  `+2.081 / +2.734 / +2.930` percentage points.
- Latest DVAC weight p05 / median / mean / p95:
  `0.367 / 1.038 / 1.088 / 2.000`.
- Latest per-query weight ESS: `0.897`, equivalent to about `44.83/50`
  future-action positions; coefficient angle: `18.22 degrees`.
- GPU peak: `30.37 / 30.22 GiB`; cgroup memory peak: about `240 GiB`;
  OOM and OOM-kill deltas: `0 / 0`.

## Where to start

1. `docs/31_GLOBAL_Z_W0TO2_G49_RESTART_CLOSEOUT_AND_SUCCESS_SMOOTHING_20260824.md`
   explains the result and evidence boundaries.
2. `analysis/FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.png` compares original
   GRPO, DVAC v1, v2, v3, and this run on raw/5-step/10-step success.
3. `analysis/GLOBAL_Z_METHOD_DIAGNOSTICS_G49.png` shows the method-specific
   signal and weight behavior.
4. `analysis/GLOBAL_Z_RESOURCES_G49.png` summarizes GPU/RAM/resource behavior.
5. `analysis/SUMMARY_G49.json` is the machine-readable headline summary.

## Included files

- `docs/`: the closeout note and the command-by-command evidence ledger.
- `analysis/`: all four high-information PNGs; success, method, horizon, and
  rank/step CSVs; the JSON summary; and the exact analysis script.
- `raw/run/metrics.log`: the main training metric stream through g49.
- `raw/run/dvac_train/actor_rank*/`: both ranks' runner metrics, rolling-state
  metadata, run manifests, and the latest representative g49-aligned NPZ.
- `raw/runtime/`: driver log, resolved configuration, launch command, and one
  copy of the full resource-monitor CSV.
- `FILE_SHA256.tsv`: per-file SHA-256 checksums for every payload file other
  than the checksum table itself.

## Deliberately excluded

- Checkpoints and videos.
- Historical per-step NPZ collections; only the latest representative NPZ from
  each actor rank is included.
- `analysis/GLOBAL_Z_RESOURCES_G49.csv`, because it duplicates the included
  `raw/runtime/resource_monitor/resources.csv` and would add about 9.9 MB.
- Any incomplete Global Step 50 metric.

The source evidence remains under
`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_stop_g49_20260824/`.
