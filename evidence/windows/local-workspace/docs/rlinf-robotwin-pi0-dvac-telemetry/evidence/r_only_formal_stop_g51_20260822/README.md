# R-only v2 formal closeout — stopped after complete Global Step 51

This is a lightweight, high-information closeout of the AutoDL run
`idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822`.

## Terminal facts

- User-authorized stop completed at `2026-08-22T21:42:04+08:00`.
- Complete metrics: Global Steps 1–51; the interrupted next step is excluded.
- Retained server checkpoints: g10/g20/g30/g40/g50.
- The complete g50 DCP has 3 files and `10,393,939,477` bytes.
- wrapper/driver/observer exited; driver rc134 is the authorized stop, observer rc0.
- No CUDA OOM, cgroup OOM, or OOM-kill was observed.
- Post-cleanup GPUs were at 0 MiB; cgroup current was about 84.95 GiB.

## What is retained here

- exact resolved config, launch command/timestamps/exit codes, and logs;
- complete g1–51 metrics plus both actor-rank step CSVs and rolling state;
- representative dual-rank NPZ at warmup g1, first apply g2, middle g25, and final g51;
- one sampled action/control-level RoboTwin trace;
- the full local resource CSV for reproducible analysis, plus a 60-second sample for the ZIP;
- original-GRPO/v1/v2 comparison plots, v2 diagnostics, resource plots, and the g51 `[0,2]`
  counterfactual;
- operation ledger, raw read-only audit, analysis scripts, CSV/JSON summaries, and hashes.

## Deliberately excluded from the ZIP

- the 49 GiB remote run tree and all checkpoint bodies;
- the full 10 MB two-second resource table (the evidence directory retains it locally);
- 94 of the 102 per-step NPZ files; four representative steps per actor rank are included instead.

The method/result discussion is in
`docs/rlinf-robotwin-pi0-dvac-telemetry/17_R_ONLY_G51_CLOSEOUT_AND_V3_RANGE_DISCUSSION_20260822.md`.
