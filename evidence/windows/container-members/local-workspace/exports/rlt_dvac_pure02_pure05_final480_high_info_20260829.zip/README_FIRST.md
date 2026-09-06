# Pure02 / Pure05 final-480 high-information package

## Outcome

- Pure02 (`strength=0.5`, shorthand `[0,2]`) completed Step 480/480 with `exit_code=0`.
- Pure05 (`strength=2.0`, shorthand `[0,5]`) completed Step 480/480 with `exit_code=0`.
- Both use frozen pi0 reference targets and mean-one C10-internal DVAC reweighting; only the weighting strength differs.
- Source HEAD for both runs: `a2ae5cbe81049fb7027c43ea85483ed3ffc3ce2f`.
- No CUDA OOM, cgroup OOM, or OOM-kill was observed.

## Final training-rollout statistics

| Run | Cumulative success | Final raw | MA5 | MA10 | MA20 | Final fixed-20 eval |
|---|---:|---:|---:|---:|---:|---:|
| Pure02 | 60.49% | 100.0% | 97.5% | 97.5% | 95.0% | 20/20 |
| Pure05 | 59.74% | 100.0% | 97.5% | 98.75% | 93.13% | 17/20 |

The final fixed evaluation is held out from the training-rollout curve. A single final checkpoint/evaluation is descriptive, not a standalone ranking.

## Final method footprint

| Run | weight p05 / mean / p95 | weight ESS | top-20% weight mass |
|---|---|---:|---:|
| Pure02 | 0.495 / 1.000 / 1.499 | 0.916 | 27.9% |
| Pure05 | 0.000 / 1.000 / 2.541 | 0.590 | 45.0% |

Thus Pure05 produced a substantially more concentrated C10 BC-gradient allocation than Pure02 while preserving mean weight 1.

## Resource summary

- Paired monitor peak cgroup RAM: approximately 240.0 GiB; `oom=0`, `oom_kill=0`.
- GPU memory peaks: approximately 25.17 GiB on GPU0 and 25.14 GiB on GPU1.
- Pure02 wall time: 34 h 26 min; Pure05 wall time: 34 h 44 min, including final evaluation and closeout.

## Package map

- `pure02/`, `pure05/`: full metrics, resolved config, exact command, source identity, start/finish/exit status, a concise foreground tail, and Step-480 checkpoint inventory. Model weights are intentionally omitted.
- `pair/`: paired launch identity, resource trace, shared-Ray status, and cleanup completion.
- `analysis/RLT_FOUR_SINGLE_GPU_SUCCESS_FULL_AXIS_G480.png`: clean single-GPU RLT, old executed-target DVAC-BC, Pure02, and Pure05 on one raw/MA5/MA10/MA20 figure.
- `analysis/RLT_DVAC_PURE_SUCCESS_THROUGH_G480.png`: Pure02/Pure05 raw, MA5, MA10, and fixed-eval comparison.
- `analysis/*.csv`, `analysis/*.json`, and `analysis/*.py`: plotted data, summaries, and reproduction scripts.

Excluded to keep the package small: checkpoint tensors, replay buffers, videos, Ray session logs, and redundant full file inventories.
