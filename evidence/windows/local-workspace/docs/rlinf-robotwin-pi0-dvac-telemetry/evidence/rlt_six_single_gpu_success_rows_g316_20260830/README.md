# Six single-GPU RLT settings

This wide four-row plot adds live Pure03/Pure04 to the previous single-GPU comparison:

- Clean single-GPU RLT: Step476
- Old DVAC-BC, executed target: Step476
- Pure02, reference target, strength 0.5: Step480
- Pure03, reference target, strength 1.0: Step316 live snapshot
- Pure04, reference target, strength 1.5: Step313 live snapshot
- Pure05, reference target, strength 2.0: Step480

Every row uses the real Step1–480 axis. Pure03/Pure04 stop at their actual current step; no values are extrapolated. Colors and line patterns both distinguish the six settings.

Artifacts:

- `RLT_SIX_SINGLE_GPU_SUCCESS_ROWS_STEP1_480.png`
- `success_curves.csv`
- `summary.json`
