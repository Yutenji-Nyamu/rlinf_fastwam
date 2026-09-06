# GRPO-DVAC W[0,5] live snapshot through Step 12

Capture boundary: 2026-08-25 23:18--23:21 CST. The wrapper was alive, fatal matches were zero, complete Step 12 had been logged, and Step 13 rollout had started.

- [`01_w0to5_vs_matched_grpo_success_step12.png`](01_w0to5_vs_matched_grpo_success_step12.png): per-step success, trailing-5, and fixed-64 against the strict-matched original GRPO.
- [`02_w0to5_optimization_and_resources_step12.png`](02_w0to5_optimization_and_resources_step12.png): KL, clip fraction, gradient norm, DVAC weighting, host RAM, and GPU4--7 memory.
- [`summary_step12.json`](summary_step12.json): exact plotted summary values.
- `driver.log`, `resource.csv`, and the TensorBoard event are the small read-only source files; no checkpoint, video, or large Ray log was copied.

Current evidence is too early for a method-effect claim. The live engineering risk is host available RAM continuing to decline despite zero memory PSI at capture time.
