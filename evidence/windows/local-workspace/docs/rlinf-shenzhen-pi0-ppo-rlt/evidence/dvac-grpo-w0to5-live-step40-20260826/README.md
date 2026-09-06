# GRPO-DVAC `[0,5]` live snapshot through complete Step 40

Snapshot time: 2026-08-26 09:40 CST. Server inspection and downloads were read-only.

- wrapper alive; no exit code yet; fatal-pattern count 0;
- complete Step 40; Step 41 rollout was at `3/4`;
- Step 40 training-rollout success `0.9238281=473/512`;
- Step 40 fixed-64 `63/64=98.4375%`;
- optimization: KL `0.015`, clip fraction `0.049`, pre-clip grad norm `39.185`;
- DVAC: weight mean `1.644`, ESS fraction `0.610`;
- strict paired original-GRPO comparison through Step 40: train mean delta `-1.47 pp`, latest-5 delta `-3.48 pp`;
- common fixed checkpoints Step10/20/30/40: W[0,5] `240/256`, original GRPO `242/256`;
- host available RAM `238.15 GiB`, minimum observed `221.17 GiB`, last-hour delta `-20.91 GiB`;
- latest/peak max-card memory `58.83/74.93 GiB`.

Files:

- `driver.log`: downloaded live console log;
- `resource.csv`: one-minute observer samples;
- `events.out.tfevents.*`: TensorBoard scalars;
- `summary_step40.json`: parsed high-information summary;
- `01_w0to5_vs_matched_grpo_success_step40.png`: train/five-step/fixed64 comparison;
- `02_w0to5_optimization_and_resources_step40.png`: optimization and resource curves.

The run is incomplete. These figures do not establish a final method effect.
