# 50-task CP execution evidence

Source, settings and validation for Clean50 x 50 tasks. See [execution notes](../../pi05_distillation/14_MULTITASK_EXECUTION_20261022.md). No model weights, replay buffers or videos are included. Deployment paths in evidence are normalized to `${DATA}` / `${HOME}`.

`data-pipeline-source.tar.gz` records the actual initial preparation and missing-task reconciliation workers. It reuses the accepted September 28 conversion chain: RoboTwin `c3ddfa8b97d5519efa828b075999bd0006778e5e` raw-to-Aloha conversion, followed by the accepted LeRobot converter with the existing API compatibility adapter. The pinned dataset is TianxingChen/RoboTwin2.0 `9dc9299c163db059931898a9f0852098a61155a1`. The wrappers preserve partial conversions for inspection and publish the formal manifest only after all 50 tasks are complete. Four CPU conversion jobs may run concurrently; they do not allocate CUDA contexts.

The source archive is a deployment snapshot, not a new self-contained data installer. Expand the normalized locations and retain the accepted converter environment when reproducing it. The runnable training/evaluation implementation is under `rlinf/algorithms/pi05_distillation/` and `tools/pi05_distillation/`.

Training smoke uses 14 available tasks only to validate mixed batches, full resume and the exported N1 overlay. Formal CP starts from the original base and requires all 50 tasks. Paired original N1/N10 baselines overlap the remaining data preparation. Normal failures count as failures; simulator errors remain explicit missing data.
