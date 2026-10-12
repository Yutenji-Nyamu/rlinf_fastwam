# Native pi0.5 attention recording and offline atlas

This observer records native B16/H50/M10 inference without training. The model, task configurations, seeds, preprocessing, and execution budget are inherited from accepted DV50 commit d74e7edc93d346caa9758dadd4537197eeaf0081. Run on the assigned physical GPU with its existing compute/graphics scope.

- `observer.py`: post-RoPE native Q/K at action expert layers 10/14/18, all 8 Q heads, all 10 denoising calls. Original eager kernel remains in use. Prefix K is shared once per query; all action Q/K are saved. Includes real mask/positions/token-source/camera mapping, original full FP32 random draw, actual solver initial latent, full D32 chain/velocity/t/dt, and final environment actions.
- `run_batch.py CONFIG.json`: original task/seed inference. The first two batches repeat the native first observation and compare observer on/off with exact actions, full chain, and RNG. Full observations and action submission mappings accompany each query. Terminal TOPP physical execution prefix remains unknown, with an explicit validity mask.
- `analysis.py` and `analyze_batch.py CONFIG.json`: one pass over saved raw tensors, FP32 attention reconstruction, probability average over 8 heads then 3 layers, modal conditioning, normalized and raw entropy, token counts, mass, JS/span, all-other/future Anchor, head/layer/step diagnostics, native final-five-step DV over effective D14, action/gripper comparisons, camera previews, and the interactive atlas.
- `lane.py 6|7`: one formal queue per card. The completed expansion gate is removed; a missing or temporarily unreadable progress file means unknown, not a fatal error. Keep the existing batch deadline and skip recorded native task failures. On exit, write the receipt consumed by the existing per-card RLT owner after compute/graphics clearance. The current fallback request determines the checkpoint.
- `test_math.py`: seven focused CPU checks covering signal math and missing/fresh/stalled/transient progress files.

For `open_laptop`, `place_object_scale`, and `put_object_cabinet`, the native temporary setup task initializes success-check attributes in `get_info()`, but the actual reset task does not. The runner reuses that task's own `get_info()` immediately after reset. It does not execute expert actions, draw randomness, or render an extra frame. The first repaired laptop B16 batch passed full raw-record and video validation on 2026-10-12; remaining native rendering failures are a separate issue. Lane configurations are prepared once, not rewritten during collection.

## Numerical acceptance v1.1

The initial generic per-head FP32 max-absolute threshold 0.02 was exceeded in the two calibration attempts (BF16 reference vs FP32 reconstruction), while actions/chain/RNG remained exact. Those failed attempts are retained, not silently relabeled successful. The initial implementation stopped before saving the first raw query; this evidence gap was fixed so any subsequent numerical rejection preserves its raw tensors and unexecuted observation.

v1.1 reproduces the actual native arithmetic from saved Q/K and requires zero error against native A for the slot0 calibration reference (all three layers and ten calls). Action, chain and RNG parity covers the entire batch. It separately checks the intended head-then-layer mean FP32 distribution using the original 0.02 absolute / 0.08 row-L1 limits. Every per-head FP32 discrepancy, peak agreement, and entropy error remains in `parity.json`; the legacy threshold result is retained explicitly. Original native attention, actions, and RNG are not modified to meet a check.

## Evidence and interpretation

The two initial real batches passed v1.1: native attention reconstruction error 0, exact action/chain/RNG, FP32 main-distribution max error about 0.00166. Full capture completed for 32 episodes (turn_switch 9/16, adjust_bottle 11/16). These success rates are native inference outcomes, not a training comparison.

Each query saves all B16 slots and an active mask. Failed episodes remain in the atlas. Native reset retries and actual seeds are recorded; totals of requested records and distinct task/actual-seed pairs are separate. Position source labels do not imply pure causal modality content; mixed boundary tokens are `other`. Different cameras have separate patch coordinates. The FP32 path is an analysis reconstruction, not a bitwise replacement for BF16 inference.

Data and atlas: `/data/chenyiteng/results/pi05-attn/20261010-video-v4` on SZ3. Transaction receipts: `/data/chenyiteng/deployment-20261010/pi05-attn-video-v4`. 500 GiB is the planning budget, updated from measured batches; it is not an artificial hard cap. Original weights, raw Q/K and credentials are not published to Git. The user-authorized 2026-10-13 export includes computed action scalars and seven selected executions with short videos/images; see [lightweight results and interpretation](../../evidence/derived-20261013/README.md).

The accepted runner initializes the model with one native B16 forward before creating the simulator. Keep this order, the original chunk-endpoint video, and the 3600-second batch deadline. The temporary observation lock and CUDA loading override were removed after failing to resolve the stall. Current acceptance state is in `docs/pi05-attn-20261010/EXECUTION.md`.

Historical setup, smoke, publication scripts and old source archives (24 files) are preserved with per-file SHA256 in `E:/Codex/home/visualizations/2026/10/03/01a0ffb6-b309-7fa1-9ae0-b20e6d63a11c/attn-setup-history-20261011.zip`. They are not runtime dependencies; obsolete v1 status tools are archived there too. Keep this folder for the runner, observer, offline analysis, lane, atlas and preview only.

## Reuse correction (2026-10-10)

The initial no-video collection was stopped and excluded from the new video-enabled total. The rejected control-step video observer exceeded the native 120-second environment timeout. The accepted 2026-10-03 signal runner is now the video baseline: native head-camera observations before/after each chunk, OpenCV mp4v at 4 fps, query q mapped to video frames q and q+1, and complete decode/frame-count validation. No additional camera/render/physics call is inserted. Video explains the chunk stage, not the exact action/contact instant. The rejected per-card two-worker scheduler was never launched.

Formal collection remains the accepted B16 protocol. The same runner also supports separately labeled native B32 and B48 capacity probes; the single noise stream is not equivalent to separate old B16 noise streams, so probe records are excluded from formal counts. Larger batches require `capacity_only=true`. Whole-card memory includes model, simulator and rendering; Torch tensor peaks alone are insufficient for scheduling. See REUSE_AUDIT.md for provenance and incident scope.

Current data: `/data/chenyiteng/results/pi05-attn/20261010-video-v4`; receipts: `/data/chenyiteng/deployment-20261010/pi05-attn-video-v4`. The previous v1 directory is history.

2026-10-12: removed the obsolete v1-only `start_preview.py` launcher from this working folder; its original bytes are in `maintenance-oneoffs-20261012.zip` on E:. The running HTTP service and current reconnecting preview are unchanged.

2026-10-12 22:10: the user chose task-level deferral. All remaining six batches belong to move_stapler_pad, place_fan and place_mouse_pad; they are marked deferred, with failed attempts preserved. The other 47 tasks each have 32 validated episodes: 1504 records, 1496 distinct task/actual-seed pairs, 858 task successes. This is a 47-task dataset, not completion of the original 50-task target. Exact Attn processes exited and GPU6/7 cleared; the unchanged original owner automatically launched both RLT fallbacks from CP225 at 22:09. New RLT rounds are not yet confirmed. No renderer or observation changes, new retry loop or persistent manager were added. The earlier old-runner and OIDN2.3.3 diagnostics reproduced the native renderer wait; root cause remains unresolved. See evidence/recovery-20261012/task-deferral.json and render-retry.json.
