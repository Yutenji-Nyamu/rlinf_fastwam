# Native pi0.5 attention recording and offline atlas

This observer records native B16/H50/M10 inference without training. The model, task configurations, seeds, preprocessing, and execution budget are inherited from accepted DV50 commit d74e7edc93d346caa9758dadd4537197eeaf0081. Run on the assigned physical GPU with its existing compute/graphics scope.

- `observer.py`: post-RoPE native Q/K at action expert layers 10/14/18, all 8 Q heads, all 10 denoising calls. Original eager kernel remains in use. Prefix K is shared once per query; all action Q/K are saved. Includes real mask/positions/token-source/camera mapping, original full FP32 random draw, actual solver initial latent, full D32 chain/velocity/t/dt, and final environment actions.
- `run_batch.py CONFIG.json`: original task/seed inference. The first two batches repeat the native first observation and compare observer on/off with exact actions, full chain, and RNG. Full observations and action submission mappings accompany each query. Terminal TOPP physical execution prefix remains unknown, with an explicit validity mask.
- `analysis.py` and `analyze_batch.py CONFIG.json`: one pass over saved raw tensors, FP32 attention reconstruction, probability average over 8 heads then 3 layers, modal conditioning, normalized and raw entropy, token counts, mass, JS/span, all-other/future Anchor, head/layer/step diagnostics, native final-five-step DV over effective D14, action/gripper comparisons, camera previews, and the interactive atlas.
- `lane.py 6|7`: per-card queue and one receipt-only barrier before expansion from 12 to 50 tasks x 32 requested episodes. No raw reload at the barrier, no silent failed-batch retry, no new cache quota. Completion plus exact compute/graphics clearance produces the release receipt consumed by the original RLT CP150 fallback owner. Existing slots4/5 stay intact.
- `test_math.py`: focused CPU tests of entropy vs mass, N0/1, JS and batch independence, and column/future Anchor direction.

## Numerical acceptance v1.1

The initial generic per-head FP32 max-absolute threshold 0.02 was exceeded in the two calibration attempts (BF16 reference vs FP32 reconstruction), while actions/chain/RNG remained exact. Those failed attempts are retained, not silently relabeled successful. The initial implementation stopped before saving the first raw query; this evidence gap was fixed so any subsequent numerical rejection preserves its raw tensors and unexecuted observation.

v1.1 reproduces the actual native arithmetic from saved Q/K and requires zero error against native A. It separately checks the intended head-then-layer mean FP32 distribution using the original 0.02 absolute / 0.08 row-L1 limits. Every per-head FP32 discrepancy, peak agreement, and entropy error remains in `parity.json`; the legacy threshold result is retained explicitly. Original native attention, actions, and RNG are not modified to meet a check.

## Evidence and interpretation

The two initial real batches passed v1.1: native attention reconstruction error 0, exact action/chain/RNG, FP32 main-distribution max error about 0.00166. Full capture completed for 32 episodes (turn_switch 9/16, adjust_bottle 11/16). These success rates are native inference outcomes, not a training comparison.

Each query saves all B16 slots and an active mask. Failed episodes remain in the atlas. Native reset retries and actual seeds are recorded; totals of requested records and distinct task/actual-seed pairs are separate. Position source labels do not imply pure causal modality content; mixed boundary tokens are `other`. Different cameras have separate patch coordinates. The FP32 path is an analysis reconstruction, not a bitwise replacement for BF16 inference.

Data and atlas: `/data/chenyiteng/results/pi05-attn/20261010-v1` on SZ3 only. Transaction receipts: `/data/chenyiteng/deployment-20261010/pi05-attn-v1`. 500 GiB is the planning budget, updated from measured batches; it is not an artificial hard cap. Original weights, trajectories, images, raw Q/K and credentials are not published to Git.
