# Fast-WAM online success BC — 2026-09-13

Independent branch: `codex/sz-fastwam-bc-20260913`. Clean BC base: `63349ad45f0d6a209f94e5d770ab568fb29d3082`; native Fast integration source: `0d5daf6fa98de14865a74be143f51ef2972dab49`; official Fast-WAM checkout: `7faa711`.

## Implementation

The existing online success-BC collector, replay sampling and optimization loop now dispatch to native Fast-WAM. The teacher predicts 32 actions and submits 24; successful samples store the actual 24 submitted commands plus the raw Fast observation. Native action normalization converts these commands to targets. The remaining eight prediction positions have zero loss mask. No predicted, unexecuted tail is labeled as successful, and no large KV cache is stored in replay.

Only the action expert is trained. VAE, text, video and proprioception components remain frozen. Training uses the official scheduler's flow-matching target and training weight (training scheduler shift 1); ODE collection retains the native inference override `sigma_shift=5`. Rollout seed 42 advances continuously, with evaluation RNG isolated. Existing OpenPI dispatch and default behavior are preserved.

## Candidate full-run configuration

This file records a prepared configuration; completing smoke does not mean a long training run has started.

| Setting | Candidate | Basis or difference from historical clean π0.5 8/U10 |
|---|---|---|
| Collection / updates / batch | 8 trajectories, U10, global batch 1024 | Same method budget |
| Optimizer | Adam, constant LR 2.5e-5, betas 0.9/0.95, epsilon 1e-8, weight decay 1e-10, clip 1 | Same |
| Initialization and data | Original Fast checkpoint, empty success pool; no demonstration mixing, no DVAC, no success-length filter | Same BC rule; native model |
| Seeds | Actor 1234; rollout 42; fixed 32 evaluation scenes | Same protocol |
| Schedule | 100 rounds, evaluation every 5 rounds, checkpoint every 10 | Same |
| Native model dimensions | Prediction 32, execution 24, action dimension 14, 10 ODE steps, three cameras | Fast model interface |
| Episode limit | 192 action steps | Changes from 200 to fit execution chunks of 24 |
| Resource settings | Microbatch 2, BF16, FSDP2, DCP checkpoints, bucket sync, environment offload | Changes from microbatch 32 / FSDP1; fit native Fast memory use |

Configuration: `examples/embodiment/config/robotwin_move_pillbottle_pad_online_bc_fastwam.yaml`.

## Verification

- CPU: **17/17 tests passed** (six native Fast BC tests and 11 existing online-BC tests). Full Hydra configuration composition and model/actor/rollout imports passed without CUDA initialization; `git diff --check` passed.
- Real clean50 data on CPU: 7,188 frames, 50 FPS. A raw frame provides the main camera, both wrist cameras, 14-dimensional state and instruction. Its real 24×14 action label has no padding. The loader supports the installed LeRobot namespace and the older namespace.
- GPU6 direct model check: real demonstration observation, ODE output `[1,24,14]`, targets `[1,32,14]`, 24 valid positions. BC loss and an independent official-scheduler oracle both equal **0.0032649245113134384**. Gradient norm **0.69921875**; action-encoder bias changed by **3.0517578125e-5**; frozen components received no gradients. Peak CUDA allocation **30.7963 GiB**. This verifies a real demonstration update, not an online success claim.

## Real online smoke

**Passed on GPU6.** The real RoboTwin/Ray run completed one round with 8/8 successful episodes, 56 archived successful queries, one learner update and a written checkpoint. Driver exit code was 0 and all recorded metrics were finite. The smoke used eight environments, at most 192 action steps per episode, U1, global batch 2 and microbatch 1; it validates collection → actual-command labeling → success replay → FSDP2 update → checkpoint. The 8/8 result is a small startup smoke result, not an estimate of trained policy performance.

Evidence acceptance, read after completion at 2026-09-13 21:48 CST:

```json
{
  "rounds": 1,
  "collection_success": [{"step": 0, "value": 1.0}],
  "returncode": 0,
  "all_metrics_finite": true,
  "success_episodes": 8,
  "success_queries": 56,
  "learner_updates": 1,
  "checkpoint_present": true,
  "passed": true
}
```

Source evidence receipts: `bc/cpu-result.json`, `gpu-smoke-v1/bc-demo.json` and `bc/online-acceptance.json` under the dedicated `server-maintenance-20260913/fastwam-bc-rlt` run. The CPU test pair was `test_fastwam_online_bc.py` and `test_online_bc.py`; the direct model check is reproducible using `tools/fastwam_bc_real_query_probe.py` with a real local LeRobot demonstration.

Large checkpoints, replay tensors and environment/token dumps are excluded from this source publication. Validation evidence remains in the task-specific server maintenance directory; only this compact summary accompanies the reviewed source files.
