# RoboTwin Attn-I collection and Clean-weak C20 scan

User-authorized handoff, 2026-10-20. `SZ3 6/7`: Dojo stopped → RoboTwin
attention/U/DV/noise4/full-video collection → original per-card RLT.
`SZ1 4/5`: early-two-chunk sweep stopped → Clean CP140 weak vs Uτ3 CP70
strong C20 calibration/sweep → original RLT. Other cards remain unchanged.

## Reuse and exact changes

The accepted `tools/pi05_u_video` runner, full native video recorder, 50-task
seed/config set, B16, H50, N10, M5 side calculation and B48 three-extra-noise
batch are reused. There are 32 episodes per task, 1,600 total. Main noise0
still runs as B16; the three additional samples never alter executed actions.
The native main model is warmed before SAPIEN.

`legacy_observer.py` is copied unchanged from the accepted
`pi05-attn-recovery-20261012/tools/pi05_attn/observer.py`.
`observer.py` adds actual prefix Q/mask/position IDs and the final
`action_out_proj` input. Existing `record_signals=False` scopes exclude M5
and extra-noise passes. The native eager kernel, solver, policy and RNG are
not replaced. Layer indices are 9/13/17, all native heads and 10 evaluations.

For each layer, reconstruct full attention first, then sum object-token
columns. `I_direct` is action-to-object-word mass. `I_via_visual` multiplies
that layer's action-to-image attention by its image-to-object-word soft map,
then averages heads and layers. It is descriptive spatial overlap, not causal
flow. `I_total = I_direct + I_via_visual`; this sum is not a probability.
Conditional versions divide the aggregated numerators by instruction/visual
mass respectively. Zero denominators and unmatched spans are missing.

Object noun/phrase spans use a transparent English lexicon and surface
preposition role heuristic. These roles are not annotated ground truth.
Actual complete prompts, token IDs, character offsets, match/missing flags
and raw QK remain available to correct the spans without rerunning inference.
All-object columns use a token union. Source/goal columns are also retained.
No smoothing, min-max scaling or sharpening is applied.

## Files and plotting

Each query retains the accepted observation NPZ, U/noise4 arrays and execution
summary. New files:

- `query_NNN_attention.pt`: native-dtype main action Q/K, full prefix Q/K,
  actual masks and positions, latent/velocity chain, hidden state, token IDs.
- `query_NNN_attention.json`: actual prompt/offsets, camera/patch order,
  object spans, head/scale metadata, timings, sizes and validation results.
- `query_NNN_attention.npz`: `[B,10,50,6]` direct/via/total/conditional values,
  layer/head diagnostics, visual maps and DV.

Object axis: `all_noun, source_noun, goal_noun, all_phrase, source_phrase,
goal_phrase`. Main curve uses denoise index 9; 1/5/10 and last-five mean are
predeclared diagnostics. Plot direct, via and their raw sum together; U, DV
and noise consistency remain separate signals with their own units.

Use `episode_XX_control.jsonl` and `episode_XX_frames.json` to map real
executed query/h positions to video frames. Retain full action-level curves,
exclude unexecuted prediction tails, and select real same-episode peak/phase
frames. Full head-camera video is inherited at 25Hz plus query endpoints;
both wrist views are retained in each query's input observations, not claimed
as full wrist videos. The three previously repaired native tasks reuse their
accepted `get_info()` initialization.

First real B16 batch on each card verifies unchanged main action, complete
chain, M5 endpoint and RNG; native prefix/action attention reconstruction,
full-softmax row sums and score bounds; then complete video decoding/mapping.
These batches count toward the formal fixed budget. Native task failures use
the existing timeout/cleanup/skip loop. Attention-method validation errors stop
expansion instead of being misclassified as task errors.

## New C20 family

Protected weak: `pillbottle/clean_c17_cp140/full_weights.pt`.
Protected strong: `pillbottle/grpo_u_tau3_cp70/full_weights.pt`.
Both use strict full-state overlays on the accepted base wrapper and
normalization. A `weak-loaded.json` receipt prevents accidentally using the
original checkpoint as the weak policy.

Both execute C20, predict H50, use N10, native 200-action budget and unchanged
four B32 seed batches (two cards, 64 concurrent). Run 128 weak-only calibration
episodes and 128 strong-only baseline episodes. Derive high/low thresholds
from all valid new weak-only decisions, without outcomes. Never mix original
checkpoint U into this calibration. Historical dedup identity now includes
the weak overlay path.

Each direction gets 10 target fractions: 5%,15%,25%,35%,45%,55%,65%,75%,85%,95%.
Coarse coverage is run first, with the order fixed before outcomes. Every
condition uses 128 episodes: 2,560 sweep + 256 anchors = 2,816 episodes.
Gating U remains the accepted mean of H50 action-level U; both policies then
execute the first 20 actions. There is no early-two-chunk restriction.
Actual measured takeover fractions, not target fractions, are plotted.

Launch commands, resolved configurations and finite stopping rules are in
server transaction `prepared.json`; `clean_c20.py` reuses the accepted
policy-eval lanes and condition barriers. `launch_gate.py` is a finite
preparation gate that execs those lanes in the same PID, not an extra GPU
worker or permanent monitor.

## Closeout

Early2: 27/33 complete conditions, 3,520 valid episodes including 64 from an
incomplete condition; two in-flight B32 batches were interrupted. Near
64% first-two-block takeover, high-U is 80/128 and low-U 81/128. Both curves
rise mainly with takeover fraction; this run does not establish a clear
high-U advantage. The early-only-all anchor is 92/128 versus weak 55/128.

Dojo: 1,045 completed episodes, 106 successes. Stack bowls:83/125;
stack blocks:17/125; swap blocks:2/125; fasten screws:4/125. The remainder
has no completed successes. Partial/native-error runs are separate, not
silently counted as full-budget failures. Selected D09–D12 drawing materials
remain in the existing `uaca_cv27_draw_bkp/materials/dojo_selected4_20261020`.

`evidence/sz1-closeout.tar.gz` and `evidence/sz3-closeout.tar.gz` contain the
lightweight configurations, decisions, episode records, summaries and logs.
Large raw arrays, videos and protected checkpoints remain on their servers.
