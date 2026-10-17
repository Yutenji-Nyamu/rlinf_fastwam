# Frozen π0.5 policy and inference-budget evaluation

2026-10-17. User-authorized SZ1 physical GPUs 6/7; one process and 32 native
RoboTwin environments per card. Each test condition has 128 predetermined
episodes, in two waves of 64. Task: `move_pillbottle_pad`, clean, 200 submitted
action slots, original normalization, three cameras, native success flag.

Source parent: `3319b5ed4fb4e8ad406cc583f2eaa33b687f9103`, the accepted
20261003 signal inference tree. Runner initialization, native chunk stepping,
chunk-end video and simulator cleanup follow that implementation and the
accepted 20261012 B32 attention capacity path. Training/model code is unchanged.

## Frozen conditions and order

1. Original Sidney π0.5, N10/C50.
2. Clean C17 CP140, N10/C50: base construction and original normalization,
   followed by strict full-state loading.
3. Original weights, N5/15/20/30, each C50.
4. Original weights, C10/20/30, each N10 and prediction horizon H50.
5. Independent 64-episode weak-policy calibration; freeze U quantiles 1/3, 2/3.
6. High-U, Low-U, Random strong-policy replacement, each 128 episodes.

Total: 12 test conditions ×128 +64 calibration =1600 episodes. C100 is not
valid with the checkpoint's H50. No repeated/padded stale chunks or silent H100
change. The user explicitly chose to omit C100 and retain the clean H50 comparison.

Official eval seeds are fixed before any outcomes; exclude the official train
set, previous fixed GRPO eval seeds and the first32 historical signal seeds.
Tests share 128 seed/slot assignments. Calibration uses a disjoint64.
Native reset retries and actual seeds remain recorded; summaries only report a
full paired128 result when actual seeds really match uniquely.

Each corresponding policy query uses the same natural full-batch initial noise
seed. Different denoising counts and additional strong calls cannot shift the
next query's RNG. Main N10 +same-noise side N5 uses the existing accepted sampler
and U formula (mean over normalized D14, then H50). No attention hooks/raw hidden
arrays are needed. One calibration warmup per card verifies exact main-action
and RNG parity. Weak U is recomputed at every new chunk, including after strong
chunks. Strong selection acts on entire chunks, not individual joints.

Thresholds aim at one-third takeover on calibration; changed state distributions
can change test takeover rates. Report actual decision and submitted-slot shares.
Strong inference runs a full fixed batch whenever any active slot selects it,
preserving slot/noise alignment; timing therefore reflects this implementation,
not a hypothetical optimally packed strong subset. Terminal TOPP physical action
prefix remains unknown; submitted-slot counts are not exact physical-action counts.

## Execution and stopping

Deployment: `/data/chenyiteng/deployment-20261017/pi05-policy-budget-eval-v1`.
Results: `/data/chenyiteng/results/pi05-policy-eval/pillbottle-20261017-v1`.
Resolved configs and official seed subsets are under results `configs/`.
`lane.py 6` and `lane.py 7` launch the existing venv Python with a single physical
CUDA device and the accepted per-card NVIDIA graphics profile. Environment files
are scoped to these processes. The original owner keeps the exact RLT fallbacks;
its external dependency changes only for6/7 to these finite queues.

Each condition finishes on both cards before the next begins. A native batch
error, invalid weights/actions, 600s without query progress or 3600s batch limit
stops the lane without automatic replay. Exact owned-process cleanup precedes
release; the original owner independently checks identity exit and C/G clearance
before returning that card to RLT. Existing EXPO4/5 and shared Ray are unchanged.

CPU validation on the server: `python -B -m unittest -v test_protocol` from this
directory. Tests cover active-slot gating, calibration-only quantiles and invalid
C100. Native startup/strict weight load and live C/G binding are checked once.

Outputs: per-episode seeds/success/counts, per-decision U/selection, per-query
timings, small videos, `summary.json` with Wilson95 intervals and paired
gain/loss counts against the original baseline. Large weights stay on the server.
