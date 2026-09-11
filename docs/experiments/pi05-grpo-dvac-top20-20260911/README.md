# GRPO DVAC top20 packet — 2026-09-11

Implementation overlay for original clean π0.5 GRPO commit
`1d015a2aa03ec8132d8207ba47a2be3dbe1d9591`.
This is **256 attempts/round**, 64 parallel × 4 serial, G8, U2,
global batch1024 / micro32 on two actor GPUs. The packet does not include the
later half-sampling or seed42 patch. It changes no original budget/RNG field.

## Method

`algorithm.dvac_top20` is absent/off by default. Explicit `dvac_grpo/top20`
recipe enables batch-domain top20 with 0.1 full-update bypass probability.
`selection_domain=chunk|batch`, `top_fraction`, `selected_l`,
`full_update_probability`, and `method_seed` are configurable. Only these two
domains and the user-selected random-full bypass are implemented; no soft mix,
Prism, recent-five statistics, MinMax, or old DVAC transformations are mixed in.

The native OpenPI rollout's existing endpoint previews produce raw last-L
variance per action. No extra sampling/forward or policy RNG draw is added.
Valid positions follow the original filtered **chunk** loss mask, including the
original predicted tail supervision. This is not a new executed-prefix mask.

Before each full Adam batch is split into microbatches, actor ranks gather the
small frozen V/mask/stable-ID payload and compute exact top-k. `chunk` gives each
valid 50-position chunk ten positions of coefficient5; `batch` gives global
valid-position top20 across both ranks and may leave entire chunks unselected.
Coefficients are `valid_count / ceil(q * valid_count)` on selected positions,
zero otherwise. Exact ties use deterministic query-position hashes, not global
Torch/Python RNG. The unchanged native shuffle carries canonical IDs across U2.
For larger multi-batch rounds, selection follows each actual Adam batch.

Each Adam update draws one **shared** full-update decision: with probability p,
all native valid positions get1; otherwise weights remain top20. The independent
counter-based method RNG (`blake2b_counter_v1`) uses only method_seed and update
index. Rank0 computes the decision, broadcasts it, all ranks advance the same
counter, and every microbatch in that Adam update uses that decision.
No extra optimizer step is inserted. p0 restores pure top20; p1 restores clean
direct-PG values and gradients on the same fixed data/model parameters.

Weights enter the existing clean `logprob_st` gate before native joint-chunk
aggregation. Forward logprobs/ratio/clip stay unchanged; unselected logprob leaf
derivatives are zero. Do not additionally rescale the loss denominator. The
shared model still changes other outputs, and no 80% compute saving is claimed.

## State and diagnostics

New `dvac_top20_state_rankXXXX.json` sidecar stores full method identity,
selection/bypass settings, method RNG counters, rank/world and resume budget.
Exact restore rejects missing or mismatched sidecars **before** loading the
model. Switching from old clean/DVAC checkpoints is not silently exact resume.
Checkpointing during an incomplete Adam update is rejected. Legacy DVAC
sidecars/algorithms are untouched and old DVAC cannot be enabled simultaneously.

`dvac_top20/actor_rankXX/updates.jsonl` records full-Adam decision/index, global
selection statistics, stable query IDs, per-query selected counts and selected
position histogram. Per-query advantages, native mask/length contribution,
top/final mean coefficients allow analysis of positive/negative-side mass and
native-loss-weighted coefficient shifts across ranks. Metrics distinguish proposed top mask fraction from final
nonzero fraction after the bypass. Audit entries are prepared before Adam and
are not treated as completed-update state; the checkpoint is authoritative.

## CPU-only checks (run on server, no CUDA visibility)

Overlay src into an isolated worktree from the pinned base, then:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
 tests/unit_tests/test_dvac_top20.py \
 tests/unit_tests/test_dvac_top20_actor.py
```

The tests include CPU/Gloo two-rank selection and one shared decision per full
Adam update, independent RNG/resume, method-identity rejection, and the native
chunk PPO loss gradient gate. No GPU smoke or training is part of this packet's
current authorization. GPU FSDP/model execution and real CUDA resume remain for
a later explicitly scheduled smoke.
