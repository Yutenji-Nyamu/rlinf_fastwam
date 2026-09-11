# SARM clean mix (2026-09-11)

Independent source overlay on SARM production `44f9086ad5a9c315dfabf5084f6afbba824af77f`,
closeout branch HEAD `4a3d1222b57ff826b4d4edfff239906596e457c5`.
The original `bc_rynnvalue_rabc_packet` is unchanged. Copy `src/` over the new
SARM worktree and `tests/` to that worktree's `tests/`; all other SARM sources are
inherited, including data, scorer, FM loss, and the parent update loop.

## Method

`algorithm.online_bc.rabc.clean_mix` defaults to `0.0`; candidate overlay is `0.5`.
Keep the original raw SARM mapping and complete-optimizer-batch normalization.
For a nonzero batch use `c + (1-c)*W`, without further clipping/normalization.
For a valid all-zero batch, c=0 preserves the old skip and c>0 uses all ones.
The original bool action mask is validated before fallback; it still controls
which FM targets contribute to the loss. Invalid scores/masks do not fall back.

The overlay changes only this method parameter: inherited 4/U5, 1024/32,
kappa=2 seconds, length filter off, optimizer/scorer and environment settings
remain in the authoritative resolved config. This file is not a launch script.

Existing raw/normalized SARM diagnostics keep their original meaning. New
`final_weight_*`, `final_effective_sample_size`, `base_skip_update`, final
`skip_update`, and `clean_fallback` describe the actual coefficients. Batch
metrics describe the last batch; separate `*_updates_this_round` counters
accumulate fallback/skip events and reset at the beginning of each training call.
Debug files contain raw, normalized and final weights plus action masks.

## Checkpoint contract

Method identity v2 records `clean_mix`, derived `all_zero_policy`, and
`weight_postprocess=after_full_batch_normalization`. Exact old v1 is accepted
only with c=0 and recorded as `rabc/resume_legacy_identity=1`. A different c or
method schema is rejected; old checkpoint to c=.5 requires a separately defined
method fork, not silent resume. Scoring cache identity and stats v1 do not change.

The actor load path now restores offloaded parameters/optimizer before FSDP
checkpoint load, matching the existing save path and the prior BC resume fix.
CPU method tests cover ordering and flags; they cannot validate a real CUDA/FSDP
restore. No GPU smoke or training is performed by this packet.

## Targeted CPU checks (run on the server)

From the independent, fully overlaid SARM worktree:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD" /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_online_bc_rabc.py tests/test_online_bc_rabc_actor.py
```

The numerical tests include legacy identity and tensor preservation, c bounds,
epsilon behavior, all-zero fallback and invalid inputs. Actual actor-method
tests exercise full 1024/micro32 masked loss and gradient equivalence, debug
receipts, fallback counters, Adam/scheduler/learner advancement versus old skip,
identity rejection and deterministic restored coefficients. The update-loop
test uses real CPU Adam and tensors with mocked GPU synchronization only.
The server's existing data/client tests may be run alongside these unchanged
tests; they are inherited from the original SARM branch.
