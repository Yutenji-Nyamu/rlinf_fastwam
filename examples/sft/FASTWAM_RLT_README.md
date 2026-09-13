# Fast-WAM current RLT implementation

Based on clean current-AR RLT `77d0a673`, with pinned Fast-WAM integration
`0d5daf6f` and official model code `7faa711`. Fast-WAM is frozen throughout.

## Data and method

- Stage1 raw clean50 cameras/state/task go through the existing Fast-WAM
  RoboTwin adapter. Cache only final current-frame video hidden (120 x 3072).
  The complete original two-layer encoder + two-layer causal-AR teacher-forcing
  decoder is trained; z remains 2048. No pi0 Stage1 weights are reused.
- Stage2 keeps original clean student/Q objectives and replay. The teacher
  emits H32, student/reference use C24; student uses explicit identity output
  in Fast-WAM's native z-score action coordinates. Old pi0 defaults to tanh.
  Before physical execution, replace teacher template's first 24 positions,
  then use official action denormalization. No teacher target clipping.
- Stage2 full recipe preserves 8 envs, B512/micro256, UTD5, critic/actor ratio2,
  replay80k, warmup pool20k, initialization30k updates, original20k+50k BC/Q
  schedule, 1600 per-round update cap, eval20 every25, 600 rounds.
- Model-specific deltas: C10→24, H50→32, native 10-step ODE/shift5,
  Fast stats and feature; episode/rollout/step_lim200→192 for C24 divisibility.
  Thus the count of physical commands and queries is not an identical budget.
- Stage1 global32, 2000 updates, LR2.5e-5, betas.9/.95, warmup100 then cosine,
  clip1 are retained. Microbatch1 bounds frozen-model/AR memory, accumulation
  retains the global batch. Stage1 uses AdamW and the same tiny1e-10 decay.

## Sequential GPU6 smoke commands

Use the server's Fast-WAM environment, project PYTHONPATH and pinned official
FASTWAM_CONFIG_DIR. Set CUDA_VISIBLE_DEVICES=6 only for these standalone probes;
the shared-Ray launcher must keep it unset and use global placement6.

```bash
python examples/sft/train_fastwam_rlt_stage1.py extract \
  --config examples/sft/config/robotwin_rlt_stage1_fastwam_current_ar.yaml \
  --out "$RUN/stage1-features" --max-frames 8
python examples/sft/train_fastwam_rlt_stage1.py train \
  --config examples/sft/config/robotwin_rlt_stage1_fastwam_current_ar.yaml \
  --features "$RUN/stage1-features" --out "$RUN/stage1" --steps 2
python examples/sft/train_fastwam_rlt_stage1.py train \
  --config examples/sft/config/robotwin_rlt_stage1_fastwam_current_ar.yaml \
  --features "$RUN/stage1-features" --out "$RUN/stage1-restored" --steps 3 \
  --resume "$RUN/stage1/stage1.pt"
python examples/embodiment/fastwam_rlt_probe.py \
  --config examples/sft/config/robotwin_rlt_stage1_fastwam_current_ar.yaml \
  --stage1 "$RUN/stage1-restored/stage1.pt" \
  --observation "$RUN/stage1-features/first-observation.pt" \
  --out "$RUN/interface-probe.json"
```

The probe compares every retained layer K/V bitwise to upstream prefill,
checks physical teacher/reference decode equality, and updates student and Q
twice with the original `RLTACLossMixin`. Its transition is explicitly
synthetic: it is an interface check, not online success or task performance.

The real environment/replay smoke uses config
`robotwin_adjust_bottle_rlt_fastwam_smoke`. It intentionally reduces env/batch,
warmup and updates for three rounds, with unchanged model C24. Set
FASTWAM_RLT_STAGE1_PATH and the existing RLT_STAGE1_MANIFEST_* /
RLT_NORM_STATS_SHA256 contract environment variables to the fresh Fast files.
The formal recipe requires at least2000 Stage1 updates; smoke requires2.
Short Stage1 artifacts are marked `smoke:true`, not mature trained features.

CPU checks: `pytest -q tests/unit_tests/test_fastwam_rlt_adapter.py` plus the
existing RLT loss/route/resume tests. Full Stage1/formal training is a separate
launch decision after smoke and model-specific budget review.
