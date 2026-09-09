# π0.5 online BC: frozen RynnValue + SARM RA-BC

Independent branch from clean BC `01d770db3988da7862454e97434d4ff08f726fa2`.

Enable `algorithm.online_bc.rabc.enabled` explicitly. No DVAC or AttenA code is
enabled, and the disabled path retains clean BC behavior. First implementation
supports one actor rank, no demo mixture and no DrQ.

Raw main-camera RGB and task text are captured at query boundaries. A completed
successful episode is scored by a separate localhost service before admission.
For each boundary the locked RynnValue-8B uses four uniformly sampled causal
history frames and returns its last absolute slot, in remaining seconds. Progress
is `v_before - v_after`; terminal frames precede environment reset. The frozen
model runs BF16 on one GPU and moves back to CPU after every episode request.

Only raw progress and scorer identity are stored in replay. New admitted unique
queries update population mean/variance once. Repeated replay draws never update
these moments. The SARM soft ramp uses clamped mean and floored standard deviation:
`clip((r - (mu - 2*sigma))/(4*sigma + epsilon_signal), 0, 1)`;
negative progress gives zero, progress strictly above `kappa_seconds` gives one.
The threshold is a seconds-valued method parameter, not SARM's original 0.01
dimensionless progress copied into seconds.

Raw weights are normalized across the complete optimizer batch before splitting
into microbatches. The original masked query FM losses are weighted with
`g = B*w/(sum(w)+epsilon_weight)` then averaged. An entirely zero-weight batch
skips optimizer, scheduler and update counter. No reward-model gradients flow.

Checkpoint identity includes model/code/processor/runtime fingerprint, metadata,
threshold, numerical epsilons, statistics scope and admission limit. Resume checks
raw cached boundary values, unique episode keys and reconstructed replay moments.
Weights are recomputed when a batch is sampled; model inference is cached.

Start the scorer from a neutral working directory in its dedicated environment:

```sh
CUDA_VISIBLE_DEVICES=7 /path/to/rynnvalue-env/bin/python -u -B -c \
  "import runpy; runpy.run_path('/path/to/checkout/rlinf/utils/rynnvalue_scorer.py', run_name='__main__')" \
  serve --model-path /path/to/locked/RynnValue-8B --manifest /path/to/manifest.json \
  --device cuda:0 --port 18797
```

Use this entrypoint rather than invoking the utils file directly: its sibling
`logging.py` otherwise shadows Python's standard-library logging module.

This packet authorizes only a two-round GPU7 smoke: 4 new attempts/round, U5,
batch1024/micro32, clean LR2.5e-5, denoise10, horizon50. Success-length admission
is inherited from clean BC (off). The full resolved snapshot and method-only
configuration diff are generated before launch. Formal training is a later decision.
