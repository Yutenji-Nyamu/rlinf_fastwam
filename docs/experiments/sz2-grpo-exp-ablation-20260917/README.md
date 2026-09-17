# sz2 GRPO128 DVAC exponential ablations

Based on 8aa03740785f298a72943d345f6b6e117e0a509d. Fresh 200 rounds, 128 trajectories per round, G8, B512/micro32, U2, lr5e-6, seed42, fixed32 every5, save10. Algorithm source files unchanged.

- t2-notricks: physical 4,5, both tau=2, positive advantages, initial local/chunk alpha=1; dropout and annealing disabled.
- t15-notricks: physical 6,7, both tau=1.5, positive advantages, initial local/chunk alpha=1; dropout and annealing disabled.

Direct personal Ray on one unmasked 8-GPU node; each experiment uses only its assigned pair and a separate namespace. No Slurm configuration changes. User authorized formal launch and observation of 1–2 rounds; no extra smoke. Runtime contracts preserve the original SZ1/source and config provenance.
