# OGPO pi0 RoboTwin formal v2: live 21.8k snapshot

Read-only snapshot captured while the 90k run was still active. It contains only light logs,
TensorBoard/config provenance, the full one-second resource trace to the snapshot, and derived
tables/plots. It contains no replay or checkpoint tensor.

- Progress: 21,802/90,000 replay-valid primitive rows; 590/4,000 paired updates.
- Fixed-policy eval: 1/20 at row 0, 1/20 near 10k, 3/20 near 20k.
- No checkpoint exists yet; the first full resume checkpoint is due at 30k.
- GPU peaks: 57,231/56,834 MiB; cgroup peak 155,821,133,824 bytes; OOM/OOM-kill 0.
