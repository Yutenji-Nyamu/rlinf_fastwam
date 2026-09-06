# Metric glossary

- `env/success_once`: eight-episode training rollout wave from the EMA actor.
- `eval/success_once`: 20-episode fixed evaluation from the online actor.
- `ratio`: burst mean of the online/EMA same-chain likelihood ratio; it is not a clip fraction.
- `replay_rows` and `success_rows`: per-rank means; global progress is `total_online_rows`.
- cgroup `oom=0`: no kernel OOM; Ray killed a worker proactively at its 95% threshold.
