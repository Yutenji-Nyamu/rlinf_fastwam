# EXPO cache / renderer repair evidence

- `cpu-checks.json`: five server CPU checks and two real replay payloads, A → B → A.
- `learning-timing.json`: call 839 cold fill and calls 840–842 warm-cache timings; baseline previous 20 calls.
- `cache-migration.json`: original two-file cache/timer deployment; latest and last1 retain all six learning payload hashes.
- `renderer-migration.json`: renderer allocation boundary patch; both complete checkpoints retain their six learning payload hashes.
- `formal-acceptance-20261009.json`: exact formal restore, five new saved learner calls, live GPU/CPU/queue evidence and N4 evaluation real-action progress.
- `fallback-startup-checks.json`: exact GPU4/5 scope-file permission corrections and CPU Python startup checks; file content hashes preserved.

Source changes: `formal_replay.py` adds a 64 GiB on-demand CPU cache; `train_expo_formal.py` adds host timing and releases unused CUDA allocator memory before simulator creation. Algorithm, data sampling, checkpoint payloads, 60,000-action budget, GPU4/5, N1 collection, B64/Q20, 8+8 candidates, evaluation N4 × 20 every 10 episodes remain fixed.

Runtime evidence is sampled, not a guarantee of future progress. GPU6/7 BC driver identities were retained. RLT remains lower priority and waits for EXPO release. Large checkpoints, replay files and full logs remain on the server.
