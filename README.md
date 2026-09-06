# RL experiment evidence archive — 2026-09-06

This artifact-only branch archives the owned Shenzhen results and Windows rl workspace.
It is NOT a training/runtime branch and does not change any running experiment.

- `evidence/server-results/`: point-in-time configurations, logs, metrics, plots, tiny sidecars.
- `evidence/source-snapshots/`: source/records from known server worktrees, including local edits.
- `evidence/uncommitted/`: unvalidated diagnostic patches, preserved as evidence only.
- `evidence/windows/`: local historical notes, scripts, results, filtered container members.
- Each archive manifest records included/excluded files, hashes, redactions and compression.

Excluded by explicit user choice: replay/raw tensor datasets even when each shard is small;
videos, model/checkpoint weights, large binaries, caches and credentials. Large text is
losslessly gzip-compressed when practical. Original local/remote files are not rewritten.
This is an evidence backup, not a complete model/data backup or proof of resumability.
Recent still-running logs reflect the copy time, not final experiment completion.

The final documentation/script delta is recorded in `evidence/windows/late-final/`; its hashes override earlier copies in the full manifest.
