# Explicit retirement of three completed multitask runs

2026-10-30: user explicitly selected the old C50 Direct, old C50 OneDP, and completed C10 Direct runs for deletion. Removed their 10 weight/export files, 8 matching protected hardlinks, and 32,458 raw evaluation arrays/videos/per-action control logs. Unique allocated storage released: 49.0555 GiB; Data available after deletion: 94.32 GiB.

Directories, all JSON configurations/results, scalar training logs, summaries, source, current single-task checkpoints, GRPO protected weights and RLT dependencies remain. The four C50 protection-registry entries are marked retired by explicit user authorization. These three retired runs no longer have local resumable checkpoints.

`receipt.json` summarizes the transaction; `exact-manifest.json.gz` records file identities and sizes, not file contents. Scope, ownership/inodes, hardlink completeness, open handles and active/queued dependencies were checked before unlinking.
