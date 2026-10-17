# Post-ICLR experiment artifacts — 2026-10-17

Shenzhen owned runs updated since 2026-09-26 CST: 54,328 files, 1.73 GiB compressed, 167 archive parts plus 12 manifests. This is an artifact-only branch; original source branches remain unchanged.

- [Research and experiment discussion](research/00_INDEX.md)
- [Archive audit and scope](research/04_ARTIFACT_AUDIT.md)
- [Retained checkpoint candidates](research/03_CHECKPOINTS.md)
- [Family curve data](figures/curves.csv) and [figure inventory](figures/family-inventory.json)
- [Download raw light artifacts](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/post-iclr-artifacts-20261017)

Release assets are independent tar.gz parts, named by host; each part preserves original absolute paths without the leading slash. Manifests map every file to its part and include SHA256, size and observed modification time. Extract each host into a separate directory. Active logs are observed prefixes, not a globally atomic or final snapshot.

Excluded: model/optimizer tensors, replay and datasets, videos, caches and large individual logs. Source branches and remote HEAD checks are under evidence/. Live training is unchanged. AutoDL and unknown temporary directories are outside this collection.

## Using the plotting artifacts

1. Start with `figures/curves.csv` for the restored family charts. `figure` distinguishes panels and `series` identifies the line; `raw_y` contains unsmoothed values. The HTML and SVG files reproduce the 2026-10-17 11:34 snapshot.
2. For a different plot, filter `evidence/artifact-files.csv` by host/path and inspect the source config. It gives the archive batch, part, original path and SHA256. TensorBoard events, per-action JSON, derived NumPy arrays and evaluated episode receipts are preserved.
3. Download the matching Release assets. Main names look like `sz1-part-001.tar.gz`; supplemental batches use `sz1-extra-`, `sz1-plots-` and `sz1-text-`. Use the manifest from the same host and batch. Extract all selected parts into a new host-specific output directory, preserving their internal paths.

The source inventories distinguish training rollouts, fixed evaluations, restarted segments and historical baselines. Matching a task name alone does not make two curves directly comparable. See `research/03_CHECKPOINTS.md` for the retained-policy candidates and their historical evaluation evidence.

Large raw arrays and observations are listed in `evidence/numeric-exclusions.json`. Oversized text logs are listed in `evidence/text-exclusions.json`. These files remain on the servers. The metadata records their observed sizes; it does not promise future retention.
