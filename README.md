# Post-ICLR experiment artifacts — 2026-10-17

Shenzhen owned runs updated since 2026-09-26 CST. This is an artifact-only branch; original source branches remain unchanged.

- [Research and experiment discussion](research/00_INDEX.md)
- [Archive audit and scope](research/04_ARTIFACT_AUDIT.md)
- [Retained checkpoint candidates](research/03_CHECKPOINTS.md)
- [Family curve data](figures/curves.csv) and [figure inventory](figures/family-inventory.json)
- [Download raw light artifacts](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/post-iclr-artifacts-20261017)

Release assets are independent tar.gz parts, named by host; each part preserves original absolute paths without the leading slash. Manifests map every file to its part and include SHA256, size and observed modification time. Extract each host into a separate directory. Active logs are observed prefixes, not a globally atomic or final snapshot.

Excluded: model/optimizer tensors, replay and datasets, videos, caches and large individual logs. Source branches and remote HEAD checks are under evidence/. Live training is unchanged. AutoDL and unknown temporary directories are outside this collection.
