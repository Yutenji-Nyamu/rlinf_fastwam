# Offline action-level signal review (2026-10-04)

The completed inference records are replayed on CPU to preserve each submitted action position, screen both localized peaks and phase contrasts, and pair candidate chunks with recorded before/after frames. Terminal successful chunks have unknown physical execution prefixes and are excluded from ranking.

- [Method and limitations](METHOD.md)
- [Current research context](context.md)
- [Inference terminal state and queue observations](execution-terminal.md)
- [Source hashes](source-manifest.json)

`code/*.py.txt` are exact execution-source archives, not new production entrypoints. The `.txt` extension makes their provenance role explicit and keeps them outside runtime imports. To replay, use the original server paths and receipts; do not blindly launch these archives. The separately maintained manual visual notes remain on the server.

Validation: all 1,077 batch queries recomputed; all score masks match. Seven scores match exactly; SR's equivalent Gram-eigenvalue calculation differs by at most 1.33e-15. All 368 marked entries exclude unknown terminal prefixes; all 1,732 gallery links resolve. A/B/C and visual labels are descriptive selection aids, not statistical significance or training evidence.

Plots, videos, NPZ files, per-episode candidate tables and full visual notes stay on the server. Only source archives, compact terminal/QC evidence and documentation are published here. Frozen inference source and training configs are unchanged. No additional GPU inference or training was run for this analysis.
