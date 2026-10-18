# Completed policy evaluation evidence, 2026-10-18

The previous public branches contained implementation and startup records. This directory adds the completed22-condition comparison, paired episode/decision/timing records, and a bounded source archive inventory. All groups use128 formal episodes; the separate64-episode calibration remains calibration. Original timing-only/partial failed v2 records are kept in the raw archive and are not counted as completed groups.

`summary.csv` and `summary.json` are plot-ready. `records/` contains primary episode/decision/count/timing JSON and reused summaries with their original source paths. The source roots and original protocol/source commits remain in the archives. For CP180, high/low/random achieved59/70/66 of128, at27.39%/39.79%/32.48% realized decision takeover. These are unequal budgets, so the threshold sweep is needed to compare curves at similar rates.

The two raw compressed archives are published as GitHub Release assets under `inference-artifacts-20261018`; they include available text/JSON logs, complete-video frame maps, and failed-run evidence. Videos, per-query endpoint arrays, replay and model weights remain on the authorized servers. No model weights are placed in Git or the release. Existing protected-eval copies and exclusion registry remain authoritative for cleanup.

The new30-condition threshold sweep is still running; its final results are not included in this snapshot. Use the sibling threshold-sweep-20261018 execution record for current routing. The plot helper `tools/pi05_policy_eval/plot_takeover_rates.py SUMMARY OUTPUT_DIR` draws only completed points against actual takeover rates and exports CSV.
