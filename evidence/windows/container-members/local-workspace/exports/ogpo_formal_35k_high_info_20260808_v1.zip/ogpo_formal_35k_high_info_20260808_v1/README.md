# OGPO pi0 RoboTwin formal 35k: compact evidence package

This package contains the high-information artifacts from the completed 35k run. It deliberately excludes the 58.23 GiB full checkpoint and all replay tensors.

## Result at a glance

- Exit 0 after 35,000 replay-valid primitive transitions, 10,000 warmup rows, and 2,500 paired actor+critic updates.
- 26 eight-environment waves = 208 simulated train episodes; 57 succeeded (27.4%).
- Fixed-policy evaluation: 1/20 at row 0, 7/20 at row 20,081, and 1/20 at row 35,000.
- Numerical training stayed finite, but the final policy lost the mid-run evaluation gain.
- Wall time was 12:23:42: paired updates 68.9%, rollout 28.5%, evaluation about 1.9%.
- GPU peaks were 57,231 and 57,676 MiB; no cgroup OOM or OOM-kill event occurred.

## Main files

- `SUMMARY.json`: machine-readable final outcome and checkpoint composition.
- `metrics_table.csv`: one row per 8-env train wave.
- `evaluation_points.csv`: the three fixed-policy evaluations.
- `metrics.log`, `driver.log`, TensorBoard event/config: original training evidence.
- `resources_1s.csv`, `resource_summary.json`: complete one-second resource trace and summary.
- `resolved.yaml`, `source_config.yaml`, `exact_command.txt`, `run_provenance.tsv`: exact launch provenance.
- `checkpoint_complete.json`, `checkpoint_inventory.tsv`: completion evidence without model/replay tensors.
- `visuals/*.png`: phone-readable training and resource plots.

## Interpretation boundary

The run establishes that the end-to-end OGPO+CA path executes and learns transiently. Three evaluation points are too sparse to locate the post-20k regression, and low critic TD loss alone does not establish reliable candidate ranking.
