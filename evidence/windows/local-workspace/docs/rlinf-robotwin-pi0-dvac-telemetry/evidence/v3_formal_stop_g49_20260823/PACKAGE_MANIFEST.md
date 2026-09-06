# v3 g49 high-information closeout package

This package represents the R-only DVAC v3 `[0,2]` run through the last fully completed Global Step 49.
The interrupted next rollout is excluded.

Included:

- `25_V3_G49_CLOSEOUT_AND_METHOD_ANALYSIS_20260823.md` and the operation ledger;
- four final PNGs, compact derived CSV/JSON tables, and reproducible analysis scripts;
- complete scalar `metrics.log`, the two rank-local metric/state/manifest files, and one representative g49 NPZ per rank;
- driver log, exact launch command, resolved YAML, complete resource CSV, and the TensorBoard scalar event.

Excluded to keep the package small:

- server checkpoints (`global_step_10/20/30/40`, about 9.7 GiB each);
- the other 96 rank-local step NPZ files;
- full control-trace video/frame bodies.

Server-side retained artifacts remain under:

```text
/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
```
