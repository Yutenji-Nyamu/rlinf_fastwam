# GRPO-DVAC v4 live snapshot through Step 31

Read-only snapshot of small logs and scalar artifacts used to refresh the matched GRPO comparison figures. No checkpoint or video is copied here.

Capture boundary: complete Step31; Step32 was in progress. The later read-only live refresh reached complete Step33, but these figures deliberately retain one exact shared input boundary.

Contents:

- `driver.log`, `resource.csv`, `resolved.yaml`, selected TensorBoard event/scalars.
- `dvac_v4_console_scalars_step1_31.csv` and `summary_step31.json`.
- `01_*`: DVAC through Step31, full matched baseline through Step52, explicit fixed64 marker legend and Step30 comparison.
- `02_*`: optimization and DVAC weighting scalars.
- `03_*`: one-minute resource timeline.
- `04_*`: AutoDL and Shenzhen within-machine global-z `[0,2]` comparisons.

Key paired results through Step31: train mean DVAC-GRPO=`-0.50 pp`; latest5=`-1.25 pp`. Fixed64 Step10/20/30 is `57/59/63` versus `57/60/62`; cumulative is `179/192` on both sides.
