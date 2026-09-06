# π0.5 GRPO pair live snapshot — 2026-09-01 18:04 CST

- Control and DVAC both completed Step 11 and were in Step 12 rollout `3/4`.
- Step 11 train success: Control `70.70%`; DVAC `[0.5,1.5]` `80.47%`.
- Trailing 5-step mean: `80.94% / 83.98%`; trailing 10-step mean: `83.36% / 83.40%`.
- Fixed-32: Control `28/32` at Step 5 and `27/32` at Step 10; DVAC `29/32` and `30/32. Overall `55/64` versus `59/64`.
- Step 11 Control KL/clip/grad=`0.104/0.196/30.929`; DVAC=`0.078/0.019/26.979`.
- DVAC weight mean/ESS=`1.026/0.966`; high/low clip fractions=`0.048/0.000059`.
- Both wrappers alive; Step 10 local-shard checkpoints complete; fatal/OOM/Vulkan/worker-death/nonfinite counts all zero.
- GPU4--7 used `62.8--66.3 GiB/card`; host available memory about `813.6 GiB`, memory PSI zero; `/data` had about `1.8 TiB` free.

Inputs are the two live `driver.log` and `resource.csv` files under `raw/`. `curves.csv`, `summary.json`, and the PNG are derived locally.
