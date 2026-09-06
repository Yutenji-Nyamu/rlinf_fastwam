# formal100 live snapshot through complete step 22

- Machine/account: `SZ-H100` / `chenyiteng`.
- Run: `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1`.
- Initial read-only metric audit: `2026-08-22T01:59:08Z` (`2026-08-22 09:59:08 CST`).
- Final boundary audit: `2026-08-22T02:08:21Z` (`2026-08-22 10:08:21 CST`).
- Complete metric tables: steps 1–22; training continued after the Step 22 boundary.
- Source: server `metrics.log`, parsed after removing terminal ANSI escapes. No server file or process was changed.
- `cumulative_mean_step_time_s` is the metric-table header's cumulative mean. `current_step_time_s` is that step's `Time/step` value; it includes periodic eval at steps 10 and 20.
- Eval fields are intentionally null/blank except at configured eval steps 10 and 20.
- CSV contains the fuller compact schema; JSON contains the fields requested for plotting and fast inspection.

## 可视化补充（2026-08-22 10:08 CST）

- `ppo_step22_dashboard.html`：响应式交互仪表盘；曲线支持 hover，图例可点击隐藏。
- `ppo_step22_training_trends_mobile.png`：手机可读的训练趋势静态图，明确区分 train 与仅在
  Global Step 10/20 执行的 fixed-64 eval。
- `ppo_step22_resource_snapshot_mobile.png`：GPU、主机内存与磁盘的单点资源图；资源值不是时间序列。
- CSV/JSON 已刷新为完整 Step 1–22；可视化直接读取其中 22 个互异 step，不再额外拼接 Step 22。
  最新 train success `0.9277344`、KL `0.014`、clip fraction `0.053`、grad norm `28.012`、critic EV
  `0.419`、value loss `0.020`，耗时曲线也覆盖 Step 1–22。
- 同时使用 10:08 单点资源刷新：四卡显存约 `62.1/61.4/60.7/61.0 GiB`、MemAvailable
  `597.00 GiB`、cgroup current `1,535,529,234,432 bytes`（约 `1.397 TiB`）、EnvWorker RSS
  求和 `1324.07 GiB`；未把 09:58 的瞬时 util 混入新快照。
- `resource_step22_boundary.json` records the follow-up memory samples through the complete Step 22 boundary. At the `10:08:21 CST` resource snapshot, cgroup memory was `1,535,529,234,432` bytes (`1.397 TiB`), host `MemAvailable` was `597.00 GiB`, the four EnvWorker RSS values summed to `1324.07 GiB`, and cgroup `high/max/oom/oom_kill` events were all zero. The very large EnvWorker allocation did not materially release at that boundary.
