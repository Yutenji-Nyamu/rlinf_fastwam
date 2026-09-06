# R-only DVAC formal：Global Step 23 快照分析

本目录分析服务器在 `2026-08-22 09:56 +08:00` 左右制作的只读快照。训练日志包含完整
Global Step 1–23；`rollout_step0022.npz` 对应 Global Step 23。所有 success 都是每步 256 条
on-policy 训练 rollout 的 `success_once`，不是 held-out 评估。

## 一眼结论

- R-only g1–23 rollout success 均值 `82.52%`；历史成功 GRPO 同一 global-step 区间为
  `84.97%`，相差 `-2.45 pp`。g23 单步为 `86.72% vs 85.55%`，但最近5步仍为
  `84.30% vs 87.73%`，所以当前没有形成稳定领先。
- 优化强度整体接近：平均 KL `0.0387 vs 0.0423`，joint-query PPO clip fraction
  `0.1281 vs 0.1467`，pre-global-clip gradient norm `35.52 vs 34.43`。
- R-only 权重确实显著不同于1：g2–23平均 p05/median/p95约
  `0.715/1.031/1.198`；约`37.56%`位置降权、`62.44%`位置增权。平均权重仍接近1
  (`0.9995`)，因此主要改变50个future action的相对梯度构成。
- g23原始 `log V_L3` 后25格比前25格高`0.191`，recent per-h位置baseline本身高`0.141`；
  去位置后的 residual 只剩`+0.023`，最终weight只剩`+0.0034`。这说明新版已经基本消除
  “chunk越靠后天然越高”的主趋势，同时保留同一h下的状态/action异常。
- GPU峰值`30.37/30.22 GiB`，空间宽松。cgroup RAM峰值`226.62/240 GiB`，快照末
  `217.56 GiB`；所有memory max/OOM/OOM-kill事件仍为0。主要RAM占用来自EnvWorker。

历史GRPO是工程参数对齐的历史参考，不是本次同seed并行control arm；Global Step 1在DVAC尚未应用时
两条run已经相差`-1.17 pp`，因此当前差值只作同轴描述。

## 图和表

- `TRAINING_VS_GRPO_G23.png`：success、KL、PPO clip、pre-clip gradient norm和step时间的
  同轴对照。
- `DVAC_R_ONLY_DIAGNOSTICS_G23.png`：g2–23权重分布、升/降权比例、robust residual和g23逐h权重。
- `RESOURCE_OVERVIEW_G23.png`：GPU、cgroup/进程RSS、余量与周期性资源结构的详细资源图。
- `RESOURCES_G23.png`：较简洁的GPU/cgroup/host资源时间线。
- `ANALYSIS_SUMMARY_G23.json`：训练、DVAC和基础资源的机器可读摘要。
- `RESOURCE_SUMMARY.json`：资源与process RSS的详细机器可读摘要。
- `TRAIN_METRICS_G23.csv`：解析后的23个完整global step。
- `TRAINING_VS_GRPO_G23.csv`：R-only与历史GRPO逐step合并表。
- `DVAC_RANK_METRICS_G23.csv` / `DVAC_STEP_METRICS_G23.csv`：双rank原表与逐step聚合表。
- `LATEST_HORIZON_G23.csv`：g23精确合并两rank后的50个future-h统计。
- `SNAPSHOT_FILE_INVENTORY.csv`：本次下载快照的文件清单与大小。

复现训练/DVAC分析：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  .\docs\rlinf-robotwin-pi0-dvac-telemetry\evidence\r_only_formal_live_g23_20260822\analysis\analyze_r_only_g23.py
```

资源细分由同目录 `analyze_resources_g23.py` 复现。
