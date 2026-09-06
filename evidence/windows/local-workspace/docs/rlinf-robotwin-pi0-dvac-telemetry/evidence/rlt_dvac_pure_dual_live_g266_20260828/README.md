# RLT-DVAC-Pure 双实验 Step 268 现场

快照时间：2026-08-28 20:01:56 CST。该目录是只读下载和离线汇总，不改变训练。

## 1. 生命周期与资源

- `s0p5`：完整到 Step 273，driver 存活；UI 估算剩余 `14:56:34`。
- `s2p0`：完整到 Step 268，driver 存活；UI 估算剩余 `15:32:30`。
- 两条 compose error 均为 0；扫描未见 fatal；cgroup `oom=0`、`oom_kill=0`。
- 现场 RAM `223.71 GiB`，本轮资源 CSV 峰值 `224.60 GiB`。
- GPU0/1 显存历史峰值约 `25.17/25.14 GiB`。

## 2. 共同 Step 268 对照

| 指标 | s0p5，`[0,2]`简称 | s2p0，`[0,5]`简称 |
|---|---:|---:|
| Step 268 raw train success | 87.5% | 100.0% |
| trailing 5-step | 82.5% | 80.0% |
| trailing 10-step | 87.5% | 86.25% |
| cumulative train success | 36.71% | 35.45% |
| Step 200 fixed-20 | 16/20 | 13/20 |
| Step 225 fixed-20 | 19/20 | 13/20 |
| Step 250 fixed-20 | 19/20 | 15/20 |
| weight p05 / mean / p95 | .477 / 1 / 1.511 | 0 / 1 / 2.546 |
| weight ESS | .912 | .589 |
| top-20% weight mass | 28% | 45% |

当前事实是：较温和的 `s0p5` 在最近三次 fixed-20 均高于 `s2p0`；这不能单独证明最终方法效果，但已说明继续放大同一“高 DVAC 增强 reference-BC”映射没有显示出收益。

## 3. 文件

- `RLT_DVAC_PURE_SUCCESS_THROUGH_G268.png`：逐步、5步、10步和 fixed-20 四图。
- `success_curves.csv`：作图数据。
- `summary.json`：上述数字和资源摘要。
- `raw/`：两条原始 metrics log 与共享资源 CSV。
