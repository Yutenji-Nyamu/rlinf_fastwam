# 四条单卡 RLT 训练成功率对比：共同 Step 195

2026-08-28 14:40 CST只读刷新。为保证横轴一致，结束于Step476的旧两条实验也只画到当前四条共同拥有的
Step195；图中均为training-rollout success，不是fixed20评估。

| run | Step195 | 最近5步 | 最近10步 | Step1--195累计 |
|---|---:|---:|---:|---:|
| RLT clean | 25.00% | 62.50% | 65.00% | 20.96% |
| RLT-DVAC-BC previous | 62.50% | 72.50% | 66.25% | 20.19% |
| Pure `[0,2]` / s0p5 | 87.50% | 82.50% | 70.00% | 17.95% |
| Pure `[0,5]` / s2p0 | 100.00% | 82.50% | 68.75% | 20.00% |

当前两条Pure在最近5步共同最高；最近10步为`s0p5 > s2p0 > previous DVAC-BC > clean RLT`，差距仍小。
累计轴则clean RLT最高，说明Pure目前主要是最近阶段上升更快，尚不能概括为全程稳定领先。

- `RLT-DVAC-BC previous`：成功episode切换到executed-action target，再用DVAC重分配BC。
- `Pure`：始终保留原RLT的π0 reference target，只用DVAC重分配成功episode内C10 BC。
- 颜色：clean蓝、previous紫、Pure s0p5绿、Pure s2p0橙；旧两条另用虚线区分。

文件：`RLT_FOUR_SINGLE_GPU_SUCCESS_THROUGH_G195.png`、`success_curves.csv`、`summary.json`。
