# 单卡RLT control vs success-episode DVAC-BC：Step111现场

现场时间：2026-08-26 09:40 CST。两条均完整到Global Step 111/480并继续运行。

## 结果摘要

| 指标 | Control | DVAC-BC | 方法减Control |
|---|---:|---:|---:|
| Step111单步train success | 62.5% | 62.5% | 0.0 pp |
| 最近5步train success | 25.0% | 32.5% | +7.5 pp |
| 最近10步train success | 20.0% | 20.0% | 0.0 pp |
| Step1--111均值 | 14.75% | 14.41% | -0.34 pp |
| Step100 fixed-20 | 0/20 | 4/20 | +20 pp |

Step25/50/75 fixed-20两边均为0。Step100是第一次出现held-out分离，但只有一次20-episode评估，尚不足以
判断稳定领先。train每步只有8条episode，因此单步以12.5个百分点跳动，应优先看滑动均值。

方法版Step111权重p05/mean/p95=`0.732/1.000/1.259`、ESS=`0.975`；baseline已冻结，说明方法分支正常
apply，但仍属于较温和的C10内部重分配。control/method update step约`62.8k/62.9k`，训练预算对齐。

两条wrapper均alive，fatal匹配0，cgroup OOM/OOM-kill=0。RAM约154--155 GiB，历史峰155.6 GiB；
GPU0/1显存随rollout/offload波动，峰值约25.2/25.1 GiB。run目录约1.1/1.2 GiB，Step100 checkpoint
目录已经出现。

## 时间对比

| 指标 | 单卡Control | 单卡DVAC-BC | 历史双卡RLT |
|---|---:|---:|---:|
| 到Step111墙钟 | 9h34m20s | 9h35m29s | -- |
| 截至Step111平均 | 5m10.5s/step | 5m11.1s/step | -- |
| 最近10个普通step | 5m12.3s/step | 5m07.8s/step | -- |
| 到Step100墙钟 | 8h35m36s | 8h37m15s | 3h41m05s |
| 完整480实测/当前外推 | 约41h23m | 约41h29m | 20h12m49s |

历史双卡由fresh 1--250的10h46m16s与resume 251--480的9h26m33s组成，完整平均约
2m31.6s/cycle。当前两条单卡任务的DVAC-BC额外墙钟仅69秒（约0.2%），方法计算开销相对总训练可忽略。
当前完整外推取runner在Step111给出的ETA，后续实际速度仍会随eval、checkpoint和update阶段波动。

同Step100并非完全同阶段比较：当前单rank在cycle67开始online优化，而历史双rank约cycle135才达到两rank各
10k replay门槛。因此当前Step100已包含大量SAC更新，历史Step100仍主要在采样；除单卡计算串行化外，这也让
当前早期墙钟比历史高约2.33倍。完整480的总倍率约2.05倍，更适合用来估计总体耗时。

## 文件

- `RLT_CONTROL_VS_SUCCESS_BC_G111.png`：成功率、方法权重和资源一图；
- `curves.csv`：Step1--111原始、5步和10步train success；
- `summary.json`：本页汇总数值。

数据来自服务器只读下载的两份`metrics.log`与shared resource CSV；绘图脚本为
`tmp/plot_rlt_success_bc_formal_live.py`。本轮没有干预训练。
