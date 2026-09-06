# RLT teacher-DVAC fresh-480：g342只读快照

采集时间：2026-08-25 09:38--09:56 CST。服务器训练未被修改、暂停或发送signal。

## 结论

- 最新纳入图表的完整cycle为`g342/480`，进度71.2%，ETA约6小时03分；formal仍在连续运行。
- `g136`起replay warmup结束、DVAC baseline冻结并进入apply；`g342`累计`update_step=161,000`。
- `g342`训练rollout为`7/8=87.5%`，5步/10步均值为`92.5%/90.0%`。
- fixed20评估从`g175=40%`开始上升，`g300=90%`，最近`g325=80%`。它比每步仅8条on-policy rollout更适合读策略效果，但还不是与uniform RLT的配对对照。
- 最新DVAC权重`p05/median/p95=0.319/0.952/1.861`，mean=`0.993`，ESS=`0.896`，top20质量=`28.9%`；说明平均尺度接近1，但action间的相对梯度分配已明显改变。
- cgroup current/peak=`101.8/102.5 GiB`，EnvWorker RSS current/peak=`60.6/61.3 GiB`；GPU峰值`19.37/19.51 GiB`。`high/max/oom/oom_kill`与PSI压力事件均为0。
- 最新完整checkpoint为`g325`；服务器数据盘剩余约725 GiB。

## 文件

- `RLT_DVAC_LIVE_G342_SUMMARY.png`：训练rollout、fixed20、DVAC权重、资源四面板。
- `step_metrics.csv`：只含`g1--g342`完整metric table；未完成cycle不计。
- `summary.json`：本快照的关键数值。
- `raw/metrics.log`、`raw/resources.csv`：服务器原始只读副本。

绘图只使用完整metric table和可解析资源样本；图像已人工检查标题、图例、坐标和注释。
