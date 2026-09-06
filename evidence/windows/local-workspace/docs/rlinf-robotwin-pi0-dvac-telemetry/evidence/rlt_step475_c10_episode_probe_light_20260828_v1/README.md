# RLT Step475：C10 episode DVAC 数值时间线

本包从已停止的matched-width旧方法Step475 replay中只读抽取。扫描最后4,000个query row，重建329条完整
student-route episode：成功281条、失败48条；再按episode平均C10 DVAC的低/中/高各选3条展示。

## 主要结果

- replay完整保存`teacher_dvac_v=[L2,L3,L4,H50]`，但RLT/Pure训练只消费`L3,h=0..9`。
- 329条episode中，单query的`h10..49`均值比`h0..9`高`0.222 log10`，即约`1.67×`；
  87.39%的query都是后40格更高。这正是H50远期位置效应，不能把后40格混入C10 BC权重。
- 成功episode通常在第10--12个query结束，正reward出现在最后query；失败episode通常跑满20个query。
- 聚合时间线没有出现“越接近成功，C10 DVAC越高”的单调模式。成功均值由开头约`-1.96`降到末端
  约`-2.15`；失败由约`-1.96`降到约`-2.25`。个别episode会在早期或中段出现明显高峰，但仅凭无图像
  replay不能把峰值命名为接触、抓取或放置阶段。
- 用旧run冻结baseline对这4,000行做Pure反事实映射：
  - `s0p5`：p05/mean/p95=`0.515/1.000/1.488`，mean ESS=`0.922`，top20质量=`27.7%`；
  - `s2p0`：p05/mean/p95=`0/1.000/2.539`，19.74%位置归零，mean ESS=`0.593`，top20质量=`45.0%`。
  这是旧数据上的离线幅度预览，不是新formal的训练指标。

## 图与原始数据

- [六条episode时间线](FIG01_EPISODE_TIMELINES.png)
- [实际使用的C10热图](FIG02_C10_HEATMAPS.png)
- [同query的H50上下文热图](FIG03_H50_CONTEXT_HEATMAPS.png)
- [329条episode归一化进度聚合](FIG04_PROGRESS_AGGREGATE.png)
- [episode摘要](episode_summary.csv)、[逐query表](queries.csv)、[原始数组](selected_episodes.npz)、
  [机器可读摘要](summary.json)

## 能恢复与不能恢复

按`trajectory_index.json -> trajectory_id_list`读取，并以当前action对齐的`done`切段，可以恢复synthetic
episode序号、query顺序、成功/失败、reward、动作、reference和DVAC。旧配置没有保存图像/视频，也没有
reset ID、原env slot或精确runner step，因此本包只能做数值阶段分析；要解释具体机器人阶段，需要另做一次
fixed-reset带control trace的评估。
