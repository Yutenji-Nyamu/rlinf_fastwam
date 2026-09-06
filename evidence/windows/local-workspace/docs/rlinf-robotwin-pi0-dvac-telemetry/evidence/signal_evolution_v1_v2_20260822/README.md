# DVAC训练信号演化：v1 g54 与 v2 g35

本目录只分析已经记录的train-SDE telemetry，不修改或重放训练。

## 产物

- `DVAC_SIGNAL_EVOLUTION_V1_G54_V2_G35.png`：raw DVAC、v1 global-z、v2位置基线/residual及最终权重的逐step图。
- `SIGNAL_EVOLUTION.csv`：图中逐step数据。
- `SIGNAL_EVOLUTION_SUMMARY.json`：主要端点数值和g35四通道分解。
- `build_signal_evolution.py`：从v1 closeout、v2 g35 snapshot复现上述产物。

## 先区分模型信号、派生信号和训练干预

1. `V_L3(q,h)`是最后三次flow endpoint preview的方差；`y=ln(V_L3+eps)`便于处理长尾。这是模型信号。
2. v1把所有`q,h`与recent-5的global mean/std比较，得到global z；future-h位置趋势仍在其中。
3. v2先为每个`h`建立recent-5中位数`b_h`和MAD尺度，再得到dimensionless residual `R(q,h)`。
4. `weight`是由z或R映射出来的训练干预，不是DVAC本体；`A_q*weight(q,h)`才是GRPO方向与DVAC音量合成后的effective credit。

## 这次能看到什么

- v1的mean raw `ln V_L3`从g1的`-4.4128`升至g54的`-3.9470`，对应几何均值V约`+59.34%`。
- v2从g1的`-4.4236`升至g35的`-4.0209`，约`+49.58%`。
- 这是on-policy曲线：参数在变，策略访问到的state也在变，所以不能仅由该趋势判断“模型自身变得更不确定”。固定state/checkpoint telemetry才能隔离模型变化。
- v2 g2–35的residual median平均为`0.303`，g35为`0.238`；p95经常达到`+2`上限，说明正尾持续存在。
- g35把clipped residual继续拆成`S_q=mean_h R(q,h)`和`I_qh=R(q,h)-S_q`后，约`39.8%` residual方差是query-wide，`60.2%`是query内部local shape。这里是统计分账，不等于已经得到任务关键性标签。
- g35 raw `log V`后前差为`+0.223`，位置基线`b_h`的log尺度后前差为`+0.085`；标准化R后前差
  `+0.181`、weight后前差`+0.0237`。R是无量纲，不能与前两个log值直接相减；这组数只说明当前
  horizon shape在去recent典型位置曲线后仍有残留结构。

## 哪些量有逐step记录

| 量 | v1 | v2 |
|---|---|---|
| raw `V_L2/V_L3/V_L4(q,h)` | 每step NPZ；本地完整raw到g39，server保留到g54 | 每step NPZ；server当前到g35 |
| raw `ln V_L3` mean/std | runner CSV逐step | runner CSV逐step |
| global z / clipped-z | 实际训练信号，NPZ与runner汇总 | 不使用 |
| per-h `b_h/MAD/scale` | 训练时未建立；可由raw NPZ离线重算 | 实际训练基线，NPZ直接保存；runner保存中位汇总 |
| residual `R(q,h)` | 训练时未使用；可离线重算 | NPZ直接保存；runner保存p05/median/p95 |
| `S_q`、`I_qh` | 可由离线R派生 | 可由已存R精确派生；目前不是单独字段 |
| weight及正/负advantage mean weight | NPZ与runner逐step | NPZ与runner逐step |
| `A_q*weight(q,h)` | 可由NPZ派生 | 可由NPZ派生 |
| update后的per-h `Delta logprob` | 未记录 | 未记录 |

## 与LLM entropy的关系

可借用的统计结构是：raw position-level uncertainty、固定位置趋势、sample-wide shift、local anomaly。
但Shannon entropy来自categorical next-token distribution；DVAC来自连续flow去噪过程中endpoint estimate的
revision variance。二者数值、单位和生成过程都不同。因此`P_h/S_q/I_qh`可以借鉴entropy工作的校准与
对照方法，不能把DVAC数值直接称为token entropy。
