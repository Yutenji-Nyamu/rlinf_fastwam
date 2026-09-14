# 深圳 RLT / GRPO 曲线与共同窗口结论

数据时间：2026-09-14 15:41:20 CST，使用停止旧 GRPO 后的 [poststop.json](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/poststop.json)。本图册只包含既有 RLT 五组与 GRPO128 三组，不包含新 smoke / top20 组。本专题仅整理已有快照与[机制诊断](C:/Users/86136/Documents/rl/docs/server-admin/RLT_GRPO_DIAGNOSIS_20260914.md)，没有追加服务器分析或训练。

## 图册与数据

[打开完整 HTML 图册](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/index.html) · [下载独立图册 ZIP](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/shenzhen_rlt_grpo_comparison_20260914_154120.zip)

每项均为独立宽图；图册提供对应 PDF。每组保留完整真实历史和终点，不截到共同100轮，也不补齐不同终点。MA 为包含当前轮的向后连续窗口；采集与 fixed 分开。RLT 每次 fixed 为20个状态，GRPO为32个状态。

| 图型 | RLT：Clean / Pure04 / Scale1 / Scale2 / 反转 | GRPO：Clean128 / Linear128 / Exp128 |
|---|---|---|
| 逐轮采集 | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/rlt_round_success.png) | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/grpo_round_success.png) |
| MA50 | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/rlt_ma50.png) | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/grpo_ma50.png) |
| MA10 | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/rlt_ma10.png) | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/grpo_ma10.png) |
| MA20 | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/rlt_ma20.png) | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/grpo_ma20.png) |
| fixed | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/rlt_fixed.png) | [PNG](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/grpo_fixed.png) |

补充：[Linear − Clean 的共同轮次 MA20 差值图](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/grpo_linear_minus_clean_ma20.png)。

CSV：[原始成功率](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/raw_success_series.csv) · [全部绘图曲线](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/plotted_curves.csv) · [Linear / Clean 阶段均值](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/grpo_linear_clean_phase_means.csv) · [共同窗口指标](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/matched_window_metrics.csv)。[Manifest](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots/plot_manifest.json)记录选组、端点与统计口径。

## GRPO：线性整体相当，指数下降更明确

三组采集共同至R109；fixed共同至R105。以下均为窗口内逐轮标量的算术平均，窗口含首尾；fixed不补齐非评估轮次。

| 指标与窗口 | Clean128 | Linear128 | Exp128 |
|---|---:|---:|---:|
| 采集，R1–109，109点 | 50.44% | 50.57% | 46.45% |
| fixed，R5–105，每5轮一次，21点 | 49.55% | 51.79% | 44.20% |
| 后段采集，R61–105，45点 | 54.69% | 52.97% | 49.70% |
| 后段fixed，R65–105，每5轮一次，9点 | 55.56% | 55.21% | 48.26% |

Clean与Linear单独共同R1–116的采集均值为50.889% / 50.950%，仅差 **+0.061个百分点**。R61–105 Linear采集略低、fixed近乎相同。因此，图上看不出Linear稳定更好与统计一致；累计小正差不能解释为持续优势。

下表训练行为统一使用 **R61–105的45轮**；权重指标是日志中每轮批次聚合量的均值。

| 后段训练指标 | Clean128 | Linear128 | Exp128 |
|---|---:|---:|---:|
| PPO clip比例 | 5.163% | 5.010% | 5.027% |
| approx KL | 0.01651 | 0.01604 | 0.01620 |
| clip前grad norm | 27.98 | 30.26 | 38.85 |
| W的ESS比例 | 未记录 | 90.02% | 63.20% |
| 正优势总系数倍率 | 未记录 | 0.9777 | 0.9099 |
| 负优势总系数倍率 | 未记录 | 1.0206 | 1.1013 |
| 有效chunk数 | 未记录 | 409.09 | 408.87 |

指数版更集中权重、减少正优势侧份额并增加负优势侧份额，但后段clip和KL相近；不能把下降简单解释为普遍PPO越界。ESS描述权重集中度，不是独立有效数据量；clip前梯度范数也不等于Adam后的参数位移。每组只有一次训练，fixed重复使用同一批状态，以上是描述性比较。

## RLT：反转前已有差距，fixed最新已到100%

反转组最新采集为 **R411=87.5%**；最新fixed为 **R400=20/20，即100%**。这与其采集爬升较慢同时成立，不能称它在所有口径上始终最差。

方向tag显示R188为正反混合轮，R189起整轮反向。反转前 **R160–187的28轮采集均值** 已为：Clean40.63%、Pure04 23.21%、Scale1 27.23%、Scale2 30.80%、反转组23.66%。组间差异在切点前就存在，各组也不是同一个随机前缀在切点分叉的配对实验，不能把整条差距都归于flip。

原诊断中的外层候选机制仍应按其证据范围理解：**低V表示teacher同一去噪链最后三次终点预测较一致，不表示预测正确，也不表示reference更接近成功动作。** 外层反转增加低V成功query的整体模仿份额，可能把student拉向与成功执行动作偏离的teacher reference。

CP350小回放里，反转组有 **152条成功执行动作与reference不同的历史记录**。只在这152条已保存记录上改变权重，两层正向→反向使加权“执行动作−reference”MSE相对增加 **8.54%**（相对均权MSE的比值0.9628→1.0450）。这是历史动作差的代理量；不是当前student残差、当前训练梯度，也不是成功率变化。它支持进一步检查外层，尚不能量化flip造成多少训练劣化。详见[已有机制诊断](C:/Users/86136/Documents/rl/docs/server-admin/RLT_GRPO_DIAGNOSIS_20260914.md)。

## 交付回执

ZIP为2,885,917字节，共28文件：11 PNG、11 PDF、HTML、4 CSV、manifest。SHA-256：`f807cab4cb19658a4711f9f62157fce8b3d5a530246c5ebead9e3e3d8ea15cd8`。[完整回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/plots_zip_receipt.json)
