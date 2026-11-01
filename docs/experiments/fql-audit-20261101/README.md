# FQL-fix谷底审阅 · 2026-11-01

数据来自已完成FQL-fix：27个128回合点、108批Q记录、405条真实转移及训练指标。图中训练MSE/LR来自每个保存点末批，Q/回报来自对应实际执行。相同实际评估种子/噪声/协议，训练评估种子无重叠。

最重要发现：160轮15.6%→400轮88.3%，同时Q继续高估至12.27、实际回报0.84，不能说Q已学准。回升与动作偏移减小、学生步长减小相伴，因果待对照。保留全部原始聚合点，不平滑。

[四联图PNG](diagnostics.png) · [SVG](diagnostics.svg) · [PDF](diagnostics.pdf) · [CSV](points.csv) · [数据JSON](analysis.json) · [完整审阅](../../small_flow_distillation/36_FQL_FIX_VALLEY_AUDIT_20261101.md) · [GAD耗时](../../gad/12_SIGNAL_COST_20261101.md)

现场11月01日 12:08：45为GAD Direct180轮评估，上一完整160轮105/128；6卡最后stamp_seal，7卡临时RLT，之后双卡CP v2＋GAD→DMD new＋GAD。此处仅记录快照，不改变队列。

本轮无GPU实验、无参数修改、无删除；公开聚合成绩和图稿，不含模型、原始图像/视频、凭据或其他用户快照。网页http://127.0.0.1:8194/fql1101a/。
