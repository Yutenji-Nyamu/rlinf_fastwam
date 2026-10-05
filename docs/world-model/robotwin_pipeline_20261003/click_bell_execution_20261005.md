# click_bell WMRL 执行记录 · 2026-10-05

## 当前状态

22:31快照：Rynn32已完成，按铃进入`native_bell32`原生采集。Rynn把15条真实成功和17条真实失败全部判为No，不接GRPO奖励；数值头分数重叠，本次不拟阈值。详见[Rynn结论与轻量证据](rynn_binary_probe_20261005.md)。按铃专用RM、smoke与正式首轮尚未完成。

修复后的v2入口于21:54启动。v1原生标签入口因namespace未同步失败，未产生有效样本；其原finally已清理并派发RLT，归还无错误。下文保留已冻结的执行安排。

用户授权顺序：小批 Rynn 真实成败测试 → 整理结论 → 按铃专用奖励模型 → 同并行短 smoke → 正式训练。WM 优先，RLT 原断点排在后面。仅管理本人在物理 4–7 卡上的明确进程；0–3 只检查本人有无越界，不清理其他用户。

## 保留与更换

沿用 π0.5、OpenDW 三视角/14D、RLinf GRPO、两路 B16 WM 服务、卸载等待、资源记录及 RLT 借还流程。任务从 adjust_bottle 改为 click_bell，重新使用原始 SFT，不继承摆瓶子 CP70。CP70 仅用于前置 Rynn 原生标签采集。

奖励使用 WorldArena 的 `click_bell/resnet_rm.pth`，不使用 LPIPS。每个 C32 生成 8 张主视角图，任一分数 ≥0.9 时在动作块末端给一次 1 并结束轨迹，否则为 0；关闭分数差分。原 adjust_bottle 配置/源码保持。

| 项目 | 冻结配置 |
|---|---|
| GPU | 3机 4–7；actor/rollout 两 rank，WM 在6/7 |
| 正式采样 | N64/G8/R8；512条/轮；C32，最多384动作 |
| WM batch | 两服务各B16 |
| 更新 | global2048、micro8、2 update epochs |
| 预算 | 200 runner轮；每10轮保存和原生32回合评估 |
| 工程 smoke | 相同N64/G8/B16与卡位，R1、C32、1轮；global64 |
| 原生标签小测 | 每任务N16×2=32回合，C32/384；仅GPU4 |

smoke 仅减少串行工作量，不将单块零梯度误称完整任务无法学习。正式从原始 SFT 重新开始，工程 smoke 权重不续接。

## 已完成的准备

- 按铃权重588,338,878 bytes，SHA256 `5c4141b599b35f6f46064e11410e376668659e669fdda6e938f497bf7d34d464`；CPU strict load通过，122,382,017参数。
- 深圳1现有50条clean示范全部提取同时刻三图及14D起点，无重新筛选。约3MB reset包跨机校验通过；大轨迹文件保持原处。
- 原生评估沿用已有click_bell种子列表前32项，没有新筛种子或专家规划器准备。
- 服务器CPU：binary6、bell8、prechecks6通过；实际正式计划校验通过。
- 前置测试顺序固定为 `native_rynn32 → rynn32 → native_bell32 → bell_rm`。Rynn语义判别差只记录，不阻塞独立的按铃路线。
- 按铃固定0.9阈值先核原生成败。需要正负两类，且不是常数或反向判断；不进行阈值拟合。此小测不能证明对OpenDW生成图可靠。

## 执行位置

服务器根：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003`。

- 准备及入口：`click-bell-v2/prepared/owner-plan.json`、`click-bell-v2/generated/owner/opendw_formal_owner.py`。
- 唯一执行输出：`runs/click-bell-v2`；owner PID1443064/start717950867。旧v1 owner3119746已完成退出。
- 原生标签和Rynn结果分别在 `native_rynn32`、`rynn32`；按铃在 `native_bell32`、`bell_rm`。
- 新训练checkout：`rlinf-opendw-bell-v1`；原运行checkout未修改。
- RLT借还：`click-bell-v2/prepared-cycles/cycle`；沿原链保留4/5/6/7完整CP225/250/150/150。成功、失败均由本owner精确清理自己的进程并续接；不要重放旧owner。

未新增常驻守护层，使用原owner的顺序执行与finally归还。源文件、轻量结果和本记录纳入独立Git发布；权重、完整日志、视频及checkpoint不放Git。

## 结果

v1启动问题已定位：只设置Ray namespace，未同步RLinf的`Cluster.NAMESPACE`及worker环境`CLUSTER_NAMESPACE`，辅助actor查找管理器失败后重复初始化Ray。v2补齐两者，并删除官方WorkerGroup不存在的`stop()`调用，收尾交回原owner。没有重启共享Ray、改算法或扩大试验；v1失败/清理/归还回执全部保留。

Rynn32于22:30:36完成：工程通过，语言Success全No（TP0/FN15/FP0/TN17，unknown0），数值AUC0.702且成功/失败范围重叠。因此本次Rynn到此留档，按授权继续按铃路线。按铃32回合当时仍在采集，待其专用RM、工程smoke与正式启动结果补入；当前不能称已验证按铃奖励准确或已开始正式学习。
