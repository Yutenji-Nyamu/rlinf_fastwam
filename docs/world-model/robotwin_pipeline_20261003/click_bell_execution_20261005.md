# click_bell WMRL 执行记录 · 2026-10-05

## 当前状态

23:14快照：Rynn32及按铃原生/RM小测已完成；工程smoke退出0、保存CP1并完成精确清理。23:07启动的正式训练现已实际采样，两路WM均完成第二C32块中的真B16批。**目前是正式首轮采样中，尚无完整正式更新；smoke本身没有有效学习信号。**

Rynn把15条真实成功和17条真实失败全部判为No，不接GRPO奖励；数值头分数重叠，本次不拟阈值。详见[Rynn结论与轻量证据](rynn_binary_probe_20261005.md)。按铃原SFT在固定32回合中32/32成功；专用RM在48张有标签画面上TP14/FN2/FP3/TN29，另16张历史成功辅助图不计准确率。

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

Rynn32于22:30:36完成：工程通过，语言Success全No（TP0/FN15/FP0/TN17，unknown0），数值AUC0.702且成功/失败范围重叠。因此本次Rynn到此留档，按授权继续按铃路线。

### 按铃原生基线与RM

原始SFT在这批固定原生32回合全部成功（N16×2，C32，最多384动作），22:45及22:48两次逐回合标签快照一致。**这组小样本已满分，后续同一组评估没有可测的提升空间，应先看能否保持、是否退化；这不代表其他种子或更大评估也达到100%。**

RM按固定样本顺序取64图：16张首次观测到成功、32张成功前最后一次false、16张成功后的辅助图。没有按RM分数挑样本，也没有拟合阈值。部署阈值仍为0.9：

| 画面类别 | 本次结果 |
|---|---:|
| 首次成功里程碑 | 14/16判成功，2/16未检出 |
| 成功前false画面 | 29/32判未成功，3/32提前判成功 |
| 历史成功辅助图 | 16张，标签-1，不计主混淆矩阵 |
| 平衡准确率 | 89.06%，仅为本批原生画面指标 |

这32张阴性图来自最终成功回合的成功之前，**不是32条失败回合**；当前没有按铃最终失败轨迹覆盖。`near_false`只表示时间上最近的false，不保证画面中机械臂确实紧贴铃铛。

CPU strict load通过；本次GPU4真B16处理64图，加载45.51秒，纯推理1.59秒，峰值allocated0.596GiB、reserved0.697GiB。完整RM预检含进程/汇总开销67.40秒。该资源结果仅属单独RM推理，不包含WM或策略训练。

原生图每C32保存一次，“首次成功图”可能已在接触发生后，未必拍到接触瞬间。因此两次未检出应描述为未识别这批成功里程碑画面，不直接断言模型漏掉接触本身。WM则每C32评分8张C4图，任一分数达阈值即给一次1、在块末结束；判定机会更多，逐帧小测的误判率不能直接当作整条轨迹的假成功率。OpenDW生成画面上的可靠性仍未验证。

本次固定0.9门槛通过的是原生画面基本区分检查，足以继续已配置工程smoke；不代表已证明真实环境收益。轻量结果含64条分数/类别及稳定回执哈希：[按铃RM结果](../publication_bell_20261005/bell_reward_result.json)。

### 执行与发布边界

工程smoke完整耗时1070.08秒（约17.8分钟），退出0、清理回执`all_stopped=true`。CP1全权重及rank0/1两份本地shard均已落盘。actor/rollout在4/5卡，env/WM在6/7卡；没有扩大卡位。

| smoke检查 | 结果及边界 |
|---|---|
| WM采样 | 64条，各执行一个C32块；该采样段约41.93秒 |
| WM并行 | 两服务各32条，内部真B16；denoiser每次batch均16，输出finite |
| WM资源 | 末B16分别3.32/3.28秒；单服务生成峰值allocated45.44GiB、reserved53.35GiB |
| 卸载 | 两服务`residual_model_cuda_tensors=0`；仍各有约33MiB分配，不能称进程显存完全为零 |
| 学习信号 | 64条WM回报全0；grad_norm、policy_loss、total_loss均0，没有有效成败学习信号 |
| 统计异常 | rollout advantages/rewards显示nan；全0组被原filter屏蔽后统计集合为空，不能称非零学习验收通过 |
| 短原生评估 | 32条、每条同样最多32动作，全部未成功；不能与384动作的32/32基线直接比较 |

这轮smoke证明并行生成、策略前反向调用、保存与卸载清理链路能够完成，不要求单个C32块完成整个按铃任务。正式从原始SFT开始，不续接smoke权重，恢复完整384动作、每轮512条采样。

正式driver于23:07:08启动：PID528981/start718385161，namespace `opendw_sz3_bell_formal_v2`；正式卡位回执同样为4–7。23:14:46/47，两路WM分别完成`seed0-call1`/`seed1-call1`的B16批，denoiser batch均16、输出finite；这是正式启动之后第二个C32动作块的实际生成证据，不是旧smoke日志。driver当前显示`Generating Rollout Epochs 0/8`，表示首个采样批尚未全部结束，不表示没有执行采样。

当前可以确认正式首轮在采样；不能称已采完本轮512条、已完成有效更新或已有真实环境收益。轻量JSON只保存必要日志摘录及本次取回快照哈希，持续变化的driver/service日志没有加入稳定文件pins。

源码、文档和Rynn结论的首发60文件已推`codex/wmrl-bell-reward-20261005`，远端核验提交`ff3ec40307950f4c22975cb1d68dffdd7bcb1dbf`，运行checkout保持不变。按铃结果为后续四文件补充，当前仍待该补充发布回执；模型、视频和完整日志不在Git。
