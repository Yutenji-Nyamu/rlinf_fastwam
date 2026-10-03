# 2026-10-03：本轮改为讨论，暂不实施

**历史暂停阶段（2026-10-04补注）：**本页保留2026-10-03中途“先讨论、暂不实施”的当时指令及停止回执。用户随后重新授权实现、资源smoke及条件具备后的训练，执行已转入四卡N64/G8/R8 smoke；以下“最新”“当前边界”“未发布”均只描述本页形成时刻，不构成新的暂停指令或现行状态。当前路由见[四卡记录](multigpu_smoke_20261003.md)和[云端发布回执](cloud_publication_20261003.md)。本机原始回执未随本页公开。

## 最新用户边界

用户先授权几次资源smoke自主试串并行，随后强调RLT低优先级，最后明确：**“这轮还是讨论，还先不实现，继续吧。”**最后一句是当前边界：停止实施准备，不借卡、不启动GPU，不因旧授权自行恢复。继续维护研究结论与方案。

## 截止时已经做了什么

- 接线草稿：OpenDW环境、三相机/动作/state适配、HTTP模型服务、任务RM、配置生成器、卸载等待补丁和借还草稿。保留在`local_patches/opendw_smoke_20261003`及`tools/opendw_smoke_gpu4_cycle.py`，没有称GPU验收完成。
- SZ3独立准备目录：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003`。新RLinf worktree从`2151a08e`建立；独立venv继承现有只读包路径，新增依赖装在新venv。未修改正式RLT源码和运行环境。
- 服务器CPU检查：13项通过，覆盖适配、reset/快照/独立随机数、NPZ/HTTP合同。两次测试失败是NumPy断言直接比较Tensor，改成numpy数组后通过；不是GPU模型验证。
- reset：已有adjust_bottle clean50每条episode第0帧，同步头/左腕/右腕、14D控制状态及同episode指令；50条全部保留，没有成功筛选。原JPEG不含新格式`XPL-RGB1`标记，按原converter直接cv2解码。
- N8/N16、G8、C32/H50、一runner轮候选配置已生成。GPU4借还计划仅prepare，核到完整CP100；**stop未调用，借卡未发生**。
- OpenDW公开权重下载已被本次停止指令中止，部分文件仍未完成；T5下载进程此前已经结束。保留文件供日后明确恢复时复用，不删除、不自动续传。
- 最终停止核验：下载worker PID1331979/start699924328及其唯一子进程1333681/start699927499已退出；深圳3RLT四driver4170403/3476963/3476983/3477005身份仍匹配。没有WM GPU任务、没有RLT启停、没有共享Ray变更。

服务器唯一停止回执：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/discussion-only-receipt.json`。

本地回执：opendw-discussion-only-stop.out（本机留存、未发布：`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003/opendw-discussion-only-stop.out`）。CPU结果：opendw-deploy-cpu3.out（本机留存、未发布：`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003/opendw-deploy-cpu3.out`）。

## 讨论结论：哪些清楚，哪些待实验

| 项目 | 清楚的方案 | 仍需回答 |
|---|---|---|
| 图像/动作 | 三图顺序固定；π0.5反归一化为14D绝对命令，OpenDW再用自身统计归一化 | 生成图在当前策略动作下是否准确 |
| state | 当前policy读控制目标；next-state用所消费C32的末条控制命令，夹爪裁剪[0,1] | WM是否充分模拟真实控制器插值、失败和接触造成的视觉变化 |
| 时间 | H50不改，执行/训练logprob取前32；32动作对应8张未来图，返回末图 | 正式400动作尾16如何严格计数，不静默跑416 |
| 奖励 | 摆瓶子先复用WorldArena连续成功score；0.9作模型成功阈值 | OpenDW图上的假成功率；阈值是否合适；原生成功是否改善 |
| 并行 | G8同起点，环境独立历史，WM服务可逐样本排队 | 真正显存/吞吐；N增加是否只是队列增长；哪些阶段值得增加卡数 |
| 方法 | 首先是冻结OpenDW+RM的π0.5-GRPO | PACE需补采真实轨迹和更新WM；MBPO另需value/GAE/bootstrap/PPO |

state的直观例子：控制目标0.7弧度，碰到物体实测只有0.65。当前RoboTwin给策略的字段仍是控制目标0.7；因此缺next-state可以先按末命令补0.7。图像是否正确反映接触后实际姿态，仍由WM质量决定。不能将命令状态近似解释为模型已经精确预测实际机器人状态。

并行区分：N是逻辑轨迹数，WM batch是一次GPU推理的样本数，actor microbatch是一次反向传播的样本数。公开入口先按B1处理，则N8→N16主要增加排队时间；不能期待吞吐翻倍，也不能把旧Wan micro64直接搬给当前组件。

## Git边界

本轮实时`git ls-remote`核旧LIBERO WMRL分支`codex/sz3-wan-goal-20260930`远端为`21b5d590a9116d94e38c139a3ec4aa9723abbcc3`，包含既有修复、配置、监控、评估结果和轻量文档。大检查点及原始日志不在Git，也没有独立云备份回执。本次RoboTwin/OpenDW草稿和新讨论文档未发布，当前不推进发布/训练。

后续先从[组合上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)、[state与C32合同](c32_adapter_decisions.md)、[奖励训练与候选](reward_training_and_candidates.md)和[方法地图](method_map.md)继续讨论。所有owner/借还脚本均是待复核草稿，禁止直接重放已prepare的初次入口。
