# A2World 接入 RoboTwin / RLinf：公开实现核验

**研究快照（2026-10-04补注）：**公开代码、论文和权重的核查截止2026-10-03；本文“未运行”“尚缺闭环”等叙述限定于该次调查，不作为当前实验进度。后续组合和执行路由见[OpenDW主上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)，本次仅补充归档，没有重新检索或启动实验。

核验日期：2026-10-03。仅检索论文、官方源码与资产元数据；未下载模型、安装环境或运行训练。

源码固定到 [`077e10ad6cee07342b5e779f11fea78247584834`](https://github.com/LogosRoboticsGroup/A2World/tree/077e10ad6cee07342b5e779f11fea78247584834)（2026-08-30）。HF 固定到 [`411846321a75f1b64af2ab79af3db939c85d6a72`](https://huggingface.co/Fleurrr/A2World-World-Model/tree/411846321a75f1b64af2ab79af3db939c85d6a72)。

## 结论

A2World 不是仅做 LIBERO：有多机器人预训练与双臂实机研究。但**公开可下载的任务适配 WM 只有 LIBERO；未核到 RoboTwin 配置、转换器、专项权重或实验结果**。能接收双/三相机不等于已经能模拟 RoboTwin；这是值得适配的多视角 WM 起点，还不是替换当前 Wan 的现成插件。

此外，**公开 `a2world-libero.pt` 是否完整覆盖 Spatial、Object、Goal、Long 四组，资料没有明确交代**。model card、manifest、WM 配置仅标 LIBERO；论文四组 98.6% 是独立的 A2World-policy，不能当作这份 WM 权重的覆盖清单或 π0.5 RL 收益。仓库 `world_action_model/shell/train_libero_4suites.sh` 也属于策略训练。当前不能保证它直接支持我们全部 LIBERO Goal 任务。

## 公开资产与证据边界

| 项目 | 已核实 | 没有被证明 |
|---|---|---|
| WM 权重 | `a2world-pretrained.pt` 5,054,966,852 bytes；`a2world-libero.pt` 5,029,465,374 bytes；EMA / BF16 | RoboTwin 专用权重、发布 LIBERO 权重逐任务覆盖 |
| 推理 | 多视角动作条件推理、history-aware 自回归 CLI；有输入图片与动作样例 | RLinf Gym 环境、与现有 π0.5/RoboTwin 已跑通 |
| 训练 | LIBERO HDF5 转换、全参数微调、FSDP 配置与 launcher | RoboTwin 数据配方、低显存 LoRA 成功记录 |
| 数据 | 原始 LIBERO 转换工具、少量本地 demo；发布模型目录没有训练集 | 论文全部清洗后的多机器人数据、RoboTwin WM 训练集 |
| RL | WM 可以接受外部策略动作并生成图像 | 发布完整 reward / termination / GRPO / PACE 闭环 |

权重列表来源：[HF 模型页](https://huggingface.co/Fleurrr/A2World-World-Model)、[源码 manifest L7–24](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/release/weights.json#L7-L24)。训练与推理入口：[WM README](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/README.md#L99-L136)。HF 作者名下按 A2World 检索 datasets 返回空，仅支持“未核到该发布数据集”，不能推断所有原始数据均未公开。

## 接口契约：哪里相同，哪里不同

| 项目 | 发布实现 | 对 RoboTwin 的含义 |
|---|---|---|
| 机器人 / 动作 | 14D = 两臂各 `[Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper]`；单臂后 7D 补零。通用 pose converter 使用前一帧末端坐标系中的相对位姿 | 若 π0.5 输出 14D 关节位置，维数虽相同，语义不同。必须定义关节目标到末端条件的映射，或用 RoboTwin 原动作重新训练动作条件模块 |
| LIBERO 特例 | 第一只臂的六维动作乘 0.05；夹爪 `(1-a)/2`；servo 格式与预训练 pose converter 分开 | 不能照搬 LIBERO 的比例、夹爪符号到 RoboTwin |
| 相机 | LIBERO 主相机 ID=2、腕相机 ID=0；预训练例有头部+双腕。4 个视角 embedding，联合多视角生成 | 可复用多视角架构，但需对齐 head/left/right、分辨率、方向与时间戳，再做目标域适配 |
| 时序 | 默认 256×256；20 动作→21 帧（含起始帧）；6 latent frames / view；35 步采样；发布例按 10 fps 描述 | C8 与现有控制时钟不是同一契约；帧率 metadata 不能代替动作持续时间对齐 |
| state 参数 | LIBERO rollout 中由已执行动作历史与待执行动作 chunk 计算 22D 路径统计 | 不需要查询真实未来 pose；但它也不是生成的关节状态/末端状态 |
| 输出 | 各相机生成视频，并追加生成图像与动作历史 | 没有同步输出 RoboTwin qpos / 物体状态 / 成功判据；π0.5 所需 state 仍要单独闭合 |

动作来源：[数据契约 L18–40](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/docs/DATA.md#L18-L40)、[pose converter L61–94](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/a2world/actions.py#L61-L94)。相机与时序：[rollout 常量/参数](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/a2world/rollout.py#L28-L65)、[输入/输出链](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/a2world/rollout.py#L161-L212)。

**history 还有一个双臂细节：**抽取历史帧的距离统计覆盖两只臂；但公开 LIBERO 的 `action_path_features` 只用第一臂六维动作及夹爪生成 22D 特征。不能因外层接受 14D 就称其双臂 history 全部已适配。需要扩展双臂特征并微调对应模块，或明确采用另一条已验证的历史方案。[history.py L13–17、L32–64](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/a2world/history.py#L13-L64)

**未来 pose 是否是硬依赖：不是。**公开 demo 用初始图重复填充历史、零动作初始化；随后 `state = action_path_features(history_actions, action_chunk, indices)`，用生成图与已输入动作更新历史。训练转换器虽存绝对 pose，这不意味着公共推理需要真实未来 pose；也不能把这一 action-derived state 误当真实机器人 proprioception。[rollout.py L164–181、L209–212](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/a2world/rollout.py#L164-L212)

## C20 改 C8：padding 是候选适配，尚不是等价替换

当前动作张量 `[B,20,14]` 整体展平成 280D，经 MLP 后把同一个动作条件加到**所有预测时刻**。checkpoint checker 固定校验首层输入 280D。因此直接把 T 改 8 会与旧权重 shape 不符。[整体动作嵌入 L1103–1107、L1147–1152](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/cosmos_predict2/models/video2world_multiview_action_state_pred_dit.py#L1103-L1152)、[权重形状校验](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/a2world/checkpoints.py#L13-L17)

有两条可以讨论的工程路线：

1. **保留 C20 模型，以 8 个真实动作加 12 个 padding 生成，再取第 8 步。**shape 可兼容，但从整体动作条件可推知 padding 可能影响前 8 帧，不能声称因果独立或已验证；全零也不保证保持当前夹爪状态，应定义“保持”的语义。需改公共 CLI 的完整 20 步检查，只用前 8 帧及动作更新历史。计算量仍主要是 20 步生成，并未获得 C8 的等比例提速。
2. **真正训练 C8 版本。**改 280→112 的动作输入，以及 21→9 帧、6→3 latent / view 等相关时序配置与数据切片；复用主干，重新适配动作条件层与相关时间表示，再用 RoboTwin 数据微调。不能仅改一个 YAML 常数，也不能承诺少量训练即可达到原 C20 质量。

若策略能提供 20 个计划动作，也可一次条件完整计划、只执行前 8 步再规划；但未来计划同样参与联合预测，这是另一套需要确认的控制协议。本文不改变当前实验 C8 或 π0.5 horizon。

## 真正替换 RLinf Wan 还缺什么

1. **RoboTwin 数据与动作对齐：**采集/导出同步相机、实际执行动作、机器人 state、成败及时间信息；固定机器人和相机布局。当前 KIR reset 图片不够训练动态模型。
2. **目标域微调：**以通用预训练 WM 为起点，增加 RoboTwin loader；决定关节还是末端动作、双臂 history、C8/C20。LIBERO 专项权重不是已验证的 RoboTwin 初始化最优选择。
3. **RLinf backend 与状态闭环：**实现 reset/KIR、vectorized step、双/三相机 observation、历史缓存、动作归一化、资源装卸；为策略 state 建立与生成图一致的更新办法。
4. **奖励与终止：**RoboTwin 成功检测器通常依赖模拟器内部状态；仅有生成 RGB 不能直接调用同一真值判据。需任务适配的视觉 reward / done 或其他已验证机制，不能默认当前 LIBERO reward 能迁移。
5. **真实环境校验：**先验证动作响应与多视角一致性、native RoboTwin 基线与 WM rollout 误差，再讨论策略 RL；多视角生成质量本身不证明策略会提升。

论文基于 Cosmos-Predict2-2B，增加模块后的 WM 报 2.5B；微调示例使用 8 张 H200。论文展示 π0.5 在自训实机 WM 内闭环评估，成功由人工核验；不是现成自动 RL 系统。默认推理 CLI 可以配置 1 GPU，但无本次实测显存/吞吐，不能把权重约 5 GB 当运行显存，也不能据此承诺共享机器配置。[论文 §4.1、§4.3、补充 §7.2](https://arxiv.org/html/2606.29501v1)

**建议定位：**A2World 是有真实多视角结构、权重和微调入口的可行研究起点；RoboTwin + π0.5 还需要目标域数据、动作/state 契约与 RL 环境闭环。当前最准确的状态是“可开展适配研究”，不是“换模型路径即可开训”。
