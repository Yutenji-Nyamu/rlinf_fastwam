# RoboTwin与LIBERO多视角WM-RL：选型与接入梳理

**研究快照（2026-10-04补注）：**公开代码、论文和权重的核查截止2026-10-03；本文“未运行”“尚缺闭环”等叙述限定于该次调查，不作为当前实验进度。后续组合和执行路由见[OpenDW主上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)，本次仅补充归档，没有重新检索或启动实验。

2026-10-03。范围：回顾已有工程、实时核查官方论文/代码/权重、讨论替换方案；没有启动训练或修改实验参数。

**同日后续纠正与讨论主入口：**[三者组合上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)。深圳2实码核到policy观察的14D state来自drive targets与夹爪命令；缺实测next proprio不自动意味着必须新训状态预测器。具体执行目标与图像时间仍须对齐。

**现有RLinf＋RoboTwin＋π0.5＋GRPO训练底座可以复用。当前更贴近三相机RoboTwin的是OpenDW的Robotwin模型，再补RL环境、任务奖励、时间和状态接口。A2World/VLA-MBPO有价值，但公开配套主要面向LIBERO，不能视为RoboTwin现成闭环。**

## 先找回之前的路线

1. **OpenDW**：已有Robotwin三视角、14D绝对关节动作条件预测，目标是保持π0.5/GRPO，替换环境动态。难点在C32时间、下一state、奖励和并行。
2. **WorldArena 2.0**：后来考虑先学其RoboTwin＋π0.5＋GRPO单任务闭环，再接回我们的工程。代码框架已具备，但对应14D Wan/依赖/reset模型包仍未配齐，且默认单头图、零state。
3. **当前LIBERO Wan**：使用已配套发布的LIBERO WM/reset/reward，再接π0.5；冻结WM阶段已真实运行。这不等于完成RoboTwin多视角版本，也不等于完成PACE。

旧专题：[OpenDW](../OPENDW_PI05_GRPO_CONTEXT.md)、[WorldArena](../WORLDARENA_FIRST_REPRODUCTION_PLAN.md)。动态执行状态应另看现场，本页不声明GPU空闲或正在训练。

## 四个候选解决不同问题

| 候选 | 实际可复用物 | 最重要缺口 | 当前适合的角色 |
|---|---|---|---|
| OpenDW DW05-Robotwin | Robotwin配套权重、三视角、外部14D绝对关节动作→未来视频、WM训练代码 | 控制目标state与图像对齐、任务reward/done、C32与C50时序、batch=1推理 | 三相机RoboTwin WM的优先工程候选 |
| WorldArena 2.0 | RLinf衍生runner、π0.5/GRPO、RoboTwin任务reward、Wan env/HTTP | 匹配14D Wan权重和依赖/reset bundle；仍单头图/零state | 学习环境封装，借对应任务奖励 |
| A2World | 多视角WM预训练和LIBERO权重、转换/训练/自回归推理 | C20→C8、动作/历史协议、RLinf env与reward；未核到RoboTwin配套 | LIBERO双图动态模型候选 |
| VLA-MBPO | π0.5＋双图Bagel WM＋短分支RL代码、Goal早期示例权重 | 发布样例任务/资源范围、C10→C8、末端双图输出、RLinf接入；RoboTwin组合未完成 | LIBERO完整方法参考 |

“支持相机”须指能够在**同一段外部动作下共同预测多个未来视角**，不只是策略能读取多张图，也不只是WAM会联合生成视频与自己的动作。OpenDW当前确有外部动作条件入口，所以不需要替换π0.5为DW策略。

## A2World与VLA-MBPO是否都只做LIBERO

**都不能概括成只研究LIBERO，但目前公开可照着运行的仿真配套主要是LIBERO。**

- A2World预训练跨多种真实机器人，论文有模拟器和策略两个分支；公开WM权重是通用预训练＋LIBERO微调。未核到RoboTwin专用配置/转换器/权重。LIBERO WM权重究竟覆盖哪几组任务，模型卡没有明确到suite；论文A2World-policy四组成绩不证明WM权重四组覆盖。
- VLA-MBPO论文仿真是LIBERO四组，另有真机；真机双臂不能当RoboTwin仿真证据。**纠正：README虽提`train_robotwin.sh`，但本轮官方与作者Git树均无该文件。** 公开Goal例程实际默认`libero_goal_task_3`、早期step2500 WM，不能据此宣称十任务或论文最优权重已完整发布。
- 新VLARLKit确实同时列RoboTwin和VLA-MBPO，但RoboTwin配置需要离散state输入，branch rollout当前不支持该条件；RoboTwin `reset_to_states`仍未实现。两项分别支持，不代表组合已支持。

精确论文、发布revision和代码锚点见[A2World审计](a2world.md)、[VLA-MBPO审计](vla_mbpo.md)。

## 替换当前LIBERO Wan的实际差异

| 项目 | A2World | VLA-MBPO |
|---|---|---|
| 输出与时间 | 默认20动作→21帧，多视角视频 | 当前训练/服务配套C10，输出块末头图＋单腕图 |
| C8要求 | 动作整体展平为20×14输入，补零/补末动作后取第8帧不等价已验证C8 | 可变文本动作长度不等价已有C8权重；仅块末图无法直接取第8帧 |
| 动作 | LIBERO7D经尺度/夹爪处理放入14D表示；不可直接传策略内部归一化tensor | LIBERO有效7D，有其自己的action normalizer和文本编码 |
| reward | 需外接；可将head裁切接当前RM，须核预处理/生成域 | 已有末端head的Yes/No任务评分；采用它会同时更换reward语义 |
| state | 动作历史派生特征不是预测真实关节state | WM env沿用current state，不预测next state |
| RL框架 | 新backend和多视角env可保留现有GRPO | 只借WM可以保留GRPO；完整方法还要搬短分支/value/PPO等 |

两条路线都需要双相机reset/历史数据。旧单图reset不能凭空补出同期腕图；能从完整模拟器状态恢复时可重渲染，否则重新采集。当前LIBERO π0.5实现并不使用真实proprio条件，因此state缺口对RoboTwin原合同更直接，不能一概当作两个域同样的阻塞。

## RoboTwin具体需要补什么

可复用：π0.5权重与normalizer、Flow-SDE/GRPO、actor/rollout同步、保存恢复、RoboTwin原生评估、已有数据采集工程。

新增或适配：

1. **动态模型backend/service。** 载入OpenDW Robotwin，接受π0.5反归一化后的14D关节动作。A2World即使也是14D，表示末端位姿增量，不能直接接同一tensor。
2. **环境观察。** 联合生成头＋左右腕，再分别还原π0.5相机键/尺寸/布局；保留各环境自己的历史与随机数。
3. **下一state。** OpenDW没有实测next proprio输出；但本轮实核RoboTwin的policy state是drive targets＋夹爪命令，应先按实际执行的目标重建，核插值、部分执行及图像时刻。此前将其直接等同于缺实测qpos的阻塞，现予纠正；不能静默改成全零。
4. **任务奖励和结束。** 生成图像不能直接运行依赖物体位置/接触的RoboTwin原生成功判定。可复用WorldArena对应任务reward，或训练/引入另一个视觉奖励；成功、超时和final observation都要明确定义。
5. **时间和批处理。** 默认32动作→9帧，与我们历史C50不一致。把50动作补到64后拿末帧，会取到错误时间。还要处理公开batch=1入口的并行/队列；不能照搬N64。
6. **数据与验证。** 需要同步三图、实际动作、policy可见控制state、原生成功标签的完整轨迹；实测qpos可另外留作动力学诊断，不能与控制state混名。现有reset和种子不是完整WM训练集。WM对当前π0.5行为偏差大时需微调，PACE再加阶段性采集和更新WM。

这些改动集中在环境与数据层，无需为了换WM重写策略网络和GRPO。更具体框架证据、RISE/τ0-WM/TOPReward等补充路线见[框架及扩展资料](framework_and_extra_sources.md)。

## 讨论建议

- **目标是改善当前LIBERO双相机训练：**A2World作为较独立的WM候选，VLA-MBPO作为“短分支＋奖励＋策略更新”方法参考。先明确C8与双相机接口、公开任务覆盖，不能仅换checkpoint路径。
- **目标是三相机RoboTwin：**优先OpenDW Robotwin预测能力＋现有RLinf训练底座；借WorldArena的对应任务奖励/环境设计。它更接近我们的相机和关节动作合同。
- **目标是完整方法复现：**VLA-MBPO或RISE要单独按其配套方案理解和运行；“只换WM保留GRPO”不能写成复现了整篇论文。

最有价值的首个证据是：同一真实RoboTwin起点、同一段π0.5外部动作，WM生成的三路未来与真实执行是否一致，奖励能否正确区分成败。这比直接开始长RL更能回答“接上后值不值得训练”。本页是建议，尚未安排该实验。
