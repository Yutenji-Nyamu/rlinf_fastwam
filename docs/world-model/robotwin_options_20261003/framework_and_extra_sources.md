# RLinf接入边界与补充路线

**研究快照（2026-10-04补注）：**公开代码、论文和权重的核查截止2026-10-03；本文“未运行”“尚缺闭环”等叙述限定于该次调查，不作为当前实验进度。后续组合和执行路由见[OpenDW主上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)，本次仅补充归档，没有重新检索或启动实验。

2026-10-03，只读官方源码、文档与发布元数据。未安装、下载模型或启动实验。

## 已有训练底座与需要补的环境

本项目已经有RoboTwin原生环境中的π0.5/Flow-SDE/GRPO、Sidney权重转换与动作一致性验证、固定种子评测和保存恢复。历史核定合同是三相机、14D绝对关节动作与14D state、H50/C50/M10；见[原接口与基线](../OPENDW_PI05_GRPO_CONTEXT.md)。这些是已保存工程证据，本文不推断当前GPU运行状态，也不将别的任务最新参数当作本实验配置。

RLinf官方现在也明确支持RoboTwin π0.5在线RL，观察包含head、可选左右腕和14D proprio，训练与原生评估入口公开。[训练教程](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)及[评估教程](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robotwin.html)。这证明物理仿真训练底座可复用；其SFT/RL结果并不是WM训练的结果。

实时GitHub API核定RLinf main仍为`c70606f08cdca259b8dec03d4430926b5b8fac9d`（2026-10-02）。当前WM backend目录实际只有Wan和OpenSora实现及NPU补丁，没有A2World、OpenDW或Bagel backend。

- [backend契约](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/world_model/backend/__init__.py#L43-L104)：`open_session(init_frames, init_actions, seeds)`，`generate(actions)`，`close_session`与onload/offload；生成返回单视频`[B,C,T,H,W]`。
- [通用WM env](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/world_model/env.py#L528-L584)：仍返回`wrist_images=None`和16D零state。新增backend本身不会使上层自动返回三视角和真实状态。
- [RoboTwin env](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/robotwin/robotwin_env.py#L329-L398)：处理chunk reward、终止/超时、实际执行动作数与final observation。接WM时要明确继承这些语义，不能把视频帧数当作执行动作数。

建议工程边界：沿我们已经验证的训练树新增模型backend/service、RoboTwin WM env、动作/图像/state转换器、reset/轨迹读取与独立YAML。π0.5、GRPO、权重同步、checkpoint及原生RoboTwin评估继续复用；不为取得一个后端接口而整体升级框架。此处是设计建议，没有写训练实现。

## 数据到底需要什么

WM训练需要同步的`task / episode / time / head / left_wrist / right_wrist / executed_actions / policy_state / native_success / termination`；可额外记录`measured_qpos`用于动力学诊断。**同日纠正：**深圳2共享RoboTwin的`policy_state`来自12轴drive targets＋2个缓存夹爪命令，不是实测qpos。仍须区分实际生效的控制目标、策略内部归一化32D张量和实测关节角；不能仅因OpenDW没有实测state输出就要求新训预测器。见[实时源核验及接口审计](../robotwin_pipeline_20261003/interface_audit.md)。

reset包只负责“从哪一帧开始”，不能替代动态训练轨迹。现有演示和策略采集流程可作数据来源；本轮未扫描服务器完整轨迹库存，因此不声称已有数量足够。若要PACE式闭环，需要把当前π0.5的成功、失败和偏离专家的轨迹重新交给WM训练流程，阶段性更新WM后再继续策略RL。

视频模型通常没有RoboTwin物体位置、接触等内部状态，原生`check_success`无法直接在生成RGB上运行。可选任务视觉奖励、视频/语言奖励，或同时预测相关状态；都要用原生成功标签判断其误判。末动作近似下一关节state解决不了物体状态和任务判定。

用户此前明确的RoboTwin step limit 400应按真实动作计数继承；本轮不把它改成400视频帧，也不改C50或当前LIBERO C8约定。

## 本轮补查的其他实现

### RISE：三视角WM＋π0.5＋RLinf的完整工程参考

官方[RISE](https://github.com/OpenDriveLab/RISE) main=`5fac1e6ab9d50d4cc1ba4daeadf14023c8415955`（2026-06-03）。已有多视角动态模型、progress/value模型、离线训练、RLinf衍生在线训练；[动态模型文档](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/docs/dynamics_model.md)明确头＋左右腕三视角，并给LTX基座、预训练权重、LeRobot数据和任务微调入口。[在线训练](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/docs/online_training.md)已有env/rollout/actor资源分配和chunk奖励配置。

其公开动态预训练来源为Galaxea与AgiBot，发布示例/配置面向自身真机任务；本轮完整Git树未查到名为RoboTwin的配方。`rl_release.yaml`有π0.5、C50、两chunk imagined rollout与独立value模型。它能帮助我们理解三相机和短想象训练如何接线，但仍需RoboTwin数据微调WM和任务奖励，不能把真机配套权重当作RoboTwin即用模型。更不需要把RISE、WorldArena和现用RLinf三套runner全部混合；选择现用底座，只借相关接口。

### τ0-WM：已公开动作条件模拟器，但需要域适配

[官方仓库](https://github.com/sii-research/tau-0-wm) main=`a3e4c0fe58b88974ab6d9b1522af97e5e46be04d`。除了联合视频动作策略，确实有独立action-conditioned simulator、微调入口与服务；[实际返回](https://github.com/sii-research/tau-0-wm/blob/a3e4c0fe58b88974ab6d9b1522af97e5e46be04d/web_infer_utils/simulator/TauSimulator.py#L228-L260)含未来图像和预测reward。公开模板是TACO等自定义数据，预训练动作采用双臂末端位姿表示，不是我们14D关节合同。未核到RoboTwin配套权重或RLinf适配。可作未来候选，当前工程距离大于OpenDW Robotwin bundle。

### WorldLoop/TOPReward：减少专用奖励训练的一种参考

[RLinf官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/resources/blog/worldloop_topreward.html)公开了冻结Qwen3-VL token概率奖励接Wan/GRPO的路线，当前实验证据是LIBERO Spatial/Object；实现引用独立分支/PR。它说明不必所有任务都从头训练分类器，但不证明RoboTwin能直接复用阈值，也不能替代动作条件WM或下一state。该文章报告的资源来自MI300，不能套为我们的H100显存预算。

### 未列入首选的项目

- [DreamX-Phi](https://github.com/AMAP-ML/DreamX-Phi)在WorldArena有结果，但本轮官方仓库仍只有占位说明，承诺赛后发布模型/推理；不能作为当前可下载替代。
- [MultiWorld](https://github.com/CIntellifusion/MultiWorld)提供多视角动作条件建模，但配套机器人任务是RoboFactory，需要另做RoboTwin动作/数据域适配；旧[多视角调研](../multiview_20261003/public_options.md)保留细节。
- [EVA](https://github.com/RobbinW/EVA)虽有RoboTwin与Flow-GRPO，公开任务是用逆动力学奖励训练视频生成器；不是直接保留π0.5、以外部动作驱动WM训练π0.5的现成闭环。

## 如何理解投入

只换WM：主要改环境、数据和模型服务，策略RL算法可保持。迁入完整VLA-MBPO/RISE：还包括短分支初态采样、value/reward、原生状态恢复及训练目标，属于另一个方法实现。若重新训练三视角Wan：是新动态模型数据/训练工作，不是普通接口转换。

预训练checkpoint大小不等于运行显存；官方4卡/8卡训练例程也不等于我们现有并行度可直接运行。此轮仅有源码/发布证据，没有本地GPU容量、吞吐或收益承诺。
