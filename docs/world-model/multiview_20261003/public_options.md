# 双相机动作条件世界模型：公开实现核查

**研究快照（2026-10-04补注）：**公开代码、论文和权重的核查截止2026-10-03；本文“未运行”“尚缺闭环”等叙述限定于该次调查，不作为当前实验进度。后续组合和执行路由见[OpenDW主上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)，本次仅补充归档，没有重新检索或启动实验。

核查日期：2026-10-03。只读检索官方论文、仓库、文档、Hugging Face 文件清单；未下载模型、安装依赖或运行训练。本文区分“策略能看多图”“模型联合生成视频和动作”“给定外部策略动作后同步生成多视角环境”，最后一种才是当前 π0.5 WMRL 需要的接口。

## 结论

**当前 RLinf-Wan 的单视角限制是真实实现/权重边界，不是 Wan 架构永远不能做双相机。** 最新官方主干仍把腕图返回为 `None`。公开领域已经有动作驱动的双/多视角模拟器和完整多视角 VLA-RL 代码，因此自行接入现实；但没有核到“现有 Goal Wan 权重打开一个开关，就能正确生成腕图”的方案。

最相关的方向是：A2World 提供实际 LIBERO 双视角 WM 权重；VLA-MBPO 提供 π0.5 + 双视角 WM 的 RL 闭环；MultiWorld 提供 Wan2.2 多视角动作条件生成，可借鉴结构。三者解决不同部分，都不是当前冻结 Wan 的直接换配置替代品。

## 六项比较

| 方案 | 多视角和外部动作接口 | 实际开放资产 | 接入当前工作流的边界 |
|---|---|---|---|
| RLinf 官方 Wan/WoVR | 当前单视角；给动作生成未来主图 | 三套 LIBERO 配方、WM、奖励和 reset 数据 | 当前可运行路线；腕图仍缺失 |
| RLinf DreamZero / NVIDIA DreamZero | 多视角输入，并联合生成动作与视频 | RLinf LIBERO 双视角变换、Wan2.2-5B SFT/评估和权重；NVIDIA 完整训练及14B权重 | 是策略模型接入；不是现成给 π0.5 动作驱动的双视角 WM 环境 |
| A2World / A2World-sim | 当前主/腕图 + 动作块 → 两视角未来；支持历史与自回归 | LIBERO HDF5 转换、全参微调、推理；已核实际 `a2world-libero.pt` 5.03 GB | 最直接的双视角 LIBERO WM 候选；Cosmos2B，公开动作块20步，需要新后端/奖励/reset适配 |
| VLA-MBPO | 头/腕交错解码，动作条件动态与奖励；π0.5 RL闭环 | OpenPI训练、LIBERO/WM WebSocket、WM训练；Goal toy WM 权重29.2 GB | 最贴近“π0.5双图WMRL”目标；Bagel而非Wan，算法为短分支rollout，当前权重不是论文最强模型 |
| MultiWorld | Wan2.2-5B，多视角同步生成；VGGT共享场景特征，外部动作条件 | 训练/长程推理、RoboFactory数据、机器人权重11.19 GB | 可参考Wan多视角改法；公开机器人动作是8维关节控制、多机器人，缺现成LIBERO/C8/RL闭环 |
| OpenWAM（Stanford框架） | 多图RGB拼接；提供严格FDM：当前观测/状态 + 给定动作 → 视频 | 训练/接口代码和视频预训练权重 | 可参考拼图与FDM设计；不能把视频预训练权重当已验证LIBERO动作条件WM，任务权重仍需核验/训练 |

## RLinf 的精确边界

实时 GitHub API 核到主干 `c70606f08cdca259b8dec03d4430926b5b8fac9d`，提交时间2026-10-02 08:56 UTC。新源码已重构至 `rlinf/envs/sim/world_model/`，不能继续只按旧目录寻找支持。

- [`env.py` 551、581–583行](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/world_model/env.py#L551)：`v=1`，返回主图且 `wrist_images=None`。
- [`backend/wan.py`](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/world_model/backend/wan.py)：每环境只有一个画面历史；`input_image4`是最近4帧时间历史，不是4台相机；生成结果为 `[B,3,T,H,W]`。
- [DreamZero官方文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_dreamzero.html)确实支持LIBERO双视角、DROID三视角，但归于SFT/策略部署。其[`predict_action_batch`](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/dreamzero/dreamzero_policy.py)调用联合生成再返回动作，`default_forward`仍未实现。不能把它描述成已有双视角Wan RL环境。
- 文档的[Step18000权重链接](https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000/tree/main)当前重定向到Step26000；选型时要锁实际资产版本。

## 哪些公开路线最值得参考

### A2World：有现成双视角LIBERO模型，但接口不同

10月3日深入核查补充：公开`a2world-libero.pt`没有明确逐suite覆盖；论文A2World-policy四组成绩不能当作此WM的覆盖清单。C20动作经整体280D嵌入条件所有帧，补齐后截取前8帧尚非已验证的C8接口。RoboTwin的14D关节动作与A2World的14D末端增量也不等义，详见[本轮审计](../robotwin_options_20261003/a2world.md)。

[官方代码](https://github.com/LogosRoboticsGroup/A2World)、[模型文件](https://huggingface.co/Fleurrr/A2World-World-Model/tree/main)、[模型卡](https://github.com/LogosRoboticsGroup/A2World/blob/main/world_model/release/MODEL_CARD.md)。已核预训练和LIBERO两份实际权重，合计10.1 GB。基座是Cosmos-Predict2-2B；LIBERO使用agent-view和eye-in-hand两个相机ID，不是两份互相独立的生成。

[数据文档](https://github.com/LogosRoboticsGroup/A2World/blob/main/world_model/docs/DATA.md)明确：20动作对应21帧；LIBERO七维动作填入14维表示，平移/旋转乘0.05，夹爪映射也不同。这里不能直接塞当前归一化C8。配套转换保留末端位置、姿态和夹爪元数据；全参数训练例子是 `GPUS=8`，这不是四卡必然训不动的证明，也不是已核四卡预算。尚未看到现成RLinf后端或π0.5 RL收益回执。

### VLA-MBPO：双视角RL流程参考，但公开默认样例范围有限

10月3日进一步逐文件核查补充：默认Goal RL配置实际使用`libero_goal_task_3`；README提及的`train_robotwin.sh`在官方/作者当前Git树均未找到，不能视为已发布RoboTwin训练支持。WM输出为C10块末主/腕两图，并不预测下一state。详细代码与替换边界见[本轮审计](../robotwin_options_20261003/vla_mbpo.md)。

[官方仓库](https://github.com/LAMDA-RL/VLA-MBPO)已公开π0.5四套件配置、真实LIBERO和WM服务、策略更新、WM训练代码；[Goal模型](https://huggingface.co/Rhx11111/Bagel-goal-256-2500/tree/main)API核到 `model.safetensors` 29,214,685,336字节。作者明确这是早期step2500示例，配套策略是一条演示SFT初始化，不能期待论文最终分数。

[WM训练脚本](https://github.com/LAMDA-RL/VLA-MBPO/blob/main/world_model/scripts/train_libero.sh)为四卡FULL_SHARD；仍带站点模块和路径配置，实际移植需整理。不能从“四卡脚本”推断当前四卡共享配置能以同样并行度装下。它的价值是双视角生成及短分支采样已有完整实现，而不是轻量直接替换现有Wan。

### MultiWorld：Wan自身可以扩展，不必从零发明

[官方代码](https://github.com/CIntellifusion/MultiWorld)、[机器人实配](https://github.com/CIntellifusion/MultiWorld/blob/main/robots/configs/inference.yaml)、[已开放权重](https://huggingface.co/Haoyuwu/MultiWorldCheckpoint/tree/main)。实配为Wan2.2-TI2V-5B、256×320、81帧、训练2视角/评估3视角，动作条件8维每机器人，并加入相机与共享场景信息。机器人权重API核到11,194,929,712字节。公开推理例子8进程是所给启动方式，不能称单个轨迹最低必须8卡。

它证明多视角动作条件Wan现实可做；但是替换到LIBERO需要动作、时序、数据和训练适配，不能加载后就输出当前C8主/腕图。[论文/项目说明](https://multi-world.github.io/)侧重动作跟随、画质与多视角一致性，未提供当前π0.5/RLinf学习闭环。

### DreamZero与OpenWAM：可复用多视角编码经验，别混淆策略与模拟器

[NVIDIA DreamZero](https://github.com/dreamzero0/dreamzero)默认三视角、联合动作视频生成；原14B分布式推理文档最低2卡，已有较小Wan2.2-5B训练路径。[RLinf双视角数据变换](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/data/datasets/dreamzero/data_transforms/libero_sim.py)可参考相机布局，但要当世界模型需明确外部动作条件和未来图像返回路径。

[Stanford OpenWAM](https://github.com/OpenWAM/OpenWAM)区分联合策略与严格forward dynamics，有多图拼接和FDM接口设计。其公开30层训练配方报告四张48 GB卡；该成本不能移用到本项目N64/C8。预训练模型能生成多图，也不能直接证明它按π0.5动作准确模拟LIBERO接触。

## 搜过但不能当现成双相机解法

- [DreamDojo论文](https://arxiv.org/html/2602.06949v1#S5)明确把“不自然支持多视角模拟”列为限制；代码树出现multiview camera模块，不足以证明机器人后训练权重支持同步头/腕模拟。官方主要评估使用单视角策略。
- [WorldArena2 RLinf集成](https://github.com/WorldArena2/WorldArena-2.0/blob/main/RL_env_benchmark/README.md)的RoboTwin Wan数据实际读取head RGB，输出 `[T,1,H,W,3]`；通用接口写可选 `wrist_images` 不等于Wan已生成腕图。
- 把“相机运动控制”“多张参考图”“联合生成动作视频”搜到的结果直接归为“外部动作条件双相机模拟器”，会高估开源即用程度。

## 对自行接和训的判断

现实。最轻的Wan路线是同步主/腕图组成固定画布、沿用动作条件生成、输出后拆开，再用配对双视角轨迹微调；这样无需一开始就做复杂几何网络。但现有单图权重没有学过这种布局，光拼接输入不能算支持。保留每相机原分辨率会增加token与算力；保持总画布像素则损失每视角细节。

更正规的路线是每相机独立编码、加相机ID并共享注意力/场景表示，MultiWorld/A2World提供参考。此时真正难点是两张图要描述同一个物体和接触状态，腕相机随机器人移动，以及策略改进后产生的新动作是否仍能被WM正确预测。

所以“双相机”是重要的训练部署一致性缺口，但当前推盘退化还不能只归因于它。现成多视角模型仍可能有动作不跟随、奖励误判或长程漂移；论文/代码中的多视角支持和本任务真实收益需要分别判断。本轮讨论不改变现有WM训练方法和并行预算。
