# WoVR闭环参考与可复用代码

2026-10-08抓取；下表区分“论文主张”“已公开代码”“我们的适配”。源码缓存与SHA清单：`E:/Codex/home/visualizations/2026/10/03/01a0ffb6-b309-7fa1-9ae0-b20e6d63a11c/followup-plan-20261008/reference/`。本聊天追加核验仍使用这些锁定源码，仅补下载一个multimodal.py；未下载模型或完整数据集。

| 来源 | 已核验的内容 | 对本pipeline的用途/限制 |
|---|---|---|
| [WoVR论文](https://arxiv.org/html/2602.13977v1) | PACE、KIR、时序稳定与masked优化；实验只做一次共同演化 | 借鉴闭环顺序；不把“每10轮重训WM/RM”说成论文规定 |
| [RLinf Wan说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html) | 公开例子使用冻结WM；明确共同演化需自行串接 | 支持现有RLinf策略训练和原生评估框架 |
| [RLinf环境](https://github.com/RLinf/RLinf/blob/0067f7d5ce525c18bda439ed8dbb638abafd6a06/rlinf/envs/sim/world_model/env.py) / [Wan backend](https://github.com/RLinf/RLinf/blob/0067f7d5ce525c18bda439ed8dbb638abafd6a06/rlinf/envs/sim/world_model/backend/wan.py) | 当前上游已改为sim/world_model/backend结构，存在KIR和reset适配 | 借鉴接口；不要再引用旧main路径world_model_wan_env.py（404） |
| [RLinf DiffSynth训练入口](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/examples/wanvideo/model_training/train_rlinf.py) | VLA轨迹NPY数据、train/val目录、action条件训练 | 可参考训练/验证结构；不是DW05 checkpoint的直接训练器 |
| [OpenDW训练入口](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/playground/example_dw_exp.py) | train/inference/norm-stats/smoke入口 | 最接近我方现用模型，优先复用 |
| [DW05数据集](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/dw_dataset.py) / [变换](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/transform/action.py) | episode JSONL、三视角、动作/状态预处理 | 默认14→16→17→32、delta/quantile；与现用14D发布bundle不一致，须适配 |
| [DW05训练器](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/exp/dw05_trainer.py) / [通用训练器](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/exp/generative_trainer.py) | weights-only加载、完整恢复、权重导出及验证 | 14D模型/数据适配后复用；默认验证还需保留mask和固定留出集 |
| [WorldArena2 RoboTwin集成](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/README.md) / [环境](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/envs/world_model/world_model_wan_env.py) | 原生HDF5的head RGB与joint_action/vector合同；HTTP环境代码确实存在 | 借鉴RoboTwin动作/图像映射；其单head NPY并非我方三视角数据。README提到的diffsynth-studio/encode_vla_to_rlinf.py未在该锁定仓库树中找到，不列为已取得代码 |
| [WMPO官方代码](https://github.com/WM-PO/WMPO/tree/c836d74ec6f4525c93fe980d54d0ca870118615a) | OpenSora轨迹训练、想象GRPO与奖励模型脚本 | 对照“固定WM策略优化”的方法；不为闭环先迁移到其框架 |
| [World4RL项目](https://world4rl.github.io/) | 视觉world-model策略RL的另一方法参照 | 本轮仅作方法对照，不把项目介绍当作已验证的DW05可用模块 |

## WoVR原文的规模和边界

论文每个LIBERO suite含10任务：先采1500条基础策略轨迹，再用更新策略采1000条，更新WM一次后继续策略学习。这是**suite级**预算，不是要求我们单个lift_pot也采2500条。论文未明确给出PACE阶段同步重训RM的周期；不能把每10轮重训WM/RM写成官方规定。KIR与时序建模/目标函数是另外的机制，完成一次PACE不等于完整复现WoVR。

## 锁定版本

- OpenDW `e33befa8005a1585e0140dbf464566e90bc79aa1`，与当前使用源码pin一致。
- RLinf `0067f7d5ce525c18bda439ed8dbb638abafd6a06`（仅参考，不原地升级训练仓库）。
- RLinf/diffsynth-studio `2a2e05fa1f724828b243f272540989b19a6e54f8`。
- WorldArena2 `5978ce5c81e55b8c8358f4f5966a13ce385ff155`。
- WMPO `c836d74ec6f4525c93fe980d54d0ca870118615a`。

OpenDW训练器、模型核心、数据变换、推理器均已逐处读到；尚未运行微调/做显存验收。参考是否“能直接执行”与“具有可复用源码”明确分开。

## 本聊天复核：补足先前审计，避免两份配方混用

| 证据 | 核验位置 | 得出的适配结论 |
|---|---|---|
| [10-03动作/state审计](../robotwin_pipeline_20261003/action_state_followup_20261003.md) | §2发布bundle与默认训练不同 | 此差异已有记录，必须继承到本轮规划；未证明发布权重训练有错 |
| [现有lift服务源码](../../../local_scripts/lift_two_gpu_20261006/reference/lift-pot-v1__generated__service__opendw_service_batched.py) | L330–341、477–487 | 显式zscore_14d、absolute、robotwin_resize，model action/proprio维度14/14 |
| [官方policy](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L309-L368) | 从checkpoint识别维度，选择归一化并构建对应模型 | trainer的resume加载不等于自动完成同样配置推导 |
| [默认dataset](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/dw_dataset.py#L775-L799) | ArrangeState、AddTerminationState、AddAction、DeltaAction、quantile、Pad32 | 直接复用该transform链会与当前推理不一致 |
| [训练loss](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/model/dw05/dw05_core.py#L137-L211)及[样本输出](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/transform/output.py#L29-L73) | 失败has_action=False只mask动作loss；action数组保留、video loss另算 | 失败数据可训练WM动态；传真实episode标签，不删所有失败 |
| [默认验证](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/exp/dw05_trainer.py#L33-L135) | 转batched sample未传四类mask；每次随机一条 | 固定留出清单并保留mask；否则val_loss不与训练同义 |
| [模型可训练范围](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/model/dw05/dw05_arch.py#L314-L345) | dit/MoT和proprio encoder可训；VAE/T5冻结 | 不是只有小adapter，训练显存另测；公开入口不是已验证的两卡LoRA配方 |
| [图像处理](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/transform/multimodal.py#L164-L225)及[未来索引](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/transform/multimodal.py#L559-L571) | 默认未来图letterbox；时刻为start+4/8/…/32 | 改为现服务direct resize；保留已对齐的时间索引 |

补抓`multimodal.py`位于`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wovr-cycle-review-20261008/`，SHA256 `5aa7f2a3a8ac0035fa2a2ac5e92a8b47187ebfedb99c6b8b10df031aec3439a3`。其余复用上述reference目录与verified_files.json。网页工具部分固定commit链接读取失败时，使用已缓存的锁定源码及公开raw源复核；没有用访问失败冒充已读。

仍未取得当前发布checkpoint的完整训练manifest/数据导出时钟，故不能保证恢复了其原始训练配方。本轮要做的是忠实延续已跑通的推理合同，用新原生数据做受控适配，而非宣称精确复现预训练。

## 本轮补充：全貌、轮换和验证的直接依据

| 用户问题 | 官方依据 | 我们应怎样表述 |
|---|---|---|
| WoVR除了PACE还有什么 | [论文§4.1–4.3](https://arxiv.org/html/2602.13977v1#S4)：动作条件WM、固定参考/近期历史/带噪历史训练、KIR、成功后mask与有效长度归约、PACE | 是一组配合的办法；可以共同规划，不能把“先PACE”写成论文的强制阶段 |
| 历史条件是不是多传几张图即可 | [OpenDW模型构造](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/model/dw05/dw05_arch.py#L58-L95)拒绝num_hist_frames非1；[RLinf Wan backend](https://github.com/RLinf/RLinf/blob/0067f7d5ce525c18bda439ed8dbb638abafd6a06/rlinf/envs/sim/world_model/backend/wan.py)传input_image参考图、input_image4近四帧和历史动作 | 现用OpenDW缺的是模型条件能力；要改条件组装、时序位置、训练和服务历史缓存，不能直接加载WoVR Wan权重替换 |
| 官方怎样循环 | [论文§5.2](https://arxiv.org/html/2602.13977v1#S5.SS2)：每LIBERO suite基础策略1500条、更新策略1000条，实际一次WM共同演化；[RLinf官方例子](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html)说明冻结WM且共同演化需自行实现 | 每10轮只是我方可选间隔；公共代码不是完整外层循环的一键命令 |
| 是否同时训练RM | 论文§4.1定义二分类RM，但§4.3/§5.2未明确同步RM重训周期；[RLinf RM指南](https://rlinf.readthedocs.io/en/latest/rst_source/extending/reward_model.html)提供原生采集、标签、训练、推理流程 | 同一批真实数据可以分别训练WM和RM；固定RM是首轮区分贡献的建议，不是假称官方永久冻结RM |
| 论文的GPU轮换是不是WM微调 | [论文附录A](https://arxiv.org/html/2602.13977v1#A1)：rollout内策略与WM推理，随后policy optimization | 图里的Training指策略优化；PACE的WM训练是另一个低频阶段。我们的分阶段卸载仍可采用 |
| 论文有没有独立验证WM | [§5.1](https://arxiv.org/html/2602.13977v1#S5.SS1)：3000训练、200留出轨迹；[§6.1](https://arxiv.org/html/2602.13977v1#S6.SS1)：1500训练、24留出，比较视频预测指标 | 这些是不同实验的预算，不能合并为策略实验的2500。支持留出验证，但未规定每次切换必须完整重跑论文基准 |
| 已有done是否等于WoVR损失 | 论文公式11按有效轨迹长度归约；[RLinf公开示例配置](https://github.com/RLinf/RLinf/blob/0067f7d5ce525c18bda439ed8dbb638abafd6a06/examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml)写token-mean；我方env已有首次成功奖励/停后续动作块 | 配置名称和done本身不足以证明公式等价。后续只审查首次成功块内的有效范围与最终归约，不重复重造已有功能 |

上述事实已合并进[通俗全貌](WOVR_EXPLAINED.md)。新实验“从头”指原始预训练π0.5从RL第0轮起步，OpenDW仍由现用预训练权重初始化。默认训练配方与发布推理配方不同，不能推导为该权重不能续训。

## 本轮收敛与预算核对

用户选择保留OpenDW现模型、chunk级处理，不加KIR/历史改造；每10轮切阶段。上文有关完整WoVR的参考代码保留为背景，不再列作本轮实现要求。

- 论文§5.2明确初始1500＝150/任务；后续1000/10＝平均100/任务，只更新一次。采集150初始/100每10轮是我方多轮扩展，不能称论文原预算。
- [锁定RLinf策略例子](https://github.com/RLinf/RLinf/blob/0067f7d5ce525c18bda439ed8dbb638abafd6a06/examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml#L29-L78)：max_epochs1000、N64、rollout_epoch16。官方页面明确WM冻结，故既非WM训练epoch，也不证明论文各阶段实际跑满这些轮次。
- [OpenDW通用训练配置](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/exp/base_dw_exp.py#L97-L124)：num_epochs5、learning_rate1e-4；[示例入口](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/playground/example_dw_exp.py#L43-L51)覆盖batch_size2。这是通用默认，未验证为当前14D发布模型的最佳微调配方。
- [训练步数计算](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/exp/generative_trainer.py#L305-L326)：按dataset长度、每卡batch、进程数、累积与epoch计算；推理B16与策略每轮512轨迹不能直接当WM优化batch。
- 本轮未核到WoVR正文对策略每阶段具体RL轮数/WM具体梯度更新步数的完整记录。公开用户issue中的100000epoch疑问不是作者推荐，不作为实验参数。

前版文件完整存于archive/20261008_before_chunk_only_cycle/，当前取舍以CONTEXT/PLAN为准。
