# 新 RoboTwin 任务的 WM 奖励从哪里来

2026-10-06 官方资料刷新。本文只做文献、代码与已有实验记录核查，没有改训练、借卡或部署新模型。当前现场与云端发布另看本轮执行记录；10月3–5日旧专题保留为历史。

**可以换到其他任务，奖励也有明确的制作流程。**真实 RoboTwin 可以直接判断成功；OpenDW 生成的是视频，没有物体真位置和接触真值，所以要再接一个“看画面判断完成”的模块。优先办法是：**让原生仿真自动给成功/失败标签，训练一个小视觉分类器，再放到 WM 里用。**并不要求我们手工标注全部视频，也不要求从零训练一个大 VLM。

**不能把目前两个任务都说成“原始成功率太高”。**按铃有明确证据：原 SFT 同组32/32，10月6日已有CP10/20/30/40分别32/32、32/32、31/32、32/32，这组测试基本到顶。摆瓶子此前25%–53%是已训练检查点的原生结果，99.7%是WM代理成功率；截至10月5日专题，尚缺同C32/384的未训练baseline。因此摆瓶子的主要已知问题是代理分数和原生结果不一致，不能据代理分数断言原策略已很强。[按铃执行](click_bell_execution_20261005.md)、[摆瓶子奖励饱和分析](reward_saturation_discussion_20261005.md)

## 官方各自提供什么

| 来源 | 已公开且本次可查的内容 | 换新 RoboTwin 任务还缺什么 |
|---|---|---|
| **OpenDW** | Robotwin 动作条件三视角生成、模型权重、WM 训练代码与 JSONL 格式 | **不提供现成通用成功判定。**README 仍写 Value Expert 后续发布；即使以后有 value，也须核它是否表示成功概率 |
| **WorldArena 2.0** | `adjust_bottle`、`click_bell` 两个任务 RM；ResNet18＋T5 融合模型；奖励训练与预处理代码 | 新任务的原生成败数据、对应权重与独立验证；这两份权重不能只换指令就当作全任务 RM |
| **RLinf / WoVR** | 采集→转换→训练→接入的奖励工作流；WoVR 用小成功分类器提供稀疏奖励 | 框架能训练 RM，不等于自带全部 RoboTwin 任务权重。用新域仍要标签与适配 |
| **WMPO** | VideoMAE 短视频成功分类器、训练脚本、阈值验证、四个 MimicGen 任务权重 | 有可借的配方；现成权重不是 RoboTwin 专用，仍要自己的目标任务数据 |
| **RLinf WorldLoop / TOPReward 路线** | 冻结 Qwen3-VL-8B，读取完成回答 token 概率，官方有 LIBERO 接入和结果 | 可免 RM 微调，但仍要验证 RoboTwin 的成功/失败可分性，不能照搬 LIBERO 阈值 |

来源：[OpenDW README](https://github.com/dexmal/opendw#architecture-scope)、[WorldArena 权重目录](https://huggingface.co/WorldArena/WorldArena2.0/tree/main/reward_model)、[RLinf Reward Model Guide](https://rlinf.readthedocs.io/en/latest/rst_source/extending/reward_model.html)、[WoVR §4.1](https://arxiv.org/html/2602.13977v1#S4.SS1)、[WMPO 发布清单](https://github.com/WM-PO/WMPO#prepare-datasets-and-pre-trained-checkpoints)、[WorldLoop 官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/resources/blog/worldloop_topreward.html)。

## 最接近我们现有框架的自训办法

**目标是训练“完成了吗”，不是“像不像示范”或“还剩多久”。**数据和接线可以沿现有原生标签采集器走：

1. 选一个原 SFT 仍有提升空间、关键结果能从图像看清的任务。用同一评估协议确认起点，避免按网上排行榜或 WM 代理分数选任务。
2. 在原生 RoboTwin 跑策略，保存图像、指令、episode/时间和成功信号；加入成功示范与真实失败，尤其是“差一点完成”的失败。原生判定自动提供 0/1 标签。
3. 按**整条轨迹**划分训练、验证、测试，避免同一视频的相邻帧同时出现在训练和测试。先训练小 ResNet；如果成功是“刚才发生过接触”，再用短视频而非仅最后一张图。
4. 在未参与训练的原生数据上看漏成功、假成功，再检查 OpenDW 生成域。通过后替换奖励模块，保留 π0.5、OpenDW、B16、GRPO 的主体。

这是我们建议的实施流程；没有在本轮启动新采集或 RM 训练。现有几十条样本足以发现“全 No”这类明显失败，不能据此承诺新模型训练数据已经足够。

WorldArena 的公开预处理从 `info.success` 取逐帧标签，缺失时回退到 `info.episode.success_once`，默认按 episode 做 80/20 划分、负正约 2:1。实际适配时必须保留**图像时刻与标签时刻一致**，不能把成功 episode 的全部前期画面都标成 1。对于按铃这种一次接触后锁存成功的任务，若成功已经发生而当前画面看不出来，短历史或接触附近帧更合适。[官方预处理](https://github.com/WorldArena2/WorldArena-2.0/blob/main/RL_env_benchmark/examples/reward/preprocess_reward_dataset.py)

RLinf 已给出 `data_collection.only_success: False`、pickle 导出、`preprocess_reward_dataset.py`、`run_reward_training.sh` 及在线接入示例。其默认 ResNet 路线的输入是图像/标签；它还给出短历史 VLM 微调路线。我们可以借这些工具，不必升级或合并整套最新 RLinf。[最新工作流](https://rlinf.readthedocs.io/en/latest/rst_source/extending/reward_model.html)

## 训练规模到底有无依据

WorldArena 论文报告：共 **3000 条混合轨迹**，由两档策略与原生 planner 采集，同时用于 WM 和 RM；另有 **1000 条专家轨迹**用于策略 SFT。该处没有给出每任务正负分布，也没有说“训练任一新任务 RM 最少要 3000 条”。其论文 SFT 起点为按铃 **43.75%**、摆瓶子 **55.08%**，所以论文确实有明显提升空间，不能把这个起点套到我们的原 SFT 上。[论文表2、附录B.1](https://arxiv.org/html/2605.17912v1#A2.SS1)

当前两任务 RM 的默认结构共 **122,382,017 参数**；T5 冻结时可训练参数约 **12.75M**。该完整结构的总参数已由我们 10月5日 strict load 复核，远小于 RynnValue。新增单任务也可用不带 T5 的 ResNet 分类器，是否需要文本取决于是否共用多任务模型。[模型代码](https://github.com/WorldArena2/WorldArena-2.0/blob/main/RL_env_benchmark/rlinf/models/embodiment/reward/robotwin_reward_model.py)、[我们参数与训练链审计](reward_training_and_candidates.md)、[实际加载与运行](click_bell_execution_20261005.md)

**训练入口已有，但不能说完整复现文本 RM 一键完成。**10月3日固定版本审计发现：通用 dataset 返回 image/label，reward worker 未传 instruction；仅把 `model_type` 改成 T5 结构会走无文本分支。若沿用发布 RM 的文字条件结构，须补 dataset/collate/forward 的指令传递。单任务也能固定传该任务一句话。这个旧固定版本结论仍作为适配检查项；本轮成功读取模型和训练 YAML，dataset/worker 网页抓取失败，没有声称重新核到了其最新完整调用链。[默认训练配置](https://github.com/WorldArena2/WorldArena-2.0/blob/main/RL_env_benchmark/examples/reward/config/reward_training.yaml)、[固定版本具体调用链](reward_training_and_candidates.md)

## WMPO 给了另一个更贴合“事件”的做法

WMPO 用一段视频判断是否成功：成功结尾片段是正例；成功之前的片段和失败视频片段是负例；滑动窗口检查整条生成轨迹，阈值在验证集选。它的公开脚本确实包括训练和验证，并非只有论文描述。[论文 §3.3](https://arxiv.org/html/2511.09515v1#S3.SS3)、[训练脚本](https://github.com/WM-PO/WMPO/blob/main/reward_model/videomae.py)、[阈值验证](https://github.com/WM-PO/WMPO/blob/main/reward_model/find_thre.py)

本次源码读到示例为 VideoMAE-base、8帧、224像素、二分类；输入 WebDataset 的 `video.npy` 与 `meta.json`，后者含 `finish_step` / `complete`。示例训练/验证路径写成相同的作者本地模式，不能直接照抄；我们应明确不相交的 episode 划分。论文写 BCE，公开实现用双 logits＋CrossEntropy，都是二分类目标，但不是字面相同的实现。200k 更新和8卡命令只是脚本示例，不是我们新任务的最低预算。[同一脚本配置、dataset、main](https://github.com/WM-PO/WMPO/blob/main/reward_model/videomae.py)

它更值得借鉴的地方是**用真实失败和近完成片段教会模型分辨**，不是更大的网络。对按铃，“刚才确实碰到，现在抬手了”需要时间信息；对静态摆放，单张主图通常更简单，但遮挡仍可能要求腕图。这是根据任务可观测性给出的适配建议，不是已做完的比较实验。

## 为什么 LPIPS、通用 VLM 或更准 RM 都不能自动解决全部问题

LPIPS 衡量当前图与目标图相似程度，视角、背景和物体外观都会影响它；图像相似不等于完成了正确接触。WorldArena README 把它和任务 RM 都列为可选接口，公开按铃 YAML 默认 LPIPS，而我们实际已明确使用专用分类器。[官方环境与奖励接口](https://github.com/WorldArena2/WorldArena-2.0/blob/main/RL_env_benchmark/README.md)、[按铃配置](https://github.com/WorldArena2/WorldArena-2.0/blob/main/RL_env_benchmark/examples/embodiment/config/env/wan_robotwin_click_bell.yaml)

RLinf 的 WorldLoop 提示另一条零训练路线：不让 VLM 自由生成 Yes/No，而取指定完成回答 token 的概率。官方是在 LIBERO 校验窗口与阈值后接入，不能推出 Rynn 的任意 Yes/No 概率也会有效，也不能把 0.46 直接搬到 RoboTwin。该路线可作为未来离线备选，本轮未部署。[官方 WorldLoop / TOPReward](https://rlinf.readthedocs.io/en/latest/rst_source/resources/blog/worldloop_topreward.html)

有两种不同错误要区分：

- **RM 看错：**画面里没完成，却给了成功。新标签、难负例、适当视角/短历史可能改进。
- **WM 画错：**动作实际会失败，视频却画出“完美完成”。即使 RM 把画面读对，也会给成功。单靠更换 RM 不能知道这段视频违反了真实动力学。

第二种需要对照原生轨迹、改进 WM 的策略/失败分布覆盖，或缩短误差累积。WoVR 明确讨论幻觉造成虚假成功，使用 KIR 与后续策略—WM 对齐；它没有承诺一个奖励模型就能替代这些步骤。[WoVR §4.2](https://arxiv.org/html/2602.13977v1#S4.SS2)

## 本轮建议与边界

**下一条有学习价值的新任务，优先“原生基线有空间＋成功可看清＋小 RM”路线。**数据用现有原生采集链自动标，模型先简单，靠真实失败检验。Rynn 的已有结果继续用来决定是否值得专门适配；原始数值或文本输出都不能仅凭名称当成功奖励。[Rynn 成败实测](rynn_binary_probe_20261005.md)

本轮重新读了上述官方网页、论文和公开代码；GitHub API 本机 TLS 访问失败，未声称完整仓库 HEAD/树已独立重新枚举。OpenDW README 仍明确 Value Expert 后续发布；WorldArena 的公开 RM 目录本次可见 `adjust_bottle`、`click_bell`、`t5-base`。没有找到可直接替换为“所有 RoboTwin 任务成功判定”的官方公共权重。Robometer 按用户决定不再列入执行候选。
