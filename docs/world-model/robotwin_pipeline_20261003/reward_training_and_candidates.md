# RoboTwin 任务成功奖励：训练线索与公开候选

2026-10-03 专项检索。承接 [奖励语义与既有实现](reward_options.md)，本轮重新查了官方代码、论文、HF 文件目录和模型元数据。只研究与维护文档，未启动服务器任务、未下载完整权重。

## 当前决定

**首个 smoke 使用 WorldArena 的 `adjust_bottle` 已发布奖励模型。**它读取 OpenDW 拆出的主相机 RGB 和摆瓶子指令，返回成功分数；腕图仍给 π0.5。奖励模块可独立更换，无需先训练新 RM，也无需把下列通用模型一起部署。

**其他 RoboTwin 任务可以照着训练。**直接标签来自真实 RoboTwin 的任务成功判定，模型学习从图像近似该判定。当前公开材料给出了可复用模型与通用训练组件；完整复现 WorldArena 文本条件 RM 还要补一处文本传递，见下文。这是明确的工程接线，不是新算法问题。

## WorldArena 模型有多大，训练配方公开到哪里

固定源码：[WorldArena-2.0 `5978ce5c`](https://github.com/WorldArena2/WorldArena-2.0/tree/5978ce5c81e55b8c8358f4f5966a13ce385ff155)。模型文件为 [`robotwin_reward_model.py`](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/models/embodiment/reward/robotwin_reward_model.py)。

| 部件 | 默认结构 | 按源码与配置计算的独立参数量 |
|---|---|---:|
| 视觉 | ImageNet ResNet18，去除原分类头，保留空间 tokens | 11,176,512 |
| 文本 | T5EncoderModel / T5-base，默认冻结 | 109,628,544 |
| 融合与输出 | 768→512 投影、512维8头 cross-attention、LayerNorm、512→256→1 MLP | 1,576,961 |
| **合计** | 图像＋任务文本→sigmoid；BCE训练 | **122,382,017，约122M** |
| **默认可训练** | 冻结T5，训练视觉与融合/输出 | **12,753,473，约12.75M** |

参数量是按公开默认结构计算，未在GPU实例化计数；T5共享词嵌入按独立参数计算，不重复计数。T5配置来自[同一发布包](https://huggingface.co/WorldArena/WorldArena2.0/blob/main/reward_model/t5-base/config.json)。已核 `adjust_bottle/full_weights.pt` 与 `resnet_rm.pth` 是同一LFS对象，各588,338,878 bytes，约588 MB；不能用这个文件名推断它只有一个11M的ResNet。通过131,072-byte HTTP Range静态检查，checkpoint键确实包含视觉、T5 encoder、文本投影、cross-attention和reward head；未反序列化执行checkpoint。[发布目录](https://huggingface.co/WorldArena/WorldArena2.0/tree/main/reward_model)

现有训练链的边界很具体：

- [`train_reward_model.py`](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/examples/reward/train_reward_model.py) 和 [`reward_training.yaml`](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/examples/reward/config/reward_training.yaml) 已有训练、验证、保存；默认是**不读文本的ResNet**，micro32/global64、LR1e-4、BCE分类。
- [`RewardBinaryDataset`](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/data/datasets/reward_model.py) 只返回image/label；[`FSDPRewardWorker`](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/workers/reward/reward_worker.py) 调用 `model(images, labels)`，没有传instruction。
- T5 RM已经注册且有BCE `forward`。**只改model_type能走无文本分支，不能因此称复现了发布的文本条件RM。**保留同结构训练，需要给dataset/collate/forward补任务字符串；单任务也可固定传同一句指令。

因此后续选择清楚：想最少改动，单任务使用纯ResNet成功分类器；想沿用本次发布RM的结构，补文本传递再微调。二者均远小于4B–8B的VLM，当前无需为奖励训练另建大模型体系。

## 数据到底要多少，标签怎么来

**最直接的论文线索是3000条混合轨迹。**WorldArena附录B.1称，用两档不同水平策略和原生planner采集total 3000条轨迹，同时训练WM与RM；另有1000条专家轨迹用于策略SFT。论文没有在这段拆分两个任务各多少、正负各多少，也没给单任务RM的最少数据量。因此不能把“3000”说成每个新任务的必需量，也不能把“1000专家SFT”当RM数据配方。[论文附录B.1](https://arxiv.org/html/2605.17912v1#A2.SS1)

公开[预处理脚本](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/examples/reward/preprocess_reward_dataset.py)给了更可操作的线索：读逐帧success，按整episode切80/20训练验证，默认fail:success为2:1；CLI默认使用全部帧（内部函数默认5帧，不能混称为CLI默认）。它针对RLinf采集的`.pkl`，不是RoboTwin专家HDF5的现成转换器。

对新任务建议如下，属于我们的实施方案，非论文宣称的样本复杂度：

1. **先用已有目标任务的成功和失败rollout。**记录主图、任务文本、当前帧的原生成功标签，以及episode ID；额外保留失败末帧和“差一点成功”的帧。纯专家数据能提供大量未完成的中间帧，但通常缺“看起来快成功、最后却失败”的负例。
2. **由任务判定自动标注。**例如摆瓶子，真的满足任务几何条件才标1；未满足则0。不能把成功episode每一帧都标1，也不能统一把最后一帧当成功。需要区分“当前成功”和“曾经成功”的事件锁存。
3. **按episode切分，保留全部原始验证分布。**训练可沿用2:1负正采样；公开脚本连验证集也采样，我们做阈值判断时应另保留未重采样的验证集，避免误判率因采样变得好看。
4. **是否加数据看误判种类。**对新布局、近成功失败和WM生成图分别看假阳性/漏判，不用单一accuracy替代。若主图看不见关键接触事件，再加腕图或短历史；这不是当前摆瓶子smoke的前置工程。

RoboTwin公开超过100k轨迹是跨任务/配置的数据资源，**不是单任务RM需求量**，也不能假定所有发布轨迹自带逐帧失败标签。[官方数据说明](https://robotwin-platform.github.io/doc/usage/collect-data.html)

## 专项检索：哪些候选对RoboTwin有用

表中“可选”指代码/资产可获得，不代表已在我们OpenDW生成视频上验证。

| 候选 | RoboTwin相关证据与公开资产 | 输出真正代表什么 | 对我们的排序 |
|---|---|---|---|
| **WorldArena任务RM** | `adjust_bottle`、`click_bell`权重，约122M默认结构；当前先选摆瓶子 | 单帧＋文本成功分数 | **首个smoke直接复用**。仅这两任务的资产已核，不当50任务通用模型 |
| **Robometer-4B** | 官方完整训练/推理/HTTP、LoRA指南和权重；LeRobot也有封装 | **分别输出逐帧progress与success**，还有轨迹偏好 | **通用success候选优先考察**；本轮未核到RoboTwin专用checkpoint。官方模型HF统计约4.45B参数，约8.9GB BF16权重 |
| **RynnValue-8B** | 官方公开代码与权重；训练混合数据包含27,414条RoboTwin episode | 剩余完成时间＋文本 `Success: Yes/No` | 有RoboTwin训练覆盖的备选。要判断done应检查Success输出，不能把剩余时间接近0自动等同成功；HF参数约9.57B，名称8B不等于整模型精确参数数 |
| **PRIMO-R1-7B** | 公开RoboTwin clean/randomized训练子集、ID/OOD评测、SFT/RL代码和7B权重 | 初始图＋视频＋当前图＋指令→0–100进度与推理 | RoboTwin过程评分候选；100是进度语义，不是已经校准的done概率；公开Robotwin媒体子集约12.5GB |
| **RoboReward-4B/8B** | 权重、视频推理与数据；主要基于OXE/RoboArena真实机器人 | 终局1–5级完成程度 | 终局复核或微调备选；5级可作为成功候选，但不是即用的RoboTwin阈值 |
| **Robo-Dopamine / GRM** | 多视角训练/推理、4B/8B Preview；2.0论文有RLinf＋RoboTwin实验 | 前后画面相对进展；2.0加历史/OOD处理 | 后续稠密奖励。2.0训练仍叠加原环境任务奖励，不能直接用progress替掉成功判定 |
| **WorldLoop / TOPReward** | RLinf公开接线，冻结Qwen3-VL-8B | 从VLM的True token概率构造完成评分 | 接口近；现有阈值/实验属LIBERO，需要RoboTwin输入与阈值适配 |
| **CreFlow组合约束判定** | 论文直接评8个RoboTwin任务的生成视频，组合时序/对象约束与VLM判定 | 视频是否同时满足任务约束，二元reward＋错误定位 | 方法值得借鉴；本轮未核到可直接下载的完整官方判定实现/模型包，不排进当前smoke |
| **EVA/IDM平滑奖励** | RoboTwin公开IDM、训练代码与权重 | 逆推出的动作是否平滑、符合机械约束 | 适合动作/视频质量辅助，**不是任务成功分类器** |

直接来源与细节：

- **Robometer**：[官方仓库](https://github.com/robometer/robometer/tree/352d160389daa964788de1ec933d1925f3a6de4f)、[作者权重](https://huggingface.co/robometer/Robometer-4B)、[LeRobot success输出配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rewards/robometer/configuration_robometer.py)。[LoRA指南](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/FINETUNE_ROBOMETER.md)已有新数据微调例子，并强调成功cutoff必须匹配轨迹真实结束位置；这正是能直接借用的适配工具，不能拿示例1000更新当所有任务标准预算。
- **RynnValue**：[官方代码](https://github.com/alibaba-damo-academy/RynnValue)、[权重与训练来源表](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B)。其HTTP奖励路径主要服务value/reward；Success文本作为独立判定还要接解析接口，不能默认已有RLinf done接线。
- **PRIMO-R1**：[官方代码和发布清单](https://github.com/10-OASIS-01/PRIMO-R1/tree/f746e1f83af001337dabffafa948129353ac962e)、[7B权重](https://huggingface.co/LeonOverload/PRIMO-R1-7B)、[论文](https://arxiv.org/abs/2603.15600)。公开进度数据值得复用，但先确认目标任务、相机和标签含义；全量PRIMO条数不等于新任务所需条数。
- **RoboReward**：[4B模型卡](https://huggingface.co/teetone/RoboReward-4B)、[论文](https://arxiv.org/abs/2601.00675)。反事实换任务指令和时间截短用于构造负例/近成功，是可借鉴的数据增强。
- **GRM**：[官方源码](https://github.com/FlagOpen/Robo-Dopamine)、[Preview权重](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview)、[2.0论文](https://arxiv.org/html/2608.15680v1)。公开Preview与8月论文完整新模型的对应关系仍需另核；不把论文RoboTwin成功率归到任一Preview。
- **WorldLoop**：[RLinf官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/resources/blog/worldloop_topreward.html)。这条路线已接RLinf，但不因它“无需训练RM”就认定可免任务验证。
- **CreFlow**：[论文§4–5](https://arxiv.org/html/2605.14274v1)。实验中用Claude生成一次任务监视逻辑，用Gemini判断属性状态；它训练的是视频生成模型，奖励逻辑可借鉴但整套方法不是我们的π0.5-GRPO。[EVA官方实现](https://github.com/RobbinW/EVA)则明确以速度/加速度/jerk等动作可执行性作奖励。

还有一个更窄的公开线索：[blocks_ranking_rgb生成轨迹与判定证据](https://huggingface.co/datasets/VincentNi/robotwin-blocks-ranking-rgb-rollouts)。作者发布160条生成视频及SAM3跟踪、IDM动作、FK/顺序/不重复等判定结果；它提供可研究的单任务规则例子，不能据此称已有覆盖RoboTwin的通用success服务，也不能把模型自己打的pass率当真实环境成功率。

## 结论与接线边界

**现在已经清晰可做：**摆瓶子使用发布RM，保持主图＋同任务文本和官方预处理，分数/预测成功/termination/truncation分开记录；接入OpenDW后直接并行smoke，不需要等待新奖励研究。

**以后换任务：**优先目标任务的小分类器；若希望跨任务复用，先比较Robometer的success头和RynnValue的Success输出。PRIMO、GRM、RISE/Feat2Go等progress/value路线留作稠密奖励，不把它们和成功检测混为同一组件。

**仍需经验回答的点只有效果：**已发布摆瓶子RM在OpenDW画面上是否可靠、新任务需要多少混合数据、加入腕图/历史能改善多少。这些不妨碍接口smoke；本轮没有凭空声称模型已通过我们的任务精度验收。

源码/HF元数据摘录：[E盘证据清单](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/robotwin-reward-targeted-20261003/source_manifest.json)。模型参数为源码算术结果；HF文件体积为2026-10-03只读API结果；没有完整权重、训练数据或服务器访问凭据写入本目录。
