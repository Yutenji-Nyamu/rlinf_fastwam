# WoVR / PACE 与 LIBERO 四套件：公开实现可用性

2026-10-03，只读公开来源与本地已归档证据；未 SSH，未改服务器、训练或旧审计。本文回答“公开了什么、当前离可执行闭环差什么”，不承诺复现收益或工期。

## 直接结论

1. **WoVR 的冻结 Wan＋KIR＋GRPO 阶段已在 RLinf；完整 PACE 没有现成一键例程。** 最新官方 Wan 文档仍明确说明例程冻结世界模型，co-evolution 需自行实现。公开了训练 action-conditioned Wan 的官方 DiffSynth fork，因此不是“只能看论文、没有训练代码”。[Wan 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html)、[官方 WM 训练入口](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/examples/wanvideo/model_training/train_rlinf.py)
2. **我们已跑通的是 π0.5＋冻结 Goal Wan 的适配路径；尚未跑通 π0.5 的 PACE。** 目前没有官方 Wan＋π0.5 YAML 或其收益基准；现有官方完整结果以 OpenVLA-OFT 为策略。当前动作专家训练、单图、H10/C8 与原策略双图/H10/C5 的区别仍需保留，不能只更新 WM 就假定观测契约已解决。[本地已核对的接口/配置](../audit_20261003/official_method_audit.md)、[官方维护者说明](https://github.com/RLinf/RLinf/issues/831#issuecomment-4102834159)
3. **目前是 LIBERO-Goal，全套 10 个任务。** Spatial、Object 有配套 Wan、奖励、reset 数据及 OFT YAML；Long 有 Wan/VAE 和奖励权重，但缺现成 reset 数据与 RLinf loader/config 接线，支持程度不同。不能拿 Goal 权重只改 `task_suite_name` 就认为完成迁移。

## 固定版本与最新版本

|层级|2026-10-03 核验结果|含义|
|---|---|---|
|当前固定 RLinf|`d34d4c320d08cb982de034aa9a011f08dc0fa217`|此前运行与诊断依据|
|最新 RLinf main|`c70606f08cdca259b8dec03d4430926b5b8fac9d`，提交时间 2026-10-02 08:56 UTC|本轮重新读取 GitHub commit/tree API|
|WanBackend 对比|两版本文件 SHA256 均为 `72271ececc307211196a4c3f7a595f48cea21e8d3440e502a6dd7a893c058d14`|升级 main 本身不增加 PACE、π0.5 Wan 配方或 Long 奖励 loader|
|官方 DiffSynth fork|最新 `2a2e05fa1f724828b243f272540989b19a6e54f8`，2026-02-13|与已用 DiffSynth 固定版本相同；WM 训练代码已经在该版本内|
|最新官方 Wan 配置|Goal、Spatial、Object 三套 OFT 训练 YAML|未发现 Wan Long、Wan π0.5 或 PACE 编排 YAML|

直接来源：[最新 commit](https://github.com/RLinf/RLinf/commit/c70606f08cdca259b8dec03d4430926b5b8fac9d)、[最新 WanBackend](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/world_model/backend/wan.py)、[固定 WanBackend](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/backend/wan.py)、[DiffSynth commit](https://github.com/RLinf/diffsynth-studio/commit/2a2e05fa1f724828b243f272540989b19a6e54f8)。

2026-09-21 的官方 WorldLoop/TOPReward 博客也明确：当前冻结 Wan，只复现静态世界模型阶段，PACE 属于后续工作。其奖励实现当时在 PR #1554 审阅，固定实验分支独立给出；博客存在不代表当前 `WanBackend.load_reward_model` 已自动支持该奖励。该博客讨论的奖励与复现现象由本轮主研究另行汇总。[官方博客 §03、§05–06](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/resources/blog/worldloop_topreward.html)

## PACE 方法已明确，哪些工程已公开

论文的 PACE 是低频更新：用基准策略真实环境轨迹训练 WM Base，在其中更新策略，再采集进化策略轨迹形成 WM Evo，继续策略优化。LIBERO 实验每套 1500 条初始轨迹＋1000 条更新轨迹，只做一次 WM 刷新；策略是 one-trajectory SFT 的 OpenVLA-OFT。Spatial 去掉 PACE 为 71.0%，完整 81.5%。这支持分布对齐的重要性，不保证强 π0.5 也有相同增益。[WoVR §4.3、§5.2、§6.2](https://arxiv.org/html/2602.13977v1)

|环节|已有公开组件|仍需补齐|
|---|---|---|
|策略在真实 LIBERO 采集|RLinf `CollectEpisode`，pickle 或 LeRobot 导出|选择匹配的相机、实际执行动作、任务/种子；保留成功和失败；逐步而非仅 chunk 末尾帧的采集契约要核准|
|动作条件 WM 微调|DiffSynth `train_rlinf.py`，full DiT 与 LoRA 参数入口|本地路径、适配训练/验证数据、冻结/可训练参数、checkpoint 导出与加载核验|
|基础奖励训练|RLinf 通用 ResNet reward trainer、数据预处理入口|现用 Goal task-embedding 架构/预处理/权重格式兼容；不是任意 trainer 输出直接覆盖 `.pth`|
|KIR/reset|已有 Wan KIR 消费代码与三套 reset 资产|若刷新起点数据，需要从新轨迹构造相同 schema 和关键帧；不是把整段 WM 训练数据直接传入 reset|
|阶段调度|现有策略训练与模型路径配置|收集→训练 WM→质量验证→载入 WM Evo→策略续训的明确 checkpoint/数据版本交接|
|外部评价|现有真实 LIBERO 评估|在同一输入/执行协议下比较更新前后；WM 内 reward 不代替真实成功率|

采集来源：[官方数据采集文档](https://rlinf.readthedocs.io/en/latest/rst_source/guides/data_collection.html)、[CollectEpisode 源码](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/wrappers/collect_episode.py)。

### 世界模型训练代码的具体边界

官方实际可找到的启动文件是 `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf.sh`；README 指向旧路径，且保留训练时手动调整 action MLP 初始化的备注。因此应沿实际代码准备，不能照 README 一行命令宣称现成可跑。[实际启动脚本](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf.sh)、[README](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/README.md)

其训练数据 loader 为 `RLinfNpyDataset`，路径是 `train_data/<step>/<seed>/{rgb.npy,actions.npy}` 及对应 `val_data`；时间、并行环境为数组前两维，当前通常需 HWC 图像及 7D 动作。源码只对作者特定旧路径作 CHW→HWC，且包含 13 帧特例及长轨迹采样常数。适配时必须明确图像布局、参考帧、连续片段和动作对应时刻。文档注释里的 CHW 不能替代对实际分支的检查。当前 HF 那些单文件 reset dict npy 是另一种格式，不能直接用于此 trainer。[loader 源码 L364–464](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/trainers/utils.py#L364-L464)

最新 `CollectEpisode` 的 pickle 保存 `observations` 长度 N+1、`actions/rewards/infos` 长度 N；需转换为上述 WM 数组并核对 o_t、a_t、o_(t+1)，不能简单堆叠同索引。训练/验证应按完整轨迹切分，避免同轨迹片段跨集合。此为工程判断，尚未执行转换。

### 奖励再训练支持程度

最新 RLinf 有 `examples/reward/train_reward_model.py`，训练数据 `.pt` 为 `images/labels/metadata`。示例使用无 task embedding 的 `ResNetRewardModel`、224×224、ImageNet normalization；现用 Goal WM 奖励是 DiffSynth `TaskEmbedResnetRewModel`，输入归一化与模块命名也不同。**“有通用 reward trainer”不等于“已有现用 Goal reward 的直接再训配方”。** 应先复用或补兼容 trainer/导出；仅因做 PACE 也不必自动重训 reward，论文定义的核心更新是动力学，是否更新判分器取决于新数据上的误判证据。

来源：[reward 配置](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/reward/config/reward_training.yaml)、[reward 数据格式](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/data/datasets/reward_model.py)、[通用 ResNet 架构](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/reward/resnet_reward_model.py)、[现用 task-embedding 架构](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/models/reward_model.py#L326)。

### 卡数和 LoRA：已知与未知

- **官方 action-WM 全参数示例：单节点 8 个 GPU 进程，BF16，DeepSpeed ZeRO2，DiT 可训练，VAE 不在可训练列表，256×256、13 帧、LR 1e-5。** 作者给出的 `num_epochs=100000` 和绝对路径是研究脚本参数，不能当作我们的推荐预算；8 卡是发布示例配置，不能推导“至少 8 卡才可训练”。未找到该脚本的明确卡型、单卡显存下限、峰值或训练时长承诺。[full 脚本](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf.sh)、[Accelerate 配置](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/examples/wanvideo/model_training/full/accelerate_config_14B.yaml)
- **LoRA 在 action-WM trainer 的参数和模块注入中已支持。** 但仓库现成 `lora/Wan2.2-TI2V-5B.sh` 使用普通视频 `train.py`、CSV 数据和单图条件，不是已验证的 π0.5 PACE 配方。需要确定 action MLP 是否训练、adapter 如何导出/合并以及 RLinf 如何加载；最新 WanBackend 只有基础模型和 VAE 加载，没有现成 LoRA 配置入口。未核到 action-WM LoRA 官方最低卡数或显存保证。[LoRA 支持代码](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/trainers/utils.py#L545-L579)、[普通视频 LoRA 例程](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh)
- 当前四卡能做冻结 Wan＋策略 RL，不等于四卡能容纳 full WM 训练；LoRA 可能降低优化器/梯度内存，但实际激活峰值仍需独立测量。此为资源机制判断，不是已实测容量。

## 四套件定义与公开资产

VLA/WoVR 常说的四套件是以下四个，每个 10 任务。原始 LIBERO 的总数是 130，其中 LIBERO-100 再拆 90 个预训练任务与 10 个下游任务；**Long=libero_10，不是 100 个任务，也不是第 5 套新 benchmark。** [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)、[官方目录定义](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/code_structure.html)

|套件 / 配置名|主要变化|Wan / VAE|奖励权重|reset npy|RLinf 原生 Wan 配方|
|---|---|---|---|---|---|
|Goal / `libero_goal`，当前使用|相同对象与空间关系，不同任务目标/动作行为|有|`taskemb_resnet_rm.pth`|742，已核 496 initial＋246 KIR|有，OFT|
|Spatial / `libero_spatial`|相同对象集中的碗，依据位置/空间关系区分目标|有|`resnet_rm.pth`|692|有，OFT|
|Object / `libero_object`|pick-and-place 中目标对象类型变化|有|`resnet_rm.pth`|692|有，OFT|
|Long / `libero_10`|对象、布局、目标组合变化的长时序任务|有|6 个分组 task-embedding `.pth`|该模型仓库 0 个|无现成 Long Wan YAML，需接线|

任务变化定义：[LIBERO 原论文](https://arxiv.org/abs/2306.03310)、[论文官方 OpenReview PDF](https://openreview.net/pdf?id=5zigAK7HPY)。

HF 资产在本轮 API 核验时全部公开且未 gated，逐套固定 revision：

- [Goal@bd395971](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Goal/tree/bd395971c3467de3dd19e7e6c7562af48a2894a6)：747 个文件，742 个 npy。
- [Spatial@cfffe6b8](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Spatial/tree/cfffe6b86babb4a9464d4ba53339e6fa31c3e0fb)：697 个文件，692 个 npy。
- [Object@ad71e787](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Object/tree/ad71e7875837b6b705d75450c9ebcc8f97d02cf5)：697 个文件，692 个 npy。
- [Long@d543adce](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Long/tree/d543adce87ac220f06f03c962417b6481d4340de)：10 个文件，含 WM、VAE、6 组 reward 与一份评价文本，无 README、无 reset npy。

Long 的 `RewModel_TaskEmbed_only4libero10` 类确实在 DiffSynth reward_model.py；最新 RLinf `load_reward_model` 只接受 `ResnetRewModel` / `TaskEmbedResnetRewModel`，且只给后者传任务指令。因此 Long 还要接分组奖励加载/任务路由，并准备兼容 reset/KIR。不能声称“Long 权重没开源”，也不能声称“四套都一条现成命令能跑”。[Long 类](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/models/reward_model.py#L172)、[最新 loader](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/world_model/backend/wan.py#L62-L79)

空间、物体和 Goal 各自训练的 Wan 与 RM 应成套切换；reset 数据、任务描述映射、策略归一化统计与官方评估环境也需一致。Goal task embedding 只是该套件内任务条件，不代表已学其它套件的成功判据。WorldLoop 博客提到的“仅改 task_suite_name”特指通用 Qwen 奖励跨套件配置，不授权沿用 Goal 的 Wan/RM/reset 整包。

## 如果要做，现实分阶段路线

1. **先明确目标是验证方法还是保留 π0.5 做研究。** 最接近公开复现的是 OFT＋现成套件冻结例程；沿 π0.5 继续则保留已有训练链，补齐单图/动作执行协议和同协议真实基线，归类为 π0.5 的方法扩展。
2. **先补可复用的数据出口。** 用选定 π0.5 在同一真实 LIBERO 套件采集包含成败的逐步 RGB/动作/终止/任务标识数据，生成 WM train/val 与独立 reset/KIR；先证明单个样本时序、布局与奖励标签正确。当前数百个 reset npy 不能替代完整动态训练数据。
3. **对齐 WM 后再完整闭环。** 从目标套件 action-Wan 权重继续训练可作为工程起点；全参数路径最接近官方发布脚本，LoRA 路径需额外验证 adapter/action 层和导出。先取得对 π0.5 轨迹的独立质量证据，再用该 WM 更新策略。
4. **做一次 PACE 迭代。** 第一阶段策略取得可用检查点后，在真实 LIBERO 采集新轨迹、得到 WM Evo、载入新 WM 进行第二阶段策略优化，继续同口径外部评估。论文的 1500＋1000 是复现预算参考；若从已有公共 WM 继续训练或减少预算，应明确是新实验协议。
5. **换套件分开处理。** Spatial/Object 的基础资产接入成本较低；Long 先补 reset 和分组奖励接口。不要同时更换策略输入、套件、reward 方案与 WM 更新方式，否则很难解释效果。

这里给的是准备次序，尚未执行。公开材料不足以可靠报“几天完成”或指定少于 8 卡的 full-WM 容量；工程工作主要在数据/接口/阶段编排，效果工作主要在策略分布对齐与真实评估。

## 取证文件

公开 API 元数据、完整轻量源码及 SHA256 清单：

`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-discussion-20261003/`

读取脚本：`.tmp/wm_audit_20261003/discussion_sources.py`、`discussion_code_sources.py`。模型权重和视频未下载；没有运行任何训练或追加实验。
