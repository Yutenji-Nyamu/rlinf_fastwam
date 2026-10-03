# WMRL 奖励与成功判定：现状、候选和推荐

2026-10-03。仅研究：读取官方代码、论文与 HF 元数据；未连接服务器、下载权重或修改实验。当前 LIBERO 实配引用工作区最近已核验记录，本轮重新确认其公开配置与奖励源码，未声称刷新运行现场。

## 先分清四件事

| 名称 | 通俗含义 | 不能直接等同 |
|---|---|---|
| success | 任务要求是否已经满足，通常是布尔值 | 看起来接近目标、分数上升 |
| reward | 交给优化器的学习分数；可由 success、进展或差分构造 | 环境真实成功率 |
| value / progress | value 通常估计未来累计回报；progress 描述已完成多少。论文有时把 progress 网络命名为 value | 当前已成功、必须终止 |
| done | 本回合结束；应区分任务终止 termination 与步数到限 truncation | 所有 done 都是成功 |

WM 提供“动作以后画面怎样变化”，RM 提供“这些变化是否有益”。当前 GRPO 不需要 critic；接入 PPO 的 value/GAE 或 RISE 的 progress advantage 会改变方法，不是单纯换 RM。

## 我们的 LIBERO Goal 当前用什么

最近已核验实配是官方 `taskemb_resnet_rm.pth`，45,379,978 bytes，`TaskEmbedResnetRewModel`。它看**单帧主相机图＋十个 Goal 任务的离散 ID**：ResNet 图像特征拼接 task embedding，再经 MLP、sigmoid、`round`，返回 0/1。任务指令先查固定字符串表，不是自由语言理解器。Spatial 官方配置使用不接任务文本的 `ResnetRewModel/resnet_rm.pth`；不能把 WorldLoop 对 Spatial 非任务 RM 的描述套到我们 Goal。

- [Goal 配置 L53–55](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/embodiment/config/env/wan_libero_goal.yaml#L53-L55)；[Spatial 配置 L54–56](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/embodiment/config/env/wan_libero_spatial.yaml#L54-L56)。
- [任务映射与网络](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/models/reward_model.py#L326-L397)；[发布权重](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Goal/tree/bd395971c3467de3dd19e7e6c7562af48a2894a6)。

环境再做 `r_t = s_t - s_(t-1)`，当前 π0.5 配方系数 1；chunk 任一帧判成功，则在 chunk 末记录 termination。因此分数先 0→1→0 时，可能记“曾成功”却累计差分回报为 0。训练 success_once 是 RM 对生成图的判断；原生 LIBERO 才是实际任务成功评估。更多本项目已核证据见 [机制记录](../WAN_GOAL_MECHANISM_AND_PARALLEL_20261001.md)。发布 checkpoint 的完整训练划分与逐任务误判率未公开；本轮没有证明推盘子退化就是 RM 误判。

## RoboTwin 原生判定与 WorldArena 默认到底是什么

原生 RoboTwin 有物理状态可查。例如 `adjust_bottle` 检查瓶子功能点的位置和高度；`click_bell` 检查夹爪与铃铛接触点附近的位置并锁存事件。视频 WM 只有生成 RGB，不能直接运行这些判据。它们适合给真实仿真轨迹生成训练标签，也仍是最终 native evaluation 的依据。[adjust_bottle L63–67](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/adjust_bottle.py#L63-L67)；[click_bell L71–83](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/click_bell.py#L71-L83)

| WorldArena 任务 | 实际默认奖励 | 输出与限制 |
|---|---|---|
| adjust_bottle | `RoboTwinT5CrossAttn`：ResNet18 图像 tokens＋冻结 T5 文本编码＋cross-attention | 返回连续 sigmoid 分数；训练用 BCE。可借鉴并复用对应任务资产，但不是任意 RoboTwin 任务通用 checkpoint |
| click_bell | `LPIPSLastFrameRewardModel`，VGG，`one_minus` | `1-clamp(LPIPS(生成图,所选数据片段末帧),0,1)`；衡量外观接近程度，不能证明按铃接触事件发生 |

代码固定 `WorldArena2/WorldArena-2.0@5978ce5c81e55b8c8358f4f5966a13ce385ff155`：

- [adjust_bottle 默认 L49–52](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/examples/embodiment/config/env/wan_robotwin_adjust_bottle.yaml#L49-L52)；[click_bell 默认 L49–56](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/examples/embodiment/config/env/wan_robotwin_click_bell.yaml#L49-L56)。
- [T5 RM 训练/推理 L159–197](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/models/embodiment/reward/robotwin_reward_model.py#L159-L197)；[LPIPS 公式 L77–104](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/models/embodiment/reward/lpips_reward_model.py#L77-L104)。

两任务环境默认对分数逐帧差分；原始分数的 chunk 最大值达到 0.9 就估计成功，在 chunk 末 done。env 子配置系数为 1，正式 π0.5 训练 YAML 覆盖为 5；阈值作用于原始分数，不能把它和乘 5 后的奖励混为一谈。[阈值与差分 L228–258](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/envs/world_model/world_model_wan_env.py#L228-L258)；[正式训练配置](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/examples/embodiment/config/wan_robotwin_adjust_bottle_grpo_openpi_pi05.yaml#L64-L91)。

[HF 发布目录](https://huggingface.co/WorldArena/WorldArena2.0/tree/main/reward_model)实核有 adjust_bottle 的 `full_weights.pt`、`resnet_rm.pth` 和 click_bell 的 `resnet_rm.pth`，各 588,338,878 bytes，并有 T5 资产。**click_bell 有分类器文件，不代表默认 YAML 使用它**。本轮未实际加载验证 checkpoint 与任意本地依赖兼容。

OpenDW 固定 [`e33befa8005a1585e0140dbf464566e90bc79aa1` README L21–27](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/README.md#L21-L27)仍明确 Value Expert 留待后续发布；当前不应依赖它自动给奖励。

## 候选如何选择

| 路线 | 标签 / 已有资产 | 最适合我们的用途 | 仍需解决 |
|---|---|---|---|
| 同任务 success classifier | WorldArena adjust_bottle 已有资产；新任务可用 native success 给真轨迹逐帧/短窗标注，BCE 训练 | **最小接线首选**，保留 RLinf＋GRPO，明确成功与终止 | 覆盖失败、接触临界帧、错误对象及 WM 画面；单图不可观察事件可能需短历史/腕图 |
| WorldLoop / TOPReward | 冻结 Qwen3-VL-8B，无专用 RM 训练；RLinf 公共分支已有接口 | 现有流程最接近的通用 VLM 对照或离线复核 | 额外显存/推理时间、视频预处理、RoboTwin 阈值校准；已有结果不保证跨域可靠 |
| RoboReward 4B/8B | 公开模型；视频＋指令→终局 1–5 完成程度 | 可下载的通用终局判断候选，也适合筛疑似假成功 | 5 分是类别而非校准成功概率；原生 RoboTwin/生成视频仍要核验 |
| Robo-Dopamine / GRM | 公开多视角推理、数据与微调代码；3B/8B、2.0-4B/8B Preview | 想引入多视角、细粒度进展时的候选 | 主要输出相对进展；不能直接充当 success/done，完整新论文权重对应关系未确认 |
| RISE progress value | π0.5 初始化的三视角网络；成功演示进度监督＋成功/失败 TD | 稀疏奖励长期不够时，研究稠密且识别失败的价值信号 | 新任务数据与训练；其 advantage-conditioned 方法不等于现用 GRPO |
| VLA-MBPO BAGEL RM＋critic | 已公开 LIBERO Yes/No 服务、标签转换、短分支 PPO | 借鉴任务判定和短分支 bootstrap 的分工 | BAGEL 较重、RoboTwin 训练标签与模型需适配；critic 不能替代成功检测 |
| LPIPS 末帧相似度 | 不需要任务 RM 训练，但需参考末帧 | WorldArena click_bell 的原配方对照、可视化辅助 | 相似度可受背景/姿态主导，不建议作为新任务唯一成功依据 |

### TOPReward / WorldLoop 的实际边界

TOPReward 从冻结 VLM token 概率提取信号；WorldLoop 接的是 Qwen3-VL-8B 的 `P(" True")`，16 帧窗口、阈值 0.46 后转 0/1，复用 Wan/GRPO。阈值来自其 LIBERO 实验，不能直接宣称适用于 RoboTwin。现有 Spatial 结果早于视频采样修复，Object 在修复后能运行但没有超出评估噪声的增益。[RLinf 官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/resources/blog/worldloop_topreward.html)

固定实现 [`ZJLi2013/RLinf@0c7a3d5f8d43a9c2ae242236a5699fb9d0a7243b`](https://github.com/ZJLi2013/RLinf/blob/0c7a3d5f8d43a9c2ae242236a5699fb9d0a7243b/rlinf/models/embodiment/reward/topreward_vlm_model.py#L144-L196)已处理 video metadata 和 episode history 清空。它降低了训练专用 RM 的门槛，不等于不需要任务验证；增加一个 8B RM 的成本也不能按小 ResNet 估算。TOPReward 官方源固定 [`4877a0ee5098cbec18485125466631b1dcc4a573`](https://github.com/TOPReward/TOPReward/tree/4877a0ee5098cbec18485125466631b1dcc4a573)。

### 两个额外的公开通用机器人 RM

**RoboReward：**[4B 模型卡](https://huggingface.co/teetone/RoboReward-4B)定义从“无进展”到“完整完成”的 1–5 级终局评分，使用 Qwen3-VL 视频推理接口。HF API 实核 4B 两个 safetensors 分片合计约 8.88 GB；[8B](https://huggingface.co/teetone/RoboReward-8B)四片约 17.53 GB，均不是仅有模型名。其训练数据包含对成功轨迹的反事实任务重标和截短，补充失败/近成功样本。[论文](https://arxiv.org/abs/2601.00675)没有给我们这套 OpenDW→RoboTwin 生成图的即用成功阈值。本轮核到了权重、推理说明和数据，未确认独立的完整官方训练仓库。

**Robo-Dopamine / GRM：**[官方代码固定 `2c714abca66329f8e2113587226bc9556726f6c2`](https://github.com/FlagOpen/Robo-Dopamine/tree/2c714abca66329f8e2113587226bc9556726f6c2)公开三视角、前后状态比较、数据生成和微调；[2.0-4B Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview)有 9,653,631,848-byte 权重，支持有/无目标参考图，但分数表达进退程度。

需额外区分 2026-08 的 [Robo-Dopamine 2.0 论文](https://arxiv.org/html/2608.15680v1)：它确有 RLinf＋RoboTwin＋OpenVLA-OFT＋GRPO，但使用 `r = 原环境任务奖励 + γΦ(next) - Φ(now)`。因此它证明进展 shaping 可以帮助 RL，**没有证明可以去掉任务成功判定**。已核官方仓库 head 仍为 5 月；未核到公开 Preview 和该论文完整 OOD/history 模型的对应回执，不能把论文 RoboTwin 分数算作 Preview 的已验证表现。

### VLA-MBPO 与 RISE 不能混称为“现成稠密成功奖励”

VLA-MBPO 公共 BAGEL 服务看主图与任务，生成 Yes/No；这是 success classifier 的一种实现。其转换脚本只把“成功回合的最后一帧”标 Yes，其他标 No；迁移到成功后继续录制的 RoboTwin 数据会错标，需要改成真实逐时刻成功语义。PPO critic 另外估计未来回报，在短分支末尾 bootstrap，不能在没有 RM 时自动知道任务要求。[判定服务 L287–329](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/world_model/world_model_client/world_model_inference_server.py#L287-L329)；[标签 L260–298](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/world_model/scripts/convert_libero_data.py#L260-L298)。

RISE 的 progress value 用专家轨迹 `t/T` 暖启动，再加入成功/失败轨迹 TD，使模型不只学习“时间越长分越高”；其候选 chunk 优势取预测未来 value 均值减当前 value。实现已公开，但 RoboTwin 仍要准备目标任务数据。[论文 §III-A](https://arxiv.org/html/2602.11075v2#S3.SS1)；[公开 value 训练入口](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/docs/offline_learning.md#L19-L41)。

## 当前组装方案的建议

**保留现有 RLinf＋π0.5＋GRPO，OpenDW 负责三视角动态，奖励作为独立模块。**若首任务是 adjust_bottle，优先核验并复用 WorldArena 对应 RM；若换任意其他 RoboTwin 任务，优先用其 native success 标签训练轻量任务分类器。相机需要三视角并不自动要求奖励网络也上三视角，先依据成功条件是否在当前输入可观察来选。

通用候选按用途选：**WorldLoop/TOPReward** 最接近现有 RLinf 接口；**RoboReward** 适合作终局复核；**Robo-Dopamine/RISE** 适合后续稠密进展研究。不用等待 OpenDW Value Expert，也不需要把所有候选同时装进训练。

标签质量比“奖励看起来平滑”更重要：应区分已完成与接近完成，保留失败/偏离专家的轨迹，按 episode 划分训练与验证，避免相邻帧泄漏。对 WM 生成图，错误高分可能推动策略利用画面漏洞；缺失成功标签则可能让真正有效的动作得不到奖励。用真实任务成功标准检查误判，尤其 false positive，并保持 `reward_score / success_pred / termination / truncation` 分别记录，才能知道收益来自哪里。这些是候选筛选原则；本轮没有启动新实验或改变现用奖励。
