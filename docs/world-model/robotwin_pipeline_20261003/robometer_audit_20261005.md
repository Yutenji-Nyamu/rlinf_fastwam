# Robometer：接口、证据与 OpenDW WMRL 接入审查

2026-10-05。按用户要求重新核官方论文、代码、模型卡及 LeRobot 实现。本文件是研究和下一步测试方案；本轮没有下载 Robometer 权重、连接服务器、占用 GPU 或切换训练。现场与 Rynn 数值实验以本轮执行文档为准。

## 结论

Robometer 值得作为下一个通用奖励候选：**同一前向直接输出进展和成功概率，支持 batch，不必生成 Yes/No 文字**。现成代码、权重、HTTP 服务、LoRA 流程都有；接入工程距离比从头训练奖励模型近。我们现成的 K8 历史、请求身份、批队列、同步卸载和 GRPO 接点可以复用。

当前最大的未知是 **RoboTwin `adjust_bottle` / `click_bell` 和 OpenDW 生成图上的判断能力**。本次没有找到专用 RoboTwin checkpoint、这两个任务的精度报告或 OpenDW 验证，不能因为它有独立 success 头就认为一定比 Rynn 准。推荐先用同一组原生专家、失败、前缀和生成轨迹做离线对照。

## 它具体算什么

| 项目 | 核验结论 | 对我们的含义 |
|---|---|---|
| 基座 | Qwen3-VL-4B-Instruct；已发布实配 BF16、非量化 | 不是小型单图分类器；仍需单独测显存/延迟 |
| 输入 | 任务文字＋按时间排序的 RGB 图像；公开权重训练 K8、multi-image、每帧 progress token | 可直接用现有主相机 K8 历史；不需要 π0.5 动作、14D state 或目标图 |
| `progress_pred` | 每帧的 10-bin 分类分布取期望，范围 0–1 | 表示模型估计进展，不是成功概率，也不是经过校准的剩余时间 |
| `success_probs` | 独立 success MLP 的 logit 经 sigmoid，每帧一个数 | 可做完成候选，但阈值需本任务验证；0.5 只是官方示例默认值 |
| preference | 两段同任务视频一起输入，另一个头判断哪段更好 | 首次接 GRPO 不需要成对比较所有轨迹，可先用单轨迹输出 |
| 历史语义 | 因果注意力下，每个 progress token 读取本帧及此前帧 | 一次 K8 forward 可得 8 个时刻的分数，不是为每帧调用 8 次 |

实配：[HF config.yaml](https://huggingface.co/robometer/Robometer-4B/blob/main/config.yaml)。张量接口与解码：[官方模型](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/models/rbm.py)、[服务 `compute_batch_outputs`](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/evals/eval_server.py#L470-L539)。本地数组入口接受 `uint8[T,H,W,C]`：[示例](https://github.com/robometer/robometer/blob/main/scripts/example_inference_local.py#L28-L117)。

`use_multi_image` 在这里表示多张时间帧，不能理解成已经训练好 RoboTwin 同刻三视角融合。我们先保持主相机；三视角拼图或独立视角投票属于新协议。主相机末段出画的问题仍然存在，需同时评可见成功前缀和终局。

## batch 是现成的，但需要自己限制大小

官方 `ProgressSample(trajectory=Trajectory(frames=..., task=...))` 可以组成列表交给 collator；HTTP 有 `/evaluate_batch` 和 `/evaluate_batch_npy`，返回每条的 progress/success 序列。服务按空闲 GPU **分配整个请求**，每张 GPU 放一份模型；不是把一次大请求自动切碎，也不是张量并行。

本次源码中 `process_batch_helper` 将收到的全部样本一次 collate/forward，不能仅设置配置 `batch_size=16` 就假定 512 条自动拆成 32 批。我们应显式 B4/B8/B16 分片，验顺序、帧数和最后一帧索引，记录实际 batch。原服务也没有现成 `/offload`；要复用我们同步卸载接口，防止与 actor 重叠。

来源：[数据对象](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/data/dataset_types.py#L13-L62)、[官方 batch 服务](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/evals/eval_server.py#L164-L297)、[多卡调度](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/evals/eval_server.py#L300-L452)。

`use_frame_steps=true` 会把 T 帧扩成 T 个历史前缀，每个前缀抽 4 帧，显著增加 forward 样本数。首轮建议 **K8、`use_frame_steps=false`**，一条历史作为一个样本；取其末槽表示当前时刻。这是我们的初始方案，不代表所有历史采样方式等价。

按我们计划的 N64/G8/R8＝512 轨迹、C32/max384 上限计算：

| 评分方式 | 每轮最多逻辑视频样本 | B16 时最多 GPU forward |
|---|---:|---:|
| 每条终局一次 | 512 | 32 |
| 每个 C32 块一次 | 512×12＝6144 | 384 |
| 每个生成帧另建一个前缀评分 | 512×12×8＝49152 | 3072 |

这是整除情况下的计算，未计重试；提前终止会减少调用量。N、WM batch、RM batch 和 actor microbatch 是四个不同参数。Robometer 一次 K8 返回 8 个分数，不要求第三行的昂贵用法。

## 成本：能支持的结论到哪里

- 10月3日已保存的官方 HF 元数据对应 `beef63bc914c5c189329d49c6d712d96d632aa34`：4,450,286,861 个 BF16 参数，纯权重约 **8.29GiB**。这是旧锁定快照的权重下限，不是总显存或本轮实测；当前网页仍列 BF16，但本轮 API 未成功刷新参数清单。
- 直接数值头没有自回归生成开销；因此从结构上比“同等输入后继续生成文本”少一段计算，不能据此承诺比已测 Rynn B16 快多少。
- 作者 DreamZero 案例报告 Robometer forward 约 0.6–1 秒，但未给我们 H100/K8/B16 合同的基准，不能直接乘出正式每轮耗时。
- 默认上游依赖为 Python3.10、Torch2.8、Transformers≥4.57、Unsloth 等；应独立环境，避免覆盖现有 RLinf/Rynn 环境。默认 BF16 先建立参考，量化精度和装卸另外测。

来源：[成本快照说明](general_reward_integration_20261003.md)、[依赖](https://github.com/robometer/robometer/blob/main/pyproject.toml#L20-L102)、[论文附录G](https://arxiv.org/html/2603.02115v2#A7)。

## 可靠性：有哪些实证，哪些没有

训练不仅看成功示范，还用成功/失败的相对比较、错任务文字和倒放等数据；独立成功头有直接监督。论文列出 RBM-1M 超过百万条、21种机器人，数据清单包括 OXE、LIBERO、MetaWorld 等；全文及这次所核公开清单**未出现 RoboTwin**，所以我们按“未证实训练覆盖”处理。

最相关的实验是 DreamZero：对世界模型生成的候选视频用 **progress 排序**，再执行相应动作。这支持“奖励模型可用于生成视频”的可行性，但只是一个任务、10次试验，不是 OpenDW/RoboTwin/GRPO 的复现，也没有验证我们需要的 success 阈值。论文在线 RL 与失败检测另有实证，不能把相关性、排序或失败 F1 当成通用 success 准确率。[论文方法、附录A/E/G](https://arxiv.org/html/2603.02115v2)、[作者项目结果](https://robometer.github.io/)。

对我们仍要防的具体情况：瓶子出画后漏判、铃铛短接触不可见、生成图幻觉被当成功、仅凭轨迹变长分数上升、静止/倒放也能赚奖励。不同 reward 模型即使一致，也不能替代原生 `check_success()` 的真实环境评估。

## 现成接入路径与建议

| 路线 | 已有 | 还差 | 建议 |
|---|---|---|---|
| 上游官方独立服务 | 完整权重加载、collator、batch、raw success/progress | 隔离环境、冻结版本、我们的小批队列/身份回执/卸载 | 首次离线对照优先，最接近作者接口 |
| LeRobot 官方移植 | 纯推理模型、processor、batch `compute_reward` | 依赖与权重转换版本对齐；验证与上游逐样本一致 | 可参考简化部署，不必为了奖励移植整个训练框架 |
| RoboTwin LoRA | 官方转换/训练/LoRA教程 | 带成败及成功时刻的真实任务数据、验证集、少量生成难例 | 零样本失效时再考虑，不先训练 |

**LeRobot 的一个重要接口细节：**当前 `compute_reward(reward_output="success")` 最终返回阈值化的 0/1，虽然模块文档称 success probability。首轮审查需要原始概率，应读取上游 `success_probs` 或 LeRobot 低层 decode，而不是拿已经二值化的数据画校准曲线。[LeRobot 实际返回](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rewards/robometer/modeling_robometer.py#L227-L260)、[默认配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rewards/robometer/configuration_robometer.py#L44-L67)。

若以后微调，官方教程提供单卡 LoRA 例子；示例的 1000 updates 和论文 RoboFAC 数据量不能当作我们任务的最低需要量。先用我们原生环境的成功判据给轨迹明确成功时刻，保留任务/种子独立验证；不要默认每个专家视频最后一帧才成功。官方专门要求按数据源校准 success cutoff。[LoRA教程](https://github.com/robometer/robometer/blob/main/FINETUNE_ROBOMETER.md#2-lora-fine-tuning)。

## 可执行的小规模候选试验

1. 用已有原生初始/专家、可见成功前缀、真实失败、静止与倒放，对同一主相机 K8 分别取 progress 和 success 概率；生成轨迹作为独立一类，不能自己充当标签真值。
2. B4→B8→B16 少量测吞吐、峰值及批间输出一致性。先成功加载一个锁定版本，再测批量，不同时改帧数/量化/奖励形式。
3. 判断 success 能否分清终局与未完成，progress 能否给真正改善排序且不奖励静止。需要原始逐样本分数，不以“模型有输出”作为通过。
4. 合格再进入当前 RLinf scorer 接点：先选独立 success 或数值进展的一种用法，明确 reward、done 和 G8 过滤语义；同步装卸保持。**换成连续 progress 后，原先为二元成功设计的组过滤上下限不能原样解释。**

我们已经做完 Rynn 接口工程，Robometer 的新增量主要是“替换模型加载/预处理/解码＋小样本校准”；不会自动解决世界模型的动力学偏差，也不需要先重写 π0.5、OpenDW 或 GRPO。

## 版本与研究边界

本次主查官方 main、arXiv v2（2026-05-13）与现有锁定源码快照 `352d160389daa964788de1ec933d1925f3a6de4f`；固定版本 `dataset_types.py` 已重新在线读到。HF/Git API 当前 SHA 刷新失败，故不把 main 视作该旧SHA的别名；实际部署前再冻结下载版本与哈希。本文件没有宣称已跑 Robometer smoke。
