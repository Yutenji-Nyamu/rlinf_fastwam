# Fast-WAM GRPO 学习链源码复核与候选排序

2026-09-05；只读分析，不修改训练、模型、优化器、环境或运行预算，不执行新评估。

## 1. 结论与本轮证据范围

**当前已完成预算里，Fast-WAM 未展示令人信服的持续增益；但还不能判定 GRPO 接入失效，更不能用总 loss 小判定它没有更新。**本轮重新核对服务器实际模型桥、旧/新 logprob、通用 GRPO、mask、优化器与同步路径，没有发现“完全漏更新、错用最终动作重算旧 transition、把模型 H32 与执行 C24 错接”这类确定性错误。

新增一个比泛泛猜 LR 更具体的高优先候选：**Fast 的整个 action expert 以 BF16 原始参数进入 AdamW，并用 BF16 累积每 rank 256 个微批次。**本轮只读抽查 6 个张量发现，40 次 Adam 更新后，输入投影仍有 99.25% 元素与 SFT 完全相同，首尾层 q 投影仍有约 94—95% 元素相同，但这些元素的 Adam 一阶矩都非零。参数更新高度稀疏已经实测；BF16 舍入限制是强嫌疑，仍不能把它直接定为无增益唯一原因。它和“MB2 的梯度被错误除小了 16 倍”是两回事，后者从源码和日志可以排除。

现场数据为 [本轮完整快照](discussion-audit-20260905/live_snapshot.json)，取样开始 11:23:45 CST；源码保存于 `discussion-audit-20260905/sources/`。Fast HEAD `f3a1689e5ceb`、Sidney `f50e235c5ab1`、旧 π0 `ab0988498a04`，所查 working tree 均 clean。Fast 的模型桥、advantages/losses、FSDP manager/strategy 和真实 embodied actor 均与本地对应源码 LF 文本一致；RoboTwin wrapper 多出已知 Env-local scene-fence 接入。只引用这轮实际源，不把旧 review 的结论当新现场。

## 2. 曲线能说明什么

| 同一口径 | 当前 Fast / stapler / noOIDN | Sidney π0.5 / pillbottle | 深圳旧 π0 / adjust_bottle |
|---|---:|---:|---:|
| 现场最新完整 step | 17 | driver 96，TB 已落盘到 95 | 96，历史 run 已结束 |
| 首 10 步训练均值 | 23.13% | 41.68% | 78.36% |
| 最后 10 步训练均值 | 26.33% | 67.42% | 94.73% |
| 当前可用 fixed32 | Step5/10/15 = 11/11/13 | 最新 Step95 = 22；Step70 最好 26 | 最新 Step95 = 31 |
| 前 17 步 logged KL 均值 | 0.001907 | 0.022519 | 0.021701 |
| 前 17 步 logged clip 均值 | 1.42% | 5.74% | 7.56% |
| 前 17 步 logged ratio 均值 | 0.85394 | 1.00535 | 1.00199 |
| 前 17 步裁剪前 grad norm 均值 | 9.246 | 20.164 | 30.684 |

Fast fixed 从 11 到 13 多成功 2 条，是小幅改善迹象；3 个评估点和没有同协议 Step0 的条件下，不能说已稳定学会。前后十步还重叠 Step8—10，不是两个独立实验。另一个旧 OIDN/128 档跑到 33 步，首末十步基本持平，是此前独立失败运行的结果，不能与本 run 接起来画成 50 步。

Sidney 与旧 π0 的任务和 SFT 起点不同。表中绝对成功率用于描述各自实验，不能推出 Fast 模型较弱或 GRPO 算法不同。Fast 先前官方 standalone 的 11/16 也没有对齐这次 fixed32、192-action、无 OIDN 协议，不能当本 run 的 Step0。

## 3. 模型桥与 GRPO 主链核对

| 边界 | 当前源码行为 | 判断 |
|---|---|---|
| 图像/状态/动作 | 官方 head + 左右腕拼图、PIL resize、14D absolute qpos、自身 stats；输出只反归一化一次 | 未发现相机顺序、norm 或重复 decode 的明显错接 |
| rollout 随机量 | 对完整逻辑 batch 一次生成独立初始噪声、独立 SDE epsilon；单个随机 denoise index 广播 batch，之后才按 MB2 前向 | **纠正 09-04 旧 review：当前 Fast 不是逐 sample 独立抽 denoise index**；此处与 OpenPI 同为 batch 共享 index |
| stochastic transition | 10 次去噪中恰好一次 Flow-SDE，其余 ODE；保存真实 chain[k]、chain[k+1]、k 和当时 old logprob | 不是重新抽噪声伪造 behavior density |
| actor replay | 复用保存的图像/text/proprio，重建冻结 conditioning，仅对原 k 的真实 next latent 计算新密度 | 旧/新概率条件与 transition 对齐 |
| H/C 边界 | 模型预测 H32、保存完整 H32 chain；执行 C24，old/new logprob 都裁 C24；通用 loss 对 24×14 维求和 | 结构自洽；它是单去噪 transition 的 surrogate，不是最终物理动作分布的精确边际密度 |
| GRPO | 同 seed 的 G8，各轨迹稀疏成功 reward；组内减均值/除标准差，广播给 query | 与两个 π 系列共用 `compute_grpo_advantages` |
| mask/filter | done 后 mask；平均组奖励在 [0.1,0.9] 才保留 | 二元奖励 G8 时全成/全败组剔除；需要实际保留组数 |
| 优化/同步 | 每个 global batch 累积完才 step；两 update epochs；每外层 step 进行 actor→rollout 同步；实际学习率恒定 | 没有发现只前向不 backward、不 optimizer.step 或缺少周期同步 |
| 可训练集合 | `mot.mixtures.action.*`；video、VAE、T5、proprio 冻结；canonical module 去掉重复注册 alias | 824 是参数张量名称数，不是 824 个参数元素 |

来源：[Fast policy](discussion-audit-20260905/sources/fastwam/rlinf/models/embodiment/fastwam/fastwam_policy.py)、[Flow-SDE/replay](discussion-audit-20260905/sources/fastwam/rlinf/models/embodiment/fastwam/fastwam_rl.py)、[builder](discussion-audit-20260905/sources/fastwam/rlinf/models/embodiment/fastwam/builder.py)、[实际 embodied actor](discussion-audit-20260905/sources/fastwam/rlinf/workers/actor/embodied_fsdp_actor_worker.py)、[advantages](discussion-audit-20260905/sources/fastwam/rlinf/algorithms/advantages.py)。

既有 08-31 真实模型验收记录为：B1 wrapper/official 动作最大差 0；同权重 stochastic behavior/replay logprob 最大差 0；action expert 梯度有限且非零；DCP fresh/save/reload/update 通过。见 [原始实施账本](CURRENT_IMPLEMENTATION_AND_SMOKE_LEDGER_20260831.md)。这降低了核心桥错接嫌疑，但不替代当前长程训练期间每次同步和更新幅度的观测。

## 4. MB2 与总 loss：已能排除的误判

当前每 rank 累积次数为 `1024 / 2 GPU / MB2 = 256`；两个 π 系列为 `1024 / 2 / MB32 = 16`。实际 loss 是每个微批次先求均值再除累积次数，累积完统一 step。固定 global batch 下，代数上得到相同的全局均值，不会因为 MB32 改成 MB2 就天然把梯度缩小 16 倍。

**记录的 `actor/total_loss` 则正是在除累积次数之后写入的微批次 loss。**现场前 17 步均值：

| 模型 | policy_loss | total_loss | 比值 |
|---|---:|---:|---:|
| Fast | 0.000128278 | 0.000000501084 | 256 |
| Sidney | 0.000446486 | 0.0000279054 | 16 |
| 旧 π0 | 0.000188824 | 0.0000118015 | 16 |

这里 total_loss 的差异主要含记录口径；此外正负 advantage 本来就会抵消，policy_loss 接近 0 也不等于没有梯度。Fast grad norm 非零，logged KL/clip 也非零。

KL/clip/ratio 还有另一层口径问题：每个微批次先按有效 mask 求指标均值，全 mask 微批次返回 0，actor 再对所有微批次算术平均。MB2 更容易碰到全 mask 微批次，因此 logged ratio 0.85 不能直接解释为有效样本平均概率掉到 0.85。现有日志缺少逐微批次有效计数，无法离线准确反算 valid 加权统计。

实际优化 loss 不走上述指标平均，而是 `mean(value / (loss_mask_sum / horizon) * mask)`，再按累积数缩放；mask_sum 在组过滤前计算，因此被过滤的组确实不提供梯度，但不会在每个小批次按有效个数随机重新归一化。源码：[masked_mean 与 masked_mean_ratio](discussion-audit-20260905/sources/fastwam/rlinf/utils/utils.py)、[compute_loss_mask](discussion-audit-20260905/sources/fastwam/rlinf/utils/metric_utils.py)、[losses](discussion-audit-20260905/sources/fastwam/rlinf/algorithms/losses.py)。

## 5. 新增具体候选：参数与累积精度

Fast resolved 为 `actor.model.precision=bf16`、FSDP2 `param_dtype=bf16/reduce_dtype=bf16`。builder 将该 dtype 传入 official model；锁定 official `ActionDiT.from_pretrained` 对整个 action expert 执行 `.to(dtype=torch_dtype)`。通用 FSDP manager 将可训练参数直接交给 AdamW，没有单独建立 FP32 master copy。

11:27:28 CST 的 [checkpoint 元数据只读证据](discussion-audit-20260905/native_and_dtype_metadata.json) 已确认实际保存 dtype，而非只读 YAML 猜测：

| 可训练模型参数 | BF16 张量 / 元素 | FP32 张量 / 元素 | Adam 状态 |
|---|---:|---:|---|
| Fast Step10 DCP | 824 / 1,020,900,366 | 0 / 0 | exp_avg 与 exp_avg_sq 各 824 个 BF16 张量；824 个 FP32 张量全是 step 计数 |
| 旧 π0 Step90 DCP | 127 / 574,750,720 | 47 / 3,286,048 | moments 跟随相应参数 dtype；FP32 为 norms、action/state/time 投影 |
| Sidney Step90 local shard | 127 / 574,750,720 | 82 / 118,671,392 | rank0 的 moments 为 19 个 BF16、41 个 FP32 flatten shards；元素各为全模型对应集合一半 |

上表模型元素按元数据 global shape 与可训练 FQN 统计；Sidney optimizer 是 rank0 的真实局部分片，不能拿其张量条目数与另外两者未 flatten 的张量数量直接相比。Sidney FP32 部分包含 AdaRMSNorm 的 dense conditioning 以及 action/time 投影。**两个 π 系列也有大量 BF16 可训练参数，不能把这个差异写成 Fast BF16 vs π 全 FP32。**

PyTorch 2.11 的 [MixedPrecisionPolicy 官方合同](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy) 明确：优化器使用 sharded 参数的原始 dtype；关闭梯度同步时，累积采用 reduce_dtype。因此不能把“计算用 BF16”自动理解成“优化器主权重仍是 FP32”。

机制上，BF16 的参数刻度可能大于 LR 5e-6 所产生的实际 Adam 增量。例如权重在约 0.01 一带，相邻 BF16 可表示数间隔约 6.1e-5；比半个刻度还小的增量可在写回时被舍去。这只是数值示例，不是说所有权重、所有步都不动。Adam moments、梯度方向、权重大小都会影响实际增量；非零梯度本身不能排除这种情况。

OpenPI builder 走不同路径：模型先构造、加载权重，再对 `paligemma_with_expert` 的 selected params 转 BF16；action/time/state 投影不在整模型 BF16 转换范围内。Fast 与 π 系列“配置里都写混合精度”不代表相同的可训练参数 dtype。实际 checkpoint 元数据已经确认差别；本项列为高优先候选，**尚不是已证实的无增益根因**。

另外，Fast 每 rank 256 次 BF16 梯度累积，π 系列 16 次且 reduce dtype 走其实际模型/框架合同。数学上相同的均值在有限精度里未必同样准确。这也是可测候选，不能由此直接宣布累积错了，更不应未测就加 LR 或扩大显存占用。

### 5.1 既有 Step10 与原始 SFT 的有限参数读数

11:32:28 CST 读取 6 个 action 张量，共 6,588,416 元素；原始 SFT 使用 mmap，DCP 按 storage offset 仅读取 39,596,538 bytes。没有运行模型、优化器或 GPU，没有恢复训练。来源：[完整参数读数](discussion-audit-20260905/weight_delta_probe.json)。

| 模块 | SFT→Step10 有变化的元素 | 相对 L2 增量 | Adam step | 一阶矩非零比例 |
|---|---:|---:|---:|---:|
| action_encoder.weight | 0.7533% | 1.73e-5 | 40 | 100% |
| head.weight | 10.7561% | 1.088e-3 | 40 | 100% |
| time_embedding.0.weight | 28.9898% | 1.707e-3 | 40 | 100% |
| blocks.0.self_attn.q.weight | 4.7271% | 2.704e-4 | 40 | 100% |
| blocks.29.self_attn.q.weight | 5.7888% | 3.245e-4 | 40 | 100% |
| blocks.0.modulation | 1.8555% | 2.868e-5 | 40 | 100% |

比较基线是 SFT 转为训练实际 BF16 dtype 后的数值，避免把 dtype 转换当成学习变化；本次这些原始张量本身也已是 BF16。40 次 optimizer step、全部非零 moments 以及一部分真实权重变化，直接排除了“完全没训练”。同时，首尾 q 投影和输入投影的大多数权重并未产生任何净变化，结合没有 FP32 master 的已知合同，使舍入限制比仅看小 KL 更有依据。

这仍有明确边界：抽样不是完整 10.21 亿参数；净变化小可能包括方向抵消、原本应保持稳定的权重；一阶矩非零不意味着每个坐标都该有大增量。当前证据不量化 BF16 对最终成功率的因果贡献，也不证明用 FP32 一定能解决任务学习。最合适的表述是“已看到实际更新稀疏，精度成为优先检查/修正候选”，而非“已经找到了唯一根因”。

## 6. 参数、预算与可塑性的实际差别

| 维度 | 当前 Fast | Sidney π0.5 | 深圳旧 π0 |
|---|---|---|---|
| 任务 | move_stapler_pad | move_pillbottle_pad | adjust_bottle |
| 每轮轨迹 / G8 组 | 256 / 32 | 256 / 32 | 256 / 32 |
| 环境×rollout | 32×8 | 64×4 | 64×4 |
| H / 执行 C / M / horizon | 32 / 24 / 10 / 192 | 50 / 50 / 10 / 200 | 50 / 50 / 4 / 200 |
| 每轮 query records | 2048 | 1024 | 1024 |
| GB / MB / epochs | 1024 / 2 / 2 | 1024 / 32 / 2 | 1024 / 32 / 2 |
| 每轮 optimizer calls | 4 | 2 | 2 |
| LR / SDE noise | 5e-6 / 0.3 | 5e-6 / 0.5 | 5.6e-6 / 0.5 |
| denoise 时间网格 | shift5 | 线性 | 线性 |
| 内部动作表示 | absolute，Fast stats | absolute，Sidney stats | delta，旧 π0 stats |
| 可训练范围 | action expert；proprio encoder 冻结 | action expert + action/time 投影 | action expert + action/time/state 投影 |
| 观测 | 3-camera composite + VAE，none | 3-camera OpenPI，OIDN | 3-camera OpenPI，OIDN |

来源：[Fast resolved](discussion-audit-20260905/fastwam_resolved.yaml)、[Sidney resolved](discussion-audit-20260905/sidney_resolved.yaml)、[旧 π0 resolved](discussion-audit-20260905/pi0_resolved.yaml)。两个 π 系列对比的内部 delta/state/norm 证据见 [原专项复核 §5—7](../../rlinf-shenzhen-multitask-pi05/evidence/PI05_VS_PI0_LEARNING_REVIEW_20260904.md)。

**当前 Fast 已不是“每轮只有 π0 一半 GRPO 组”的旧 128 档。**但 17 步只有 4352 轨迹、68 optimizer calls；旧 128 档 33 步为 4224 轨迹、66 calls。两者是不同 run、不同观测协议，总预算不能拼接，也不能把 current 17 vs Sidney 95 当同预算比较。

Fast 的每条轨迹有 8 个 query，这 8 个 query 共享一次成败结果，不是 8 个独立 outcome。G8 组全部成功或全部失败时没有相对优劣信号；总体成功率 25% 也可能由一些组全成、其余组全败形成，不能由总体成功率反推有效组充足。

Fast 冻结 proprio 是已有接入合同，并非遗漏 requires_grad：其 conditioning 处于 no_grad，若未来决定训练 proprio，必须同时重接 replay conditioning 的梯度，不能只翻一个开关。192 与 200 只相差 8 个动作，本身优先级不高；真正值得看的时间尺度是 C24/8 次闭环、任务失败阶段和末段能否完成放置/松爪。

## 7. 我会如何排序

这里排序的是下一步排查价值，不是已经计算出的根因概率。

1. **真实有效策略更新：参数精度与实际权重增量。**本轮已经确认全 BF16 action/Adam 状态，并在既有 Step10 看见更新高度稀疏。下一次获批短测优先核对单次 step 的实际增量与表示刻度，讨论保留 FP32 trainable master、BF16 计算及 FP32 累积的最小实现；不先盲调 LR，也不直接把冻结的整套 video/VAE/T5 转 FP32。
2. **有效 G8 混合成败组与 valid 加权策略变化。**现有低 KL/clip 是信号，但需同时知道有多少组真正提供 advantage、多少 query 被保留。若主要是同类失败导致全零 advantage，单纯增 update 次数不会产生新信息。
3. **探索与任务闭环。**Fast 的 shift5/M10/noise0.3、动作 stats、C24 和冻结 proprio 与 π 系列不同；应先看现存 fixed 视频中失败在抓取、搬运、放置还是超时。相同 noise 数字不表示相同物理扰动，不宜直接照搬 0.5。
4. **原始 SFT 在当前观测/评估协议下的起点。**关 OIDN 会改变 RGB，可能使起点下降；旧 OIDN run 本来也没有稳定上升，故不能把全部问题归因于关闭降噪。需要同一协议的 SFT 与当前 checkpoint 才能隔开“基础能力弱”和“RL 没改善”。
5. **长程预算受事故截断。**这是已经发生的限制，但修复 reset/fence 只能恢复采样预算，不会自动解决学习。当前 17 步不足以判定上限，也不足以把此前所有失效都归咎于没跑够。

低证据、当前不优先的说法：GRPO advantage 算反、MB2 把梯度错误缩小 16 倍、824 个可训练元素、无梯度/无同步、当前每轮少一半组、scene-fence 补丁直接改了训练 loss。现有源与记录不支持这些判断。

## 8. 本轮只读账本

- 完整读取根 PROJECT_CONTEXT/HANDOFF、近期窗口入口与 Fast current SSOT；旧学习 review 只作定位线索。
- 阅读本地模型桥及完整现场保存的对应源码，按 LF 文本核对一致性；重新读实际 embodied actor，避免错用 reasoning actor。
- 从服务器完整快照解析三 run 前 17 步优化统计与完整成功率；未额外运行 rollout/eval 或加载整模型。
- 从 DCP metadata 与 FakeTensorMode 保存的 Sidney local-shard 元数据分类统计可训练参数和 Adam moments；按 FQN/shape 汇总，没有在本地加载 checkpoint storage。
- 读取 root 汇总的服务器有限参数比较：6 张量、6.59M 元素，DCP 39.6MB、SFT mmap；仅现有数值比较，不运行模型、optimizer 或 GPU。
- 核对 PyTorch 2.11 官方 FSDP2 dtype/accumulation 合同；引用只用于界定底层机制，不代替当前实验因果证据。
- 明确纠正旧 review 的 denoise index 粒度；写本证据，没有更改根 HANDOFF、服务器代码、运行进程或共享依赖。
