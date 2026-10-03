# Robometer / RynnValue：通用奖励的接入与成本

2026-10-03。官方源码、权重元数据及本地历史回执专项核验。本文保留后续讨论方案；**当前 OpenDW＋π0.5-GRPO 继续用 WorldArena `adjust_bottle` RM，不因本次研究自动接入大模型。** GPU smoke / 正式训练状态由本轮 owner 回执记录，本文不宣布其完成。

## 1. 先讲结论

- **Robometer 更直接回答“现在成功了吗”**：任务文字＋多帧 RGB，一次前向输出各帧进展和成功概率，不需要生成一段回答。
- **RynnValue 主要回答“估计还要多久”**：数值头输出剩余秒数；取得 `Success: Yes/No` 要走同一模型的语言生成分支，不能把“剩余秒数很小”直接当成功。
- 两者可以做候选通用 RM，但都还没有证明能可靠判断我们的 OpenDW 生成图。优先用于保存轨迹的离线核验；若以后需要在线补充，先做每回合终局的一次二级判断。
- 我们已有 RynnValue-8B 权重、独立环境和数值服务的历史实现。本轮已核资产目录仍在；**已有数值服务没有接入 Success 输出，也不是现成的 OpenDW 成功判定器。**

来源：[Robometer 模型卡](https://huggingface.co/robometer/Robometer-4B)、[RynnValue 语言与价值接口](https://github.com/alibaba-damo-academy/RynnValue/blob/353a3caa732107aff374ef410069a78c4b830d6d/README.md)。

## 2. Success、进展和剩余秒数分别怎么来

| 模型 / 路径 | 输入 | 输出 | 成功判定的接点 |
|---|---|---|---|
| Robometer-4B | 任务文字＋按时间排序的 RGB 帧 | `progress_pred`、`success_probs`；另外有双轨迹 preference | `success_logits` 经一次 sigmoid，得到各输入帧成功概率；末槽可作终局候选分数 |
| RynnValue 数值路径 | 任务文字＋机器人/相机说明＋采样帧 | `value.pred_value`：每槽剩余秒数；另有相对时间头 | 没有可直接搬用的成功阈值。`v_before-v_after` 是进展信号，不是 success |
| RynnValue Analysis 路径 | 同一多模态提示 | 生成 Video Description / Match / Success | 解析 `Success: Yes/No`；缺失、截断或解析失败必须单独记 unknown，不能冒充预测失败或成功 |

Robometer 的 progress 和 success 是不同头，不能用“progress 接近1”代替 success。官方本地可视化示例的 success 阈值是0.5，**它不是我们摆瓶子任务已经校准的阈值**，也不能直接继承 WorldArena 的0.9。[官方前向与 sigmoid](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/evals/eval_server.py)、[本地推理示例](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/scripts/example_inference_local.py)

RynnValue 的官方 demo 先完成数值 forward，再对终局前缀调用 `generate(do_sample=False, max_new_tokens=128, use_cache=True)`；文本不是另外装一个大语言模型，而是**同一8B模型多走一次自回归生成**。如果以后只要 Success，可以另封装仅调用 generate 的入口；这不是我们现有数值服务已经实现的功能。其 `Match: Yes` 只表示画面/任务匹配，也不等于任务成功。[官方 demo](https://github.com/alibaba-damo-academy/RynnValue/blob/353a3caa732107aff374ef410069a78c4b830d6d/rynn_infer/inference.py)、[官方两条评分路径](https://github.com/alibaba-damo-academy/RynnValue/blob/353a3caa732107aff374ef410069a78c4b830d6d/robometer/robometer/evals/baselines/rynnvalue.py)

## 3. 一次调用到底看几帧

**Robometer：8帧是已发布权重的训练配置，不是“一帧一调用”的硬接口。** 它支持把一段采样视频作为一个样本，单次前向得到多个帧分数。官方服务也有 `use_frame_steps`：把每个时刻的历史前缀分别抽成4帧，再取各前缀末槽。后者会显著增加样本数。官方本地脚本允许读取最多512帧，这只是取帧上限，不证明一次512帧有合适显存或可靠精度。[权重实配](https://huggingface.co/robometer/Robometer-4B/blob/beef63bc914c5c189329d49c6d712d96d632aa34/config.yaml)

**RynnValue：每个评分样本可给K帧，但“给K帧”不等于只算K个时刻。** 官方 demo 默认对每个视频时刻都建立一个历史前缀，每个前缀再抽64帧；`num_steps=0` 表示所有视频时刻。官方 benchmark 的 episode 路径默认K8、推理batch4；我们已验证的旧服务是K4、batch1，每个query边界独立评分。直接照搬64帧默认值会远比当前小服务昂贵。

首个候选通用核验合同建议：主相机、同一真实任务文字、保留起点和末帧、从当前完整历史均匀抽8帧。早期不足8帧时记录明确的重复/补齐方式。不要把左腕、右腕依次插进时间轴，伪装成时间前进；三视角拼图或多图提示属于新的输入方案，需另验证。

## 4. C32、384动作、G8的调用量

以下是**候选384动作预算的成本计算，不代表已部署配置**：R1、每块32动作、每块8张未来图，因此每回合12块、96张未来图；N表示全部轨迹数，已经包含G8。N8是一组，N32是四组，不再额外乘8。假设没有提前终止、不计初态评分。

| 评分方式 | 每条轨迹的评分样本数 | N8 | N32 | 若每样本抽8帧，N8 / N32的图像呈现次数 |
|---|---:|---:|---:|---:|
| 每张生成帧都独立构建历史前缀 | 96 | 768 | 3,072 | 6,144 / 24,576 |
| 每个C32块末构建一次历史前缀 | 12 | 96 | 384 | 768 / 3,072 |
| 每个回合终局构建一次前缀 | 1 | 8 | 32 | 64 / 256 |

这些是**逻辑样本数，不等于HTTP次数，也不一定等于GPU forward次数**。若每次GPU处理B个样本，理论forward数为 `ceil(样本数/B)`；比如B8时，上表N8对应96/12/1次，N32对应384/48/4次。B8能否容纳取决于分辨率、token和实际峰值，当前未测。

另有一个重要方案：**Robometer把每块新生成的8帧一起送入，一次就输出这8帧的success/progress**。这样仍得到每条96个帧分数，但仅12个8帧样本，而非96个前缀样本。它与“每个时刻都包含从起点到现在的历史”不是同一输入协议，应在日志里分开。

若RynnValue在每个样本同时要数值和语言Success，照官方两阶段实现，还要对应768/3,072、96/384或8/32次Analysis生成；每次最多128个新token，实际可提前结束。**只做终局Success不需要先把全部96个时刻的value都算完。**

R增加时上述数量乘R；提前成功会减少数量。G8共享起点仅允许去重完全相同的初态输入，分叉后的8条轨迹不能共用评分。训练U轮可复用冻结RM在同一批轨迹上的评分，不应重新推理。

## 5. 显存：区分纯权重下限与真实运行峰值

2026-10-03从官方HF API读取已发布BF16 tensor数量，按“数量×2字节”计算：

| 已发布模型 | 全部已保存参数 | BF16纯权重 | 本项目运行证据 |
|---|---:|---:|---|
| Robometer-4B | 4,450,286,861 | 8.901GB = **8.29GiB** | 尚无本任务GPU实测 |
| RynnValue-4B | 5,144,979,456 | 10.290GB = **9.58GiB** | 尚无本任务GPU实测 |
| RynnValue-8B | 9,574,961,904 | 19.150GB = **17.84GiB** | 9/9旧任务：BF16/B1/K4的Torch allocated峰值约18.02GiB |

模型名称4B/8B没有完整表达视觉/价值头的额外参数。表内下限不包含激活、KV、CUDA上下文、临时张量和allocator余量，不能当作部署显存预算。[Robometer元数据](https://huggingface.co/api/models/robometer/Robometer-4B)、[RynnValue-4B元数据](https://huggingface.co/api/models/Alibaba-DAMO-Academy/RynnValue-4B)、[RynnValue-8B元数据](https://huggingface.co/api/models/Alibaba-DAMO-Academy/RynnValue-8B)

9/9记录中，RynnValue8B一条成功episode的4个边界（每边界4帧，B1）耗时12.09/12.60秒，结束回CPU后Torch allocated约32MiB。该耗时包括当时服务的请求、评分和装卸流程，**不能除以4当成今天新增一帧的固定耗时，也不能外推文本生成耗时**。它证明我们曾真实跑通过该服务，不能证明与OpenDW/actor同时常驻一定放得下。[本地部署与smoke实录](../../rlinf-robotwin-pi0-online-bc/BC_RYNNVALUE_RABC_IMPLEMENTATION_20260909.md)

## 6. batch、cache、量化：有什么，什么还没证实

| 优化 | 已有支持 | 我们需要保持的边界 |
|---|---|---|
| 批推理 | Robometer collator与HTTP batch；RynnValue episode前缀批处理、CPU预处理流水线 | 先显式切小batch。Robometer服务接收整个请求列表，不要以为只设配置里的batch_size就一定替你拆好；Rynn示例的torch.cat适合同长度/同形状前缀，异长度指令需正确padding |
| 结果缓存 | 我们旧client已按episode、模型revision、输入哈希保存数值 | 同一输入重放和U次更新可复用；新生成分叉不能复用别条结果；增加相机/采样/Success提示后必须改cache身份 |
| KV缓存 | RynnValue文本生成内部明确use_cache=True | 这是一次生成内部的token复用。当前前缀评分会重新选历史帧，不能直接把前一次KV当新前缀；Robometer公开接口也未提供跨块KV复用合同 |
| 原生量化接口 | Robometer setup有bitsandbytes 8-bit与Unsloth 4-bit分支 | 已发布权重实配为BF16、quantization=false；有开关不等于公开reward checkpoint重载、success校准和CPU卸载链已通过 |
| 第三方NF4 | OpenRAL已发布Robometer NF4包，并自报8帧推理约3.56GB峰值 | 是OpenRAL作者的运行报告，不是Robometer论文复现或我们的OpenDW测试。没有核到RynnValue官方量化后的奖励精度/显存配方 |

代码依据：[Robometer batch服务](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/evals/eval_server.py)、[官方量化分支](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/utils/setup_utils.py)、[RynnValue batch实现](https://github.com/alibaba-damo-academy/RynnValue/blob/353a3caa732107aff374ef410069a78c4b830d6d/robometer/robometer/evals/baselines/rynnvalue.py)、[OpenRAL NF4模型卡与其验证范围](https://huggingface.co/OpenRAL/rskill-robometer_4b-any-general-nf4)。

因此成本路线应先减少评分频率、固定少量采样帧，再试batch；量化是后续备选，当前不把“理论4-bit省显存”写成已验证方案。冻结推理一般不保存训练optimizer，但输入长度仍会明显影响临时显存。

## 7. 我们已有的RynnValue资产和真实差距

本地可复用实现：

- `local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/utils/rynnvalue_scorer.py`：HTTP纯数值评分、BF16/B1/K4、每episode搬上GPU再同步卸载；固定 `pred_slot_isolated_eager`、`use_cache=False`。
- 同目录 `rynnvalue_client.py`：原始RGB、任务原文、前后边界对齐、数值身份和内容缓存。它不输出success。
- [服务说明](../../../local_scripts/bc_rynnvalue_rabc_packet/SCORER_README.md)、[9/9两轮smoke结果](../../../local_scripts/bc_rynnvalue_rabc_packet/smoke_summary.json)：当时实际10次Adam、2个成功episode、6个入池query；不是本次WMRL实跑。

本轮根agent只读服务器回执确认这三个目录存在：

```text
/data/chenyiteng/models/RynnValue-8B-8738c5e4
/data/chenyiteng/projects/RynnValue-10e0d333
/data/chenyiteng/venvs/rynnvalue-8b-py310
```

模型目录列出四个权重分片、config/index及manifest；这是当前资产存在证据，未在本专题重新逐片校验哈希，也未启动旧服务。[本轮资产回执](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003/opendw-reward-cpu-v1.out)

**版本差异必须保留：** 当前HF8B仍是 `8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e`；实际config为absolute symlog `[0,512]`、relative `[-256,256]`。上游Git新README已写quantile支持与causal推理，不能据此把现有权重描述成新版quantile模型，也不能无声换attention后还声称沿用旧实验。未来选新版要重新锁权重、config、源码和评分合同。[现有8B实配](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B/blob/8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e/config.json)

真要接入，缺的是明确的小接口：保存每环境独立RGB历史→固定采样→新独立scorer→带版本的score/success/unknown结果；再补装卸同步和资源记录。若选RynnValue Success，必须新增语言生成与解析、失败处理、延迟测量；不能只将现有 `remaining_seconds` 改字段名。若选Robometer，需要独立依赖和权重加载路径，不必合并其完整训练仓库。

## 8. 推荐的后续用法

**第一优先：离线核验。** 使用已保存的原生成功/失败轨迹和OpenDW生成轨迹，对照WorldArena分数，集中看“失败但高分”“短暂达标又滑落”“图像不合理却报成功”。先报告每类样本和未知比例，再讨论阈值；这一步不改变正在跑的GRPO信号，也不要求新模型常驻GPU。

**第二候选：每回合终局二级判断。** 每条完整轨迹最多一次，当前N8/N32就是8/32个逻辑样本。先仅记录agreement/disagreement；只有原生标注集验证过，再决定是否影响termination/reward。若以后选择双判定，应明确“双方同意才成功”会同时减少假阳性、增加漏报，不能当零代价增强。

**暂不推荐：逐帧替换主RM。** 代价最高，同时改变奖励分布与结束规则，很难判断改善来自WM、策略还是RM。先把当前WorldArena主奖励闭环跑清楚；本文件只保留Robometer/RynnValue方案和证据，后续不一定使用大模型。

轻量来源快照与BF16元数据保存在 [研究目录](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/opendw-reward-research-20261003/source-manifest.json)。本次未下载上述大模型、未SSH、未运行GPU；未给未测方案承诺吞吐或收益。
