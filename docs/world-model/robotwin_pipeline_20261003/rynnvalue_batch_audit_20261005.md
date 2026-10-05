# RynnValue Success batch：源码能力、512条成本与显存方案

2026-10-05，依据深圳3 16:12实际源码与官方固定版本交叉审查。**结论：模型具有batch维度，官方已实现数值batch；Success示例仍单条生成，我们需要封装和GPU验证，没有发现必须逐条推理的架构限制。**

17:53实施更新：Success batch已真实跑通，K8的B4/B8/B16吞吐为2.25/3.94/5.35片段/秒，reserved峰值20.43/22.58/26.88GiB；预热后固定两样本重复，不是跨任务吞吐保证。B16作为后续候选，质量仍失败，不能据工程完成启动正式。完整结果与单卡归还见[实施记录](rynnvalue_execution_20261005.md)。下文保留调研时方案。

## 1. 版本与源码证据

| 位置 | 实际行为 |
|---|---|
| Git `10e0d333`，`rynn_infer/inference.py:228–252` | 数值输入拼batch；官方评测默认B4 |
| 同文件`:267–285` | Success只取final_sample做generate，只decode `gen_out[0]` |
| HF `8738c5e4`，`processing_rynn_value_lang.py:292–335` | process_episode处理一个视频，外层需逐视频处理再collate |
| 同版本`modeling_rynn_value_lang.py:324–374` | slot mask有batch维，generation逐步传递cached prediction-key mask |
| `attention_impl.py:38–65` | custom eager attention保留B维；prefill显式产生与B×L²相关的矩阵 |
| 本地`rynnvalue_scorer.py:270–309` | 旧RA-BC服务明确B1、K4、language_generation=False，不能直接当Success batch服务 |

实际环境：`/data/chenyiteng/venvs/rynnvalue-8b-py310`，Torch2.7.1/Transformers4.57.6/Accelerate1.14.0。HF配置是symlog value tokenizer、`pred_slot_isolated_eager`；新README讲的quantile/causal不是当前锁定版本。第一轮保持权重、processor、提示与attention合同，不擅自切FlashAttention/SDPA或最新默认值。

主源：[固定Git推理](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/rynn_infer/inference.py)、[固定HF模型](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B/tree/8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e)、[当前README的不同合同](https://github.com/alibaba-damo-academy/RynnValue)。本机轻量hash回执：[source receipt](rynnvalue_source_receipt_20261005.json)；完整只读文本在回执给出的E盘位置。

## 2. 最小batch封装

1. 第一版固定摆瓶子、主图256×320、K8。处理器最小面积65536，当前主图81920像素已满足，不为调用Rynn额外放大到640。
2. 逐视频调用process_episode，然后按实际token长度、K和图像尺寸分桶；同长度直接拼`input_ids / attention_mask`，依样本顺序拼图像pixel_values及image_grid_thw。不能把多条轨迹的帧拼成一个视频。
3. 一次generate接B条，greedy/beam1/use_cache=True，沿各行解码新增token并归还对应UID；不只decode第0行。保持官方生成配置，单独记录每条结束长度。
4. 先用等长分桶避免不必要padding改动。若支持异长，采用正确左padding/attention mask，位置编码由模型处理；额外检验特殊slot mask和KV。不手造普通二维位置编码覆盖多模态位置。
5. 输出三态Success及原文；缺失/截断/冲突单独统计。B1与B4/8/16需对照Yes/No/unknown、变序后身份、尾批和峰值；浮点批量计算可能造成临界生成差异，差异要记录而非强称逐token完全一致。

只要Success就直接generate。无需先为所有历史时刻建立前缀，跑完剩余时间曲线再生成；也无需另外加载一个语言模型。K8是每条视频8帧，B8是同时8条视频，二者不能混淆。

一个具体兼容点：旧数值服务使用`logits_to_keep=0, use_cache=False`，在锁定模型`:427–454`中会跳过语言logits，不能原样传给Success入口。新生成入口应使用`logits_to_keep=1, use_cache=True`；官方generate依赖HF自动设置前者，显式指定可避免误继承数值kwargs。

## 3. 一轮512条到底多少batch

当前计划继承正式N64×R8=512，G8已含在N64内；max384/C32=12块，每块8张未来图。

| 评分频率 | 最多视频样本/轮 | 单卡B8批数 | 单卡B16批数 |
|---|---:|---:|---:|
| 每条轨迹末尾一次 | 512 | 64 | 32 |
| 每C32块一次 | 6,144 | 768 | 384 |
| 每张未来图各建一个历史前缀 | 49,152 | 6,144 | 3,072 |

均分两卡时，终局B8约32波、B16约16波。这是理想满批算术，不是已测耗时；在线有早停、候选数量和尾批，未必能凑满。每批generate内部还包含prefill与多次token解码，不能把一批叫“一次forward”。HTTP请求可以装一个队列再内部切batch，不必每条一请求。

“旧RM报成功才复核”第一次通常最多512条，但被否决后继续，下一块可能再复核，最坏仍接近6,144。不能承诺在线永久只有512次。按当前旧规则约113动作提前结束的数量，不能预测新规则后的耗时。

B16可用与否取决于K、实际token长度、生成长度与共卡负载；RM batch与WM batch分别配置，WM已B16不意味着Rynn B16已通过。

## 4. 显存和调度

Rynn8B BF16纯权重约17.84GiB；此前B1/K4数值路径约18.02GiB allocated峰值是9月旧实验，不能代替K8＋Success生成实测。生成另需视觉中间张量、KV、eager attention和allocator。权重搬上GPU后应连续消费整个队列，不能每条视频装卸一次。

最近记录的WM卡6/7峰值约55–56GiB，加Rynn纯权重已约73–74GiB，80GiB卡余量不足以承诺激活/缓存安全。在线优先候选如下：

| 方案 | 使用方式 | 取舍 |
|---|---|---|
| **策略卡4/5分时（优先测）** | 采样时原策略约17GiB，额外驻留Rynn；本rank先出动作，等待WM生成后Rynn判分；actor训练/原生评估前Rynn同步卸载 | 账面权重/常驻合计约35GiB，比WM卡更有空间；实际batch峰值、rollout共存和阶段控制仍需测 |
| WM卸载后的独立评分阶段 | WM offload确认→Rynn消费整批→Rynn offload确认→下一阶段 | 适合离线/旁路，最容易隔离峰值；在线每块切换会反复搬模型，通常不划算 |
| Rynn与WM同驻6/7 | 同卡排队生成与判分 | 当前显存余量小，不作首选 |

所有是候选调度，**本轮没有借卡、启动服务或占GPU**。上线时由当前WM唯一owner纳入Rynn生命周期，不另开一个能抢卡的训练owner；Rynn保持独立venv/本机服务。GPU固定4–7，RLT归还链仍由原owner处理。

不仅检查服务回报offloaded，还需同步CUDA并核模型cuda tensor/本进程显存回落；actor前的现有WM barrier不能自动覆盖新Rynn进程，必须补Rynn barrier。把Rynn塞进rollout旁边后，不能同时让actor升峰却忘记卸载。

## 5. 一次实测要回答什么

- B1参考后直接测B4、B8、B16，保留实际可用最大档及更小一档吞吐，不预设最大档最快。
- 固定K8/输入分辨率/提示，计首载与预热、clips/s、完整512队列墙钟、生成长度、unknown、CUDA allocated/reserved、整卡峰值、CPU RSS。
- 同时检查输出条数、UID、乱序/尾批和独立样本性；用已标注原生视频核误报/漏报，以免只测吞吐没测判断。
- 上线共卡方案只需对已选batch补一次rollout共存与卸载交接验收；不要把无载独占显存结果直接当共卡结论。

这些实测尚未执行。接入的下一决策见[主计划](rynnvalue_integration_plan_20261005.md)。
