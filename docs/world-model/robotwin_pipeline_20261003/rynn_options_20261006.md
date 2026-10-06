# RynnValue：已经试到哪里，接下来怎样用才合理

2026-10-06；本轮只核对已有实验、论文、官方模型与代码，未新增 GPU 实验，未修改正在运行的 WMRL。

**最重要的修正：官方 Rynn 的 RL 实验用数值头提供“进展辅分”，成功与结束由人另行确认。它没有证明自己的 `Success: Yes/No` 能独立承担我们缺失的 WM 成败裁判。** 我们先试语言 0/1 是合理的小测，结果不支持直接上线；后来的数值测试证明有一些进展信息，仍没有补上这个缺口。[论文 B.3、B.5–B.6](https://arxiv.org/html/2608.09853v1#A2.SS3)

## 我们实际试了什么

| 尝试 | 实测结果 | 能说明什么 |
|---|---|---|
| 16 个初始片段＋16 个专家终段，语言 Success | 全部 No；解析正常 | 不是把解析失败误当 No；专家成功没有被识别出来 |
| 可见直立前缀、重复可见帧、K64、文字／元信息／原分辨率、官方示例等对照 | 实际完成的生成仍为 No；缺元信息的输入错误另列 | 不能单独归因于最后出画、K8 太少或统一缩放；尚未定位唯一根因 |
| 16 专家×8 条件，共128个数值输入 | 15/16 初末下降；25%之后47/48相邻比较方向正确；9/16开局反升 | 已有运动历史后经常看得出进展；重复初始图偏低，不能随便阈值化或逐步奖差分 |
| 新采32条原生 CP70 回合，有 simulator 逐条真值 | 15成功、17失败；语言全部 No，unknown=0；数值 AUC=0.70196 | 语言漏掉15/15成功；数值有排序趋势，但分布重叠，**不是70.2%准确率** |

最后一批的成功数值范围0.285–4.795，失败0.326–9.894。存在失败比成功分数更低的具体反例；没有能严格分开本批两类的低值阈值。本批未搜索或部署阈值。

证据：[语言诊断](rynnvalue_semantic_diagnosis_20261005.md)、[128组数值诊断](rynn_numeric_probe_20261005.md)、[32条真实成败实验](rynn_binary_probe_20261005.md)、[逐条轻量结果](../publication_bell_20261005/rynn_binary_result.json)。这些是 `adjust_bottle` 与固定 `RynnValue-8B@8738c5e4` 的结果，不推广成“所有任务上的 Rynn 都无效”。

## 三种输出不要混用

| 输出 | 说人话 | 代码含义与边界 |
|---|---|---|
| absolute value | 看起来距离做完还差多远 | `outputs.value.pred_value`；训练目标是剩余时间，越低越接近完成。我们读每个独立历史前缀的最后槽；不是成功概率，也未在本任务标定为准确秒数 |
| relative value | 两张观察之间像隔了多久、方向如何 | `outputs.relative.pred_value`；是呈现序列相邻帧的时间位移，可能为负，不是成功标签 |
| Analysis/Success | 用文字说有没有完成 | LM 自回归生成的 Yes/No；不是独立的成功分类头；Match 只表示“视频在做这条指令”，不能当 Success |

官方 `process_history` 接受 `success` 标签并构造语言监督，所以 Success **确实有正式训练设计**，不是我们临时编一个问题问模型；但有该训练目标不保证本任务可靠。[训练样本处理器](https://raw.githubusercontent.com/alibaba-damo-academy/RynnValue/main/rynn_value/processing_rynn_value_lang.py)、[提示构造](https://raw.githubusercontent.com/alibaba-damo-academy/RynnValue/main/rynn_value/conversations.py)

官方模型输出只有 value、relative、language 等字段；评估适配器中虽保留“可选 success head”的兼容检查，不能因此声称现有权重提供独立成功概率。其 `confusion_score_mode=match_binary` 是任务－视频匹配，也不是任务成败。[模型定义](https://raw.githubusercontent.com/alibaba-damo-academy/RynnValue/main/rynn_value/modeling_rynn_value_lang.py)、[评估适配器](https://raw.githubusercontent.com/alibaba-damo-academy/RynnValue/main/robometer/robometer/evals/baselines/rynnvalue.py)

## 官方究竟怎么做 RL

论文的真实机器人实验：人工确认成败与结束；Rynn 对第三人称历史给数值，**不运行语言生成**。离线用 IQL，在线用 DSRL/SAC 学噪声控制器，在线 VLA 冻结。并不是我们的“生成视频＋语言 Success＋π0.5-GRPO”。[论文 B.3–B.6](https://arxiv.org/html/2608.09853v1#A2.SS3)

在线公式为：

$$
\Phi_t=-v_t,\qquad \gamma_c=\gamma_s^q
$$

$$
r_t=r_t^{\mathrm{sparse}}+\kappa F_t,\qquad
F_t=\begin{cases}
\gamma_c\Phi_{t+1}-\Phi_t,&\text{下一状态非终止},\\
-\Phi_t,&\text{下一状态终止}.
\end{cases}
$$

成功完成的转移 sparse=0，其余为−1；Rynn 的 $\kappa=0.1$，在线 $\gamma_s=0.999$、$q=10$。终止分支等价于终止势设0。离线 IQL 则写 $\gamma_{\mathrm{off}}\Phi_{h+1}-\Phi_h$、$\gamma_{\mathrm{off}}=0.99$；该离线公式未显式写终止势置0，不把两个分支混写。[论文式13–21与表9–10](https://arxiv.org/html/2608.09853v1#A2.SS5)

**含义：人负责“最终做没做成”，Rynn 帮助“中间哪些动作更有进展”。** 因而照搬官方辅分不会自动解决纯 RGB 世界模型没有真值裁判的问题。

### 为什么不能直接把差分塞进 GRPO

以下是对接时的数学推导，不是已实施方案。若整个轨迹只拿一个累计回报做组内比较，直接累加 $v_t-v_{t+1}$ 会得到：

$$
\sum_{t=0}^{T-1}(v_t-v_{t+1})=v_0-v_T.
$$

中间怎么走被抵消，效果主要变成“按末分排序”。若按相同折扣累计标准势差，则：

$$
\sum_{t=0}^{T-1}\gamma^t(\gamma\Phi_{t+1}-\Phi_t)
=-\Phi_0+\gamma^T\Phi_T.
$$

如果终止势统一为0、同组起点相同，结果只剩相同的起点常数，**不会凭空产生 GRPO 需要的组内成败区别**。若刻意保留末势，可以产生排序，但那就是在优化 Rynn 的末态估分，应单独验证，不能称已经取得真实成败奖励。若算法用 reward-to-go、不同折扣或特殊 mask，还需按实际聚合规则重算。

## 为什么全 No，哪些已知，哪些只是猜测

- **已知：** 模型确实生成 No；初始与专家输入不同；换成可见前缀、更多帧、不同元信息／文字／原分辨率没有解除所测病例。32条原生成功真值也证明不是只对专家数据标签猜错。
- **有依据的限制：** 单主相机可能看不到关键物体或接触；RoboTwin success 是物理阈值，而自然语言还可能要求直立等额外条件。现有简化任务文字仍失败，因此它们不能单独解释全 No。
- **仍属假说：** 发布权重语言分支的校准／训练偏置，具体任务外观或相机域差异，以及尚未定位的推理版本细节。当前没有证据把任何一个写成根因，也没有证据声称“再调 prompt 一定行”。

低数值却输出 No 并不自相矛盾：一个是时间估值，另一个是语言判断；两条训练输出没有强制使用同一成功阈值。

## 真想用，三条路及优先级

| 路线 | 怎么做 | 对我们的价值／缺口 |
|---|---|---|
| **保留 GRPO 的0/1目标：任务裁判适配** | 原生 simulator 自动标成功／失败、首次成功时刻；包含“接近但未完成”的负例。按轨迹／种子分训练与留出集。先看固定特征＋小分类头能否分开，必要时再考虑 Rynn 语言 LoRA 或独立小视觉裁判 | 目标最直接，但这是新增训练方案，未实现、无已验证所需样本数；32条只够诊断。全 No 的硬输出无法靠调阈值变好，需要读取可用分数／特征或改训练 |
| **Rynn 先作辅助排名／数据筛选** | 用负的末态 value 给已有片段排序，人工抽查；保留原生真值用于衡量误排序。可进一步比较保留／去除低进展示范的 BC | 改动小，现有 AUC 支持“小试有理由”，不支持直接当奖励。无需让它决定 done |
| **已有可信成败裁判后，再加进展辅分** | 按算法重做奖励聚合、折扣、终止势与有效 mask；与只用成败做对照。最接近官方用途的是 IQL／DSRL 这类含 critic 的路线 | 最有官方依据，但不能直接解决目前缺0/1的问题；并且不是仅换一行 GRPO reward 的等价复现 |

推荐顺序：如果下一目标仍是 **新任务 WMRL＋GRPO**，先解决任务裁判。Rynn 暂作离线辅助分析；不要继续把“看出进展”说成“可靠判成功”。若特意研究 Rynn，则优先任务级二分类适配与留出验证，或在原生环境有真值时测辅分，二者目标要分开。

生成图还需单独验证：原生环境标签可以大量自动获得，但同一动作在真实仿真中成功，不代表 OpenDW 生成的视频也表现为成功；不能把真实轨迹的 success 无条件复制给生成视频。可用少量人工判断生成片段验证视觉可判性，再决定是否补生成域样本。

## 现成代码与成本

我们已经有 K8 历史、batch、两头输出、真值采集、卸载与资源记录，推理接线不再是主要缺口。数值测试实跑过 B16，128条纯 forward 11.8秒、约10.8条/秒；峰值 allocated25.4GiB、reserved31.2GiB。512条各评一次约47秒纯 forward；每条12块都评的上限约9.5分钟，另加预处理、排队、WM与搬运。最后32条实验因指令长度分桶，实际批量2/6/4/6/4/6/4，不能把它称为 B16 实测。[本机成本证据](rynn_numeric_probe_20261005.md#batch耗时和显存)

官方已开放模型／processor／损失、推理、评估、HTTP服务、π0.5 IQL/DSRL；新的主分支还介绍 Filter-BC。**尚未核到可以直接用于 Rynn 自身专项微调的一键配方。** 仓库里的 `FINETUNE_ROBOMETER.md` 明确训练 Robometer-4B，不等于 Rynn-8B 微调已经现成；用户本轮已排除 Robometer，不因此重新部署它。[Rynn固定版本说明](https://raw.githubusercontent.com/alibaba-damo-academy/RynnValue/10e0d333f5f3811d0d130587e50f1faf48da49e5/README.md)、[Robometer训练指南的实际对象](https://raw.githubusercontent.com/alibaba-damo-academy/RynnValue/main/robometer/FINETUNE_ROBOMETER.md)

版本边界：2026-10-06访问的 GitHub 主页已出现 quantile tokenizer、0.704 排名与 Filter-BC 的更新说明；固定 `10e0d333` README、HF卡和我们已测 `8738c5e4` 记录仍为旧 symlog 路线。主分支文字变化不等于已测模型自动升级，更不能用新排名覆盖旧实验。本轮未换权重、未验证新 tokenizer/checkpoint 兼容性。[当前主页](https://github.com/alibaba-damo-academy/RynnValue)、[固定旧说明](https://raw.githubusercontent.com/alibaba-damo-academy/RynnValue/10e0d333f5f3811d0d130587e50f1faf48da49e5/README.md)
