# 参考依据与代码核对

2026-10-31。以下代码只读审阅，未复现论文成绩。固定提交及缓存路径另见sources.json。

## 信号与选择机制

|参考|可借的具体部分|适配边界/代码状态|
|---|---|---|
|[PFD](https://arxiv.org/html/2606.20754v3)|同噪声、扰动权重、终点动作方差|π0.5直接证据；未找到可确认的官方代码。原先VLM扰动、块级方差|
|[Post-hoc UQ](https://arxiv.org/html/2610.29730)|核条件方差、无标签参考库|官方代码已读；无梯度拟合不等于无统计模型或无维护成本|
|[SAVE/VFD](https://arxiv.org/html/2606.18043)|集成不确定性、主动取数；附录权重扰动|有官方代码；只在t=1训练的学生不能当作任意时刻流场使用|
|[SAFE](https://vla-safe.github.io/)|特征距离基线、STAC-single|官方代码已读；其主方法另训MLP/LSTM，需要成功失败标签|
|[Sentinel](https://arxiv.org/abs/2410.04640)|前后块重叠一致性与分布差异|官方代码已读；动作表示/重叠时间须改成我们的接口|
|[FIPER](https://arxiv.org/abs/2510.09459)|动作样本分散与熵、正常轨迹标定|官方代码已读；RND部分需额外训练；ACE并非4样本方差自动等同|
|[FLARE](https://arxiv.org/abs/2602.09170)|参数曲率与预测不确定性|官方代码已读；非π0.5现成适配，原扩散推导需重新核对|
|[MC dropout](https://proceedings.mlr.press/v48/gal16.html)|训练时已有dropout时的随机前向|经典依据；不把临时给未训练网络加dropout称为直接复现|
|[Deep ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)|独立训练模型之间的分歧|比多噪声更接近模型不确定性，但增加训练与模型存量|
|[PDF](https://arxiv.org/html/2604.18107)|输入增强与动作投票作为输入响应参考|原离散动作logit/学习扰动头，不能直接搬成FM动作熵；本轮只核论文|
|[VeriVLA](https://arxiv.org/html/2610.21985)|块内相邻动作变化与跨样本方差|前者一步可算；只是复杂阶段代理，不能独立等同教师不可靠|
|[DiffDAgger](https://diffdagger.github.io/)|重建/扩散损失作为查询指标；提醒多模态陷阱|当前一步端点学生没有随机中间t监督，不能直接借其损失当学生U|
|[GEO](https://arxiv.org/html/2607.27933)|去噪速度改变；已有采集可分析|依赖至少2个去噪步，只能用于多步模型|
|[U-GROW](https://arxiv.org/html/2609.22879v1)|自一致性信号选择rollout/接管|已有工程参考；不直接证明我们的动作级蒸馏加权有效|

## GAD训练机制参考

|要实现的部分|最直接参考|应怎样借|
|---|---|---|
|师生动作距离|[PIVOT-Q](https://arxiv.org/html/2610.05745)|同观察、共享噪声的师生动作MSE；其代码选择状态，我们保留动作维h|
|老师可信、学生尚不会|[LLKD](https://aclanthology.org/2025.findings-naacl.142/)、[DWA-KD](https://aclanthology.org/2026.findings-eacl.181/)|借选择/加权方向；分类token概率不能直接当连续动作概率|
|局部可靠性调节蒸馏|[UACD](https://arxiv.org/html/2609.39132)|借区域加权及风险覆盖验证；它混合teacher小步与student端点构造目标，不是独立教师方差|
|分歧集中与选择监督|[HOPD](https://arxiv.org/html/2609.33301)|借按困难位置集中学习的动机；不是机器人实证|
|优先抽样/偏差校正|[PER](https://arxiv.org/abs/1511.05952)、[TorchRL官方接口](https://docs.pytorch.org/rl/main/reference/generated/torchrl.data.replay_buffers.PrioritizedSampler.html)|只替换priority定义，保留抽样概率和重要性校正；无需引入RL/Q网络|
|此前讨论UA-OPD|[OpenReview](https://openreview.net/forum?id=hHUZi8oHfo)|本轮正文/API访问403，未补证完整方法或官方代码；不依据用户转述推断精确实现|

LLKD/DWA-KD本轮未找到可确认的官方实现；搜索得到第三方DWA实现不作为作者代码。UACD本轮核了正文与项目入口，未验收训练代码。不同论文对“高分歧”可能选择增加或减少监督，不能拼成一个已有理论保证。

## 实际读过的上游切口

- PIVOT-Q `2dc126a62c54ffca6cf04884ad139a517528933e`：`pi05_quantvla/pivot_q/paired_flow.py`87–126固定/传递噪声；`selection.py`18–19仅比较首动作7维，50–84为分阶段排序、间隔选择、均值归一后裁剪权重。**我们的50×14动作级权重是新增适配，不是抄完即得。**
- Sentinel `6dc89ca4cb30f75ce834811c2d1b31f7ffadf7b5`：`sentinel/bc/ood_detection/error_utils.py`167–287对齐旧块[C:H]与新块[:H-C]，239–240拒绝原代码的position模式；337以后还有最近邻、马氏距离。
- SAFE `b8e31dfe4ea27d238f8f7b92f0c0f998a7179f78`：`failure_prob/model/embed.py`19–60马氏距离，201–248按成功/失败特征库计算距离；不能忽略其数据来源/拟合状态。
- Post-hoc UQ `f658266ab68aae58f04ba6d728a118457cfe12c2`：`q5/src/lerobot/uncertainty/uncertainty_scoring/kernel_variance.py`127–162为标准化、核矩阵Cholesky、条件方差；190以后特征提取含多步flow位置，需改为一步可用读取点。不要误取同文件后面的TargetInformedKV作为论文无标签主方法。
- UQ-VLA `bcc2433d2549afee27bf5d0a73ae528375bca686`：`active_learning/src/lerobot/uncertainty/uncertainty_scoring/scoring_metrics.py`279以后InterVelDiff在指定中间时间评估两个流场，不能不改定义就套一步端点网络；Action-L2/ACE可作输出级参考。
- FIPER `13d79c5c3069def843e454787ff128defc249838`：`evaluation/method_eval_classes/entropy_eval.py`为动作样本熵评估实现；熵估计依赖离散/统计设定，不建议把K4直接叫精确熵。
- FLARE `d875c3b65c30904f4f136181c5e80d82b027d4ad`：`src/uqdiff/laplace/flare.py`是沿扩散路径传播参数不确定性的入口，依赖precision.py中的方差估计；需额外梯度统计，首版暂不选。
- GEO `1cb8af483f1f018fcbca2a0ded9c374e84c0ea83`：`fmaccel/geometry/accel.py`计算速度序列变化；与我们旧signal_math中的GEO核对概念。

上游许可证需在未来实际复制代码时按文件履行；本轮仅作本地阅读与设计，不把下载源码并入项目发布。
