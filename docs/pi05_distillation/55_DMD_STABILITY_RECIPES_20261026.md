# DMD工程做法：先让方向可信，再谈加复杂损失

2026-10-26。来源为论文与作者代码；下面是讨论候选，未部署到正式OneDP。

## DMD2相对DMD增加了什么

辅助网络像一张“学生当前行为地图”。学生一直变化，地图若落后，用它给出的修正方向就可能不准。原DMD还用老师多步答案的回归项稳定学生；DMD2移除该项后，通过更频繁训练辅助网络稳定方向，并加入真实数据的GAN监督和多步学生的训练输入模拟。对当前一步机器人学生，最直接相关的是辅助更新频率。[DMD论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.pdf)、[DMD2 §4、附录C/F](https://arxiv.org/html/2405.14867v1)。

DMD2主要设置是辅助5次/学生1次；SD1.5的一阶段还用10次。附录比较表明只放大学习率不等价，多于5次也未必更好。我们的辅助学习率虽是学生的20倍，但仍各更新一次，不能说已经实现DMD2的机制。SDXL的一步设置还先用少量回归配对做初始化；这是特定模型经验，不能当所有DMD的必需步骤。

## 干净候选与优先级

|选择|通俗含义|当前情况/实施切口|优先级|
|---|---|---|---|
|学生监督物理维对齐|不给机器人不用的18维同等打分|生成器loss单独mask14，H50不变；辅助仍估计学生分布|先做，来源是我们两组的具体差异|
|辅助多更新|地图更新几次，再让学生前进一步|现1:1；可做5:1的来源明确对照，保留共享观察编码|高；计算会增加，另记学生更新数与GPU小时|
|flow方向按样本归一化|避免某些噪声时刻/样本把更新幅度压得太小或放得太大|当前没有这种归一化；可复用FastOPD的flow-DMD公式|高；属于另一种权重方案，不是“修正明显错误”|
|从直接蒸馏学生开始|先学会一步动作，再做分布调整|可复用成功endpoint点，老师仍固定原N10；辅助需重新适应这个学生|实用备选；应标注“endpoint初始化+分布微调”，独立于从原始基座的OneDP对照|
|残差投影PDMD|从修正方向中去掉疑似辅助误差的分量|已有公式和很小的独立函数，无新增模型前向|次选；先解决基本监督和辅助质量|
|学生EMA|用最近参数的平滑版本评估，降低单点抖动|当前OneDP无学生EMA|低；能平滑，不会纠正系统性错误方向|
|GAN/新奖励/相位专家|引入更多学习信号或改变模型|DMD2/ADM/Phased DMD有依据|暂不优先，已超出简单工程对照|

## flow适配：比“换个学习率”更具体

我们的OneDP辅助损失为epsilon误差；换成速度表达，相当于`(1-t)^2 × 速度误差`。学生端有效方向为`t(1-t)^2(vT-vF)`。二者均匀取t∈[.02,.95]，与原DDPM离散时间的噪声分布不等价。

[FastOPD附录D.1](https://arxiv.org/html/2610.02832v1#A4.SS1)给出了机器人flow-DMD对照：辅助直接学习速度误差；学生的教师/辅助终点差按每样本教师修正幅度归一化。这样不同样本的更新强弱更可控。建议把“普通速度辅助损失+对应归一化方向”视为一套有来源的flow-DMD变体，单独命名记录，不能悄悄叫成原OneDP的唯一官方实现。

[NVIDIA FastGen固定源码](https://github.com/NVlabs/FastGen/blob/f7d845629b00c75527fd04d8ce66392c363b1fbb/fastgen/methods/common_loss.py#L57)也采用每样本归一化，并用FP32计算分母。实践上应记录分母、方向范数和非有限值；有问题时查明原因，不靠静默清零隐藏故障。保持teacher/fake在学生loss中stop-gradient，辅助训练样本来自当前学生并detach，这两项我们已具备。

当前速度输出投影在autocast范围内，teacher-fake差随后才乘FP32的时间因子；可将终点转换、相减、归一化和最终loss显式置FP32。此项减少低精度相减误差，不能据此声称它是目前平台期的主因。

## 新查到的小切口：PDMD

[PDMD论文](https://arxiv.org/abs/2609.35768)将辅助误差累积作为问题；[作者函数](https://github.com/ZeamoxWang/pdmd/blob/main/src/pdmd/objectives.py#L12)逐样本把更新向量在学生—辅助终点残差方向上的分量减去，使用FP32内积和小分母保护。原证据来自高维视频，未验证π0.5动作；投影也可能去掉有用方向，不能把“一行改动”理解为保证提升。适合在基础DMD对齐后再做一个单项对照。

另外核到[SGMD](https://arxiv.org/abs/2605.30116)和[DMAD](https://arxiv.org/abs/2610.02188)直接改变辅助/分布匹配目标；[Phased DMD](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Phased_DMD_Few-step_Distribution_Matching_Distillation_via_Score_Matching_within_CVPR_2026_paper.html)分噪声区间训练。它们说明问题有研究依据，但不是本轮最小修改。

建议顺序：物理监督维对齐→辅助跟踪诊断/5:1→有来源的flow-DMD权重。一次只动一项；若更看重尽快得到强一步策略，可另走endpoint初始化的路线。
