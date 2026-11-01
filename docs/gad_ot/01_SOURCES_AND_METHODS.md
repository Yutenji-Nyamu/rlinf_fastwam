# OT相关工作：先区分运输两端

2026-11-01核对；论文结果属于作者报告，本轮没有复现实验。源码下载/阅读与实际运行严格分开。

## 1. 最直接的机器人近邻：OptiFlow

**Learning Multimodal One-step Flow Policy via Value-weighted Optimal Transport**，arXiv:2609.15883v1。

[论文§4、附录D/F](https://arxiv.org/html/2609.15883v1)；[官方仓库](https://github.com/Yonsei-DILLab/OptiFlow)；[固定源码](https://github.com/Yonsei-DILLab/OptiFlow/blob/d54ad7364646bed9405bf0b7fe313eea9587f3bf/optiflow/agent.py)；[固定配置](https://github.com/Yonsei-DILLab/OptiFlow/blob/d54ad7364646bed9405bf0b7fe313eea9587f3bf/configs/agent/optiflow.yaml)。

它联合学习Q评分器、多步参考策略和一步学生。同一状态下生成两组候选：Q决定老师候选各占多少监督份额，动作距离决定怎么分配学生。部署只用一步学生。

实际代码：老师份额为softmax(Q/温度)，学生份额均匀；逐状态归一化距离矩阵后做log-domain Sinkhorn，运输矩阵停止梯度。随后**逐行选最大运输质量的老师候选**，用选中质量乘MSE。它不是对老师动作求均值，也不是严格的一对一分配；逐行argmax之后不再精确满足原列份额。默认`lambda_q=0`，Q仍影响目标分配和参考策略。

已核官方默认：学生16候选、老师64候选、老师10步、Sinkhorn 30次、归一化距离上的ε=0.05。现有GAD的4×4是成本受限的适配，不能冒称官方样本预算。其动作截断到[-1,1]也不直接移到我们的π0.5归一化接口。

**可借：** 同观察内构造候选、距离表、停止梯度的配对。**不默认借：** Q、训练参考策略、16×64采样、小MLP学习率、动作截断、选中质量造成的额外loss缩放。

## 2. 新的DMD直接近邻：sDMD

**sDMD: Sinkhorn Distribution Matching Distillation**，arXiv:2610.39889v1，2026-10-29。

[论文§5、附录E/F](https://arxiv.org/html/2610.39889v1)；[项目](https://sinkhorn-dmd.github.io/)。项目页面本轮读取到**Code (Coming Soon)**，未获得可审阅的训练实现。

它给DMD加入去偏的Sinkhorn分布差异项，关注老师分布覆盖不足的学生样本。配对使用同一条件下的多组输出；修正方向同时涉及“学生→老师”与“学生→学生”，并复用DMD2判别器决定何时开启。作者也报告小批量运输估计会干扰原本已可靠的指导。

这说明“DMD＋OT”已有直接先例。但我们的DMD new没有该GAN判别器，直接添加一项配对MSE并不等于sDMD；其样本空间、门控和去偏项必须分清。原文大模型实验为图像/视频，不能直接推出机器人收益。首轮GAD-OT不照搬新增判别器。

## 3. 相近但运输两端不同

| 工作 | 运输两端与作用 | 对我们有用的点 | 资源状态 |
|---|---|---|---|
| [COT Policy，CoRL 2025](https://proceedings.mlr.press/v305/sochopoulos25a.html) | 条件化的噪声→示范动作，训练更直的flow | 条件必须正确：不同场景的动作不能随意换老师 | [官方代码](https://github.com/ansocho/cot_policy)存在；本轮读README，子目录访问失败，未做函数级审计 |
| [OTQL，2026预印本](https://arxiv.org/html/2607.06262v1) | 用优势改变目标分布，再做噪声→动作条件OT并后训练flow | “改目标质量”与“改配对几何”是两件事 | [项目](https://ansocho.github.io/otql-flow/)仍为Code Coming Soon；SmolVLA实机设置3NFE，不是该VLA已一步 |
| [Multisample Flow Matching，ICML 2023](https://proceedings.mlr.press/v202/pooladian23a.html) | 小批量噪声→数据耦合，影响训练路径 | 多样本耦合与单样本指定配对不同 | 已读正式论文页面；本轮未审其实现 |
| [Learning Generative Models with Sinkhorn Divergences，AISTATS 2018](https://proceedings.mlr.press/v84/genevay18a.html) | 用平滑OT类损失训练生成模型 | 生成器可由集合间几何距离指导 | 已读正式论文页面 |
| [Learning with minibatch Wasserstein，AISTATS 2020](https://proceedings.mlr.press/v108/fatras20a.html) | 分析小批量OT目标及其梯度 | 小批量目标不等同完整分布的精确Wasserstein距离 | 已读正式论文页面 |

## 4. 求解器与术语

- **OT**是最优传输总称：给两边规定份额，找总体距离小的分配。
- **条件OT/COT**：条件相同或在耦合中明确约束条件。本方案每个观察独立求解，条件约束最直接。
- **平衡OT**：两边份额严格满足；均匀4对4、平方距离时，可选最优排列作为一个精确解。
- **熵正则OT/EOT**：允许较软的运输矩阵，通常用Sinkhorn求解。“正则熵”描述配对矩阵，不是我们的动作不确定性。
- **Sinkhorn divergence**：还扣除两边各自与自己的运输项。单做一次师生Sinkhorn或逐行argmax不等于它。
- **非平衡OT/UOT**：放宽份额约束，允许改变/丢弃质量；这会另改老师分布，暂不作为首版。

工程参考：[POT log-domain Sinkhorn](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn_log)、[SciPy最小总代价匹配](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)。K=4可枚举24个排列，在PyTorch中批量求解，不必为训练增加一个逐样本CPU求解依赖。

## 5. 不能从文献直接推出的结论

- 我们的同噪声Direct本来就有明确老师答案，不是把四个老师动作先平均。OT放宽了“某份噪声只能对应原老师答案”的限制，是否更易学习需验证。
- 高标准差可能是多种都正确的动作，而不是老师不可靠；GAD低教师标准差偏好可能恰好减少OT最能发挥作用的多模态位置。
- 4×4更低的匹配误差不代表更高的闭环成功率；不能据此宣布修复了蒸馏平台期。
- 本轮命名是GAD-OT候选，不写“首次将OT用于蒸馏/机器人/DMD”；上述直接近邻需要在论文中诚实比较。
