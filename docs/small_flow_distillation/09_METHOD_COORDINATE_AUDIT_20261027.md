# 五个蒸馏方法：扩散与FM适配究竟改了什么

2026-10-27。本轮逐项读取当前losses.py、mfd.py及端点/辅助更新代码。重点区分“原方法用什么模型”和“我们现在实现的数学量”。

## 先用普通话区分

扩散和FM都能把随机噪声逐步变成动作，但网络回答的问题可以不同：预测噪声、预测干净动作，或预测当前位置该向哪里走。π0.5回答的是最后一种，即速度。不能把同一个张量直接改名成噪声或分数。

|方法|原工作/来源|当前π0.5实现教学生什么|
|---|---|---|
|直接端点|与扩散/FM都兼容的回归思路|同观察、同噪声，学生一步动作接近老师多步最终动作|
|Consistency Policy|原版EDM扩散|老师搭一小段桥，学生跨步后与EMA目标保持一致；另有专家动作FM监督|
|OneDP-S / DMD new|原版基于Diffusion Policy的分布匹配|辅助网络学习学生当前动作分布，老师与辅助的方向差指导学生|
|MFD|原生流模型的均值速度蒸馏|老师/辅助从同一起点各走一段，比较沿各自路径的平均方向|
|FQL|原版离线流Q学习|端点模仿约束，加Q对执行动作价值的指导；我们是冻结多步老师的适配版|

所以当前五项都工作在π0.5的FM接口上；CP、OneDP不是原扩散代码不变地搬过来。通用的是训练思路，时间、预测量、噪声系数和梯度要重新对齐。

## 当前DMD的符号和系数检查

π0.5时间约定：t=1是噪声，t=0是动作。采用直线路径

`x_t = (1-t) a + t epsilon`，速度目标为 `v = epsilon - a`。

因此同一位置的噪声估计为 `epsilon_hat = x_t + (1-t) v`，分数为 `-epsilon_hat/t`。分数只是训练时对分布方向的表达，不是机器人执行的动作，也不是多步端点。

当前DMD代码将 `stopgrad[t(1-t)(v_teacher-v_fake)]` 与 `x_t` 相乘。通过x_t回到学生，还保留一个`(1-t)`，总梯度系数为`t(1-t)^2`。在采用OneDP的`sigma^2=t^2`权重时，该方向与反向KL的分数差一致。时间限制在[0.02,0.95]，避开除零端点；没有发现把速度直接当分数或更新符号颠倒的问题。

辅助网络拟合学生生成动作加噪后的epsilon，生成动作detach，不反向偷偷修改学生。MFD辅助则回归原生速度，时间权重与OneDP epsilon-MSE不同，不能比较两个辅助loss的绝对大小来判断谁更好。

## CP与MFD核对结果

CP的跳步映射是`x-(t-s)*v(x,t,s)`，s=t时严格回到输入。老师先走一个小时间间隔；EMA参数冻结，但预测分支经EMA的输入导数保留，梯度才能回到学生。FM项来自专家动作，不是老师生成答案。这能解释学生可能超过老师，但也意味着它与只有老师监督的直接蒸馏信息来源不同。

MFD两路从同一加噪动作出发，各自沿自己的速度场积分；老师/辅助积分不建梯度图，最终学生仍通过加噪位置获得梯度。FP32积分/累积避免很短步长在BF16中被舍掉。共享批内积分次数K、再条件抽每个样本区间，保留每个样本原本的区间分布，同时避免逐样本循环；批内样本因此有关联，不等于改变了单样本期望。

## 尚不能说“全部等同原论文”的地方

- 当前直接蒸馏、DMD new和MFD学生按H50×真实14维计分；旧CP两项仍包括内部H50×32，旧OneDP也包含填充维。模型接口保留32维有来源，但loss是否忽略18个不用的维度是适配选择。未来严格方法对照应统一或单列这个区别。
- 只保留14维学生梯度，同时辅助拟合完整32维，不能直接宣称是在精确优化14维边缘分布的KL；它是有意使用的部分坐标指导。
- 学到的老师/辅助是近似方向，辅助能否跟上学生、数值积分误差和损失尺度仍影响效果。形式检查通过不等于成功率一定改善。
- CP从EDM改到直线FM时没有原封不动继承EDM预条件和噪声时间表。应称CP思想的FM适配。
- MFD/DMD new当前仅通过短训练、续训和N1导出smoke，尚无正式长训练成功率结论。
- FQL本轮首三点评估约3.0%、3.3%、1.6%，效果明显不佳；这属于实现/数据/优化需要进一步定位的问题，不能只因公式可写就称有效。

## 主要依据

- [Consistency Policy论文](https://arxiv.org/html/2405.07503v2)、[官方实现](https://github.com/Aaditya-Prasad/consistency-policy)。
- [One-Step Diffusion Policy，ICML 2025](https://proceedings.mlr.press/v267/wang25ba.html)。
- [Mean Flow Distillation](https://proceedings.mlr.press/v306/zhao26bx.html)、[固定版官方训练源码](https://github.com/happyw1nd/MFD/blob/bd882924fe24ce59fac0e225f7c82db10db29cb8/t2i_SANA/scripts/train_sana_mfd.py)。
- [Gaussian Diffusion与Flow Matching的换算](https://diffusionflow.github.io/)：用于核对预测量转换，不把特定路径等价推广成任意模型可直接互换。

当前源码锚点：`rlinf/algorithms/pi05_distillation/losses.py`的cp_loss/fake_loss/onedp_loss/endpoint_loss；`mfd.py`的sample_intervals/average_velocity/auxiliary_loss/student_loss。沿用此前FQL和MFD规划文档的固定参考版本，不重新选择另一套网络。
