# FM、扩散与少步机器人策略

2026-10-27。研究与源码阅读；未实施新方法。主结论：FM是一种学习生成过程的方法，不等于一步策略。上轮较多关注加速论文，候选自然偏向一步/两步，不能据此推断FM单任务策略全部如此。

## 先分清三个“步数”

- N：模型内部从噪声生成动作块，要计算多少次。
- H：一次预测多少个机器人动作。
- C：预测后实际执行多少个动作，再重新观察。

例如N1/H50/C10：模型算一次，生成50个动作，只执行前10个；不是机器人只动作一次。单任务/多任务决定条件与数据范围，与N没有必然关系。

## FM和经典DP：学什么、执行什么

两者都是根据图像/状态，从随机噪声生成一段动作。区别主要在训练路径、网络预测量和采样器，不是“一个有随机性、一个没有”。

|项目|经典Diffusion Policy常见形式|我们π0.5的FM形式|
|---|---|---|
|训练输入|按噪声日程污染的专家动作|专家动作与高斯噪声的线性混合|
|网络学什么|常见为“加进来的噪声”；也可预测干净动作或其他参数化|“当前动作应该往哪个方向改变”的速度|
|部署|DDPM或DDIM等采样器逐步还原动作|沿速度场积分，从噪声走到动作|
|一步是否天然准确|否|也否；专门的一致性/蒸馏训练能改善|

为什么训练的线段很直，FM推理仍可能需要多步？每条示范与噪声之间的训练线段可以是直线，但模型要综合许多可能的动作与条件；它学到的平均方向随位置变化。推理途中重新计算方向通常比从起点只走一个大步准确。神经网络拟合误差也会影响大步结果。

Gaussian FM和扩散可在共同框架里换算；路径、预测量、时间权重和采样器必须一起对应。不能只把DP的scheduler改名，便把旧噪声预测网络当成π0.5速度网络。

本工作区采用噪声在t=1、干净动作在t=0：

```text
x_t = (1-t) a + t ε
v_target = ε - a
a_hat = x_t - t v(x_t,t)
ε_hat = x_t + (1-t) v(x_t,t)
score = -ε_hat / t       (t>0；对应这个高斯路径)
```

这里的score是对“哪些含噪动作更可能”的密度梯度，不是机器人成功分数。上述是当前线性高斯路径的换算，不是所有扩散日程的通用公式。ManiFlow/PointFlowMatch等常采用噪声0→动作1，接口必须显式反向，不能直接照搬速度符号。

依据：[FM原论文](https://arxiv.org/abs/2210.02747)、[DP作者项目](https://diffusion-policy.cs.columbia.edu/)、[Diffusion Meets Flow Matching](https://diffusionflow.github.io/)。

## 近期代表工作：并非全部一步

“有影响力”采用正式发表、作者代码与实际基准使用作为依据，不编造引用量，也不把刚出现的预印本称为已确立主流。

|工作|如何训练/使用步数|与我们的目标关系|
|---|---|---|
|PointFlowMatch，2024|普通条件FM；作者默认推理10步，U-Net；同时提供RGB配置|纯FM小网络的具体源码参考；需接入RoboTwin/Dojo|
|AdaFlow，NeurIPS 2024|FM另学方向的条件方差，根据难度调节积分步长；容易处可一步，复杂处多步|不是预先强制所有动作一步；可用于研究步数差距|
|FlowPolicy，AAAI 2025 Oral|直接训练一致性流，目标是一步|适合作为一步方法，不能默认当有明显N10/N1落差的老师|
|ManiFlow，CoRL 2025|FM与一致性联合训练，强调1–2步；RGB/点云|RoboTwin2接入最直接的小策略参考之一；先测其原生N1/N多步差距|
|3D FlowMatch Actor，2025|普通FM与工程优化，典型5步；有单任务训练入口|论文PerAct2多任务消融：N5 85.1%、N3 83.9%、N1 75.2%，证明FM也存在少步性能损失；不是我们的单任务成绩|
|EfficientFlow，AAAI 2026|等变网络＋惩罚生成路径的弯曲，支持一步|另一个“专门训练后一步”的例子，不代表基础FM天然一步|

如果研究“将多步老师压到一步”，优先普通FM老师，或者先证明现有ManiFlow确实存在步数差。若一步已与多步相当，蒸馏方法的提升空间本来就小。

## ManiFlow具体做了什么

它可以直接从专家示范训练，不要求先训一个独立的多步老师。主要有两种练习：

1. **学局部方向**：把专家动作和噪声混合，学习从当前位置往专家动作走的速度，即普通FM。
2. **学终点一致**：从同一份噪声与专家动作形成的路径上，取两个位置；希望从两处预测的最终动作一致。

固定RoboTwin2 RGB代码中约75%样本做FM，25%做一致性。后者用EMA副本给目标：EMA是学生参数的缓慢平均，不是另一个已经训练好的专家模型。

具体源码路径是：同一噪声/专家对构造x_t、x_next → EMA在x_next估算干净终点 → 将该终点换算为x_t处应预测的方向 → 当前网络拟合它。这里的x_next直接由专家与噪声插值得到，**不是CP中的“预训练老师先积分一小步”**。当前网络还接收步长条件。

模型骨架DiT-X负责融合视觉和机器人状态；并非只换一个loss。RGB与点云是两个输入版本。作者RoboTwin2代码可借接入，Dojo支持仍需自己接，不能因RoboTwin可跑就假设Dojo即插即用。

源码细节：读到的target step size复用了当前步长；论文文字描述的独立采样与此不完全相同。把一致性比例改为0还会进入空子批求均值路径。因此“ManiFlow架构＋纯FM”应单独命名、明确处理，而不是当成一项无风险配置改动。

依据：[正式论文](https://proceedings.mlr.press/v305/yan25a.html)、[全文](https://arxiv.org/html/2509.01819v1)、[作者RoboTwin2固定代码](https://github.com/geyan21/ManiFlow_Policy/tree/7ef8d76776f8991f76a29fc4470b5f11f998f1d5)。

## FlowPolicy具体做了什么

它采用点云＋机器人状态、DP3式视觉骨架和U-Net动作网络，从专家示范训练一致性流。通俗说：沿同一条噪声→动作路径，相邻位置应指向相容的终点，方向也应保持一致，从而允许大步生成。

官方代码的loss包含“分段终点一致”与较小权重的“速度一致”两项；默认把生成时间分成2段，部署`num_inference_step=1`。训练的2段不是执行2个机器人动作，也不能据此认为部署调用模型2次。

**论文和当前代码存在需保留的区别**：论文写EMA目标网络；固定源码`compute_loss`的vt/vr都由`self.model`产生，未见目标侧detach/EMA调用。不能把论文公式与仓库实现说成逐项相同，也不宜未经核对就用这个loss替换我们已验收的CP。

依据：[论文](https://arxiv.org/html/2412.04987v2)、[固定loss源码](https://github.com/zql-kk/FlowPolicy/blob/86d10c14972b4eab083abec9026749c340f8e5ed/FlowPolicy/flow_policy_3d/policy/flowpolicy.py)。原实验是Adroit/MetaWorld，本轮没有查到作者的RoboTwin/Dojo插件。

## 现有五种baseline能否迁移

|方法|FM→另一FM基座|FM→经典扩散基座|
|---|---|---|
|直接端点蒸馏|最直接：同观察、同噪声，比较完整生成动作|也可；固定老师采样过程，学生参数化需能直接给干净终点|
|CP|复用一致性、教师桥接、EMA；适配时间条件与边界|CP本来来自扩散；EDM/噪声日程与FM版本公式不能混用|
|DMD/OneDP|复用学生/老师/辅助及分布匹配；速度→score换算与时间权重要正确|原DMD/OneDP本来就是扩散路线；应使用对应噪声/score公式|
|MFD|与速度场最接近，保留两路区间积分和平均方向差|需先转换为合适的概率流速度，不能把噪声预测值当速度积分|
|固定老师FQL|复用端点约束＋Q学习；另需真实转移数据|可借相同结构，但属于扩散教师适配，不能直接称原版FQL|

所以不需要因换成小型策略而放弃这五类方法；需要一个统一且正确的模型接口。单任务/多任务不改变这些基本目标。相同公式在新基座上也不自动继承原论文性能保证，尤其是CP专家监督、FQL奖励数据与纯蒸馏的信息来源不同。

补充固定源码：[PointFlowMatch模型/默认10步](https://github.com/robot-learning-freiburg/PointFlowMatch/blob/d46cc98cf07a15fc264dddfddf5085880749a0ab/conf/model/flow.yaml)、[RGB配置](https://github.com/robot-learning-freiburg/PointFlowMatch/blob/d46cc98cf07a15fc264dddfddf5085880749a0ab/conf/experiment/pointflowmatch_images.yaml)、[AdaFlow代码](https://github.com/hxixixh/AdaFlow/tree/1ceeeb48dae2acdbe3de516ed36736ad2c4d293a)、[3D FlowMatch Actor消融](https://arxiv.org/html/2508.11002v1)、[EfficientFlow作者项目](https://efficientflow.github.io/)。
