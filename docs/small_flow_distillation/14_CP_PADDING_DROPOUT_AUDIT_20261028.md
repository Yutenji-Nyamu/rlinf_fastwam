# CP new之前：填充维与dropout核对

2026-10-28。核对本地源码、1机正式部署的实际配置及安装版OpenPI/Gemma；CP new本轮仅规划。此前CP及其200轮续训保持原方法，另起新版才能区分改动效果。

## 填充维不是18个额外关节

π0.5内部每个动作32维，这个机器人实际使用14维，数据补18个零来接统一接口。加噪和网络预测后，这18维的误差不一定为零；把它们计入损失会给共享参数额外的训练方向。移除评分并不缩小模型，也不等于省56%的计算。

不能说“所有训练都必须排除填充维”：[OpenPI变换](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py#L301)补零，[原生PyTorch模型](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py#L336)返回全部坐标误差，[训练入口](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train_pytorch.py#L478)直接求平均。我们的建议是让这组机器人蒸馏的学生目标统一评分真实14维，作为明确的适配选择。

|实际实现|学生评分|其他分支|判断|
|---|---|---|---|
|直接端点|H50×14|老师内部仍积分32维|已排除填充目标|
|当前CP|一致性、FM均H50×32|EMA仍32维|两项都需在CP new中明确mask|
|旧OneDP|H50×32|辅助32维|历史版本已停，不追改历史结果|
|DMD new|H50×14|辅助噪声回归H50×32，分开记录14/18维误差|学生已改；辅助不是漏了同一个mask|
|MFD|H50×14|辅助速度回归H50×32，积分仍32维|同上|
|FQL|端点H50×14；Q仅实际执行段×14|Q状态中的机器人状态也取14维|已排除填充评分|

当前单任务H50/C50，全50个动作执行；此前多任务H50/C10只执行前10个。**时间长度和动作坐标是两个问题**，不能把18维填充误说成尾40个动作。

DMD/MFD辅助网络拟合的是加噪后的完整内部动作分布。仅把辅助loss也裁14维，会改变其余坐标的约束，不能保证得到真实14维的边缘分布评分。先保持完整辅助拟合、保留分项误差；后续若要真正只建模14维，应一起定义填充轨迹与输入条件，单独验证。学生目前也只是使用完整模型的14维指导分量，不能把它包装成已经严格实现的14维边缘KL。

源码：`losses.py::endpoint_loss/cp_loss/fake_loss/onedp_loss`，`mfd.py::auxiliary_loss/student_loss`，`fql.py::actor_objective/TwinQ`；正式DMD配置确为`onedp_loss_action_dim=14`，辅助5:1，非仅函数默认值。

## dropout是什么，为什么CP特别在意

dropout是在训练时临时随机关闭部分内部特征，评估时关闭这种扰动。它不是随机删除机器人动作，也不是修改去噪步数。

CP先得到两份中间动作，再把两份都经过EMA映射回干净动作后比较。可能出现“中间仍不同，最后的网络把两边都修得很像”，导致差异和学习信号很小。原论文提出dropout能缓解这种现象；Square消融只关闭最后s→0的扰动，成功率92%降到86%，其配置p=0.2。该结果不直接预测我们的增益。[论文§IV-D/表IX](https://arxiv.org/html/2405.07503v2#S4.SS4)

已核官方固定提交`eed0c465d3c1aa6dcf22f10e784df07dec06beac`：

- [配置](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/configs/ctmp_square.yaml#L62)为0.2；不能用构造函数默认0误判官方不用。
- [残差块](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/consistency_policy/diffusion_unet_with_dropout.py#L50)在第一卷积块之后、残差相加之后对隐藏特征做dropout。`scale_dropout/bias_dropout`只是命名，代码并非仅给FiLM系数加噪。
- 学生及EMA参加训练期扰动；EMA参数冻结与随机前向并不矛盾。老师冻结并eval。源码没有在每条分支前重置同一个RNG；不能擅自锁死每次mask。`prepare_drop_generators`给普通nn.Dropout附加generator属性，不等于标准PyTorch实际使用了自定义generator。

## 当时为什么没照搬

首次实现提交`7a9310a5c`的`Velocity.__init__`已经调用`model.eval()`。03/05号记录清楚说明沿用已跑通EXPO的冻结前缀、原生π0.5速度接口，并迁入CTM、EMA、损失和时间条件；**没有对dropout给出省略依据或效果验证**。这是方法适配清单的遗漏，不能事后解释成“已经验证不需要”。冻结权重也不要求关闭dropout。

当前GemmaMLP没有dropout；联合前向显式调用eager attention时也未传非零dropout。只改`model.train()`不会复现CP的扰动，还可能改变共享冻结前缀的行为。需要限定在CP动作专家内部增加训练期模块。

## 其他值得处理的点

1. 当前CP一致性损失沿用官方特殊Huber写法：先平方误差，再计算平滑距离；小误差近似四次方。FM则是原生速度MSE。末段约1.6e-8与3.57e-4不能直接比较为梯度贡献；暂无分项梯度证据。
2. 原CP是EDM/预条件U-Net，原动作监督也有时间加权；我们用FM速度监督。1:1只继承了组合权重，不保证两项在π0.5上的相对强度相同。先补少量梯度诊断，不按loss倍数盲目放大一致性。
3. 时间方向、老师小步、目标分支detach、预测分支穿过冻结EMA的输入梯度、EMA导出、完整优化器/RNG保存都按既有设计保留，未发现需要在本轮重写的证据。
4. 旧CP末代108/128=84.38%，直接103/128=80.47%，老师111/128=86.72%。当前并非“CP一直更弱”；早期下降是真实观察，新版是否解决需对照。

实施候选见[15 CP new规划](15_CP_NEW_PLAN_20261028.md)。
