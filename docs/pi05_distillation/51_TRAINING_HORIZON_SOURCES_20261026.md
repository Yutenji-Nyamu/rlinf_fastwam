# 训练H50，执行C10：本轮源码与文献核对

2026-10-26。结论：**当前两组均训练整块50个动作；评估只执行前10个，再重新观察。** 不是仅训练前10个，也不是训练做50次去噪。H是预测动作数，C是每次真正执行的动作数，N是生成整块时动作专家的调用次数。

## 当前实配

|项目|直接10→1|OneDP-S-flow|已实现CP-flow|
|---|---|---|---|
|训练动作位置|0–49全部|0–49全部|0–49全部|
|损失动作维度|14个真实关节/夹爪维度|内部32维，含18维填充|内部32维，含18维填充|
|训练目标|同观察、同噪声的老师N10终点|老师与辅助网络给出的分布修正方向|路径一致性＋专家动作FM|
|专家示范动作直接作答案|否|否|是，FM项|
|当前C10评估|学生生成H50，执行前10|学生生成H50，执行前10|未在本轮重开|

依据为从1机当前部署目录取回的`losses.py`、`train.py`、`data.py`，三者与工作树对应文件一致。`endpoint_loss()`在动作维切到14维，但没有截短时间轴；`onedp_loss()`和`fake_loss()`对整个[B,50,32]求均值。训练配置没有C10参数；C10写在评估配置。C50→C10重开时按当时批准方案只改评估执行长度。

形象地说：每次学写50行计划，机器人先做前10行，然后看新画面、重写下一份50行计划。后40行仍参与学习，但这次不会直接执行。它们可能帮助学习连贯动作，也可能占用有限的学习能力；需要对照判断。

## 相关工作的实据

|工作|训练覆盖|执行与证据|对我们的意义|
|---|---|---|---|
|CP官方CTM|完整`batch['action']`，示例H16；一致性和DSM均用整块|配置`n_action_steps=8`，推理最后切出执行段|训练比执行长，有直接官方实现依据|
|OneDP正式论文|把动作块作为一个联合分布，论文H16|正文/附录明确预测16个动作；本轮没有找到足以确认所有实验执行前缀的官方训练源码|支持整块分布蒸馏；不能凭论文H16猜它每次也执行16|
|RL-100官方flow直接蒸馏|**老师、学生都先`_extract_action()`，再MSE**|`_extract_action`按`n_action_steps`截取；所核示例H5/C4，`no_pre_action`还会改变有效预测长度|只监督执行段也有直接源码依据；我们的H50全块回归并非逐行照搬|
|OpenPI原生π0.5|流匹配训练覆盖整个`action_horizon`，不是根据客户端C自动截短|预测长度、环境执行长度在不同位置控制|继承H50并不自动代表只学C10|
|Coda|论文在H×D动作块上定义残差监督|π0.5评估明确K10；未见论文明确声称loss只取前10|可以引用其执行设置，不能由K10反推训练mask|
|FastOPD|动作块上的流场/一致性目标；DMD附录也在块与动作维上归约|LIBERO执行K10；不同基座的RoboTwin执行K50或K24|论文明确区分执行频率；不能统称所有工作都H50/C10|

对应一手位置：

- [CP固定配置](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/configs/ctmp_square.yaml#L25)、[训练入口](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/consistency_policy/student/ctm_policy.py#L404)、[执行截取](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/consistency_policy/student/ctm_policy.py#L390)。
- [OneDP正式论文](https://proceedings.mlr.press/v267/wang25ba.html)，式5/6、Algorithm 1、附录D/E；[NVIDIA项目页](https://research.nvidia.com/labs/cosmos-lab/onedp/)本轮未见可核实的官方训练仓库链接。已看第三方`aminamazlin`版本，其DDPM辅助损失权重与我们所用单位epsilon误差不同，因此不把它当官方实现。
- [RL-100执行段提取及flow蒸馏](https://github.com/Lei-Kun/RL-100/blob/6b6afceb7eb12525398207b5d6e66908863dc385/RL-100/rl_100/policy/rl100_3d.py#L1213)，以及[固定flow配置](https://github.com/Lei-Kun/RL-100/blob/6b6afceb7eb12525398207b5d6e66908863dc385/RL-100/rl_100/config/rl100_3d_flow.yaml#L9)。
- [OpenPI原生训练源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)，`forward()`的完整动作目标；实际部署版本仍以本项目固定源码为准。
- [Coda §3.2/3.3](https://arxiv.org/html/2609.21216#S3)、[FastOPD §4.3、附录B/D](https://arxiv.org/html/2610.02832v1)。论文明确内容与我们从公式作的推断在上表分开。

## 下一步最清楚的对照

如要专门优化C10，**先保持模型仍输出H50，只把直接蒸馏误差改为前10×14维**，其余数据、老师、batch、学习率、评估相同。这只改变监督位置，既有RL-100依据，也最容易解释结果。不要同时把网络预测长度改H10；后者还改变注意力与老师行为，且不会只是一个loss mask。

OneDP的前缀选择需另列对照：它原本估计整块联合分布，给学生梯度截前10维不自动等于精确的前缀边缘分布KL。先做物理维度/前缀梯度诊断，再决定沿用联合辅助网络还是另作适配；不能把直接MSE的简单截取结论未经验证搬过来。

全H50不是已证实错误，尾段也不是纯粹浪费。每个动作会随窗口平移出现在其他训练样本的前部，整块联合建模也可能有帮助。本轮完成核对与规划，未改正式损失或启动新实验。
