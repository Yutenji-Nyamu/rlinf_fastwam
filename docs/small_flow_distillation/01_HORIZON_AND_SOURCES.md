# 监督整块，还是只监督执行段

2026-10-27。H=一次预测的动作序列长度；C=重新观察前实际执行的动作数；O=历史观察数；N=生成这一块调用动作网络的次数。C1与N1不是同一件事。

## 代码证据

|实现|已核训练目标|执行|结论|
|---|---|---|---|
|原Diffusion Policy图像U-Net|完整动作窗口加噪，噪声预测MSE，按condition mask处理|`predict_action`截取`O-1:O-1+C`；常用配置H16/O2/C8|完整动作序列训练、局部执行|
|RoboTwin官方DP旧固定配置|H8全块噪声MSE，14维|O3/C6，执行索引2到7|H8含与历史观察对齐的两个动作位置，不是未来8步只扔掉尾部2步|
|Consistency Policy官方CTM|`batch['action']`全块进入CTM和DSM两项|所读square配置H16/O2/C8|一致性训练本身并不要求只训执行段|
|SmolVLA原论文及现代码|H50全块FM，现代码排除无效轨迹填充及非物理动作维|论文仿真默认C1，并比较C1/10/30/50；N固定10|“预测很多、只执行很少”已有直接先例|
|HRI Flow Matching Policy示例|完整H16轨迹做速度MSE|示例O1/C8|纯FM也常如此；该示例采样噪声另有问题，不整包照抄|
|RL-100公开flow直接蒸馏|同初始噪声，教师多步/学生一步；先`_extract_action`再MSE|只比较实际执行片段，且该helper会反归一化/裁剪|支持执行段端点蒸馏；不能说所有RL-100分支均如此|
|原DP lowdim可选分支|`pred_action_steps_only=True`时截取监督动作|连生成张量长度也变为C；默认False|这是缩短生成对象，不等于保留H但只给C打分|

这里的“更常见”限定为本轮核到的主流DP、CP、FM策略代码，不是对全部文献做数量统计。OneDP论文把动作块作为联合随机变量；没有确认其官方训练源码的执行mask，故不把论文H16直接冒充代码证明。

## 为什么常用整块

预测完整轨迹可同时约束动作间的联系；一段动作窗口随训练起点移动，某次的后段会成为另一次的前段。执行时提早重新观察，是为了及时纠偏，因此训练H与部署C可以不同。完整监督也便于在不重训的情况下比较不同C。

这不是说后段一定帮助前段。有限模型容量、不同动作位置误差尺度和梯度竞争，都可能令优化偏重后段。执行段监督直接集中在立即影响环境的部分，尤其适合端点回归；代价是未监督段质量不再受该损失约束，改变C后可能不再适用。

## 对当前π0.5的判断

当前直接蒸馏H50×14、执行C10是有依据的适配。整体MSE下降不证明前10项同样下降；也不能把后40项算成80%无效计算。网络依然联合处理全部位置，仅mask掉尾段损失通常不省80%前向计算。

最干净的下一轮比较是：同老师、学生初始化、数据、种子、H50/C10、训练样本预算与学习率，只改变端点MSE使用H50还是前C10，并记录前10/后40各自误差。可另外测共享参数梯度的量级，但先不加复杂诊断。本轮不改正在训练的实验。

DMD/MFD的教师与辅助估计联合动作分布。把输出梯度后40项置零，不自动等价于精确匹配“前10项的边缘分布”；若做该变体需单列说明。不要把端点MSE的简单mask推广成所有方法都严格等价。

## 固定来源

- [原DP图像策略](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py)：`compute_loss`与`predict_action`；[lowdim可选短输出](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_lowdim_policy.py)。
- [RoboTwin DP配置](https://github.com/RoboTwin-Platform/RoboTwin/blob/fdcfc753e23c8efe04dfb0d09b4b29f177ea5645/policy/DP/diffusion_policy/config/robot_dp_14.yaml)及同提交`diffusion_unet_image_policy.py`；历史固定配置，当前主仓已将策略接入转向XPolicyLab。
- [CP策略](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/consistency_policy/student/ctm_policy.py)、[配置](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/configs/ctmp_square.yaml)、[论文](https://arxiv.org/html/2405.07503v2)。
- [SmolVLA原论文§4.3、表12/13](https://arxiv.org/html/2506.01844v1#S4.SS3)、[当前官方forward](https://github.com/huggingface/lerobot/blob/dd5313e305ff7aaa5b6c7b4769d1b26650ce250b/src/lerobot/policies/smolvla/modeling_smolvla.py)：约319–352行排除padding，未按执行C截断；模型速度目标约742–775行。
- [HRI示例](https://github.com/HRI-EU/flow_matching/blob/516e8e18875b27741bdbdb8a252e904032c90723/examples/flow_pusht.py)：训练Gaussian、推理示例uniform不一致，参考其FM结构而不照搬采样脚本。
- [RL-100 flow蒸馏](https://github.com/Lei-Kun/RL-100/blob/6b6afceb7eb12525398207b5d6e66908863dc385/RL-100/rl_100/policy/rl100_3d.py#L1239)：切片、归一化尺度、在线时教师更新语义均与我们的固定教师需区分。
