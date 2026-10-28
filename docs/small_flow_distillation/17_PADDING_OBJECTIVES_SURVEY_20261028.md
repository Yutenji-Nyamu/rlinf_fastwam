# 填充维评分：源码分类与我们应怎样选择

2026-10-28，讨论记录。只审阅源码和完善规划，未修改训练、队列或检查点。此前现场状态仍以16:54快照为准。

## 结论

**没有“所有机器人策略都排除填充维”的统一做法。** 排除填充维有多个主流实现支持；把补零后的完整向量作为生成目标也有官方实现支持。应先明确“这18个数是不是需要模型学会生成的变量”，再决定损失。此前“CP两项都需改14维”的措辞过强，降为候选方案。

模型内部32维、机器人实际14维、预测50个动作、执行多少动作，是三个独立设置。排除18个坐标不等于只训练前14个动作，也不缩小模型。

## 代表性代码，按实际损失分类

|类别|核到的实现|实际行为|能说明什么|
|---|---|---|---|
|补零坐标也评分|OpenPI JAX/PyTorch|先补到模型维度；速度误差不裁动作维，训练入口求平均|完整32维目标是正式存在的选择|
|输入告知哪些有效，损失仍全维|RDT|action_mask拼到模型输入；最后`F.mse_loss(pred,target)`无损失mask|不能因有mask字段就断言填充误差被排除|
|只评实际动作维|LeRobot π0/π0.5|先将逐元素loss裁到实际action shape，再聚合|同为π0.5，也存在只评真实坐标的实现|
|分别处理动作维与回合末尾|SmolVLA|裁实际动作维；另外处理action_is_pad时间位置|坐标填充与时间填充应分别审计|
|按有效元素归一化|GR00T N1.7|逐元素速度MSE乘action_mask，再除有效mask总数|不把被遮掉的位置计入分母|
|固定机器人，直接建真实维度模型|原Diffusion Policy、原CP|action_dim来自任务shape_meta，输出就按真实动作维构建|原代码全维损失，不代表支持给π0.5额外18维评分|

源码锁定与定位：

- OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`：[补零](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L328)、[JAX损失](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L198)、[PyTorch聚合](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train_pytorch.py#L530)。
- LeRobot `9f45b7b37cc4b33f8ecdb503aa4d209370bd528d`：[π0.5](https://github.com/huggingface/lerobot/blob/9f45b7b37cc4b33f8ecdb503aa4d209370bd528d/src/lerobot/policies/pi05/modeling_pi05.py#L1596)、[π0](https://github.com/huggingface/lerobot/blob/9f45b7b37cc4b33f8ecdb503aa4d209370bd528d/src/lerobot/policies/pi0/modeling_pi0.py#L1150)、[SmolVLA](https://github.com/huggingface/lerobot/blob/9f45b7b37cc4b33f8ecdb503aa4d209370bd528d/src/lerobot/policies/smolvla/modeling_smolvla.py#L315)。
- GR00T `944fe70c4fd54e3fcb80a7d7258a328f9c061176`：[loss](https://github.com/NVIDIA/Isaac-GR00T/blob/944fe70c4fd54e3fcb80a7d7258a328f9c061176/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L276)、[构造mask时同时去除补齐时间与补齐动作维](https://github.com/NVIDIA/Isaac-GR00T/blob/944fe70c4fd54e3fcb80a7d7258a328f9c061176/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py#L557)。
- RDT `cd79363a1387e8f81c7724d070ef7e45fd23150f`：[输入与损失](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/cd79363a1387e8f81c7724d070ef7e45fd23150f/models/rdt_runner.py#L198)。
- DP `5ba07ac6661db573af695b419a7947ecb704690f`：[动作维来自任务](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py#L46)。其condition_mask用于条件位置，不可一概解释为不存在的关节。
- CP `eed0c465d3c1aa6dcf22f10e784df07dec06beac`：[模型构造及动作维](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/consistency_policy/ctm_policy.py)。

这些是固定代码版本的行为，不是对每个项目所有历史版本的断言。OpenPI的[action-mask提案#871](https://github.com/Physical-Intelligence/openpi/pull/871)本次查询仍开放；提案意见不等于维护者已确认的统一结论。

## 为什么补零仍然有训练任务

把一行想成“14个真实机器人指令＋18个接口空位”。干净数据里的空位确实为0，但FM训练先往全部坐标加噪。

我们的时间定义为0是动作、1是噪声：`x_t=(1-t)*a+t*epsilon`，速度目标`epsilon-a`。在补零坐标，`a=0`，于是输入为`t*epsilon`，速度目标仍为`epsilon`。**全维FM是在教模型把这些含噪坐标还原到零，并非在反复比较零与零。**

因此有两个自洽目标：

1. 真实动作要对，空位也要正确还原到零。额外约束可能稳定内部表示，也可能与真实坐标竞争共享参数；目前没有测出我们的净作用。
2. 只给机器人实际使用的坐标评分。目标直接、与端点蒸馏范围一致；空位不再受到这项损失的直接约束。

两种目标都不能仅凭代码风格宣称更强。若补齐位置表示“缺失、没有标注”，直接当零目标会加入假标签；若明确把低维动作嵌入带固定零的高维分布，则全维生成训练具有明确定义。

去掉输出评分也不会消除所有影响：动作输入经过共享线性层，填充坐标的噪声仍可能影响真实14维预测。不能把loss mask说成已完全删除18维。

## 评分范围改变也会改变数值尺度

对普通逐坐标MSE，设真实坐标平均误差为R、填充平均误差为P：

`L32=(14*R+18*P)/32`；真实维归一化后的`L14=R`。

若只把填充误差乘零、却仍除32，得到`14/32*R`，不等于按14维平均。变化包括目标与梯度尺度；Adam和梯度裁剪使其不能简单等同于统一改学习率。我们的CP特殊距离还有依赖维数的delta，选择14维时也须记录对应变化，见15。

## 我们的π0.5与各方法

本地冻结基线为`d008febaa`。`rlinf/models/embodiment/openpi/openpi_action_model.py::sft_forward`默认全维平均；`use_action_chunk_loss=True`才同时截执行段与真实动作维。DAgger worker显式开启此选项。**某条SFT路径默认如此，并不能证明已下载权重当年的训练设置如此。**

Sidney多任务权重和RLinf单任务权重的本轮核查，未补齐足以断言历史mask方式的完整训练版本及配置链。当前LeRobot处理方式不能倒推Sidney权重当年必然只训练14维；模型来源、加载实现与我们当前蒸馏损失须分别记录。

当前各方法的实际评分见14。为何不要求它们所有分支一刀切：

- **端点蒸馏/FQL学生**：比较最终机器人动作，14维是直接的任务目标；FQL的Q也只看真实执行动作。
- **CP一致性**：比较两条内部路径的最终输出。可选只比较真实动作；保留32维则还要求内部空位一致。
- **CP的FM监督**：直接学专家示范。全32维遵循OpenPI式零填充生成目标，14维遵循LeRobot式真实坐标目标。两者均有来源。
- **DMD/MFD辅助网络**：拟合的是加噪后的学生完整生成分布。裁辅助loss会改变其拟合约束，不能由“学生只评分14维”推出辅助也必须裁。当前学生14维指导是完整模型指导的投影，不宣称是已严格验证的14维边缘分布目标。

## CP new保留的选择，尚未执行

|选择|一致性评分|专家FM评分|适合回答什么|
|---|---|---|---|
|现有对照|32|32|原适配结果|
|候选A|14|14|两项都集中学习实际机器人坐标；与LeRobot和现有端点学生的范围对齐|
|候选B|14|32|只修改一致性比较范围，保留OpenPI式专家监督约束|

“都改14维”仍是合理候选，但**不是查到确定bug后的必改项**。先理解并决定目标，再与dropout分开定义；不会据此自动开三组实验。证据不足以把旧CP表现归因于填充维。

来源快照在本地`cp_padding1028/sources.json`，只保存研究证据；未向公共仓库复制整套上游源码。
