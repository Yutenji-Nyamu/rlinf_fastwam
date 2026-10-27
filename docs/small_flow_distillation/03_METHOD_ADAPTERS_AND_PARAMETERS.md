# 五方法：共用哪些接口、分别增加什么

2026-10-27，规划。新基座尚未选定，以下是实现契约与有来源的候选值，不是已跑通实配。

## 共用模型接口

保留基座自己的相机、预处理、归一化、动作次序、时间方向和执行offset。政策层只暴露：

```text
encode(observation/history) -> context
velocity(noisy_actions, time, context[, target_time]) -> velocity
endpoint(initial_noise, context, steps) -> complete_action_block
execution_slice(complete_action_block, C) -> actual_actions
```

统一比较同一老师的五个学生。每任务单独一个老师和五组学生；五任务不是先混成联合训练。teacher/normalizer/source revision固定；教师评估与学生评估同协议。选择EMA教师时，五方法都从同一EMA点初始化；学生是否另用EMA是方法配置，公开记录。

SmolVLA：现有`embed_prefix`/`denoise_step`/Euler接口最接近π0.5；32是内部维，机器人实际维由feature schema确定。DP/FMP：原生14维，不增加18维填充；O3/H8/C6执行片段是索引2开始，不能沿用π0.5的`[:C]`。

时间统一封装在model adapter内。如果内部用噪声1、动作0，则`x_t=(1-t)a+t*z`，速度目标`z-a`，向小时间走`x-=dt*v`。来自HRI/ManiFlow的噪声0→数据1必须明确反向转换；不能只改变量名。CP的第二个时间输入也需训练/推理一致。

## 五方法的数据流

|方法|学生从什么信号学习|额外可训练组件|数据需求与复用|
|---|---|---|---|
|直接端点|同观察、同噪声，学生N1追教师多步的最终完整动作块|无|示范观察即可；老师在线前向生成目标。复用`losses.py:endpoint_loss`数学结构|
|CP适配|老师短桥接＋EMA跨阶段目标，另有专家动作FM监督|学生的target-time条件；EMA是平滑副本|需要观察和专家动作；复用`cp_loss/flow_map/ema_decay`结构，FM时间采样随新基座|
|DMD new|学生生成→加噪→比较教师和辅助方向，给学生梯度|辅助速度/噪声网络|观察即可。辅助拟合学生行为；5次辅助/1次学生是现有new设置，不称原OneDP默认|
|MFD|教师与辅助从共同含噪起点各积分一段，比较平均速度|同DMD辅助网络|观察即可。复用已验收批量区间采样/无梯度积分；辅助用速度MSE，不误用DMD时间权重|
|固定老师FQL|端点蒸馏约束＋提高Q评分|双Q及慢更新目标Q|必须是真实状态/实际动作/奖励/下一状态/终止；复用当前FQL的结构，重接当前基座特征和执行段|

FQL原版还训练行为流模型；此处冻结老师，是已经讨论的变体。FQL用了奖励，CP用了专家动作监督，不能把五者统称“老师信息完全相同的纯压缩”。论文比较时列出数据权限与新增训练成本。

## 已有代码切口

现有工作树`rlinf/algorithms/pi05_distillation/`：

- `losses.py`：直接端点、CP、OneDP目标可抽取；直接函数目前写死N10/14维，CP的FM时间采样也是π0.5设定，需参数化而不是原样import。
- `mfd.py`：批量区间K采样、Euler平均速度、FP32累积可直接作为数学参考；默认14维需取schema。
- `fql.py`：双Q输入目前假定π0.5前C10，必须改为基座`execution_slice`后的实际动作；保持Q对学生动作的梯度，教师/辅助不反传。
- `prefetch.py`、`timing.py`、`checkpoint_retention.py`：复用已验收供数、分项耗时、保留完整续训点的设计。
- `data.py`不整包照搬：其完整H50窗口限制排除了末49个输入观察；新读取器应有原生episode边界padding/mask，端点方法不因缺未来专家动作而丢当前观察。
- `model.py`是π0.5模型包装，不能用作通用小网络。新增独立SmolVLA或FMP包装；保持旧训练入口默认行为不变。

完整续训保存学生/辅助或Q、各优化器、scheduler、EMA、RNG、normalizer、配置、数据split和老师revision。导出只保留部署学生及其必要前缀，但不能把这种导出当完整续训点。

## 参数归属表

|参数|来源与初始依据|新实验处理|
|---|---|---|
|视觉、状态、动作顺序/归一化|所选模型官方数据和部署代码|逐项继承；不混SmolVLA mean/std与DP limits|
|模型宽深、H、O|SmolVLA H50/O1，FMP沿RoboTwin DP H8/O3|不先缩模型或改H；先只改变部署C/N|
|教师SFT优化器|SmolVLA官方AdamW1e-4、β(.9,.95)；DP官方AdamW1e-4、β(.95,.999)|分别继承本模型；核实实际checkpoint配置覆盖默认值|
|教师训练预算|SmolVLA脚本100k更新；DP配置600epochs|是参照上限，不能照抄成相同更新数；先按该任务窗口量换算，并依据中评曲线讨论收束|
|直接端点学习率|现有π0.5为2.5e-5；小DP/SmolVLA常规训练为1e-4|新基座建议从其动作模块训练尺度起步，短测更新幅度；不因batch翻倍就自动LR翻倍|
|CP一致性与专家监督|官方CTM+DSM权重1:1、80bins、EMA power.75|flow适配保留两项结构；FM替DSM是模型适配，日志分别记两项及有效维度|
|DMD学生/辅助LR|OneDP论文1e-6/2e-5、β(0,.999)；现有π0.5new为用户指定×10|若迁移DMD new，候选1e-5/2e-4和5:1保留并标适配；不把×10说成官方小模型值。基座确定后用梯度/更新量锁定|
|DMD时间范围|OneDP DDPM离散[2,95]；现有flow适配[.02,.95]|使用连续flow约定，记录端点截断；不能套用原DDPM生成输入时刻65|
|MFD积分/辅助|作者实现Euler最大步长.02，辅助速度回归|继承已验收实现；LR和5:1属于我们的DMD-new继承，不是作者机器人参数|
|FQL双Q、目标软更新|原FQL双Q平均、目标tau=.005、MLP512×4|可继承；输入改为新观察特征/真实执行段|
|FQL蒸馏权重/Q学习率|原FQL α10、3e-4，小网络默认|Q头可从3e-4起；学生仍按基座，不把3e-4直接套动作专家。记录Q与蒸馏项量级|
|FQL折扣/终止|原FQL按环境步；现有π0.5按动作块适配|锁清“每动作”还是“每块”。若每动作γ，则块目标用累计奖励和γ^实际执行数；同一Q不能偷偷换C。时间耗尽与成功/失败终止分别处理|
|训练batch|官方模型batch参考＋单卡实测|选实际样本/秒较高的值；统一学生样本预算，记录学生/辅助实际更新数|
|评估并发、频率|我们现有RoboTwin/Dojo已跑通调度＋本轮用户意图|见04；不从batch推断仿真可用并发|

新基座未定，不能现在假装已经有可靠总时长、最佳LR或最大并行。定下基座后，正式运行表应将每行从“来源/候选”变成“实配/短测结论”。本轮未调整任何旧方法参数。

## FQL额外数据

HF专家示范足够给教师/CP做SFT，其观察也可用于直接/DMD/MFD；纯蒸馏不必预采大批老师答案。FQL需要约128回合的真实交互作为首任务候选预算，按稳定并行的整数倍采，成功失败都保留；不是只采失败。

保留观察及必要历史、归一化前实际动作、完整预测块、执行mask、奖励、下一观察、done/truncated、剩余动作预算、seed与行为模型版本。旧π0.5 C10决策记录不保证存在FMP C6的真实中间下一状态，不能按动作序列长度猜出来。跨C时重新构造有证据的转移或独立采集。只要成绩的普通评估仍不保存视频/逐动作大数组。

## 方法来源

- [CP官方实现](https://github.com/Aaditya-Prasad/consistency-policy/tree/eed0c465d3c1aa6dcf22f10e784df07dec06beac)、[论文](https://arxiv.org/html/2405.07503v2)：原版EDM/CTM，不是原生flow CP。
- [OneDP正式论文](https://proceedings.mlr.press/v267/wang25ba.html)：三网络、学习率、时刻、梯度归属；[DMD2](https://arxiv.org/html/2405.14867v1)：增加辅助更新次数的依据。两者不要合并宣称为同一原版。
- [MFD论文](https://proceedings.mlr.press/v306/zhao26bx.html)、[固定源码](https://github.com/happyw1nd/MFD/blob/bd882924fe24ce59fac0e225f7c82db10db29cb8/t2i_SANA/scripts/train_sana_mfd.py)：原图像模型参数不直接搬到机器人。
- [FQL官方JAX](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/fql.py)、[正式论文](https://proceedings.mlr.press/v267/park25f.html)、[ReinFlow PyTorch对照](https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/model/flow/ft_baselines/fql.py)。
- 我们的[π0.5模型适配](../pi05_distillation/03_PI05_ADAPTATION.md)、[FQL实现](../pi05_distillation/49_FQL_IMPLEMENTATION_20261026.md)、[DMD new](../pi05_distillation/59_DMD_NEW_SMOKE_AND_FQL_HANDOFF_20261026.md)、[MFD](../pi05_distillation/66_MFD_IMPLEMENTATION_20261026.md)。
