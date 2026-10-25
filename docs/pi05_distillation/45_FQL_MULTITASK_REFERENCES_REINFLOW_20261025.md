# 多任务一步策略、多任务Q与ReinFlow实现

2026-10-25。与[44号数据核对](44_FQL_OFFLINE_MULTITASK_DATA_20261025.md)配套；本轮无FQL训练。

## 相关工作按能借鉴什么分开

|原始来源|确切相关点|我们能借什么／不等同什么|
|---|---|---|
|[FastOPD](https://arxiv.org/html/2610.02832v1)，§4、附录B|统一语言条件学生在RoboTwin50任务上蒸馏；支持一步，使用多任务示范|多任务一步蒸馏有先例。其RoboTwin老师是LingBot/FastWAM，π0.5老师对应LIBERO；不是已经提供我们Sidney基座的FQL|
|[SnapFlow](https://arxiv.org/html/2604.05656)，附录I/K/L|π0.5一步、冻结VLM而训动作专家与时间适配；LIBERO评估|可参考π0.5一步模型切口；真实机器人四个布置分别训练，不能把“多个任务结果”一概说成一套联合模型|
|[RAPolicy](https://arxiv.org/html/2609.22888v1)，§III/IV|π0.5一步actor，真实机器人5任务联合；有任务条件的状态与块价值学习|最接近“一步VLA＋多任务价值”。可参考冻结VLM特征、动作块编码、成功失败回放；它是在线RL，Q/V目标和actor加权方式均不是FQL|
|[Q-Transformer](https://proceedings.mlr.press/v229/chebotar23a.html)|多任务机器人离线RL，混合人类示范和自主数据学习Q|证明共享多任务Q路径；但动作离散化且逐维自回归，不能原样拿来给我们的连续动作做dQ/da|
|[TD-MPC2官方world_model.py](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py)|共享Q集合接受task embedding，`Q(z,a,task)`显式拼任务条件|可参考多任务Q输入/共享参数；它不是VLA一步蒸馏，也不应把世界模型规划整套引入|

结论是“已有几块可复用依据”，不是已找到完全相同的“Sidney π0.5、RoboTwin50、固定10步老师、共享Q、纯离线一步FQL”可直接跑脚本。原FQL按env_name独立实验，不是算法禁止多任务。

RAPolicy更具体：状态含图像、机器人状态和语言；actor直接使用π0.5单步求解，没有另做10→1蒸馏。critic从回放动作学习Q与V，下一目标用V，避免FQL那种生成新动作后查Q；actor依据优势给回放动作加权，复用采集时初始噪声，不通过Q对动作的梯度改策略。它的52%→88%是论文联合5任务结果，不能搬作我们50任务预期。20 critic/5 actor及2560 critic预热也不是本项目已选参数。

## ReinFlow本身是什么

[ReinFlow](https://reinflow.github.io/)是NeurIPS2025在线强化学习方法：让机器人执行并收集奖励，在流模型每个生成步骤加入可学习高斯噪声，使这些转移的概率可计算，再用策略梯度/PPO类目标更新预训练流策略。噪声模块训练时用于探索与概率计算，部署可移除；兼容一步或少步策略。它本身不是“冻结老师＋一步学生＋Q”的FQL蒸馏。

仓库中`ft_baselines/fql.py`是作者提供的FQL对照方法，注释明确从官方JAX代码翻译到PyTorch。将这份文件称为“ReinFlow算法”会混淆。

## 这份PyTorch FQL代码逐块读

固定版本：`e722e151bed767f3ffef47527cf697f2358af55d`。

|组件／函数|通俗作用|原文件实际行为|
|---|---|---|
|`bc_flow: ReFlow`|用离线动作学一个多步行为模型|`loss_bc_flow`做FM；不是永远冻结的老师|
|`OneStepActor`|当前状态＋随机噪声直接产生动作块|两层512宽隐藏MLP，输入是`cond['state']`；不是π0.5视觉语言动作专家|
|`critic/target_critic`|打分器与较慢更新的参照打分器|双Q；目标用两Q平均，不能因为SAC习惯就悄悄改min|
|`loss_critic`|真实动作应该值多少|下一动作由一步actor产生，目标为奖励＋未终止时的折扣下一Q；拟合数据动作的Q|
|`loss_actor`|既像老师，又更高分|老师/学生同初始噪声；FM＋α端点MSE－平均Q。老师采样有no_grad；FM只训B，MSE/Q训G|
|`update_target_critic`|让参照打分器缓慢跟随|软更新；不是每次硬复制|
|`forward(mode='onestep')`|部署生成动作|只跑学生；`base_model`才走多步B|

`onestep_expert_bc_loss`虽然记录“学生对示范动作MSE”，但只作日志，不在返回优化目标里；不要把它误说成该学生另有专家动作监督。

配套`agent/finetune/flow_baselines/train_fql_agent.py`负责offline数据/online FIFO、当前与终止后的真实观察、Q/actor优化器和目标Q更新。其配置主要是状态输入的Gym/Kitchen，不是验证过的π0.5 RoboTwin50实现。它的gamma按执行块定义；我们的C10要明确折扣口径。[-1,1]裁剪亦是它的动作空间约定，不能照搬到不同归一化。

## 到我们项目，实际改哪里

1. 复用当前直接蒸馏的冻结视觉语言前缀、老师N10、学生可微N1、H50、14维物理动作mask、训练/导出/续训框架。
2. 若采用用户的冻结老师版本，去掉B优化器和FM更新；保留老师无梯度目标。不能删除学生的Q梯度链。
3. 新增共享、看得见任务的双Q与目标双Q。优先复用冻结VLM特征，接关节状态和实际执行前缀的动作编码；Q的特征输入固定，Q网络本身训练。
4. 新增真实转移读取器；现有DemoWindows保持供纯蒸馏用。数据/末帧问题见44号；复用EXPO折扣和转移公式，不复用Editor或熵温度。
5. 起点：1次Q、1次学生更新的原FQL节奏；学习率、α、Q预热和batch在真实奖励/动作量级确认后再定。先前每卡192的容量不保证加Q后仍合适。
6. 保存学生、Q、目标Q、优化器、调度、RNG和采样状态，保持完整续训；推理导出仍只需学生。

ReinFlow代码可借鉴损失拆分、目标Q与离线/在线数据接口；小MLP、状态输入、动作裁剪和具体任务预算需要替换。官方FQL是算法对照来源，ReinFlow是PyTorch实现参考，本地已跑通π0.5蒸馏是实际模型与运行接口依据。

代码来源：[FQL模型](https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/model/flow/ft_baselines/fql.py)、[FQL训练器](https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/agent/finetune/flow_baselines/train_fql_agent.py)、[老师采样的no_grad](https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/model/flow/reflow.py#L157)、[原FQL代码](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/fql.py)。
