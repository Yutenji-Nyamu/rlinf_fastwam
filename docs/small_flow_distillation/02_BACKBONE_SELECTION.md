# 原生流策略候选与双基准接入

2026-10-27。目标是成熟且快速的单任务方法试验台。单任务指训练/评估范围；VLA也可以只微调一个任务，不必等同“完全没有语言模块”。

## 候选比较

|候选|是否原生FM、规模/结构|RoboTwin证据|RoboDojo证据|适合程度|
|---|---|---|---|---|
|**SmolVLA单任务**|是；约450M总参数、100M动作专家，冻结VLM的训练已有原论文依据|RoboTwin官方XPolicyLab目录列入；共用动作/相机接口可接；本轮未确认可直接下载的这五任务高分权重|XPolicyLab有单任务训练批脚本、真实合批推理；官方榜有Single Task一行|双基准工程路径最接近，优先核验老师质量|
|**ManiFlow**|是，另含一致性；所读RGB配置DiT-X 12层、768宽，视觉R3M；不报未经统计的参数量|作者官方robotwin2.0分支，有RGB和点云训练/评估脚本|本轮固定XPolicyLab树无ManiFlow目录，未找到官方Dojo集成|成熟单任务研究代码；已针对少步优化，未必适合制造多步老师/一步学生差距|
|**FMP / Flow-DP U-Net**|纯FM；复用ResNet18+1D U-Net小策略，需训练教师|可借RoboTwin DP的数据/动作/模型骨架；不是已经跑通的现成FM权重|可借DP/XPolicyLab适配，但本轮无现成FMP Dojo成绩证据|无VLM、模型结构简单，机制研究最直接；工程移植多一些|
|**3D FlowMatch Actor**|原生FM，3D视觉条件，可做单任务；作者公开单/双臂代码与权重|本轮未核到现成官方接入；已公开代码主要是RLBench/PerAct|本轮未核到现成官方接入|有成熟3D策略谱系，但点云、末端位姿与现有关节动作接口差异较大|
|**FlowPolicy**|一致性流匹配，点云条件，本身以一步为目标|作者公开验证为Adroit/MetaWorld，未核到RoboTwin接口|本轮未核到现成官方接入|AAAI 2025 Oral，有方法影响力；更适合当少步方法参考，而非默认多步老师|
|**π0/π0.5单任务**|是；已知大VLA路线|我们已有成功流程|我们已有成功流程|兼容性最熟，但不满足优先缩小模型、加快迭代的目的|
|经典DP / ACT|经典DP是扩散，ACT是CVAE回归|接入成熟|接入成熟|不能因为有action chunk就当FM基座|

“影响力”这里采用可核证据：SmolVLA由Hugging Face/LeRobot维护且发布权重；ManiFlow是CoRL 2025工作，后续相关论文使用其作为比较对象；FMP已有研究与后续实现参考。没有用整个LeRobot仓库stars当成SmolVLA论文引用数，也没有编造引用统计。

尚未找到同时满足“小型无VLM、原生纯FM、两边现成高质量老师、广泛使用”全部条件的现成方案。SmolVLA优先的是双基准接入和模型接口；FMP优先的是轻量和机制简单。3D FlowMatch Actor的官方训练/评估还使用点云与末端位姿轨迹，不能把其RLBench速度或模型参数优势直接当成我们的RoboTwin实测。FlowPolicy、ManiFlow已经带少步训练目标，也不该仅因名称包含Flow就默认存在很大的N10/N1差距。

## SmolVLA：最值得继续核验的路径

- 原生时间为噪声1→动作0，学生可取`z-v(z,1)`；`model.denoise_step`返回速度，`sample_actions`做Euler积分并复用视觉语言KV。这与现有π0.5损失接口很接近。
- 默认H50、N10、O1，32维内部上限；真实14维和轨迹末端padding由当前官方forward处理，不能复现旧OneDP全32维误计分问题。
- XPolicyLab固定脚本B512、100000更新、32供数worker是**参考配置，不是本机已验收吞吐或必须跑满的预算**。上游默认学习率1e-4；论文预热100与当前代码1000不同，运行时必须锁定版本/明确配置。
- 推理adapter的`infer_batch_payloads`真正合并输入，调用一次`predict_action_chunk`，并按env_id返回；可以沿用模型合批结构。训练脚本默认数据名称偏Dojo，RoboTwin要显式指定正确LeRobot repo/schema。
- 原论文约100M动作专家，意味着较π0.5动作部分明显小；不保证墙钟时间按参数比例缩短，视觉前缀、数据解码、仿真仍需测。
- Dojo官方榜本轮检索的Single Task行显示约0.85%总体成功率（对应全基准条件），不支持直接假设它是合格的每任务50%老师。应先看目标简单任务的clean/in-domain成绩；不能以“有adapter”推断“有高分模型”。
- SFT阶段按上游训练范围；蒸馏阶段冻结VLM和教师前缀，明确state projector是否共享/冻结。若student state projector更新，不能无条件使用预计算的教师state KV替代学生前缀。

## ManiFlow：可以适配，但不是换个名字的原始多步FM

实读作者固定版本`7ef8d76776f8991f76a29fc4470b5f11f998f1d5`。

RGB配置H16/O2，`n_action_steps=16`；源码`predict_action`却取`O-1:O-1+C`，按这组配置实际最多返回15项，部署是否另有覆盖需实施时核实。主干DiT-X；`flow_batch_ratio=.75`、`consistency_batch_ratio=.25`；velocity有额外target-time条件。主论文为1–2步设计，发布配置`num_inference_steps=10`也不意味着教师仅做普通FM训练。

若要用它研究从多步压缩，选择有两个：保留完整ManiFlow，先实测N1与N10有没有差距；或建立明确命名的`ManiFlow-architecture + FM-only`老师。后者是消融/适配，不能称原版ManiFlow。

具体代码障碍：`compute_loss`无条件运行一致性子批并求mean，把其ratio直接设0仍会进入空batch路径；同时`target_t`传入方式在FM与采样期间不同，需要把普通瞬时速度的定义锁清楚。故不能承诺“只改比例就可以”。

本轮没有把它硬接入Dojo。若后续选它，借XPolicyLab现成batch契约与我们的Dojo故障跳过流程，移植政策侧adapter；不搬一整套新模拟器覆盖现有环境。

## DP/FMP备选的已核配置

RoboTwin历史DP配置`fdcfc753`：头部RGB+14维关节状态，O3/H8/C6；ResNet18+GN、U-Net通道[256,512,1024]、time embed128、kernel5；AdamW1e-4，beta(.95,.999)，wd1e-6，warmup500、cosine、EMA power.75；B128、600**epochs**。官方网页写600steps，与配置和论文600epochs不同，以后两者为准。图像尺寸由相机实配解析，不从π0.5强加三相机/224。

转为FM要重训教师或找到对应FM权重，不能只替换旧DDPM scheduler。统一连续时间、Gaussian起点和速度符号；现有U-Net若传Python浮点会转long，必须传浮点tensor避免时间被截为0/1。

另发现：官方安装文档锁diffusers0.11.1，其DDPM `set_timesteps(1)`给出`[0]`，纯噪声却被当接近干净时刻；低步数`step`还按`t-1`计算。不能用这个调用产生“原始一步差”的结论。经典DP要另定正确DDIM/单步端点协议；FM路径用明确1→0 Euler则没有这一歧义。

公开DP权重确有第三方来源：PACT作者提供七个RoboTwin任务的200条rand示范训练checkpoint；不覆盖这里拟选五任务，也不是FM老师，不直接挪用其成绩。

## 主来源与固定版本

- [SmolVLA论文](https://arxiv.org/html/2506.01844v1)、[HF官方发布](https://huggingface.co/blog/smolvla)、[基础权重](https://huggingface.co/lerobot/smolvla_base)。
- [LeRobot SmolVLA配置](https://github.com/huggingface/lerobot/blob/dd5313e305ff7aaa5b6c7b4769d1b26650ce250b/src/lerobot/policies/smolvla/configuration_smolvla.py)、[模型](https://github.com/huggingface/lerobot/blob/dd5313e305ff7aaa5b6c7b4769d1b26650ce250b/src/lerobot/policies/smolvla/modeling_smolvla.py)。
- [XPolicyLab SmolVLA](https://github.com/XPolicyLab/XPolicyLab/tree/c43a74653bec480ebe82c84770664e8035bc56c9/policy/SmolVLA)：实读README/model.py/train.sh/train_batch.sh/deploy.yml；[RoboTwin官方接入说明](https://robotwin-platform.github.io/doc/usage/xpolicylab.html)、[Dojo官方榜](https://robodojo-benchmark.com/leaderboard?bench=sim)。
- [ManiFlow作者RoboTwin2分支](https://github.com/geyan21/ManiFlow_Policy/tree/7ef8d76776f8991f76a29fc4470b5f11f998f1d5)、[论文/项目](https://maniflow-policy.github.io/)。实读`maniflow_image_policy.py`、`maniflow_image_timm_policy_robotwin2.yaml`。搜索初见的allenai/maniflow是另一个维护副本，本轮最终源码证据采用作者分支。
- [3D FlowMatch Actor官方代码](https://github.com/nickgkan/3d_flowmatch_actor)：本轮读README的模型API、单任务HiveFormer训练、PerAct2权重和吞吐说明；未逐层审计模型。[FlowPolicy官方代码](https://github.com/zql-kk/FlowPolicy)：AAAI 2025 Oral，原生一致性流及Adroit/MetaWorld范围。两者均留作备选，不声称本轮已验证RoboTwin/Dojo支持。
- [HRI FMP示例](https://github.com/HRI-EU/flow_matching/blob/516e8e18875b27741bdbdb8a252e904032c90723/examples/flow_pusht.py)、[CleanDiffuser RF](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/05f17fc9dbeae7c19a5e264632c9ae9aaac5994e/cleandiffuser/diffusion/rectifiedflow.py)。前者训练/推理noise分布不同，后者速度符号与我们约定相反；取公式与组件，不整包复制。
- [经典DP原库](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)、[RoboTwin旧DP](https://github.com/RoboTwin-Platform/RoboTwin/tree/fdcfc753e23c8efe04dfb0d09b4b29f177ea5645/policy/DP)、[XPolicyLab DP](https://github.com/XPolicyLab/XPolicyLab/tree/c43a74653bec480ebe82c84770664e8035bc56c9/policy/DP)。
- [旧DDPM时间网格](https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddpm.py#L164)、[RoboTwin安装版本](https://robotwin-platform.github.io/doc/usage/DP.html)、[PACT已发布DP权重清单](https://huggingface.co/datasets/Ethan-pooh/pact)。
