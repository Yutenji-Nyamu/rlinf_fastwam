# VLA-MBPO 能否接到 RLinf + RoboTwin + π0.5

**研究快照（2026-10-04补注）：**公开代码、论文和权重的核查截止2026-10-03；本文“未运行”“尚缺闭环”等叙述限定于该次调查，不作为当前实验进度。后续组合和执行路由见[OpenDW主上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)，本次仅补充归档，没有重新检索或启动实验。

2026-10-03，只读官方论文、GitHub固定版本和HF元数据；未下载权重、未SSH、未运行训练，未修改其他文件。

## 结论

**论文的仿真实验是 LIBERO 四套件，不是 RoboTwin；另有 ARX-X5、Galaxy-R1 真机实验。** 现成公开 WM 是 LIBERO Goal 早期示例。VLA-MBPO 确实提供 π0.5、主图+单腕图、模型奖励和短分支 PPO 的完整参考，不能据此推导已有 RoboTwin 三相机、14D 控制、state 条件的同等闭环。[论文 §5](https://arxiv.org/html/2603.20607v1#S5)

**需要修正“有 train_robotwin.sh 所以支持”的判断。** 官方 README 列出该命令，但本轮 GitHub API 对两个官方/作者仓库的完整递归树均未找到文件，LAMDA-RL 该链接直接404。实际 `world_model/scripts/` 只有 LIBERO 转换与训练脚本。README是意向/遗留文字，不能作为可执行支持证明。

相对当前 Wan，VLA-MBPO 更值得借鉴的是“同步两图+短分支采样+value连接长期回报”；若去 RoboTwin，主要缺的是本体/相机/动作/state对齐后的训练资产与环境接口。只换WM与完整移植MBPO是两种不同方法。

## 本轮固定的公开版本

|来源|本轮核到的revision|含义|
|---|---|---|
|`LAMDA-RL/VLA-MBPO`|`5e2b21044f2ef2189c22777643cf10e80a83bc13`，2026-08-31|论文官方主仓库；原始OpenPI/JAX策略训练、PyTorch BAGEL服务|
|作者 `Rhx11111/VLA-MBPO`|`d446866b6850cba82dde45f49409990d0dc461f8`，2026-09-10|同样未核到`train_robotwin.sh`或RoboTwin MBPO入口|
|`VLARLKit/VLARLKit`|`901ca4e9ff3fdc849170d2a65b177ccb6d919293`，2026-06-15|论文合作者的PyTorch实现，有RoboTwin普通PPO及LIBERO MBPO，两项并不等于组合已完成|
|`VLARLKit/BAGEL`|`d39d5b127b43726335391c3fbef7b6cecff8a42b`，2026-05-17|WM服务/训练参考，README资产仍标Coming soon|

核验依据：[主仓库固定树](https://github.com/LAMDA-RL/VLA-MBPO/tree/5e2b21044f2ef2189c22777643cf10e80a83bc13/world_model/scripts)、[作者固定树](https://github.com/Rhx11111/VLA-MBPO/tree/d446866b6850cba82dde45f49409990d0dc461f8/world_model/scripts)、[VLARLKit](https://github.com/VLARLKit/VLARLKit/tree/901ca4e9ff3fdc849170d2a65b177ccb6d919293)、[BAGEL](https://github.com/VLARLKit/BAGEL/tree/d39d5b127b43726335391c3fbef7b6cecff8a42b)。时间均来自Git提交UTC日期，不代表实验完成日期。

## 论文、公开配置和权重不是一个覆盖范围

|层级|已经明确的内容|不能推出什么|
|---|---|---|
|论文|LIBERO Spatial/Object/Goal/Long；单轨迹SFT π0.5；真实平台ARX-X5、Galaxy-R1|没有RoboTwin基准成绩；真机双臂不等于已适配RoboTwin|
|RL配置|四个LIBERO名字存在；公开Goal当前实际`repo_id=libero_goal_task_3`，全套项被注释|不能把默认命令称为Goal十任务完整复现|
|公开WM|`Rhx11111/Bagel-goal-256-2500`，29,214,685,336字节权重，未gated|作者明确是早期toy，不是论文最好模型；未核到可直接使用的RoboTwin专用WM|
|公开策略|`Rhx11111/pi05_libero_one_shot`，JAX/Orbax参数及策略norm stats|不是RLinf PyTorch RoboTwin策略检查点，也不是MBPO最终最强策略|

HF本轮固定revision：WM `d65f2d1c77ba4c4712c9890e042777f21a469a23`；策略 `fc07be6ab6409a535ac1666a51351e6e3db595ac`。[WM文件](https://huggingface.co/Rhx11111/Bagel-goal-256-2500/tree/d65f2d1c77ba4c4712c9890e042777f21a469a23)、[策略文件](https://huggingface.co/Rhx11111/pi05_libero_one_shot/tree/fc07be6ab6409a535ac1666a51351e6e3db595ac)。

WM仓库只列`.gitattributes`和`model.safetensors`，没有配套action normalizer/base config；原始启动还要求额外提供BAGEL基座、归一化JSON、离线分支数据。作者另有`bagel_data.tar.gz`和`libero_goal_original_data.tar.gz`公开文件，本轮未下载解包，不能断言其中没有normalizer。这里只确认“WM权重包本身不全”。[官方发布边界](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/README.md)、[当前Goal实配L865–941](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/config.py#L865-L941)。

## 实际WM接口：双图有，双腕和state预测没有

### 图像

服务有两次生成：先用当前主图+当前腕图+动作文本生成下一主图，再用下一主图+当前腕图生成下一腕图，最终返回`next_head`和`next_wrist`。策略侧接回`base_0_rgb`和`left_wrist_0_rgb`，不是左腕+右腕两图，更不是主+双腕三图。输出之后resize到224喂策略。[服务L187–270](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/world_model/world_model_client/world_model_inference_server.py#L187-L270)、[策略环境L183–225](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/libero_env.py#L183-L225)。

这比两个独立WM共享动作更强，因为后生成的腕图会读取先生成的主图；但它是学习出的跨视角一致性，不能保证几何必然正确。若要适配RoboTwin三相机，须明确联合输入与左/右腕生成顺序、视角标识和对应训练样本；不能复制一个腕图到两路。

### 动作及时间

原始RL入口把有效动作维度硬编码为7、执行chunk为10。rollout明确截取`[:, :, :7]`后送WM/真实LIBERO。WM服务的action文本循环自身能遍历输入维度与长度，但训练转换默认也是C10；这只能说明函数可接不同shape，不能说明已训练模型懂14D/C50。

送WM前先撤销策略归一化，之后WM使用自己的`min/max/clip_min/clip_max`把每维动作映射到0–256整数文本；它与π0.5的quantile norm不是同一层。数据转换默认按所给percentile统计WM边界，训练和推理必须使用同一份。RoboTwin还涉及左右臂顺序、关节/末端坐标、夹爪语义、absolute/delta及控制周期，不能只把7改14。[硬常量L27–29](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/libero_rl_utils.py#L27-L29)、[截断动作L1047–1067](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/rl_env.py#L1047-L1067)、[WM归一化L124–177](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/world_model/scripts/convert_libero_data.py#L124-L177)。

WM只生成一个chunk结束时的双图，没有C10中间每步图像。公开Goal是2个chunk的短分支，约20动作后以value bootstrap连接未来回报。C5不是其原始默认；C8也不是已发布训练协议。要保留我们的实际C值，应训练对应时序，或明确改变实验协议，而不是把“支持可变长输入”的代码当作变C已验证。

### state

`libero_env.py:223`明确保留`state=current_obs.state`，没有预测下一机器人state。该LIBERO π0.5配置关闭`discrete_state_input`；这不能自动迁移到依赖关节状态的RoboTwin策略。RoboTwin动作若为绝对关节目标，目标命令也不一定等于执行后的实际关节状态，尤其遇到接触、控制滞后；是否能解析推进必须按具体控制器核验。

### 奖励与终止

使用同一个BAGEL理解分支，只读取生成的主图+任务文本，输出Yes/No；服务通过文本是否含yes转为bool。当前公开rollout把成功锁存为`prev_rewards`，转为termination并在首次done后屏蔽loss；分支末尾的非成功状态保留value bootstrap。[奖励服务L287–329](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/world_model/world_model_client/world_model_inference_server.py#L287-L329)、[rollout奖励与done L1053–1106](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/rl_env.py#L1053-L1106)。

因此公开服务是chunk终点成功分数，不是直接输出Wan当前的逐动作相对奖励序列。论文写chunk内折扣奖励和及`gamma^k`；这份公开JAX代码则对chunk奖励sum后按chunk步用gamma（默认0.99）做GAE。移植时需明确目标口径，不能假定纸面公式与示例全部逐项一致。[GAE L155–187](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/rl_env.py#L155-L187)。

奖励训练转换目前把“成功轨迹最后一帧”标Yes，其余帧标No，再平衡类别。若RoboTwin轨迹在成功后仍保存多帧，需要按原生success逐时刻重新构造标签，不能照抄最后一帧假设。[标签L260–325](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/world_model/scripts/convert_libero_data.py#L260-L325)。

## VLARLKit里的RoboTwin支持到哪一步

这个PyTorch实现值得参考，避免只看到原始JAX就认为必须重新写全部PPO；但仍有明确未接通点：

- `robotwin_adjust_bottle_ppo_openpi_pi05.yaml`是**普通物理环境PPO**，14D、C50、`discrete_state_input=True`；三相机由对应RoboTwin数据变换处理。
- `libero_goal_vla_mbpo.yaml`才是**BAGEL短分支MBPO**，7D、C10、20动作分支，数据还是Goal task3。
- `BranchRollout` L192–201明确assert π0.5且`discrete_state_input=False`，解释当前WM只预测图像、state不得作为策略token。
- `RoboTwinEnv.reset_to_states()` L341–342直接抛`NotImplementedError`。原生任意中间态分支评估需要补恢复协议，或另设从原生initial seeds评估的完整路径；离线WM分支起点只读图像本身不要求物理恢复。

证据：[RoboTwin普通PPO](https://github.com/VLARLKit/VLARLKit/blob/901ca4e9ff3fdc849170d2a65b177ccb6d919293/examples/configs/robotwin_adjust_bottle_ppo_openpi_pi05.yaml#L59-L77)、[LIBERO MBPO](https://github.com/VLARLKit/VLARLKit/blob/901ca4e9ff3fdc849170d2a65b177ccb6d919293/examples/configs/libero_goal_vla_mbpo.yaml)、[state限制](https://github.com/VLARLKit/VLARLKit/blob/901ca4e9ff3fdc849170d2a65b177ccb6d919293/vlarlkit/rollouts/branch_rollout.py#L192-L201)、[物理reset限制](https://github.com/VLARLKit/VLARLKit/blob/901ca4e9ff3fdc849170d2a65b177ccb6d919293/env_clients/robotwin/robotwin_env.py#L341-L342)。

RoboTwin数据变换还包含左右关节符号/夹爪转换、14维absolute/delta变换；双腕输入有独立槽位。我们的移植应继承目标Control真实使用的链条，不自动拿这个外部示例的C50或state配置覆盖当前配置。[RoboTwin变换](https://github.com/VLARLKit/VLARLKit/blob/901ca4e9ff3fdc849170d2a65b177ccb6d919293/model_backends/openpi/vlarlkit_openpi/dataconfigs/robotwin_dataconfig.py#L104-L228)。

## 只换WM，保留RLinf/π0.5/GRPO：最小边界

这条路线可以复用RLinf当前actor与调度，不需搬JAX PPO，但名称应是“BAGEL-WM上的GRPO扩展”，不能称已复现完整VLA-MBPO。

1. 增加BAGEL服务/后端与独立进程资源安排；不将其权重路径塞进WanBackend。RLinf通用env目前还写`wrist_images=None`、dummy state，改动不能止于backend注册。
2. reset/history输入和返回观测升级为所需的成对或三视角；π0.5相机顺序、mask、图像预处理保持目标协议。RoboTwin另补可用state推进方案，或另行训练不依赖state的策略，后者属于额外方法改动。
3. 继承Control真实动作语义，新增BAGEL动作token转换及其专用normalizer。RoboTwin目标数据应覆盖目标任务和当前π0.5行为，训练WM动力学与任务成功判别；Goal资产不能覆盖这些任务。
4. 明确chunk终点reward在RLinf中的布局、何时done及mask；没有中间帧就不能假装有逐动作真实预测。可以只在chunk末填一次reward并定义相同的成功终止，但这是明示的新奖励粒度。
5. 保留GRPO时不能直接把rollout缩成论文的两chunk就期待相同机制：GRPO没有论文value head的长期bootstrap，两chunk没成功的组仍可能全零。若保持长rollout，又放弃了MBPO限制模型误差深度的关键设计。

对于RoboTwin，以上还有任务资产、state和第三视角训练；对已有LIBERO，则主要是接口/时序/奖励层。两者难度不能并列为“换后端”。

## 搬完整MBPO到RLinf：比换WM多什么

- 原始方法使用Flow-Noise PPO、MLP value head、GAE；我们当前是GRPO。RLinf虽已有PPO/value组件，也需对齐去噪logprob、有效动作维、chunk discount和bootstrap语义。
- 新增离线状态分支采样器，在中间观测发起短段rollout；分支结束应标truncation并bootstrap，真实成功才terminal，不能把每个短段末尾当任务失败。
- 数据同时服务WM训练与分支起点，要有相机、动作、任务、时间、原生success；真实评估需独立保持RoboTwin step limit 400与既定任务协议。
- 原始仓库为OpenPI/JAX策略+PyTorch WM，权重格式和训练状态不同；VLARLKit提供PyTorch实现但不是RLinf即插即用模块，且上述RoboTwin组合缺口仍存在。

## 现实判断与未验证项

**适合借方法和代码，不适合把公开LIBERO toy权重直接当RoboTwin环境。** 若重点是快速研究π0.5双图模型RL，LIBERO更接近现成公开链；若重点是沿我们的RoboTwin任务走，核心工作是任务数据、三视角/state、14D控制条件训练，然后才是RL接入。

未验证：任何公开检查点在我们环境的实际成功率；RoboTwin专用WM是否另有未链接发布；数据tar包内部normalizer与完整任务覆盖；真实GPU峰值/吞吐/最低卡数；新视角与动作本体微调所需数据量；把state从控制命令解析推进是否足够准确。官方四GPU full-shard训练脚本仅是示例，且含站点模块及路径，不能当当前共享四卡可直接运行的证明。
