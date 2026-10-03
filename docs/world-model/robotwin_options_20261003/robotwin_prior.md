# RoboTwin × π0.5 × WM-RL：旧方案回顾与10月3日刷新

**研究快照（2026-10-04补注）：**公开代码、论文和权重的核查截止2026-10-03；本文“未运行”“尚缺闭环”等叙述限定于该次调查，不作为当前实验进度。后续组合和执行路由见[OpenDW主上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)，本次仅补充归档，没有重新检索或启动实验。

本轮只读核查官方GitHub/Hugging Face/API和本地旧研究；没有服务器操作、模型下载或实验。此前两个专题仍保留历史事实：[WorldArena优先复现](../WORLDARENA_FIRST_REPRODUCTION_PLAN.md)、[OpenDW接口分析](../OPENDW_PI05_GRPO_CONTEXT.md)。本条只更新可用性判断，不修改旧结论原文。

## 结论与顺序

**已经有RoboTwin的Wan环境代码，也有能接外部动作的三视角OpenDW；缺的是把这些组件组成经过验证的π0.5 WM-RL训练环境。**

- 若目标是尽快给现有三相机π0.5接一个世界模型，优先研究 **OpenDW官方Robotwin bundle → 外部动作预测 → RLinf环境适配**。它的模型资产比WorldArena的RoboTwin Wan配套更完整。
- WorldArena 2.0更适合复用 **RoboTwin/π0.5/GRPO环境接口、HTTP拆分和任务reward**。若坚持先忠实复现WorldArena论文，仍先选`adjust_bottle`，但要先补齐匹配14D WM、依赖版本及reset数据，不能宣称配置直接可跑。
- 自训三视角Wan可以作为后续路线；当前公开材料已给训练代码，不需要从零写网络，但训练数据、视角同步、动作条件和训练验证都是实际新增工作。不是改相机数量即可恢复三视角能力。

这些是工程优先级判断，尚无本机时延、显存或最终成功率证据。OpenDW本身既能出动作也能预测视频，但我们使用它的外部动作入口，π0.5继续负责动作。

## 1. 与9月30日相比，主要发布物没有变化

2026-10-03通过官方API实时核验，以下HEAD/revision与旧调研相同：

| 发布物 | 当前固定版本 | 实际已发布 |
|---|---|---|
| [WorldArena 2.0代码](https://github.com/WorldArena2/WorldArena-2.0/tree/5978ce5c81e55b8c8358f4f5966a13ce385ff155) | `5978ce5c…`，9月26日Initial release | RLinf衍生工程、两任务Wan环境、π0.5/GRPO、reward、HTTP服务 |
| [WorldArena2.0模型库](https://huggingface.co/WorldArena/WorldArena2.0/tree/2c351f46da784d4d8caca210ad1ca3b11c4b4124) | `2c351f46…` | adjust_bottle/click_bell policy及norm stats、两任务reward、T5 |
| [RLinf DiffSynth](https://github.com/RLinf/diffsynth-studio/tree/2a2e05fa1f724828b243f272540989b19a6e54f8) | `2a2e05fa…` | Wan推理与训练代码，所查动作层仍7D |
| [OpenDW代码](https://github.com/dexmal/opendw/tree/e33befa8005a1585e0140dbf464566e90bc79aa1) | `e33befa8…`，7月9日Initial release | 三视角Robotwin policy/video接口、WM训练与demo |
| [DW05-Robotwin模型](https://huggingface.co/Dexmal/DW05-Robotwin/tree/6ab5f9e2636610cba440d08264663efe70c3f761) | `6ab5f9e2…` | model.pt、norm stats、VAE、text encoder、tokenizer完整离线包 |

RLinf官方HF账号本次`Wan`清单仍以LIBERO Spatial/Object/Goal/Long与Fullshot为主，未查到公开命名的RoboTwin Wan配套bundle。此结论限于官方已核列表，不声称全网绝对没有其他发布。

## 2. WorldArena已有RoboTwin Wan，为什么当时还没做成

**代码支持接口与模型训练好了，是两件事。** WorldArena 2.0的`RL_env_benchmark`确实已有：

- `wan_robotwin_adjust_bottle`与`wan_robotwin_click_bell`配置，14D绝对关节动作、C8、5帧条件＋8帧未来、256×256。
- policy观察是单head图；腕图为空、state为14D全零。它没有提供生成腕图的现成实现。`wrist_images`是接口字段，并不代表WM生成了腕图。
- adjust_bottle默认`RoboTwinT5CrossAttn` learned reward；click_bell默认LPIPS末帧相似度。两任务都公开了reward文件，但不能把“有resnet文件”读成“默认训练都用resnet”。
- 环境/HTTP服务/GRPO调用链可复用；任务成功由图像奖励估计，不是RoboTwin物理判定器。

主来源：[集成说明](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/README.md)、[adjust_bottle配置](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/examples/embodiment/config/env/wan_robotwin_adjust_bottle.yaml)、[click_bell配置](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/examples/embodiment/config/env/wan_robotwin_click_bell.yaml)。

本次仍未补齐的部分：

| 缺口 | 当前证据 | 不能采用的简化说法 |
|---|---|---|
| 14D动作条件Wan与依赖对应 | 环境传14D；公开RLinf DiffSynth动作层仍`Linear(7,dim)`、`Linear(28,4*dim)` | “环境action_dim改14便有14D WM” |
| 任务WM checkpoint | 新HF模型树无adjust_bottle所需`dit_model.safetensors`、click_bell所需`model-00001.safetensors`及对应reset目录 | “两任务π0.5已发布，所以WM也发布齐了” |
| reset/export包 | 配置需要`dataset/`；README提及的`encode_vla_to_rlinf.py`在本次完整WorldArena/DiffSynth Git树均未找到 | “拿任意原始视频目录就能reset” |
| state语义 | 主实现仍零state；对应[issue #5](https://github.com/WorldArena2/WorldArena-2.0/issues/5)仍open、0回复 | “有14D state字段就等价真实关节状态” |

动作层可直接核[固定源码411–424行](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/models/wan_video_dit.py#L411-L424)。简单把层宽改14只解决张量shape，不会产生从14D动作到机器人运动的已学习映射。

旧WorldArena库有Wan相关文件；9月30日安全读取权重元数据已经区分`wan_adjust_bottle.pt`的Video_Former/action_pred结构与Wan DiT。此轮未重新下载或读取大权重，不能将旧文件名中的Wan当作缺失checkpoint的替代证明。

## 3. 公开数据存在，但不等于已配好的RL reset包

- [WorldArena_Robotwin2.0](https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0) 当前revision `2ddc8d9b…`，官方说明是视频生成/评测的HDF5动作、初始帧、指令；树中有9个条目，主要为tar/zip。**本轮没有展开archives，不能据文件树断言压缩包里绝无可转换样本；也未建立其与当前Wan reset schema的对应。**
- HF同名[WorldArena/WorldArena2.0数据集](https://huggingface.co/datasets/WorldArena/WorldArena2.0)当前revision `af1ac34d…`，5820个条目含真实任务的三路视频/HDF5/meta；官方[Track 3说明](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description.md)将其定位为真机示范。不能因名字相同就当作RoboTwin Wan reset bundle。
- [Dexmal/robotwin2-full](https://huggingface.co/datasets/Dexmal/robotwin2-full)存在，当前revision `590d3967…`，四段tar分卷；OpenDW源码另给JSONL数据契约和训练入口。可以准备/转换RoboTwin轨迹，不等于已提供经过我们策略分布对齐的训练集。

## 4. OpenDW当前能直接提供什么，仍需什么

**目前没有新增value、下一state或批量WM推理接口。** 当前Git/HF版本未变，并重新核了实际源码：

| 项目 | 当前能力 | 接π0.5 WM-RL要做的事 |
|---|---|---|
| 多相机 | 头＋左右腕组成同一画布；默认384×320，上2/3为头、下1/3为两腕 | 将预测拼图拆回π0.5的三图格式，固定缩放/crop |
| 外部动作 | `rollout_video_with_actions(action_abs=...)`确实把动作传给`infer_joint(action=...)`，再作为`gt_action`进入视频预测 | 用π0.5反归一化后的14D环境动作，不用OpenDW生成动作替代π0.5 |
| 当前state输入 | 可给初始/条件proprio | **有输入不代表能预测下一state**；返回值仅video/action |
| 时间 | 默认32动作→9帧（首帧＋8未来帧，4动作一采样间隔） | 与现有C50或C8明确对齐；不能拿补齐后的末帧冒充第50步 |
| reward/done | 无现成任务reward/done；Value Expert仍声明后续发布 | 单独接任务奖励和终止；GRPO不需要value head，但必须有可比较回报 |
| 并行 | 训练支持batch；公开`infer_joint`明确单样本`batch_size=1` | 小规模先串行/服务化验证；批量推理要适配，不能直接说N64可接 |
| 自训 | 训练入口、数据schema、checkpoint已有 | 若现有模型的目标任务/动作分布不合适，可采π0.5轨迹微调；不从随机模型重训 |

源码主来源：[外部动作入口542–604](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L542-L604)、[单样本条件与返回275–399](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/model/dw05/dw05_core.py#L275-L399)、[发布范围与训练说明](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/README.md)。源码由官方raw重新读取，部分GitHub网页抓取返回cache miss，未将网页失败误作文件不存在。

官方online demo当前依然把末条动作作为下一base state，而非预测的真实关节状态：[1605–1629](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/playground/online_demos/robotwin_online_demo.py#L1605-L1629)。**同日后续纠正：**深圳2共享RoboTwin实际给policy的state正是drive targets＋缓存夹爪命令，非实测qpos；因而接触时命令与实测角的偏差，不自动构成policy state接口错误。要核实际生效目标、夹爪裁剪、部分执行与图像时刻，不应直接假定必须训练next-state模型。动态视频能否正确表达接触仍是另一项验证，见[接口审计](../robotwin_pipeline_20261003/interface_audit.md)。

此前已经核到两个发布差异，本次代码/权重版本不变：README提及`script/dw/infer_aloha_joint.py`仍未见于完整Git树；HF config的32D描述与实际权重14D元数据不同，runtime会从权重推断维数。应走真实存在的policy/demo接口并保持bundle统计量，不能只按README路径或config维度组装。

## 5. 对用户可以怎样简洁解释

“之前已经找到了两条路线：WorldArena给我们RL训练框架和RoboTwin奖励，但它公开的Wan模型包尚未配齐，而且仍是单相机；OpenDW给我们现成的三视角、14D动作条件视频预测，所以更适合解决相机缺口。我们主要要补的是把它接成RL环境、给奖励、处理下一状态和动作时间，再用真实RoboTwin检查效果。不是重新写π0.5或GRPO，也不是下载OpenDW就已完成WM-RL。”

建议顺序：先拿同一RoboTwin真实片段，给OpenDW输入π0.5的外部32步动作，看三视角预测是否对得上；再用WorldArena已有对应任务reward与RLinf接口组成小闭环。若选择adjust_bottle可以复用现成learned reward，但需检查其head图预处理与OpenDW输出裁剪后分布。最后才决定WM微调或更长RL训练。此处是讨论建议，不含服务器资源预留或实验启动。
