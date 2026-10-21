# adjust_bottle 首轮实配与依据

2026-10-21。用户本轮授权：深圳3物理2/3卡，原始N1/N10各128回合、H50/C50；实现CP与OneDP，简洁smoke后直接正式训练，轻量源码/结果公开推送。此前“六任务筛选、先3000步工程验证”的建议已被本轮指令取代。

**执行更新**：用户后来按IP明确将全部后续工作移至 `.105` 2/3卡，先smoke再补齐原始评估；实测后选择每卡64、全局B128，学习率与epochs保持不变。以下B64及步数是初始配置依据；最新容量、19,000/760步预算与迁移回执以[06](06_BATCH_SMOKE_SZ2.md)为准。

## 数据到底需要什么

可以直接用 [TianxingChen/RoboTwin2.0 的 adjust_bottle / aloha-agilex_clean_50](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset/adjust_bottle)。它是50条专家示范：同一时刻三相机RGB、14维机器人关节状态、任务指令，以及后续绝对关节目标。

本次直接复用服务器已有转换结果 `datasets/robotwin2/canonical/pi0-aloha-clean50-v1`，虽然目录名较旧，内容确为调瓶子：50条、7,188帧、50种具体指令、4,738个不跨轨迹的完整H50窗口。已核对episode0与官方原HDF5：state对应q[t]，action对应q[t+1]，最大差约2.8e-8；头/左右腕相机、14维顺序与既有接口一致。两方法用同一批观测，沿用当前基座的mean/std，不重新拟合归一化。

- **CP**：取一帧观测及随后50个真实动作；对动作加噪，老师在训练时算局部推进，再训练学生的一致性和动作监督。无需先采集老师10步答案。
- **OneDP-S**：核心需要观测。学生现场生成动作，辅助网络学习这些生成动作，老师现场给出学习方向。读取同一数据文件不代表用示范动作做OneDP损失；动作张量在此仅提供尺寸。
- 已有U/四噪声采集用于历史结果和接口参照，本次训练没有混入它们的测试轨迹。视频用于检查与画图，不从MP4反推动作标签。

## 模型和方法的分工

固定老师为 `models/rlinf-native/sidney-pi05-robotwin-e49e2ab`，三个角色均从同一基座初始化。模型是三相机、224输入、14维真实动作/32维内部动作、H50、绝对关节动作。OneDP学生和辅助网络训练动作专家；CP训练动作专家及新增的零初始化目标时间投影。视觉语言前缀冻结、共享一次观测编码，沿用已跑通EXPO的expert-only选择。它是π0.5适配选择；原OneDP论文训练了视觉编码器。

**CP-flow**保留官方CTM的“老师局部推进、学生跨阶段、EMA组合比较、数据监督”结构。新目标时间s进入动作专家的时间条件；t=s严格为恒等，s=0生成动作。EMA参数冻结，但预测分支对输入的导数保留。DSM换成π0.5原生flow-matching监督，权重1:1；不照搬EDM噪声尺度。

**OneDP-S-flow**保留冻结老师、学生和辅助网络、先辅助后学生的一比一交替更新。流模型噪声预测为 `epsilon_hat=x+(1-t)*v`，分布梯度采用论文 `w=sigma²=t²`；对生成动作的方向是 `t*(1-t)²*(v_teacher-v_fake)`。不引入GAN或额外梯度归一化。内积代理loss不是KL估计，不能拿其正负判断效果。

## 参数逐项依据

| 项目 | 本轮值 | 依据 |
|---|---|---|
| 任务/上限 | adjust_bottle / 400动作 | 用户选任务，RoboTwin官方任务上限；新suite显式配置 |
| 预测/执行 | H50 / C50；原始N1、N10；学生N1 | 用户本轮指定、原基座H50 |
| 评估 | 各128回合，单卡32并行，两卡分担 | 已成功的配对评估流程；同一官方eval种子列表，排除train列表，无策略筛种子 |
| 并行训练 | 双卡DDP，全局B64；起点每卡micro32 | CP官方B64、此前π0.5 EXPO成功batch；如OOM仅减micro并累积，保持B64 |
| 精度/可训练范围 | 动作专家FP32主权重，BF16计算；冻结前缀 | 既有Pi05Backend成功实现 |
| CP优化器 | AdamW lr2.5e-5，β(.9,.95)，eps1e-8，wd1e-10，clip1 | 既有π0.5动作专家实配；不用官方U-Net的lr1e-4 |
| CP损失 | 官方平滑距离 + 原生FM，1:1；自动delta=.0216 | 官方CTM/DSM权重1:1；delta=-1按维度计算常数，不是MSE。复用Huber_Loss实际公式的稳定等价写法；FM为flow适配 |
| CP时间 | 80等分flow时间；老师向0推进一个Euler小步 | 官方80bins/一个局部推进，网格和solver服从π0.5的[0,1]流约定 |
| CP数据FM时间 | Beta(1.5,1)×.999+.001 | 原生OpenPI flow-matching |
| CP EMA | inv_gamma1、power.75、上限.9999 | 官方ctmp_square.yaml；部署评估EMA学生 |
| CP训练长度 | 500 epochs，约37,500更新；每50轮评估 | 官方ctmp_square.yaml；450是OneDP论文比较设置，不能混称同一个配置 |
| CP调度 | warmup500，cosine衰减 | 官方训练配置；总步数按实际窗口数计算 |
| OneDP优化器 | 学生lr1e-6、辅助2e-5；Adam β(0,.999)，wd0 | ICML2025正式论文超参数；学生AdamW在wd0时等价Adam |
| OneDP加噪时间 | 均匀[.02,.95]；辅助epsilon MSE权重1 | 内部区间是flow适配，避开端点；不是把DDPM索引直接当flow时间 |
| OneDP交替/长度 | 辅助1次、学生1次；20epochs≈1,500更新；每5轮评估 | 原论文交替结构与模拟训练20epochs |
| 保存 | 全训练状态及另存推理专家权重 | 用户强调续训能力；固定冻结base仍是明确的外部依赖 |
| 数据读取 | 50个parquet各读一次，留压缩RGB，按batch解码 | 复用成功demo接口，避免以前单轨迹缓存导致反复整文件读取 |

50轮CP与5轮OneDP的评估均为128回合。首轮只有一个训练种子；曲线是开发评估，不能把挑出的最高点当独立最终测试。原始N10和N1是两个基线，不预设蒸馏一定提升。

## 运行与停止条件

源码：`worktrees/pi05-distill-20261021`；分支 `codex/pi05-cp-onedp-20261021`，父提交`713e706ff73c86ae35b7fa898268d8d471174cf1`。

服务器事务：`/data/chenyiteng/deployment-20261021/pi05-distill-adjust-v1`；产物：`/data/chenyiteng/results/pi05-distill/adjust-20261021-v1`。

顺序：N1/N10各128 → CP双卡12步并恢复到14步 → OneDP同样smoke → 自动核对有限梯度/恢复到14步/实际导出加载/N1只调用一次/设备绑定 → CP正式训练及周期评估 → OneDP正式训练及周期评估。通过就直接开训，无需用户再次确认；失败停在明确错误。正式训练从原始基座重新开始，不继承smoke更新。

训练命令为 `python -m torch.distributed.run --standalone --nproc_per_node=2 tools/pi05_distillation/train.py --config <cp|onedp.json> --output <run> --stop-step <segment_end> [--resume <checkpoint>]`；计算卡严格 `CUDA_VISIBLE_DEVICES=2,3`。评估沿用`tools/pi05_policy_eval/lane.py`，每卡独立绑定CUDA和EGL；不接入共享Ray。

出现非有限损失/梯度、配置与续训不匹配、进程失败立即记失败并停该训练阶段。环境批次按已验证wrapper有限重试一次，仍失败留缺失记录；不无限重试，不把错误算成功，也不因仿真基线缺失阻断独立的离线训练smoke。训练checkpoint包含学生、EMA/辅助、两个优化器、每rank RNG、step、全配置、固定数据与base依赖。采样排列与学习率均由seed/epoch/step确定，恢复无需猜测隐藏状态。

## 参考来源

- [CP官方源码](https://github.com/Aaditya-Prasad/consistency-policy/tree/eed0c465d3c1aa6dcf22f10e784df07dec06beac)：ctm_policy.py、ctmp_square.yaml。
- [OneDP正式论文](https://proceedings.mlr.press/v267/wang25ba.html)：目标、交替更新、学习率、20epochs。没有核实到官方训练源码；个人仓库不冒充官方。
- [Sidney π0.5模型卡](https://huggingface.co/SidneyXie/pi05_robotwin)：基座、三相机、14维绝对动作、H50/N10、原SFT学习率。
- 本工作区已验收EXPO `rlinf/algorithms/expo_ft/backend.py` 与 `formal_replay.py`：expert-only、FP32主权重/BF16计算、数据缓存和原生归一化。

## 本轮执行回执（2026-10-21 21:04）

[源码分支](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/pi05-cp-onedp-20261021)。5项新数学/权重加载检查、4项既有评估接口检查通过；真实π0.5模型的复制、目标时间接口和官方数据预处理通过。GPU训练smoke尚待原始评估结束，不能将CPU检查等同于GPU训练已验证。

深圳3·2与3·3分别运行原始N1、N10，均为每批32回合、共4批。首批已从query0推进到query2，尚无整批结果；现场模拟器初始化及执行较慢，暂不凭少量步数给训练ETA。自动队列已运行，顺序及失败条件见上文；无需再次审批正式训练。
