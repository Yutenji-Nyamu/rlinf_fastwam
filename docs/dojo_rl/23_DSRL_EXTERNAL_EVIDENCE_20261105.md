# DSRL外部依据：论文、官方代码与复现

2026-11-05检索。结论分成“论文/代码明确规定”“复现者反馈”“我们的推断”；不能把issue里的单次经验当成通用修复。

## 1. 原论文与官方π0代码

- [论文](https://arxiv.org/html/2506.15799v2)，重点为C.5和参数表11：完整50×32噪声难优化，因此原π0实验使用32维重复噪声。建议采用高UTD，并按任务选择模型容量和噪声幅度。
- [官方仓库固定版本](https://github.com/nakamotoo/dsrl_pi0/tree/7f48937d4553e95244cd81c79236a3256df80597)，本轮读取`examples/train_sim.py`、`train_utils_sim.py`、训练shell、pixel-SAC learner、actor/critic更新器及分布实现。
- [Aloha实配](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_aloha.sh)：B256、20次更新/新转移、幅度2、C50、3百万更新；熵目标0。Libero对应脚本为幅度1、C20、50万更新。这些是梯度更新数，不能直接与我们的“400轮×8局”比较。
- [小网络初始化](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/jaxrl2/agents/pixel_sac/pixel_sac_learner.py)：Flax默认FP32参数，Optax Adam；没有把这些网络直接降成BF16。
- [Q更新](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/jaxrl2/agents/pixel_sac/critic_updater.py)：均值Q、不向答案加入熵的π0路径有明确实现依据，不应直接拿其他SAC/FQL的默认项覆盖。

论文表格与某个环境脚本有覆盖参数差异。例如论文仿真熵目标写−d/2，而Aloha脚本为0。应说清参考哪一份配置，不能混成“官方唯一默认”。

## 2. 复现者的失败与经验

| 来源 | 实际报告 | 能借鉴什么/不能说明什么 |
|---|---|---|
| [issue 9：π0.5](https://github.com/nakamotoo/dsrl_pi0/issues/9) | 有人独立噪声能完成、重复噪声失败；也有人认为完整噪声SAC难优化，基础策略更强后重复版能工作 | 支持检查模型与噪声兼容性；没有证明完整噪声跑久必好 |
| [issue 10：稳定性](https://github.com/nakamotoo/dsrl_pi0/issues/10) | 约40%基础成功率，调熵系数/MLP仍难提升；另一复现者强调幅度敏感 | 起点40%–50%并非充分条件；幅度是具体可查项，未见统一解决回执 |
| [issue 6：重复/打乱](https://github.com/nakamotoo/dsrl_pi0/issues/6) | 报告重复、插值、固定打乱的尝试；有人初态稳定却仍无RL提升 | 可以启发受控对照，不能直接当已验证的π0.5训练配方 |
| [issue 14：熵设置差异](https://github.com/nakamotoo/dsrl_pi0/issues/14) | 询问π0与DP路径的Q熵项及目标差异 | 本次未见作者解答；不以问题本身证明某一实现错误 |
| [issue 8：Noise Aliasing复现](https://github.com/nakamotoo/dsrl_pi0/issues/8) | Araya团队分享多GPU、合批及NA实现 | 有实代码可读，但公开Aloha示例仍主要是π0，不能冒充我们的Dojo/RT2验收 |

[jesbu1的OpenPI分支](https://github.com/jesbu1/dsrl_openpi/tree/b0e3cfbfb1fb46622b4d499fbeeada95d247f4a4)主要提供π0.5/DROID关节控制和噪声接口适配；不是独立的完整1600维SAC成功对照。此次只核目录/变更，不把它列为已逐行验收的训练方案。

## 3. DSRL-NA：一种后续候选，改动比精度修复大

[Araya实现固定版本](https://github.com/arayabrain/dsrl_pi0_na/tree/afcb58f2bd95a2e3771db359b8f98eb8f4f9ebee)，本轮读取训练主程序、实配shell、learner和noise-critic updater。

普通DSRL直接给噪声评分。NA额外学一个“真实机器人动作”的Q，再用冻结π0把噪声转成动作，把动作Q的判断教给噪声Q。相近动作即使来自不同噪声，也有机会共享监督。

它新增动作Q和更多基础策略前向；公开Aloha代码仍用32维重复噪声。实际shell采用B128、动作Q/噪声Q/策略更新次数15/5/5，README默认又不同，应以具体运行配置为准。它值得保留，但不能据此认定可直接解决我们完整1600维和BF16的问题。

## 4. RLinf接入提供什么保证

[RLinf DSRL文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)、[最初接入PR746](https://github.com/RLinf/RLinf/pull/746)。PR在2026-02-26合并；提交说明当时尚未完成完整训练回路的GPU验收。这是历史接入范围，**不等于断言今天所有RLinf DSRL都未验证**。

本地沿用的小网络BF16和目标Q的FP32影子需分别判断。目标影子解决EMA缓慢更新被舍入，不自动解决在线Adam权重精度。我们本轮直接读了四组检查点与优化器状态，并做一次CPU对照，证据强于仅依据上游注释猜测。

追问已追溯：强制BF16位于[最初合并提交第169–176行](https://github.com/RLinf/RLinf/blob/42530b7225e7e8a91ad206a9e909490158d85594/rlinf/models/embodiment/openpi/openpi_action_model.py#L169)，代码注释明确为FSDP dtype兼容处理。当前本地`build_optimizers`直接构造`torch.optim.Adam`；[PyTorch Adam源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/adam.py#L168)的历史量用`zeros_like(parameter)`初始化。结合实际保存点，可定位这条BF16状态链；不能把来源误说成我们新加的独立噪声改动，也不能由此断言所有RLinf使用方式都会同样失败。

## 5. 近期其他论文的边界证据

[MiDAS](https://arxiv.org/html/2608.11363v1)附录H.2列出π0.5 DSRL对照，采用冻结VLM特征、小MLP及UTD20；作者报告DSRL的改善不总稳定，较弱基础策略尤其受限。可借鉴其特征输入和对照设置，但不同任务、不同模型与预算不能证明我们的主因。

## 对我们当前问题的取舍

1. 已证实的本地更新精度问题优先，官方FP32提供实现依据。
2. 高维噪声难学有论文与复现的双重支持；保留独立噪声的基础成功率，再逐项考虑范围、容量和数据覆盖。
3. 不能把别人提出的幅度、打乱、插值或NA全塞进同一次修复；否则无法判断何者有效。
4. 暂无可直接照搬、且已核实在本配置Dojo/RT2 π0.5上提升的完整外部配方。
