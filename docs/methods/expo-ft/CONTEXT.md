# EXPO-FT 专题上下文

更新：2026-10-01。深圳2物理4的action-expert EXPO-FT移植已通过真实fresh＋新进程resume smoke（各200动作、Q20/FM1/editor1/temp1），两个在线episode均未成功，FM使用已核验成功演示。初次cuDNN SDPA报错已用此前成功的独立进程兼容策略修复。14:31物理4–7原四RLT恢复首轮全部通过，guardian=RESTORED，原累计3000预算不变。运行依据见[实现合同](IMPLEMENTATION_BASIS_20261001.md)、[运行包](SMOKE_PACKET_20261001.md)、[实施记录](IMPLEMENTATION_LEDGER_20261001.md)和[部署路由](DEPLOYMENT_20261001.md)。现役实验路由见根HANDOFF.md。

## 已确认

- 原版EXPO-FT arXiv2605.25477v2（2026-08-17）；来源EXPO2507.07986v3；Real-Time2609.18207v1为独立扩展。
- base proposal → bounded action-space edit → 原/edited候选Q选优；rollout和TD下一候选均选优；edit用Q/entropy，base继续FM监督更新。当前官方发布实现默认base用成功episode，原版论文未明确列这个success-only开关。原配方8base＋8edit，不等于我们并行采集N8。
- 官方核心训练闭环公开，DROID/π0.5示例；未找到全部论文8任务demo/norm/SFT/最终checkpoint包，没有现成RoboTwin适配。
- 推荐在现有RLinf PyTorch接口独立实现原版；完整新增范围包括candidate sampler、edit/action-Q、图像/指令/真实执行序列replay、base FM及完整restore。
- 当前RLT Stage2输出完整chunk并冻结VLA；旧已验DSRL port训练32D生成noise。这些可借基础设施，不能直接换维度混称EXPO。

## 来源身份

- 官方主仓库：`pd-perry/expo-ft@023cf9cfcb09dab962b6e806fea47d2954b2b9bb`。
- 原版OpenPI：`pd-perry/openpi`，branch`expo_ft`，`46407a41183b037313a383ff679683f2773b766d`。
- DROID：`pd-perry/droid@04b7551ff8a4901b9914cc4dbca4d066771ae9a7`。
- 只读本地源码：`local_scripts/expo_ft_20261001/official-*`，不发布嵌套clone、依赖或模型；pins在`local_scripts/expo_ft_20261001/source-pins.json`。
- 深圳1实际π0.5源：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rlt-n8full-sz1-20260929@55c1399a50826d61e8735a64daa2f1742f1b824f`，branch`codex/sz1-pi05-rlt-n8full-20260929`，2026-10-01只读时clean。
- 现役四个Stage2：精确driver身份存活，`rlt_mlp_policy`、identity、C10/14D、8环境/200动作、U5、critic:actor=2、compact replay、bootstrap_on_truncation=true。已结束Stage1的base合同：`pi05_sidney_robotwin`，H50/C10/14D、ODE10、Sidney norm、`robotwin_aloha_canonical_v1`；其resolved无algorithm/env属于Stage1配置，不是部分Stage2 overlay。完整N8调度另见`docs/server-admin/RLT_N8_FULL_RESTORE_20260929.md`。未读取生产feature wrapper的完整调用路径；openpi_rlinf eval/SFT只作为已有接口的移植候选。
- 现场证据：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/expo-ft-source-20261001-0956.json`；六处源码及SHA在同目录`expo-ft-source-files-20261001-1003.json`。只读脚本：`local_scripts/expo_ft_20261001/target_probe.py`。未记录凭据。

## 未决/后续需验

1. official image trainable filter与论文“冻结视觉”的差异；以显式参数组和FM权重变化证明。
2. early-terminal chunk valid/continuation、timeout/final obs、实际执行K与固定C折扣；公开issue尚未给完整结论。
3. base候选batching/KV共享在当前RLinf π0.5的正确shape与成本；8env×8candidate独立预算。
4. RGB/full-H online replay内存及成功episode储存、base-weight同步与strict恢复范围。
5. method budget：edit scale/active dimensions、10Q/2-min、critic/base/editor cadence、trainable base组、demo/HIL协议；不继承RLT U5/BC-Q课程当EXPO默认，不把DROID pick的xyz/gripper mask下标直接套RoboTwin14D关节。

## 阅读入口

先读[综述](EXPO_FT_REVIEW_20261001.md)，按问题再到[论文笔记](RESEARCH_PAPER_NOTES_20261001.md)、[开源/运行审计](CODE_AUDIT_20261001.md)、[RLinf映射](RLINF_MAPPING_NOTES_20261001.md)。后续实施沿当前已跑通的π0.5模型/动作合同，仅在独立source/config/output/namespace中接方法；运行前列配置差异、命令、资源和停止条件。

实现交付分支：[codex/sz2-expo-ft-20261001](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz2-expo-ft-20261001/docs/methods/expo-ft)。仅源码、方法依据和轻量验收回执；server root `/data/chenyiteng/projects/expo-ft-sz2-20261001` 下保留大checkpoint及实际部署合同。
