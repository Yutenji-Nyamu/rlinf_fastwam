# EXPO-FT 专题上下文

更新：2026-10-01。深圳2四卡B64完整smoke和独立进程恢复已PASS，Q20、FM/editor/temp各1，action expert及小组件实际更新、冻结prefix无梯度。一次call207秒，四卡采样显存峰值约45.0/37.4/37.4/37.4GiB；详见[规模smoke](FORMAL_SCALE_SMOKE_20261001.md)。正式20k真实动作含10回合warmup已启动，原四RLT清理完成，仅EXPO退出后恢复。正式路由与现场进度见[正式运行](FORMAL_RUN_20261001.md)。此前单卡小batch smoke及14:31归还回执仍保留在[运行包](SMOKE_PACKET_20261001.md)，不能混作本次正式结果。

## 已确认

- 后续本轮用户明确：方法预算/Q视觉按论文及作者结构，暂不使用图像增强；要求正式规模短smoke，不进正式训练。新四卡B64候选合同、回放修正、资源路由和验收范围见[正式规模smoke](FORMAL_SCALE_SMOKE_20261001.md)。旧一次smoke已完成回执保持；新scope/新cycle不重放旧guardian。
- 2026-10-01正式讨论：用户明确action expert＋原生投影，不用LoRA；沿现有冻结视觉/语言prefix策略。方法侧优先论文/固定作者源码，RoboTwin基座合同保留。资源、数据/更新预算、TD窗口与Q视觉选择见[正式讨论方案](FORMAL_DISCUSSION_20261001.md)；本轮只读核验/维护文档，未借卡或启动正式训练。
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

以下为历史实施问题的定位；B64/四卡、物理步replay/demo-Q、成功FM随机批次、作者结构Q视觉已实施并通过规模smoke；图像增强按用户决定关闭。任务pick_diverse_bottles与20k预算已锁，warmup/cadence的21项服务器CPU检查通过；实际长训和固定评估正在正式scope运行，效果待结果。当前未决按正式运行页，不把旧建议当最新状态。

1. official image trainable filter与论文“冻结视觉”的差异；以显式参数组和FM权重变化证明。
2. early-terminal chunk valid/continuation、timeout/final obs、实际执行K与固定C折扣；公开issue尚未给完整结论。
3. base候选batching/KV共享在当前RLinf π0.5的正确shape与成本；8env×8candidate独立预算。
4. RGB/full-H online replay内存及成功episode储存、base-weight同步与strict恢复范围。
5. method budget：edit scale/active dimensions、10Q/2-min、critic/base/editor cadence、trainable base组、demo/HIL协议；不继承RLT U5/BC-Q课程当EXPO默认，不把DROID pick的xyz/gripper mask下标直接套RoboTwin14D关节。

## 阅读入口

先读[综述](EXPO_FT_REVIEW_20261001.md)，按问题再到[论文笔记](RESEARCH_PAPER_NOTES_20261001.md)、[开源/运行审计](CODE_AUDIT_20261001.md)、[RLinf映射](RLINF_MAPPING_NOTES_20261001.md)。后续实施沿当前已跑通的π0.5模型/动作合同，仅在独立source/config/output/namespace中接方法；运行前列配置差异、命令、资源和停止条件。

当前正式分支：[codex/sz2-expo-ft-formal-20261001](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz2-expo-ft-formal-20261001/docs/methods/expo-ft)，四卡B64源码、完整恢复/正式启动轻量回执已推送并校验远端一致。旧单卡交付分支：[codex/sz2-expo-ft-20261001](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz2-expo-ft-20261001/docs/methods/expo-ft)。server root `/data/chenyiteng/projects/expo-ft-sz2-20261001` 下保留大checkpoint及实际部署合同。
