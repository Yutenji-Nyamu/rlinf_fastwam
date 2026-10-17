# 2026-10-17：强弱策略、推理预算与U切换执行

本轮已授权并执行。物理资源为深圳1的6/7，每卡一个进程、32环境，共64并行。原GRPO τ1已收束；这批推理优先，完成后原RLT逐卡候补。用户明确C100不做，保持H50。

## 固定实验

所有正式条件各128回合，使用同一128个官方eval种子、固定分批/槽位/噪声安排，任务`move_pillbottle_pad`、clean、200动作上限、三相机及原生成功判定不变。

| 顺序 | 条件 | 固定项 |
|---|---|---|
| 1 | 原始Sidney π0.5；Clean C17 CP140 | 各128；N10、H=C50 |
| 2 | 原始模型N5/15/20/30 | 各128；H=C50；N10复用首组 |
| 3 | 原始模型C10/20/30 | 各128；N10、H50；C50复用首组 |
| 校准 | 独立64回合，只运行弱策略并记录U | 不进入128回合测试集；冻结U的1/3、2/3分位阈值 |
| 4 | High-U、Low-U、Random整块接管 | 各128；强弱均N10/C50；目标约1/3，报告实际比例 |

共12个测试条件×128＋64校准＝1600回合。候选仅CP140；本轮不再测试CP160。历史31.25%与56.25%只是选候选依据；新成功率以配对128结果为准。

种子从官方eval表按既定顺序截取，排除训练集、此前固定GRPO评估及首32个旧信号种子；无成功率筛选。各组记录requested/actual seed和原生换种。实际种子若不一致，配对统计只用真实匹配项并明确数量。

## 已实现的切口

- 在已验收的批量原生runner中，保留模型先于SAPIEN初始化、原生`chunk_step`、chunk前后视频、线程池join和环境关闭顺序。
- 主源码父点`3319b5ed4fb4e8ad406cc583f2eaa33b687f9103`，沿用10/3同噪声主10＋旁5终点U。只保留U所需端点，不采整套attention/hidden。
- 先建原底座与归一化，再严格加载CP140全部权重。每个匹配决策使用相同完整batch初噪声；旁路和强模型调用不推进后续弱模型噪声序列。
- 高/低/随机门都在完整动作块生成边界选择；每个新观察重新计算弱模型U。强模型目前整批推理，计时如实包括该成本。
- C10/20/30只执行H50输出的前缀。C30末次最多20动作，保持总上限200。用户确认省略C100，不改H。

源码：[tools/pi05_policy_eval](../../tools/pi05_policy_eval/README.md)。服务器3项CPU测试通过：有效槽位门控、仅校准数据定阈值、拒绝C100；正式U批次首次warmup检查主动作与RNG完全一致。运行中的首批原始模型不等于已经完成所有强模型/双模型验收。

## 调度与产物

- 服务器源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-policy-budget-eval-20261017`。
- 结果：`/data/chenyiteng/results/pi05-policy-eval/pillbottle-20261017-v1`；`configs/`为50批实配，`seeds.json`固定128测试＋64校准种子。
- 一次性交接事务：`/data/chenyiteng/deployment-20261017/pi05-policy-budget-eval-v1`。cutover已完成；原owner已重启，仅6/7的external identity/release变为推理lane，fallback原样保留。
- 原始两lane身份：g6 PID2546196/start511022716，g7 PID2546256/start511022872，UID1003；此为启动回执，下一次须刷新。
- 单批失败、600秒无query进展或3600秒单批上限时停止该lane，不无限重试。原owner等待进程退出及C/G清场后逐卡恢复RLT。
- 输出每回合成功/种子/动作数、每次U和选择、耗时、轻量录像。`summary.json`保留Wilson95区间、配对净差、赢/输对数及实际接管率，可直接作图。大权重不上云。

GRPO τ1收束：198/200轮，最新20/32＝62.5%，前5均值35.625%、末5均值60%、峰值75%；CP180/190保留。训练有改善但仍波动，尚无等预算Clean支持U净收益结论。[收束证据](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/codex/grpo-u-n-20261009/docs/grpo_u_n/closeout1017-tau1/RESULT.md)。

清理已完成：48个精确大文件、401.26GiB；执行后/data余554.2GiB。只含当前GRPO CP40–170、已停Editor-U τ2上份备份、旧DVAC-GRPO末代。原始模型、Clean CP140/160、RLT恢复点保留。[清单](E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a/cleanup-inf1017-manifest.json)。
