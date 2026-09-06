# RLinf 当前入口

## 当前任务与授权（2026-09-06）

用户要求：本人历史轻量实验结果推Git；大类总量很大的replay/raw数据也不推；smoke及指定正式run旧代大checkpoint精确清理；缩短上下文；诊断BC/DVAC并讨论U/权重范围。

本轮账本：docs/server-admin/SZ_ARCHIVE_CLEANUP_CONTEXT_LEDGER_20260906.md。先清单/凭据与保留依赖核验，再发布/清理；跳过活跃写入和未解决依赖。未授权改现役训练参数、增加GPU推理/训练、重启或影响共享Ray/其他用户。

## 活动实验定位（以下是旧快照，不是当前现场）

09-06 11:52：GPU6 BC52/100，GPU7 DVAC50/100，GPU4/5 GRPO160/200；fixed分别Step50=12/32、50=16/32、160=17/32，ckpt50/50/160，所查fatal/OOM0。RAM可用约656GiB但环境RSS持续增长；必须刷新后再报当前。

- BC：worktree pi05-online-bc；DVAC：pi05-online-bc-dvac。位于/data/chenyiteng/projects/rlinf-shenzhen/worktrees/；run在/data/chenyiteng/results/rlinf-shenzhen/online-bc/，各正式合同给精确路径。
- Sidney：worktree sidney-pi05-current-rlinf，run在/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/；当前运行目录runtime-resume100-to200，已从100续至200，勿重复launch。既有heartbeat由原窗口管理。
- Fast已结束，用户要求保持停止；没有新现象不重复报旧fatal。

## 按问题选一个入口，只读所需章节

| 问题 | 入口 |
|---|---|
| π0.5 BC基础、参数/模型 | docs/rlinf-robotwin-pi0-online-bc/02_PI05_ONLINE_BC_PLAN.md |
| BC＋DVAC方法 | 同目录01_DVAC_DESIGN.md（当前实现§11—12） |
| BC/DVAC差异、随机性、U/范围 | 同目录evidence/PI05_BC_DVAC_FULL_DIFFERENCE_AUDIT_20260906.md；旧种子/时间/图见其引用 |
| 原π0在线BC | 同目录00_RESEARCH_AND_PLAN.md |
| Sidney GRPO续训 | docs/rlinf-shenzhen-multitask-pi05/evidence/RESUME100_TO200_EXECUTION_LEDGER_20260905.md |
| 存储/Git/上下文清理结果 | docs/server-admin/SZ_ARCHIVE_CLEANUP_CONTEXT_RESULT_20260906.md；逐操作见本轮账本 |
| 旧实验、Fast/RLT/DSRL | docs/project-history/00_INDEX.md按需路由，不全文加载历史 |

## 当前未决点

- BC/DVAC只有17项预期配置差异，28项CPU回归通过；真实首轮16对9，DVAC首批权重全1。未见权重接线/预算错误；独立rollout动作RNG未配对控制，不能认定差距唯一由它造成。下一步若授权，修随机协议；不将train固定初态改true来替代。
- 环境内存增长未定位。参数U10、权重[0.5,1.5]仍按现役合同；本轮只讨论是否改变。
- 已清103个>1GiB checkpoint文件、852.36GiB；保护清单时正式最新完整代和Sidney活动引用Step100。不删目录/成功数据池，不追删清理中产生的新代；大权重不可恢复。
- 本地根Git仍为空；历史轻量资料已推既有个人RLinf仓库codex/sz-experiment-archive-20260906，远端SHA核验一致。按数据类别排除replay/大权重/视频；最终回执看结果文档，不盲改根remote。

旧根文件已逐字哈希校验归档于docs/project-history/context-20260906-before-trim/。不再强制读09-03窗口交接。结束时用简短结果替换本页当前状态，不继续累加时间线。
