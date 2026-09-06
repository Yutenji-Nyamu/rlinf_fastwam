# RLinf 当前入口

## 当前任务与授权（2026-09-06）

用户要求：本人历史轻量实验结果推Git；smoke大权重全部清、Sidney/DSRL/online-BC正式run非最后代大checkpoint精确清理；大幅缩短上下文，保留习惯；全面诊断BC/DVAC差异并讨论U/权重范围。

本轮账本：docs/server-admin/SZ_ARCHIVE_CLEANUP_CONTEXT_LEDGER_20260906.md。先清单/凭据与保留依赖核验，再发布/清理；跳过活跃写入和未解决依赖。未授权改现役训练参数、增加GPU推理/训练、重启或影响共享Ray/其他用户。

## 活动实验定位（以下是旧快照，不是当前现场）

09-06 10:18：GPU6 π0.5 BC45/100；GPU7 BC＋DVAC43/100；GPU4/5 Sidney GRPO156/200。latest fixed分别Step45=20/32、40=19/32、155=21/32，ckpt40/40/150；所查fatal/OOM0。必须服务器刷新后再报当前。

- BC：worktree pi05-online-bc；DVAC：pi05-online-bc-dvac。位于/data/chenyiteng/projects/rlinf-shenzhen/worktrees/；run在/data/chenyiteng/results/rlinf-shenzhen/online-bc/，各正式合同给精确路径。
- Sidney：worktree sidney-pi05-current-rlinf，run在/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/；当前运行目录runtime-resume100-to200，已从100续至200，勿重复launch。既有heartbeat由原窗口管理。
- Fast已结束，用户要求保持停止；没有新现象不重复报旧fatal。

## 按问题选一个入口，只读所需章节

| 问题 | 入口 |
|---|---|
| π0.5 BC基础、参数/模型 | docs/rlinf-robotwin-pi0-online-bc/02_PI05_ONLINE_BC_PLAN.md |
| BC＋DVAC方法 | 同目录01_DVAC_DESIGN.md（当前实现§11—12） |
| BC/DVAC种子、时间、最近图 | 同目录evidence/PI05_BC_SEEDS_TIMING_CONTEXT_STORAGE_DISCUSSION_20260906.md |
| 原π0在线BC | 同目录00_RESEARCH_AND_PLAN.md |
| Sidney GRPO续训 | docs/rlinf-shenzhen-multitask-pi05/evidence/RESUME100_TO200_EXECUTION_LEDGER_20260905.md |
| 存储/Git | 本页本轮账本；前次基线docs/server-admin/SZ_STORAGE_GIT_REVIEW_20260906.md |
| 旧实验、Fast/RLT/DSRL | docs/project-history/00_INDEX.md按需路由，不全文加载历史 |

## 当前未决点

- BC/DVAC环境seed表/轮转计划相同；独立rollout动作RNG未见显式控制，actor1234不等于全pipeline固定。不能把噪声认定为差距唯一原因，需进一步查配置/源码/数据/权重。
- 环境内存增长未定位。参数U10、权重[0.5,1.5]仍按现役合同；本轮只讨论是否改变。
- 本地根Git原为空；归档应走核实过的既有个人仓库独立分支，不盲改remote或把所有未跟踪文件塞进算法分支。

旧根文件已逐字哈希校验归档于docs/project-history/context-20260906-before-trim/。不再强制读09-03窗口交接。结束时用简短结果替换本页当前状态，不继续累加时间线。
