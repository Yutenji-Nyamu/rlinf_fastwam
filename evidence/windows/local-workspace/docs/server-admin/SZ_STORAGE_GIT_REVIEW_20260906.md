# 深圳本人存储与Git覆盖只读复核

现场：2026-09-06 10:18 CST；范围`/data/chenyiteng`和本人RLinf/RoboTwin linked worktrees，不是审计其他用户私人目录。无删除/移动/进程干预/Git push。数值以allocated bytes换算GiB，原始[JSON](SZ_STORAGE_GIT_READONLY_20260906.json)保留完整路径、文件大小、链接数、worktree状态及远端结果。

## 1. 当前存储结构

| 本人/data一级目录 | GiB | 含义 |
|---|---:|---|
| results | 1527.05 | 训练checkpoint、回放、运行日志/视频等；占本人90.1% |
| projects | 75.47 | 源代码、RoboTwin/模型项目资产；不是全是小文本 |
| models | 66.69 | π0/π0.5原SFT、Fast/FasterWAM等起点和转换格式 |
| ray | 20.61 | 包含当前共享Ray会话，不能作为普通旧日志直接删 |
| datasets | 2.94 | 数据资产 |
| cache | 2.01 | 缓存 |
| 合计（含其他小项） | 1695.39 | 约1.66TiB |

results细分（无重叠列出）：

| 类别 | GiB |
|---|---:|
| 当前Sidney π0.5 GRPO | 429.61 |
| online-bc（新旧BC、DVAC、smoke） | 271.58 |
| 旧π0 GRPO/DVAC变体 | 262.17 |
| DSRL | 187.72 |
| 旧π0.5 GRPO/DVAC | 131.50 |
| Fast-WAM GRPO | 108.07 |
| PPO | 89.40 |
| RLT | 46.66 |

目前/data盘剩630.06GiB（82%使用）、/home盘剩1246.15GiB（47%使用）。这不是本人/home用量排名，本轮未重复扫其他用户。

## 2. 推荐清什么；为什么

以下只是**讨论候选**，不是删除授权/执行清单。仅考虑大于1GiB的具体文件，保留目录、配置、日志、指标与小文件。真正执行时还需刷新活跃/恢复依赖与保留点；本轮只确认文件在，不声称恢复测试通过。

| 优先级 | 候选大文件 | 可释放GiB | 保留什么与理由 |
|---|---|---:|---|
| 1 | π0 v8 smoke Step1的full_weights＋训练分片 | 17.19 | Step2同两大文件均在；测试完成，初代复评价值低 |
| 1 | π0.5 smoke Step1的full_weights＋训练分片 | 18.91 | Step2同两大文件均在；原SFT和正式run均另存 |
| 首选小批合计 | 4个文件，均nlink1 | **36.10** | 不触碰当前正式三项 |
| 2 | DSRL正式Step65/130/195各4个大文件 | **90.35** | 保留Step200；会失去三个历史模型/续训点，需另确认依赖 |
| 次选 | DSRL fresh smoke Step1四大文件 | 30.12 | 如无恢复依赖，保留resume Step2及正式200即可 |
| 次选 | RLT Stage1 smoke Step2大权重 | 20.58 | 保留正式Stage1 Step2000，后者可能是Stage2训练起点 |

前两项精确根路径：

```text
/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8/pi0-bc-u10-eval8x4-smoke-gpu6/checkpoints
/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1/pi05-pillbottle-bc-u10-eval8x4-smoke-gpu6/checkpoints
```

各自`global_step_1/actor/model_state_dict/full_weights.pt`及`global_step_1/actor/local_shard_checkpoint/checkpoint_rank_0.pt`；保留`global_step_2`。π0大小7.5111＋9.6768GiB，π0.5为7.9410＋10.9664GiB。路径前缀须按原始JSON的`generation.path`执行前精确核验，本文不提供直接删除命令。

DSRL正式根：

```text
/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run/dsrl-current-formal-200c-v2/checkpoints
```

三个较早step各有`actor/sac_components/target_model/checkpoint_rank_{0,1}.pt`、`actor/local_shard_checkpoint/checkpoint_rank_{0,1}.pt`，合计每代30.1175GiB；整代约31.29GiB，因为还含小文件，不能把整目录大小当成本次“大文件清理量”。Step200相应文件仍在。

暂不推荐先动：

- 当前BC/DVAC的Step10—30和Sidney较早保存点。它们是大头，但还在实验比较阶段；例如Sidney Step70是当前最好fixed，Step100是续训边界，不应机械地只留最新150。
- 原SFT/转换模型、RoboTwin资产、当前共享Ray。目录名字带cache或文件可重新下载，不代表可不看依赖直接删除。
- 旧GRPO/PPO大多已经清至最后保留点；上一批约347.73GiB不能重复计算成仍可回收。`dvac-global-z-...phys4567-v4`仍有较早大文件，但最后点历史完整性有疑问，应保留待核验，不挤进首选清理批。
- 旧π0.5/Fast非最后代已执行过清理的部分不重复推荐。

建议长期保存规则：正式run按现有每10轮保存；结束后按**最后可恢复点＋最佳评估点＋必要续训起点**选择保留。smoke一般保留最后完整代和轻量报告即可。若未来自动轮转，训练状态/回放/模型要作为同代处理，不能只凭文件mtime删除。

## 3. 代码与实验记录：哪些在Git，哪些还不在

### 3.1 已经确认的部分

本轮读取两个Git common-dir下26个worktree（RLinf22、RoboTwin4）；RLinf的**22个本地codex分支HEAD与personal远端逐项一致**，包含BC、DVAC、π0.5、GRPO/PPO、Prism、ST、Fast、DSRL、RLT与诊断分支。

| 实验家族 | 已跟踪的小体积证据（代表路径） | 不能扩大成什么结论 |
|---|---|---|
| 旧GRPO/PPO及DVAC/Prism/ST | `evidence/completed_runs_20260822_20260903/`及`followup_light_20260905/` | 不是每份服务器原始日志逐字备份 |
| Fast-WAM GRPO/DVAC | completed_runs、补丁/失败记录与后续轻量证据 | 巨大视频/权重仍在服务器 |
| DSRL/RLT | completed_runs、恢复/诊断轻量记录 | 诊断worktree未提交改动仍是缺口 |
| π0 BC/BC＋DVAC | GPU6 smoke证据、online-bc-eval8/u10、online-bc-dvac | 启动记录不等于所有正式尾段已封存 |
| π0.5 BC | `docs/evidence/pi05-online-bc-smoke-20260905/`、`pi05-online-bc-formal100-20260905/` | 进行中的Step45曲线没有由本轮push |
| π0.5 BC＋DVAC | `docs/evidence/pi05-online-bc-dvac-20260905/` | 同上，不把前代继承证据当当前run结果 |
| Sidney π0.5 GRPO | `evidence/bc_dvac_review_20260905/formal100-summary-20260905.zip`、refresh110 | 已有100步总结，156步新尾段未由本轮归档推送 |

当前四条BC代码HEAD：π0 BC `2467d997`，π0 DVAC `912808c7`，π0.5 BC `6a93605d`，π0.5 DVAC `776ebc98`；Sidney `81be3193`。这些HEAD含文档提交，不能把“当前Git HEAD”无条件当进程启动时源码SHA，正式启动源锁另见运行合同。

### 3.2 明确未提交、未核实或尚未封存

1. RLT诊断worktree两处dirty未提交：`rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`、`rlinf/hybrid_engines/fsdp/strategy/checkpoint.py`。分支已push不代表这两份修改已保存到Git。
2. RoboTwin OIDN-toggle worktree有未跟踪`oidn_toggle_trial.py`。
3. RoboTwin三个codex分支本地HEAD可读，历史记录说已推；**这次远端查询HTTP/2失败，随后HTTP/1.1超时，未完成实时远端一致性核验**。不能把历史结论冒充本轮核实。
4. 当前三项运行的新尾段、今晨曲线/存储/种子调查是新证据，没有自动push；运行结束后适合用小型配置＋source-lock＋metrics＋关键日志＋图封存。
5. 本地`C:\Users\86136\Documents\rl`的`AGENTS.md/PROJECT_CONTEXT.md/HANDOFF.md`及本专题docs当前显示未跟踪。**服务器代码已推，不等于本地研究上下文已备份到Git**。
6. 独立Fast/FasterWAM、DVAC observation等非这两个Git common-dir下项目，不能用本轮26个worktree结果宣称全覆盖。run路径在说明中出现也不等于完整日志已经归档；本轮未给“所有实验100%已保存”的结论。

推荐后续：先单独整理那3个未提交文件的意图；再给已结束run补小型证据包，当前run结束后封存；本地文档单独纳入受控Git仓/目录，不把models/checkpoints或所有巨大原始JSON直接add进代码仓。本轮只报告，不代为提交或push。
