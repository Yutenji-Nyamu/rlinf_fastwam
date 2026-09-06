# 2026-09-03 深圳 RLinf 工作窗口交接

## 0. 后续专题路由（2026-09-04）

若当前问题是π0在线BC／成功过滤／成功＋DVAC，唯一SSOT为
`../rlinf-robotwin-pi0-online-bc/00_RESEARCH_AND_PLAN.md`，不要继续遍历以下历史实验。
该文已收敛为干净首版的独立上下文，含代码参考优先级；旧广搜只作history背景，不需另读。
本专题已获独立实现/基础测试授权，并确认adjust_bottle π0 SFT、参数化D0混合和优先GPU3；已完成实现及7项基础测试，GPU短测待合同确认，正式长训未授权。源码和运行状态见该SSOT，不沿用旧研究-only边界。
本文件其余部分仍是09-03历史上下文，后续动态记录在根HANDOFF，现场仍须服务器刷新。

## 1. 一句话状态

当前两项formal最后记录均存活：GPU4/5为Sidney多任务pi0.5 `move_pillbottle_pad`
GRPO；GPU6/7为修复RoboTwin renderer生命周期后的Fast-WAM GRPO续训。
本文件是近期上下文，不是动态现场；新窗口必须先只读刷新。

## 2. 当前活动实验（最后记录：2026-09-03 21:09 CST）

### 2.1 Sidney pi0.5，GPU4/5

- run：`move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1`
- 状态：完整Step2；Step1/2 success=`32.42/33.98%`，MA5/MA10均为`33.20%`；
  已进入下一轮rollout，尚未首次fixed32，fatal/OOM/OIDN/traceback=0。
- 口径：64/32 train/eval env；rollout4；256 trajectories；G8；最多1024 query records；
  GB1024/MB32/update2；H50/C50/M10；noise0.5；horizon200；fixed32/eval5/save10。
- 与刚停止的Sidney `move_stapler_pad` run相比，仅任务名和run/output路径不同。
- 账本：
  `../rlinf-shenzhen-multitask-pi05/evidence/SIDNEY_MOVE_PILLBOTTLE_GRPO_CUTOVER_20260903.md`。

### 2.2 Fast-WAM，GPU6/7

- run：`fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2`
- 状态：完整Step22，Step23 rollout 1/4；无OIDN/pthread/OOM/fatal。
- 最新训练：success 31.25%，MA5 31.72%，MA10 33.13%。
- fixed32：Step5/10/15/20=`14/32,14/32,13/32,9/32`；Step20 DCP完整。
- 口径：32 env x rollout4=128 trajectories；G8；1024 query records；
  GB1024/MB2/update2；H32/C24/M10；fixed32/eval5；DCP save10。
- 已越过旧run在Step15第三次fixed eval触发的renderer故障边界。
- 账本：
  `../fastwam-robotwin-rlinf-grpo/evidence/ROBOTWIN_VECTOR_RENDER_LIFECYCLE_FIX_LEDGER_20260903.md`。

## 3. 本窗口近期完成的核心工作

### 3.1 Sidney多任务pi0.5接入current RLinf

- 原生LeRobot只作为行为oracle；离线将checkpoint/processor/norm严格转换为RLinf-native格式。
- 转换检查：813/813 keys，0 missing/unexpected/shape mismatch，逐tensor相等；224x224模型核心
  parity通过官方action容差。
- current RLinf上层复用原生pi0.5、Flow-SDE、typed trajectory、FSDP、GRPO与checkpoint；
  没有另建训练旁路。
- B=1 RoboTwin和两卡最小GRPO smoke均通过；后者完成rollout、两次update、fixed eval和
  local-shard save，峰值约58 GiB/卡。
- branch：`codex/sz-sidney-pi05-current-rlinf`；实现commit `bab221afb8be`；含轻量smoke evidence
  的已推HEAD `f50e235c5ab1`。
- 专题入口：`../rlinf-shenzhen-multitask-pi05/00_INDEX_AND_EXECUTION.md`。
- 实现账本：
  `../rlinf-shenzhen-multitask-pi05/evidence/CURRENT_RLINF_ADAPTER_IMPLEMENTATION_LEDGER_20260903.md`。

### 3.2 Fast-WAM current RLinf及renderer故障修复

- current Fast-WAM plain GRPO：`codex/sz-fastwam-current-rlinf-grpo@7b2331c5`。
- current Fast-WAM Action-DVAC：`codex/sz-fastwam-action-dvac-adv@a6ad77ea`。
- 原Step15故障链：fixed-eval auto-reset先大量`OIDN invalid handle`，随后
  `pthread_key_create failed`，再出现Python/Ray/NCCL连锁退出；不是训练数值或磁盘故障。
- 根因边界：RoboTwin `SubEnv`在兄弟renderer仍存活时清理进程级SAPIEN cache，full close又重复清理。
- 窄修只改`robotwin/envs/vector_env.py`：child只释放local资源；partial reset不清global cache；
  full reset/close待全部child释放并GC后只清一次；reset不再被global lock串行。
- fix branch：`codex/sz-robotwin-vector-render-lifecycle-fix@8c7380c1`。
- 当前续训已经越过旧Step15故障点，是最重要的在线证据；仍需自然继续观察。
- current port SSOT：
  `../fastwam-robotwin-rlinf-grpo/12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md`。

### 3.3 轻量实验evidence回填

- 已将约15个算法分支、约1265个文件、约41 MiB的轻量证据回填对应源码分支。
- 只含resolved config、命令、指标/TensorBoard、资源CSV、关键日志、图和manifest；
  不含checkpoint、模型、数据、视频、Ray全量日志或凭据。
- 当前仍在跑的两项formal应在结束后再补其最终小证据。
- 映射：`../server-admin/SHENZHEN_LIGHT_EVIDENCE_GIT_BACKFILL_20260903.md`。

## 4. 多RLinf并发必须继承的方法

1. 使用同一个persistent shared Ray head，不为单项任务重启全局Ray。
2. 每个job使用独立branch/worktree、Ray namespace、`RLINF_CODE_WORKING_DIR`。
3. train/eval/save全部使用run-scoped绝对路径，避免Ray worker把相对路径落到共同cwd。
4. 停止或清理只针对exact owned PGID和exact namespace；保留另一项训练及其他用户任务。
5. current comparison从Control resolved逐叶复制；只允许方法字段与run/output命名不同。
6. checkpoint目录存在不代表保存成功；必须看到目标shard/metadata后才可称可恢复。

## 5. 当前服务器与资源边界

最后快照：RAM available约1.59 TiB；最近一次完整健康盘点的memory/io PSI=0；`/`、`/home`、
`/data`最后分别余223 GiB、1.4 TiB、1.5 TiB。GPU3最后为空闲；GPU0--2有其他用户任务，不能打扰。
shared Ray及Mihomo正常；HF应优先走服务器代理。

账号密码和管理员密码不得写入仓库文档。服务器访问继续使用既有固定host-key Paramiko路径，
密码只注入当前进程。

## 6. 下一步建议

- 若用户问训练状态：只读刷新两项run，输出当前step、success/MA5/MA10、fixed eval、fatal、GPU/RAM及ETA；
  有多点历史时再画一张颜色区分明显的静态PNG。
- Fast-WAM：若继续稳定，等完成或出现新fatal再处理；不要主动改采样/优化参数。
- Sidney：等首个完整step和首次fixed32后再判断`move_pillbottle_pad`在RLinf200协议下的真实起点。
- 新的实现/正式实验仍按用户当前授权逐项进行；本次交接整理不扩大服务器写权限。

## 7. 历史在哪里

- 此前根`HANDOFF.md`的约95 KB累计时间线已完整归档：
  `../project-history/HANDOFF_SNAPSHOT_20260903_PRE_WINDOW_HANDOFF.md`。
- 更老交接和规则快照索引：`../project-history/00_INDEX.md`。
- 不要在新窗口默认读取这些历史；只有核对旧事故、旧参数或旧结果时按关键词查找。
