# SZ1/SZ3 恢复完整 N8 RLT 调度 · 2026-09-29

归档边界：本文保留2026-09-29的RLT调度证据，作为EXPO参数继承的支撑材料；其中E盘原始证据和延伸RLT专题是本地历史索引，未随本次EXPO发布打包，不代表当前运行状态。当前EXPO请看[10月3日审计](../methods/expo-ft/AUDIT_20261003.md)。

**状态：12:24健康启动验收与发布核验完成。** SZ1/SZ3旧8个driver已退出，新8组均完成R1真实N8采集，92项检查全部通过；新组仍处teacher初池阶段，网络更新0，暂不判断效果。两分支已直接推送personal并回读SHA一致。SZ3启动时worker端口等待自行恢复，新48个actor均ALIVE，未改共享Ray。本文记录已发生动作，不是可重放启动命令。

用户本轮指出训练阶段缩短，要求参照以前成功的每轮8采样RLT参数检查1、3机，考虑重新放实验。历史原话已经在「0914 exp」核实：2026-09-14 23:36，用户引用完整半量方案后说“之前我们讨论过rlt每轮采样减半，训练阶段轮次不变；试试看吧”（turn `01a0a090-2e85-7f51-950f-759f84ae94f2`）。当时同时减半环境数与下面五项RLT预算，目的是大致保持阶段轮次。原半量合同（未随本包发布的本地历史索引：`RLT_CLEAN_HALF600_20260914.md`）

昨晚四任务切换按“每轮8，其他不变”仅恢复环境数，仍继承N4调度；切换记录（未随本包发布的本地历史索引：`FOUR_TASKS_RLT_20260928.md`）已明确记载。2026-09-29 12:01现场重新读取8组 `runtime/resolved.yaml`，全部确认N8、10k初池、15k初始化、cap800、课程10k＋25k。因此阶段相对旧完整N8提前有配置依据，不能仅根据当前较低成功率断言它是性能差距的唯一原因。

## 五项预算与继承范围

| 配置字段 | 原成功N8 | 半量N4 / 昨晚新N8 | 本次新N8 |
|---|---:|---:|---:|
| `algorithm.rlt_schedule.warmup_min_size` | 20000 | 10000 | 20000 |
| `algorithm.rlt_schedule.warmup_post_collect_updates` | 30000 | 15000 | 30000 |
| `algorithm.rlt_schedule.max_updates_per_train_step` | 1600 | 800 | 1600 |
| `algorithm.actor_weight_schedule.warmup_updates` | 20000 | 10000 | 20000 |
| `algorithm.actor_weight_schedule.ramp_updates` | 50000 | 25000 | 50000 |

原N8实配见 resolved.yaml（未随本包发布的本地历史索引：`../rlinf-shenzhen-rlt-dvac-pure-port/evidence/formal600-20260912/prepared-control/resolved.yaml`），调度字段位于230–244行。原成功N8是 **π0 / adjust_bottle**；之后登记成功的开关、移罐为 **π0.5 / N4**，详见[开关成功基线](PI05_RLT_TURN_SWITCH_SUCCESSFUL_BASELINE_20260925.md)、三任务配置核对（未随本包发布的本地历史索引：`RLT_THREE_TASK_SLOW_LEARNING_20260925.md`）。本次借用旧N8的调度预算，模型仍是当前π0.5，保留其归一化、student `identity` 输出及任务专属Stage1，不整份覆盖成旧π0配置。

其他保持当前各组实配：3000轮（`max_steps`与`max_epochs`均3000）、N8×rollout_epoch1、200动作、U5、C10、B512/micro256、actor/critic LR均1e-4 constant、critic:actor=2:1、回放cache/window80k、reference dropout0.5、BC/Q端点7/.05→2.5/.45、每25轮固定20回合评估及保存。Clean仅记录DV、应用权重off；combo仍τ2.5/drop0.2/anneal500。种子、任务、奖励、RoboTwin与π0.5实现保持。

## 图中灰区的含义

此前灰区标“初始收集”不够准确：它覆盖 **teacher纯采集＋teacher继续采集时的actor/critic初始化**，直到student接管。第一段网络更新为0；第二段网络已在训练，固定评估测的也是student，仍处灰区不代表没有更新。

原N8调整瓶子R136开始更新、R155 student接管、R193整轮使用BC/Q末端权重；半量N4约R135/R154/R190，因此当时阶段轮次大致保持。原阶段与预算依据（未随本包发布的本地历史索引：`../rlinf-shenzhen-rlt-dvac-pure-port/RLT_DVAC_DIRECTION_AND_BUDGET_DISCUSSION_20260913.md`） 本次N8却保留10k初池，单位轮新增transition更多，会更早跨池门槛；课程末端也仍是35k而非完整N8的70k累计critic更新。跨任务的成功早停会改变每轮有效transition，恢复预算也不保证四任务在相同轮次切换。

## 部署范围与路由

同四任务、同卡位，各自Clean/combo从 **fresh Stage2** 开始3000轮，直接复用昨晚已完成的对应任务Stage1（Clean50、B32/micro16、2000更新）。旧Stage2已经经历缩短调度，续训时改阈值不能补回早期数据和优化顺序，故使用新目录、新身份、空回放与新Stage2计数，保留旧运行全部日志、TensorBoard、DV及现存checkpoint，不清理。

| 主机 | 任务 | Clean / combo GPU |
|---|---|---|
| SZ1 | `place_phone_stand` | 4 / 5 |
| SZ1 | `pick_dual_bottles` | 6 / 7 |
| SZ3 | `move_pillbottle_pad` | 4 / 5 |
| SZ3 | `rotate_qrcode` | 6 / 7 |

已准备并派发的新入口（`{host}` 为 `sz1` 或 `sz3`，`{role}` 为 `clean` 或 `combo`）：

- 索引：`/data/chenyiteng/deployment-20260929/n8full-{host}-prepared.json`。
- 源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rlt-n8full-{host}-20260929`。
- 分支：`codex/{host}-pi05-rlt-n8full-20260929`。
- 输出：`/data/chenyiteng/results/rlinf-rlt/pi05-rlt-{task}-{role}-n8full-3000-20260929-{host}-v1`。
- 命令：准备阶段生成并审查各新run的 `runtime/command.txt` 与 `runtime/resolved.yaml`；已派发，后续按索引与回执查看，不沿旧部署稿重放。
- 停止条件：达到3000轮、用户明确停止或不可恢复运行错误；不设成功率自动中止阈值。

已在服务器逐叶验证仅五预算及新运行身份/路径改变，复用任务Stage1，按精确PID/UID/start/namespace逐任务切换；真实N8首轮采集、fresh计数与日志验收完成。SZ2 OpenWAM及其RLT恢复流程、共享Ray、其他用户和无关实验保持。

## 本轮只读证据

- SZ1完整实配与阶段数据（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8-config-audit-0929-a.json`）；SZ3同项（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8-config-audit-0929-a.json`）。`runs.current_*`下同时保存配置来源路径与SHA，现8组五值一致。
- SZ1切换前身份和进度（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8-audit-live-0929-a.json`）；SZ3同项（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8-audit-live-0929-a.json`）。这是切换前审计，当前身份改读下方新run回执。

## 执行回执

12:13准备源码：SZ1 `55c1399a50826d61e8735a64daa2f1742f1b824f`，SZ3 `56efffa988ab0001710af1199ec8f0d66063a434`，分别从昨晚当前源创建上述独立分支。准备回执核对五预算diff、Stage1复用和新路由，准备阶段未停旧。随后仅增加本人新namespace允许列表并派发，12:15左右开始切换；`inspect-v2`确认旧8driver已退出、新8driver存活，最终真实首轮完成由下方 `final-v1` 独立确认。

| 证据 | SZ1 | SZ3 |
|---|---|---|
| 五项配置差异与源码身份 | prepare-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8full-prepare-0929-v1.json`） | prepare-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8full-prepare-0929-v1.json`） |
| 允许列表及派发 | activate-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8full-activate-0929-v1.json`） | activate-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8full-activate-0929-v1.json`） |
| 启动状态 | status-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8full-status-0929-v1.json`） | status-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8full-status-0929-v1.json`） |
| 旧driver退出、新driver存活 | inspect-v2（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8full-inspect-0929-v2.json`） | inspect-v2（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8full-inspect-0929-v2.json`） |
| watch新路由更新 | closeout-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8full-closeout-0929-v1.json`） | closeout-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8full-closeout-0929-v1.json`） |
| 推送与远端SHA回读成功 | publish-v2（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8full-publish-0929-v2.json`） | publish-v2（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8full-publish-0929-v2.json`） |
| 首轮和92项最终核验 | final-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/rlt-n8full-final-0929-v1.json`） | final-v1（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz3/rlt-n8full-final-0929-v1.json`） |

12:24首轮回放池：

| 主机 / 任务 | Clean | combo |
|---|---:|---:|
| SZ1 / 手机支架 | 160 | 154 |
| SZ1 / 双瓶抓取 | 151 | 135 |
| SZ3 / 移药瓶 | 145 | 149 |
| SZ3 / 旋转二维码 | 160 | 157 |

八组均R1、`update_step=0`、`ready_for_online=0`，符合20k初池前的fresh阶段；DV invalid/missing均0，所查日志无fatal，92项最终检查全过。启动时SZ3端口暂时不足后来自行恢复，48个新actor均ALIVE，未重启或修改共享Ray。

两个分支各新增6件、修改0、删除0（四份任务配置与两份部署工具）；生产算法源码保持，head仍为上述准备SHA。旧实验产物删除0。watch已准确切到新run；`closeout-v1`在watch成功后因publisher保留旧日期断言退出，尚未push，该失败不是训练故障。修正检查器后 `publish-v2` 两机均直接推送并远端回读一致，`source_head_changed=false`、`training_changed=false`。

`inspect-v2`中的 `stage1_dependency` 检查曾错误取checkpoint的parent目录，是检测脚本路径错误；实际配置始终正确指向对应 `global_step_2000`，检查器修正后最终核验通过，Stage1未重训。禁止重复派发、重复停止或复用旧日期的发布检查稿。

## 后续监控

改读本次 `n8full-{host}-prepared.json` 及 `local_scripts/rlt_n8_audit_20260929/status.py`、`series.py`；旧four-tasks索引对应已退休运行，不应再当现役。图中的灰区应写“teacher采集期（纯采集＋learner初始化）”，或分两色标注首更与student接管，不能把整个灰区都说成没有训练。阶段/预算说明图：PNG（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/rlt-n8-audit-20260929/rlt-n8-budget-stages.png`） · SVG（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/rlt-n8-audit-20260929/rlt-n8-budget-stages.svg`）。训练各自继续到3000，后续ETA区分初池、初始化、student在线阶段。
