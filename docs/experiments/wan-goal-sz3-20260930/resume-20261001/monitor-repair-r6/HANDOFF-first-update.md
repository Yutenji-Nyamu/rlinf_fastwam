# 当前执行路由 · 2026-10-01 13:18

exp2继续深圳3 WM，exp执行深圳2 EXPO；用户最新安排两边各用物理4–7，RLT低优先级，结束后各自恢复原RLT。正常不同机不频繁交流，资源变更/交接/故障再协调。共享表见docs/server-admin/EXPERIMENT_RESOURCE_COORDINATION_20261001.md；旧完整交接已存docs/world-model/archive/HANDOFF_WAN_R6_BEFORE_20261001_1320.md。

深圳3当前Wan r6已经唯一启动，run `/data/chenyiteng/projects/wan-goal-sz3/runs/wan-goal-sz3-20261001-r6/pi05-formal`；bridge `wm-bridge-20261001-v6`，cycle `rlt-cycle-sz3-wan-goal-20261001-repair-v1`，Dojo原run的active-continuation权威。w110准备/w111启动不能重放。原四RLT完整CP125绑定；唯一v6 owner采用WM→RLT直接路线，无Dojo评测，无旧guard/竞争restorer。旧结果保留。

w123 13:18：owner活、RUNNING_WM，完成首轮step0；grad norm0.5559873、有效mask1.376953%、优势[-1.6201816,0.5400605]、loss0.0001563128均有限，下一轮采集中，monitor诊断与近期primary error为空。尚未越过原4轮故障点，未到原save40。下一步沿status_wake.sh和repair_runtime_evidence.sh只读实查；首次完整CP40且保存结束/下一轮推进后，用pi05-wan Python执行scripts/verify_formal_checkpoint40.py（CPU只读，成功回执已存在不重复）。进展专题docs/world-model/WAN_GOAL_REPAIR_PROGRESS_20261001.md。

固定d34d4c3与原SFT、头图/腕mask、seed42、GRPO/过滤、N64/G8/R8/L320/C8、global2048/micro128、H10/去噪5、1000轮/save40均保持。w108服务器18CPU检查通过；r5的AssertionError原traceback缺失，根因仍未确定。本轮修复proc FD/status UID/start、原身份pidfd、异常诊断和cleanup，不升级模型/环境后端。源码/来源见WAN_GOAL_REPAIR_20261001.md、WAN_GOAL_REPAIR_SOURCES_20261001.md；粗细日志WAN_GOAL_RUNLOG及local_logs/wan-goal-20261001/steps。

最新已发布f4e54725672728df2f728c40c244643801a7a751（w120，16A/8M/0D），分支codex/sz3-wan-goal-20260930。回执ROOT/publication-update-20261001-monitor-repair-r6/published.json。后续只发布本窗审过源码和轻量真实证据，不发布EXPO dirty。

深圳1原四RLT继续；深圳2物理4–7已授权EXPO借卡，由exp窗口唯一owner负责停/归还原四RLT，本窗不抢恢复。0–3不新增RLT副本。只保留统一rlt检查，正常推进安静；不得从历史PID/瞬时0%发信号。深圳chenyiteng固定host-key Paramiko，凭据仅RAM；SSH73160仍活。WM释放后由同owner精确归还，RLT原任务/方法/seed累计3000预算保持；共享Ray/其他用户/驱动/无关实验不动。
