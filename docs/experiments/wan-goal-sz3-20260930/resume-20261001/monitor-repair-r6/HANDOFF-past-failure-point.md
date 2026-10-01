# 当前执行路由 · 2026-10-01 14:32

exp2继续深圳3 WM，exp执行深圳2 EXPO；用户最新安排两边各用物理4–7，RLT低优先级，结束后各自恢复原RLT。正常不同机不频繁交流，资源变更/交接/故障再协调。共享表见docs/server-admin/EXPERIMENT_RESOURCE_COORDINATION_20261001.md；旧完整交接已存docs/world-model/archive/HANDOFF_WAN_R6_BEFORE_20261001_1320.md。

深圳3当前Wan r6已经唯一启动，run `/data/chenyiteng/projects/wan-goal-sz3/runs/wan-goal-sz3-20261001-r6/pi05-formal`；bridge `wm-bridge-20261001-v6`，cycle `rlt-cycle-sz3-wan-goal-20261001-repair-v1`，Dojo原run的active-continuation权威。w110准备/w111启动不能重放。原四RLT完整CP125绑定；唯一v6 owner采用WM→RLT直接路线，无Dojo评测，无旧guard/竞争restorer。旧结果保留。

w137 14:32：owner活、RUNNING_WM，已完成5轮，超过原完成4轮后中断的位置；有效GRPO梯度step0/1/2，step3/4全组过滤mask=grad=loss=0，空优势统计nan不代表权重nan，亦不声称过滤轮参数完全未变化。第6轮采集5/8，monitor诊断与近期primary error为空。尚未到原save40，长期稳定仍待验证。下一步沿status_wake.sh和repair_runtime_evidence.sh只读实查；首次完整CP40且保存结束/下一轮推进后，用pi05-wan Python执行scripts/verify_formal_checkpoint40.py（CPU只读，成功回执已存在不重复）。进展专题docs/world-model/WAN_GOAL_REPAIR_PROGRESS_20261001.md。

另一窗口同步SZ2 replay index/payload缺陷后，w129 13:42对SZ3当前cycle冻结四CP125重新调用同helper严格inspect，all_valid=true、CUDA未初始化；索引/payload样本分别19036/19020/19710/19631，全部对应，小SHA/contract/resume_dir一致。当前无缺失，不借用SZ2供体或改冻结cycle。归还前既有resume会重新inspect四份，最新证据见进展专题。

固定d34d4c3与原SFT、头图/腕mask、seed42、GRPO/过滤、N64/G8/R8/L320/C8、global2048/micro128、H10/去噪5、1000轮/save40均保持。w108服务器18CPU检查通过；r5的AssertionError原traceback缺失，根因仍未确定。本轮修复proc FD/status UID/start、原身份pidfd、异常诊断和cleanup，不升级模型/环境后端。源码/来源见WAN_GOAL_REPAIR_20261001.md、WAN_GOAL_REPAIR_SOURCES_20261001.md；粗细日志WAN_GOAL_RUNLOG及local_logs/wan-goal-20261001/steps。

最新已发布02589f43c01c60b85643afc65c6f4b43145a772a（w125，6A/3M/0D），分支codex/sz3-wan-goal-20260930；含真实首轮更新、当前资源表和CPU checkpoint reader，服务器syntax/真实路径核验通过。回执ROOT/publication-update-20261001-r6-first-update/published.json。此前修复源码w120/f4e54725已发布。后续只发布本窗审过源码和轻量真实证据，不发布EXPO dirty。

深圳1原四RLT继续；14:00收到exp窗口现场回执：深圳2物理4–7借卡完成、原四RLT精确退出，唯一guardian=PAUSED，CP1500/1525/1475/1475已独立修复冻结；EXPO smoke-v1先用物理4派发。SZ2四卡均属对方借卡/归还流程，本窗不抢恢复，等对方唯一归还回执。详细身份仅供定位见共享表，不用历史PID信号。0–3不新增RLT副本。只保留统一rlt检查，正常推进安静。深圳chenyiteng固定host-key Paramiko，凭据仅RAM；SSH73160仍活。WM释放后由同owner精确归还，RLT原任务/方法/seed累计3000预算保持；共享Ray/其他用户/驱动/无关实验不动。
