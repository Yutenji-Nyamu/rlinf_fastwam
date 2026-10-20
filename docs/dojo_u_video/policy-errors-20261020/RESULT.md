# 2026-10-20：策略记录错误自动跳过与速度复盘

深圳3·7的make_toast持续出现`Repeated query identity`：记录器在同一回合遇到已经保存过的query编号，拒绝覆盖；模拟器又重置继续，动作计数反复回退。到停止时3.58小时、0个有效回合。这个错误不是有效动作进度。

## 已执行

- 复用现有`run_task.py`外层清理路径，10秒检查一次策略日志尾部；发现该明确错误，保存原因、清理本任务token所属进程，确认本卡C/G释放，再把任务记作SKIPPED/PARTIAL，启动下一项。
- 不在同一报错任务里无限重试；如已有完整有效回合则保留，未完成部分不冒称失败回合。
- 同时修复progress.json在原生重置时短暂不存在导致误跳过的读文件竞争：保留上次进度时间，仍由原600/1200秒停滞条件处理。
- GPU清理不成功或录像验证失败仍停止该lane，避免下一任务叠上未清理上下文。

CPU测试10项通过，涵盖错误识别、正常/缺失日志不误判、清理后0回合跳过并启动下一任务、未清理/无结果不放行。当前make_toast通过唯一PID/UID/start向旧wrapper发SIGTERM，原清理回执`clear=true`；已接`pick_from_conveyor_by_image`，观察到25环境动作推进。6卡fasten_screws保持运行。

部署：`/data/chenyiteng/deployment-20261020/dojo-policy-errors-v1`。活动与补跑两层lane均使用`deployment-20261019/dojo-device-skip-g7-v1/source/run_task.py`，以后新任务自动继承。已运行中的6卡wrapper不重启，其下一任务会加载补丁。

## 为什么慢

仍为每卡25并行、每任务5批=125回合。当前stack_bowls 125回合约4.03h（48min/批），stack_blocks约3.18h（38min/批），align_blocks约1.19h（14min/批）。历史stack_bowls单批25约37min，机器/布局/成功早停不同，只能粗比。

fasten_screws上限1900动作，前100回合仅1成功，大多跑满；现场约4.1秒推进25环境的一个动作步，单批跑满约2.2h。它与550动作上限的任务不能按相同小时数估算。make_toast额外浪费的3.58h则来自明确错误循环，已用自动跳过处理。

不据此擅自减少预算或缩短动作上限；后续ETA按具体任务步数、有效批次与跳过情况更新。

本轮源码分支`codex/inference-queues-20261020`，本地`worktrees/pi05-policy-budget-eval-20261017`；小补丁与测试、部署记录公开，完整视频仍保留服务器。
