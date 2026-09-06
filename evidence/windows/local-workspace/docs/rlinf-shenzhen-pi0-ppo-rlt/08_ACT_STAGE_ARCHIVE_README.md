# `shenzhen_robotwin_act_handoff_20260821.zip` 内容说明

## 从哪里开始

1. 先读 [`05_ROBOTWIN_ACT_COMPLETED_HANDOFF.md`](05_ROBOTWIN_ACT_COMPLETED_HANDOFF.md)：完成结果、路径、
   source lock、20 步复现顺序和 artifact。
2. 遇到非官方标准流程问题时读
   [`06_ROBOTWIN_ACT_ISSUES_AND_SOLUTIONS.md`](06_ROBOTWIN_ACT_ISSUES_AND_SOLUTIONS.md)。
3. 要追每条服务器操作、退出码与关键输出时，从
   [`evidence/00_SERVER_OPERATION_LEDGER_INDEX.md`](evidence/00_SERVER_OPERATION_LEDGER_INDEX.md) 进入四个分账。
4. 下一阶段只读 [`04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md`](04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md)；ACT 的授权
   不自动扩展到 RLinf。

## ZIP 包含

- 本专题完成态/问题/来源/next-plan/C-E 存储文档；
- 现场、source/download、env install、ACT pipeline 四个实际流水账；
- official HF ACT 成功 eval 的 172,523-byte MP4；
- 本次所有 `shenzhen_*20260821*` command/probe 文件；
- 两个 resolved YAML、H100 CuRobo patch；
- `local_scripts/` 下固定 host-key 的无凭据 Paramiko helpers，以及安全 quota probe；
- ZIP 根目录 `MANIFEST_SHA256.txt`：每个 entry 的 SHA256。

## 刻意不包含

- SSH/sudo 密码、token、代理订阅 URL、Git credential；
- 服务器上的 assets、dataset、Conda env、model、checkpoint、完整 run logs；
- 冻结的旧规划账 `evidence/OPERATION_LEDGER.md`；
- 已被完成态交接取代的 pre-execution packet；
- 与深圳 ACT 无关的 AutoDL、RLT、OGPO、DVAC scripts/docs。

`local_scripts/remote_exec_autodl.py` 是历史文件名并保留旧 endpoint defaults；复现深圳机器时必须显式提供
handoff 文档中的深圳 host/port/user/host-key，不能使用它的默认连接参数。
