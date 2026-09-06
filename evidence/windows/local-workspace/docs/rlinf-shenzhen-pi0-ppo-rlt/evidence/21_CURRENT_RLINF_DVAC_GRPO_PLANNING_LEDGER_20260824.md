# current RLinf DVAC-GRPO 规划流水账（2026-08-24）

本账只记录只读源码/配置审计和一次深圳 RLT 现场刷新；没有算法实现、服务器写入、测试、smoke、
进程控制、安装或下载训练资产。凭据只存在于本地当前进程，未写入命令文件或本文。

## 1. 上下文与来源

1. 完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md` 和当前 RLT/DSRL 专题入口。
2. 读取旧 Idea2 DVAC、current GRPO、current π0 telemetry 与旧/新 RLT 的当前事实源。
3. 远端 refs 只读核对：

```text
git ls-remote --heads https://github.com/Yutenji-Nyamu/rlinf_fastwam.git
```

关键结果：`61996e15`、`145fa810`、`afdaa2e2`、`554c6dc8`、`800baf80` 与文档一致；
current remote `main=8138d670...`，本次深圳 source仍锁 `7d07a421...`，不追 main。

## 2. Git 精确 diff

为避免下载完整 checkout，建立 filtered/no-checkout 审计镜像：

```text
git clone --filter=blob:none --no-checkout \
  https://github.com/Yutenji-Nyamu/rlinf_fastwam.git \
  references/rlinf_fastwam_audit_20260824
```

- C: 操作前可用约36.06 GiB。
- 初始 pack约2.19 MiB；按需读取 diff blob后 pack约86.45 MiB。
- 未 checkout 工作树，不含模型、数据、checkpoint或实验产物。

随后对三段执行 `git log/show/diff --numstat/--shortstat`，得到：

```text
6d0db56..61996e15  6 files, +1099/-9
61996e15..145fa810 8 files, +1414/-9
145fa810..afdaa2e2 7 files, +1271/-25
6d0db56..afdaa2e2  16 files, +3759/-18
7d07a421..800baf80 7 files, +1494/-9
7d07a421..554c6dc8 1 file, +176/-0
```

逐文件与符号通过 `git show <ref>:<path>`、`git diff <left>..<right> -- <paths>` 核对；确认旧 actor
目标类在 current 已迁到 `embodied_fsdp_actor_worker.py`，current typed trajectory可原生携带嵌套
`forward_inputs`。

## 3. 深圳 RLT 现场刷新

只读 command file：

```text
local_scripts/remote_commands/shenzhen_rlt_dsrl_final_delivery_probe_20260824.sh
```

通过固定 host-key 的 Paramiko helper、普通账号执行。2026-08-24 14:49:51 CST：

```text
rlt_state=alive, exit=missing
latest runner step=110/250
global_min_replay_size=8160/10000
global_total_transitions_added=16512
actor_switch_rate=0
ready_for_online=0
update_step=0
GPU4/5=21433/21727 MiB
host available=2068613214208 bytes
DSRL state=dead, exit=0
```

本次没有停止、重启、保存、修改或清理远端内容。
