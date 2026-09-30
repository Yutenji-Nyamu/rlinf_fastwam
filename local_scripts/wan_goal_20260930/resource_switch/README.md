# SZ3 WM → Dojo 独立切换胶水（待审部署）

只在新的源码目录和新 cycle 中使用。现有 Dojo runtime 不复制、不改动；通过 `base-source-sha256.json` 核验并复用，RLT helper 在 prepare 时原样复制到新 cycle。运行账户/主机固定为 UID 20001 / h100-gpu01。

顺序：准备完整 WM → 动态精确 TERM 旧 Dojo controller → 等旧 outer 归还四个 RLT 并验证首轮 → prepare 新 cycle → 新 outer 借四卡 → WM → 精确清理 → 同一原结果续 Dojo → Dojo 最终归还 RLT。

## 文件和入口

- `stop_dojo.py`：默认只读输出绑定；加 `--execute` 才写唯一 intent 并 pidfd TERM。不会停止 outer/watchdog。
- `prepare_switch.py`：锁住旧 run，确认旧 owner 已退出且 RLT 四个首轮已验证；冻结新 cycle、原结果/manifests、WM spec 和源码哈希。不停止作业。
- `continue_pipeline.py`：复制并最小调整既有 continuation；新 WM 阶段之后继续原 Dojo。WM 命令失败但清理成功时也续 Dojo。
- `wm_stage.py`：登记 owner token、PID/UID/boot/start/命令哈希，运行 WM command 和 cleanup callback，独立核释放。
- `cleanup_owned.py`：仅向同 token / 已核身份的进程发 TERM，30 秒后 KILL；复核稳定退出与物理 GPU 4–7 无上下文。不会调用全局 `ray stop` 或连接共享 Ray。
- `check_resource_switch_cpu.py`：仅供服务器 CPU 针对性检查；不在 Windows 本地执行，不启动 Ray 或 GPU 训练。
- `audit_base.py`：只读核验 live 固定 v3 文件，差异只输出供审查，不自动更新信任清单。

## WM spec（由执行者填写实际路径与批准预算）

```json
{
  "schema": 1,
  "run_dir": "/data/chenyiteng/wan-runs/UNIQUE_RUN",
  "cwd": "/data/chenyiteng/REVIEWED_REPO",
  "physical_gpus": [4, 5, 6, 7],
  "max_seconds": null,
  "cleanup_timeout_seconds": 180,
  "command": ["/absolute/python", "-u", "/absolute/reviewed_launcher.py"],
  "cleanup_command": ["/absolute/python", "-u", "/absolute/resource_switch/cleanup_owned.py"],
  "environment": {},
  "ray": {
    "isolation": "dedicated",
    "address": "127.0.0.1:UNIQUE_PORT",
    "namespace": "UNIQUE_NAMESPACE",
    "temp_dir": "/data/chenyiteng/wr/UNIQUE_RUN"
  }
}
```

`max_seconds: null` 表示不添加墙钟时间上限，由训练自身预算或手动 TERM 结束。当前正式实验保持官方 1000 epoch，不额外设置 10 小时或其他时限；仅在用户另行授权时间预算时填写正数秒。command 是一个受审 launcher：启动独立 Ray，按“先 OFT smoke；验收通过再 π0.5 正式”的顺序运行；失败返回非零。胶水不修改任何训练参数。command 与 cleanup 均以 argv 执行，不经 shell。WM run 和 Ray temp 必须是新路径。

注入环境：

- `WM_OWNER_TOKEN`：本次唯一 token。嵌套 run_stage 使用 `父 token.新 UUID`；父 Catalog 认领精确 token 与点分隔的后代族，子 Catalog 不认领祖先/兄弟。Ray bootstrap 和所有 driver/worker 都应继承；建议也写入 Ray runtime_env 的 env_vars。
- `CUDA_VISIBLE_DEVICES=4,5,6,7`；`RAY_ADDRESS`、`CLUSTER_NAMESPACE` 为 spec 的独立实例。
- `WAN_GOAL_RUN_DIR`、`WAN_GOAL_CYCLE_ID`。
- `WAN_GOAL_IDENTITIES`：持续更新的 `managed-identities.json`。
- `WAN_GOAL_CLEANUP_RECEIPT`：cleanup 必须写出的 `wm-cleanup.json`。
- `WAN_GOAL_RAY_NAMESPACE`、`WAN_GOAL_RAY_TMPDIR`。

cleanup 返回零并写出与 owner/cycle/Ray 身份匹配的回执后，胶水还独立检查登记进程全部退出、4–7 无上下文，保存 continuation 下 `wm-release.json` 才启动 Dojo。cleanup 失败/超时则保持 `RESOURCE_RETURN_NEEDS_ATTENTION`，不启动 Dojo 或 RLT，不伪造归还成功。

不要在 launcher 内清空 token 或接共享 Ray。cleanup 只承诺清理 token 和持久登记的进程；脱离此身份链的进程必须查明，不能广泛匹配后杀掉。

同 UID 但未登记且 `/proc` 不可读的进程不会被认领/发送信号，PID/路径记在 `unreadable_unclaimed`；已登记身份变得不可核验时则报具体 PID/路径，不能据此宣称释放。路径用 resolve 核边界，但保留 `/data` 的字面路径供 launcher 使用。

## 借卡前检查（服务器 CPU）

本地仅做 AST/JSON 解析，不在本地执行项目测试。上传到全新、未运行的 glue 目录后，在深圳3 CPU 上执行：

```text
<RLT_PY> -B <GLUE>/audit_base.py --source-dir <LIVE_FIXED_V3> --write-candidate <UNIQUE_AUDIT_DIR>/base-candidate.json
<RLT_PY> -B <GLUE>/check_resource_switch_cpu.py
```

第一条仅读取 live v3 的六个固定运行文件并比较已审本地副本哈希。若不匹配，先读取/对比实际差异并核对当前 active pointer；不能因“服务器上能运行”就自动接受 candidate。确认是批准版本后，由审查者在新 glue 目录更新 `base-source-sha256.json`，重新 audit 后再 prepare。旧 v3 源码保持不变。

CPU 检查只创建自身临时目录和短时 CPU 子进程，验证：同 token 精确发信号、错误 start 不发信号、无关进程存活；未登记 PermissionError 跳过而已登记则失败；WM 非零退出且清理通过能进入续 Dojo 路径；清理失败不生成 release。GPU 状态回调在这组检查中是测试替身，真实四卡释放仍由正式入口独立检查。

现场 Python 若没有 `os.pidfd_open` / `signal.pidfd_send_signal`，`common.py` 仅在 Linux x86_64 LP64 下调用对应 syscall 434/424；不会退化成普通 PID kill。两处发信号入口仍在取得 pidfd 后复核身份。CPU 检查包含强制 ctypes 分支，以及真实子进程的 token 家族/祖先/兄弟隔离测试（共 7 项）。编号依据：[Linux v6.8 x86_64 syscall 表](https://github.com/torvalds/linux/blob/v6.8/arch/x86/entry/syscalls/syscall_64.tbl#L348)。

## 命令模板

以下均为未展开占位符，需审查后在服务器运行：

```text
<PY> -B <GLUE>/stop_dojo.py --base-source-dir <FROZEN_DOJO_V3> --run-dir <ORIGINAL_DOJO_RUN> --intent <CURRENT_ATTEMPT>/stop-for-wm-UNIQUE.json --reason 'Authorized WM then resume Dojo'
<PY> -B <GLUE>/stop_dojo.py --base-source-dir <FROZEN_DOJO_V3> --run-dir <ORIGINAL_DOJO_RUN> --intent <CURRENT_ATTEMPT>/stop-for-wm-UNIQUE.json --reason 'Authorized WM then resume Dojo' --execute

<RLT_PY> -B <GLUE>/prepare_switch.py --config <EXISTING_DOJO_CONFIG> --base-source-dir <FROZEN_DOJO_V3> --cycle-dir <NEW_CYCLE> --continuation-name <NEW_CONTINUATION> --wm-spec <REVIEWED_WM_SPEC>
<RLT_PY> -u -B <GLUE>/continue_pipeline.py --config <EXISTING_DOJO_CONFIG> --cycle-dir <NEW_CYCLE> --preparation-dir <ORIGINAL_DOJO_RUN>/prepare-<NEW_CONTINUATION>
```

继续脚本必须放在 Dojo project 内，且保留 `continue_pipeline.py` 名字；现 watchdog 以此识别新 outer。WM 阶段 phase 为 `RUNNING_WM/CLEANING_WM`，watchdog 不绑定；Dojo 阶段重新进入 `EVALUATING`，按新 active pointer 绑定。WM 阶段给新 outer 的 TERM/INT 表示结束 WM 并在清理后续 Dojo；Dojo 阶段沿用原停止与 RLT 归还语义。

验收：WM 退出和释放回执；Dojo 原计数不减少且产生首个新增原生结果；Dojo 最终退出后，四个 RLT 首轮均通过既有完整 checkpoint/计数/replay 验证。仅存活或 `resumed-dispatched` 不算完成。
