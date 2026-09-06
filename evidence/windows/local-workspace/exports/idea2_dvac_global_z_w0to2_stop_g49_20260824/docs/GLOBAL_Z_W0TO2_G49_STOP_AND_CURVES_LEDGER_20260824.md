# Global-z [0,2] g49 现场刷新与成功率曲线流水账（2026-08-24）

## 1. 请求与边界

- 请求：刷新当前 global-z `[0,2]` 训练，查看训练、方法和资源产物，并绘制逐 step 成功率、trailing 5-step mean、trailing 10-step mean。
- 操作边界：只读服务器现场；不停止、不重启、不修改训练；本地只新增轻量证据、分析脚本、CSV、PNG和文档。
- 统计口径：只计入 `metrics.log` 中形成完整 `Global Step: N/100` 表块的 step；在途 rollout 不计入。

## 2. 上下文恢复

1. 完整读取根目录 `PROJECT_CONTEXT.md`、`HANDOFF.md`。
2. 按 `HANDOFF.md` 路由读取专题单一事实源 `docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md`。
3. 确认本轮只刷新已有正式训练，不改变授权边界。

## 3. 第一轮服务器现场只读刷新

执行入口：

```text
local_scripts/remote_exec_autodl.py run --command-file tmp/idea2_global_z_w0to2_live_refresh_20260824.sh
```

连接方式：既有低层 Paramiko 密码认证；密码只进入当前本机进程环境。

结果：

- 服务器身份：`autodl-container-nekaqbwt43-6ce5babb`，用户 `root`。
- `wrapper/driver/observer` 均已退出；GPU无训练进程，当前cgroup内存约0.36 GiB。
- 最新完整训练记录为 `Global Step 49/100`；随后 step 50 rollout 只到 `14/16`，不计入训练曲线。
- 最新完整方法产物为双rank `rollout_step0048.npz`，与 Global Step 49 对齐。
- `FATAL_MATCHES=0`；两个 traceback 均是历史已知的可选Curobo导入提示；当前检查未见CUDA OOM、NCCL fatal或core actor残留。
- checkpoint已完整保存到 g10/g20/g30/g40；运行目录约39 GiB，runtime约68 MiB。
- 当前现场：GPU显存0 MiB；宿主可用RAM约955 GiB；`/root/autodl-tmp`可用约735 GiB。

待核对：为何进程在未完成 step 50 时结束。下一条只读命令专门检查wrapper/observer/driver结尾、退出标记、容器启动时间和资源监控尾部。

## 4. 退出原因窄审计

执行入口：

```text
local_scripts/remote_exec_autodl.py run --command-file tmp/idea2_global_z_stop_audit_20260824.sh
```

关键结果：

- 当前容器 PID 1 的启动时间是 `2026-08-24 16:03:26 +08:00`。
- 旧训练driver最后写入 `15:59:22`；资源监控最后采样 `16:00:27`，当时两卡仍各占约29.1/25.7 GiB，driver仍标记alive。
- 当前容器是在上述最后日志之后重新启动的。因此，训练不是从Python正常走到100步退出，而是随容器重启中断。
- driver结尾没有Python异常、CUDA OOM、NCCL fatal、`KeyboardInterrupt`或退出码；kernel OOM匹配为空。
- run期资源CSV从头到尾 `event_oom=0`、`event_oom_kill=0`。`event_max`增长表示曾碰到cgroup内存上限压力，不等同于OOM；它没有触发训练脚本内的自动停止行为。
- 当前证据能确认“容器重启导致进程消失”，但仅凭容器内日志不能判定是谁或什么外部操作触发了重启。

## 5. 轻量快照下载

执行：

```text
python tmp/idea2_global_z_download_stop_g49_20260824.py
```

下载内容：完整 `metrics.log`、两rank方法CSV与g49 NPZ、rolling state、run manifest、driver/config/launch、完整资源CSV。未下载checkpoint、历史全部NPZ、54 MiB进程RSS明细或重复视频。

本地目录：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_stop_g49_20260824/raw
```

完整资源CSV大小 `9,837,460` bytes；双rank最新NPZ约1.09 MiB；其余日志与元数据约0.77 MiB。

## 6. 分析与可视化

执行：

```text
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe \
  docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_stop_g49_20260824/analyze_global_z_g49.py
```

检查：

- 从`metrics.log`解析49个完整表块；最新严格等于g49，主数值字段全部finite。
- 双rank均存在`rollout_step0048.npz`；权重公式复算最大绝对误差`5.96e-08`。
- success图使用trailing arithmetic mean、`min_periods=1`、无forward fill；五个run各自在真实终点停止。
- 四张PNG均通过本地视觉检查，标题、图例、坐标、终点和数值可读。

核心结果：

- 当前g49 raw/5-step/10-step success=`93.359/93.203/93.320%`，g1--49平均=`90.139%`。
- 相对原GRPO共同g1--49：累计/末5/末10=`+2.081/+2.734/+2.930pp`。
- g49 weight p05/median/mean/p95=`0.367/1.038/1.088/2.000`，ESS=`0.897`，有效H=`44.83/50`，系数角=`18.22°`。
- GPU峰值=`30.37/30.22 GiB`；cgroup峰值约240 GiB；max-event增量87,054，OOM/OOM-kill增量均0。

输出：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_stop_g49_20260824/analysis/
  GLOBAL_Z_SUCCESS_RAW_ROLL5_ROLL10_G49.png
  FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.png
  GLOBAL_Z_METHOD_DIAGNOSTICS_G49.png
  GLOBAL_Z_RESOURCES_G49.png
  GLOBAL_Z_SUCCESS_RAW_ROLL5_ROLL10_G49.csv
  FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.csv
  GLOBAL_Z_DVAC_{RANK,STEP}_METRICS_G49.csv
  GLOBAL_Z_LATEST_HORIZON_G49.csv
  SUMMARY_G49.json
```

专题解释写入`31_GLOBAL_Z_W0TO2_G49_RESTART_CLOSEOUT_AND_SUCCESS_SMOOTHING_20260824.md`。
