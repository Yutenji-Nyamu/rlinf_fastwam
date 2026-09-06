# RoboTwin RLT Stage 2：8-env × 250-cycle formal 启动记录

> 任务：RoboTwin `adjust_bottle`
> 状态：2026-07-30 11:37:25+08:00 已启动；首个完整8-env cycle已闭合
> 100-cycle历史run仍只称pilot，本次才使用获批的完整多轴formal预算

## 1. 当前结论

8-env资源门已完整通过，随后同一代码、Stage 1 artifact、batch、replay、模型和评估协议的
250-cycle formal已经健康启动。2026-07-30 11:41:49+08:00只读检查确认：

- driver PID `154857`、resource monitor PID `154858` 均存活；
- 首个cycle完成8条、每条200步的reference rollout；
- 新增151条macro transitions；两rank replay平均75.5、较慢rank为71；
- warm-up尚未满足，因此`update_step=0`、本轮updates=0，符合10k rows/rank设计；
- 两卡约17,111/17,194MiB，负载对称；
- cgroup anon约32.4GiB，OOM/OOM-kill为0，memory PSI为0；
- 未出现CUDA OOM、NCCL fatal、NaN或Ray rank death。

Codex在这个节点停止主动轮询；训练、2秒资源monitor和18小时hard timeout留在服务器，
下次只有用户主动要求时才刷新现场。这里的“当前”是上述时间快照，不替代未来live检查。

## 2. 代码与隔离

| 项目 | 固定值 |
|---|---|
| worktree | `/root/autodl-tmp/RLinf_rlt_pi0_robotwin` |
| branch | `codex/rlt-pi0-robotwin` |
| HEAD | `46a2d19bae629eaa57830f5faeac71ac81a1a494` |
| 新提交 | `46a2d19b feat(rlt): add 8-env formal evaluation protocol` |
| legacy影响 | 无；只新增RLT专属overlay/seed bank/test，不改EnvWorker、RoboTwinEnv、PPO或GRPO |
| Git远端 | 本地clean；相对upstream ahead2。一次有界push在`github.com` 9秒HTTP000超时前退出，未实际push |

新提交只包含：

1. `robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml`；
2. `robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_resource_smoke.yaml`；
3. `eval_seeds_adjust_bottle_rlt_periodic20_v1.json`；
4. 连续两次20-seed游标回绕单测。

集中验证结果为27 passed；legacy base、formal、smoke三份原生compose/resolve均通过。

## 3. 8-env资源门

### 3.1 路径与结果

| 项目 | 值 |
|---|---|
| run root | `/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1` |
| runtime evidence | `/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime` |
| resolved SHA-256 | `bbcdcdcb22ca93f106dea0786a57086a2e8317caaefd610d18cabe6c4e2ff6aa` |
| wall-clock | 966s（16m06s） |
| train | 3 cycles、24 episodes、472 macro transitions |
| updates | critic/actor `3200/1600` |
| eval | 同一官方bank的20条unique seeds，`2/20=10%` |
| checkpoint | `global_step_3`，completion=true，world size2，update step3200 |
| run大小 | 63,548,290 bytes |

资源与数值：

| 指标 | 结果 |
|---|---:|
| GPU显存峰 | 17,169 / 17,252MiB |
| GPU利用峰 | 100% / 100% |
| matched RSS峰 | 51.88GiB |
| env RSS峰 | 15.07GiB |
| cgroup anon峰 | 47.47GiB |
| cgroup file峰 | 192.19GiB |
| cgroup current峰 | 240.00GiB |
| host available最低 | 934.03GiB |
| high / OOM / OOM-kill增量 | 0 / 0 / 0 |
| max回收事件增量 | 7,867 |
| actor/critic grad峰 | 3.048 / 1.022，均低于clip10 |

240GiB raw cgroup峰主要仍是可回收file cache；它确实触发了`max`回收，但anon、RSS、PSI和
OOM合同均通过，因此允许8-env formal。没有手工`drop_caches`。

### 3.2 两个准备问题及修复

1. 准备脚本的进程gate最初把Paramiko远端`bash -c`自身命令文本识别成训练进程，三次在
   创建evidence前fail closed。改为只接受`comm`为Python且args含
   `train_embodied_agent.py`，同时单独检查exact-name `raylet/gcs_server`。
2. 第一次launcher把Bash数组在heredoc内合成一个带空格的单一“可执行文件名”，训练前
   exit127，run root未创建。失败证据完整保存在：
   `/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1_failed_launcher_127`。
   修复为逐参数写出命令，`bash -n`和命令段目视通过后才重启；算法/config/hash均未改。

## 4. formal完整预算

| 轴 | 获批值 |
|---|---:|
| outer cycles | 250 |
| train env / episodes | 8 / 2,000 |
| 最大primitive action slots | 400,000 |
| 最大 / 预计macro transitions | 40,000 / 约39,105 |
| 预计critic / actor updates | 约125,525 / 62,762 |
| periodic eval | 每25 cycles × 20条，共10次/200条 |
| checkpoint | 每25 cycles，共10个 |
| replay warm-up | 10,000 rows/rank |
| critic floor / cap | 30,000 / 1,600 per cycle |
| BC/Q schedule | 20,000 warm-up + 50,000 ramp |
| replay window | 50,000/rank |
| UTD / critic:actor | 5 / 2 |
| batch global/micro | 512 / 128 |
| H/C/D | 50 / 10 / 14 |

预计wall-clock约10–13小时。外层hard timeout为64,800秒（18小时），只作故障保险；
正常终止由250 cycles控制。不会按中间成功率挑checkpoint或提前停止。

## 5. 正式运行入口

| 项目 | 值 |
|---|---|
| run root | `/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1` |
| experiment | `robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1` |
| runtime evidence | `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime` |
| resolved SHA-256 | `586644cd69461016c1dd8c653da0eea12b01c61f2d0a9b4901654d90800f2a3e` |
| driver / monitor PID | `154857 / 154858` |
| started at | `2026-07-30T11:37:25+08:00` |
| exact command | `runtime/exact_command.txt` |
| resolved config | `runtime/resolved.yaml` |
| provenance | `runtime/run_provenance.tsv` |
| budget | `runtime/budget.json` |
| 2s resources | `runtime/resources.csv` |
| full console | `runtime/driver.log` |
| stop contract | `runtime/stop_conditions.txt` |

精确调用等价于：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples/embodiment/config \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250 \
  runner.logger.log_path=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1 \
  runner.logger.experiment_name=robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1 \
  runner.resume_dir=null
```

实际由`run_foreground.sh`再包一层18小时`timeout --signal=TERM --kill-after=180s`。

## 6. 后续检查口径

下次只读刷新至少检查：

1. driver、monitor、Ray和两卡是否仍活跃；
2. 最新完整cycle、累计episodes/transitions和`update_step`；
3. cycle约128的10k rows/rank ready、约147的student接管、约172的BC/Q ramp结束；
4. 每25-cycle eval是否严格`num_trajectories=20`；
5. checkpoint 25/50/...的completion、双rank state和磁盘增长；
6. GPU、matched/env RSS、cgroup anon/file/current、PSI、OOM和磁盘；
7. actor/critic loss、Q、grad是否finite；不把单点成功率噪声当停止依据。
