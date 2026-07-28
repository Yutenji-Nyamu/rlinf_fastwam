# π0 × RoboTwin × DSRL：fresh/resume smoke 批准材料

状态：**已按本批准包执行并通过**。本文保留首轮 N=20 smoke 的批准时配置、命令、输出、
资源预期和停止条件；实际执行结果与逐操作证据见
[`SMOKE_EXECUTION_LOG_20260728.md`](./SMOKE_EXECUTION_LOG_20260728.md)。正式训练未启动。

## 1. 完整有效配置

- fresh：[`FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml`](./FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml)
  - SHA-256：`2c616159726dc9d39fbe4d011909dcba963c95c355539c606c718a6ed25da390`
- resume：[`RESUME_SMOKE_VALIDATED_RESOLVED_20260728.yaml`](./RESUME_SMOKE_VALIDATED_RESOLVED_20260728.yaml)
  - SHA-256：`c81e722d72dc06976f1011f42de1e9701c18d33857dfa1cc296af5216a491d46`
- formal 参考：[`FORMAL_VALIDATED_RESOLVED_20260728.yaml`](./FORMAL_VALIDATED_RESOLVED_20260728.yaml)
  - SHA-256：`f128166c80846ab3dddaa8e3b773b9c62db2cdb6aecaaffed452145863ef1422`

这些文件由当前服务器分支 Hydra compose 后生成，再执行除真实 `Cluster/placement` 资源发现以外的 `validate_cfg()`；生成过程中确认 Ray 未初始化。

fresh 相对 formal 只改：

1. `max_steps: 1`；
2. `save_interval/val_check_interval: 1/1`；
3. eval `rollout_epoch: 1`、固定环境 reset ids；
4. warm-up `500 → 4`；
5. smoke experiment/output 名。

resume 相对 fresh 严格只改：

1. `max_steps: 1 → 2`；
2. `resume_dir: null → global_step_1`。

两者均保持两卡、4 train env、`rollout_epoch=1`、H/N=50/20、14D state/action、32D latent repeat-H、三相机冻结 π0、单主相机小 actor/Q、batch/micro=256/64、UTD20、replay25k、10-Q、LR、FSDP、CPU patch sync 和 stochastic eval。

## 2. 固定输出目录

```bash
/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
```

目录当前不存在。fresh 和 resume 共用该根目录，但使用独立 driver/resource log；checkpoint 依次写到：

```bash
.../robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke/checkpoints/global_step_1
.../robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke/checkpoints/global_step_2
```

## 3. fresh 精确命令

执行前重新确认分支/远端一致、worktree clean、GPU/Ray 空闲、模型/venv/assets 存在和 run root 不存在。

```bash
set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
ROBOTWIN=/root/autodl-tmp/RoboTwin_RLinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
CFG=robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1

test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/resource_monitor/fresh"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="$REPO/examples/embodiment"
export REPO_PATH="$REPO"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$REPO:$ROBOTWIN"
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

FRESH_CMD=(
  "$PY" -B "$EMBODIED_PATH/train_embodied_agent.py"
  --config-path "$EMBODIED_PATH/config"
  --config-name "$CFG"
  "runner.logger.log_path=$RUN_ROOT"
  "runner.resume_dir=null"
)

printf '%q ' "${FRESH_CMD[@]}" > "$RUN_ROOT/fresh_command.txt"
printf '\n' >> "$RUN_ROOT/fresh_command.txt"

nohup "${FRESH_CMD[@]}" \
  > "$RUN_ROOT/fresh_driver.log" 2>&1 < /dev/null &
FRESH_PID=$!
printf '%s\n' "$FRESH_PID" > "$RUN_ROOT/fresh.pid"

nohup "$PY" -B "$REPO/examples/embodiment/monitor_resources.py" \
  --pid "$FRESH_PID" \
  --out-dir "$RUN_ROOT/resource_monitor/fresh" \
  --interval 2 \
  > "$RUN_ROOT/resource_monitor/fresh/monitor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/resource_monitor/fresh/monitor.pid"
```

fresh 只运行一个 collection cycle。4 个满长 episode 最多生成 40 条 macro transitions，随后最多执行 800 次 optimizer updates；提前成功时 transitions/updates 相应减少，但 warm-up=4 应确保进入 learned phase 和真实更新。

## 4. resume 前置检查与精确命令

只有 fresh 正常退出并通过 DCP1/finite/phase/replay 检查后才运行：

```bash
set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
ROBOTWIN=/root/autodl-tmp/RoboTwin_RLinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
CFG=robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
CKPT1="$RUN_ROOT/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke/checkpoints/global_step_1"

test -d "$CKPT1/actor"
test -f "$CKPT1/actor/sac_components/dsrl_trainer_state_rank_0.pt"
test -f "$CKPT1/actor/sac_components/dsrl_trainer_state_rank_1.pt"
test -f "$CKPT1/actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt"
test -f "$CKPT1/actor/sac_components/replay_buffer/rank_1/dsrl_transition_replay.pt"
mkdir -p "$RUN_ROOT/resource_monitor/resume"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="$REPO/examples/embodiment"
export REPO_PATH="$REPO"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$REPO:$ROBOTWIN"
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

RESUME_CMD=(
  "$PY" -B "$EMBODIED_PATH/train_embodied_agent.py"
  --config-path "$EMBODIED_PATH/config"
  --config-name "$CFG"
  "runner.logger.log_path=$RUN_ROOT"
  "runner.resume_dir=$CKPT1"
  "runner.max_steps=2"
)

printf '%q ' "${RESUME_CMD[@]}" > "$RUN_ROOT/resume_command.txt"
printf '\n' >> "$RUN_ROOT/resume_command.txt"

nohup "${RESUME_CMD[@]}" \
  > "$RUN_ROOT/resume_driver.log" 2>&1 < /dev/null &
RESUME_PID=$!
printf '%s\n' "$RESUME_PID" > "$RUN_ROOT/resume.pid"

nohup "$PY" -B "$REPO/examples/embodiment/monitor_resources.py" \
  --pid "$RESUME_PID" \
  --out-dir "$RUN_ROOT/resource_monitor/resume" \
  --interval 2 \
  > "$RUN_ROOT/resource_monitor/resume/monitor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/resource_monitor/resume/monitor.pid"
```

resume 从 global step 1 开始，只再运行一个 cycle 并保存 DCP2；不得出现 legacy-shadow compatibility warning，首个 rollout 必须保持 learned phase。

## 5. 预计资源与时间

- 硬件：两张 A800 80GB，actor/env/rollout 均 colocate 到 ranks 0–1。
- 历史 π0 两卡 smoke：GPU 峰约 29–40 GiB/卡，cgroup 峰约 112–125 GiB，单阶段约 27–29 分钟。
- DSRL 增加 10-Q 和最多 800 次更新，但只用 4 train env；首轮保守预计 GPU 小于 60 GiB/卡、非可回收工作集小于 180 GiB。
- fresh / resume 各预留 30–90 分钟；若真实 update throughput 更慢，单阶段最多观察 3 小时后停下诊断，不直接扩大预算。
- flat replay 全容量约 1.23 GB global；smoke 只存首轮少量 transitions，不是主要内存项。
- 当前约 236 GiB cgroup 值主要是 inactive file cache，不能单独作为内存泄漏或停止依据；启动前重新读取 `memory.stat/events`。

## 6. 立即停止条件

只停止本次 smoke 的 driver/Ray，不影响其他用户进程：

1. CUDA OOM、`memory.events` 的 `oom`/`oom_kill` 增加，或 driver/worker 异常退出；
2. loss、Q、alpha、gradient 或参数出现 NaN/Inf；
3. 冻结 π0 进入 optimizer/发生非零 delta，或 replay 存成 denoised env action 而不是 32D latent；
4. transition 的 success/reward/truncation/discount 与 N=20 合同不符；
5. fresh 没有进入 learned phase、计划 update 数不等于 `20 × global_new_transitions`，或没有生成完整 DCP1；
6. resume 出现 legacy fallback、phase/update_step/replay 回退、shadow/target mismatch，或没有生成 DCP2；
7. worker 就绪后连续 20 分钟没有 macro/update 进度；或单阶段超过 3 小时；
8. raw cgroup 使用率连续接近上限时，只有在 inactive file 已基本不可回收或伴随 `high/max/oom` 事件时才停止，避免把页缓存误判成泄漏。

## 7. smoke 通过的最低结论

- fresh：Gaussian collect → flat replay → learned actor/Q/temperature update → target EMA → actor sync → 4-episode eval → DCP1 全链通过。
- resume：精确恢复 shadow、phase、`update_step` 和 replay；再完成一轮 learned collect/update/sync/eval → DCP2。
- 记录实际 `global_new/resident/planned_optimizer_updates`、Q/actor/alpha/grad、env seconds、updates/s、GPU/RAM peak、OOM events、DCP 字段和固定环境 seed 结果。
- smoke 的固定是环境 seeds，policy latent 仍随机；不得把 4 个 eval episodes 当效果对照或 paired stochastic evaluation。
