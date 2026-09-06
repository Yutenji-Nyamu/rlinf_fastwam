# DSRL current RLinf × RoboTwin × exact π0：GPU6–7 smoke packet 草案

状态：**只读草案，未写服务器、未启动。** 现场刷新时间 `2026-08-23T13:36:21Z`。

## 1. Source lock 与现场前提

- worktree：`/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin`
- branch：`codex/sz-current-dsrl-pi0-robotwin`
- HEAD：`4b609178d10d2534f3f972435ad972e4e015c392`
- upstream：ahead/behind `0/0`，worktree clean。
- config：`examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi.yaml`
- config SHA-256：`1198f990d556650c3670b47de4a931164d374f13d4425ebe3aa8bb6737d3c6a5`
- entry：`examples/embodiment/train_embodied_agent.py`
- entry SHA-256：`02260b8b9b849331e99e9b6ef0045c68478477539761f767283017ee02ad4540`
- Python：`/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python`
- exact π0：`/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`
- RoboTwin：`/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support`
- 候选 run root 当前不存在：
  `/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823`

现场资源：8 卡均 `0 MiB / 0%` 且无 compute app；无 chenyiteng Ray/训练相关进程；GPU6–7 均空闲。主机 `2.0 TiB` RAM，约 `1.9 TiB available`，memory PSI 全为 0；`/data` 约 `3.0 TiB` 可用。

## 2. 锁定配置与预算

两阶段共同不变：

| 项 | resolved 值 |
|---|---:|
| physical GPU / actor world size | `6,7` / `2` |
| train env / rollout epoch | `4 / 1` |
| primitive horizon | 每 env 最多 `200` |
| exact π0 action horizon `H` | `50` |
| 每次 query 实际执行 `N` | `20` |
| global/micro batch | `256 / 64` |
| gradient accumulation | 每 rank `128/64=2` microbatches/update |
| replay | compact transition ring，global capacity `25,000`，每 rank `12,500` |
| smoke warm-up | global `4` macro transitions |
| UTD | `20 × global_new_transitions` |
| Q / latent | `10-Q` / `32D` |
| RTC | 关闭（与旧成功 DSRL 语义一致） |
| eval | 关闭：`val_check_interval=-1`，本 smoke 不附加效果评估 |

每个满长 env 最多给出 `200/20=10` 条 macro transition；因此每个 outer step 最多 `4×10=40` 条，计划更新最多 `20×40=800` 个 SAC update epoch。`critic_actor_ratio=1`，所以每个 update epoch 各做一次 critic、actor、temperature optimizer step，并做 target EMA；每次各含两个 microbatch。提前成功时 new transition 和更新数按实际减少。warm-up=4 的目的仅是让 fresh step 1 真实跨过 Gaussian→learned 边界并执行更新。

fresh：从 phase 0/Gaussian 采集一轮，更新并保存 `global_step_1`。  
resume：新 Python/Ray 进程从 DCP1 严格恢复 phase、`update_step`、critic FP32 shadow、target、alpha、compact ring/RNG，再执行一个 outer step并保存 `global_step_2`。

## 3. 输出路径

```bash
ROOT=/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823
FRESH_LOG=$ROOT/fresh
FRESH_EXP=dsrl-fresh-step1
CKPT1=$FRESH_LOG/$FRESH_EXP/checkpoints/global_step_1
RESUME_LOG=$ROOT/resume
RESUME_EXP=dsrl-resume-step2
CKPT2=$RESUME_LOG/$RESUME_EXP/checkpoints/global_step_2
```

分开 fresh/resume 的 `log_path`，避免 TensorBoard event/config 混在同一目录。预计每个 DCP 约 `32–35 GiB`；两阶段连日志约需 `65–75 GiB`，留在 `/data`。

## 4. 共同环境与 Hydra overrides

```bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
ROOT=/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_dsrl_openpi
  'cluster.component_placement={actor\,\ env\,\ rollout:6-7}'
  'runner.max_epochs=1000'
  'runner.val_check_interval=-1'
  'runner.save_interval=1'
  'runner.ckpt_path=null'
  'algorithm.utd_ratio=20'
  'algorithm.critic_actor_ratio=1'
  'algorithm.replay_buffer.capacity=25000'
  'algorithm.replay_buffer.warmup_size=4'
  'env.train.total_num_envs=4'
  'env.train.rollout_epoch=1'
  'env.train.max_episode_steps=200'
  'env.train.max_steps_per_rollout_epoch=200'
  "env.train.assets_path=$ROBOTWIN"
  "env.eval.assets_path=$ROBOTWIN"
  'actor.micro_batch_size=64'
  'actor.global_batch_size=256'
  "actor.model.model_path=$MODEL"
  'actor.model.num_action_chunks=20'
  'actor.model.openpi.action_horizon=50'
  'actor.model.openpi.rtc_enabled=false'
  'actor.model.openpi.use_dsrl=true'
  'actor.model.openpi.dsrl_gaussian_warmup=true'
  'actor.model.openpi.dsrl_eval_deterministic=false'
  'actor.fsdp_config.save_full_model_weights=false'
)
```

这里不设置 `CUDA_VISIBLE_DEVICES=6,7`，而沿深圳 current PPO/GRPO 已验证的方式让 Ray 看见整机，再用 placement 精确锁 physical GPU6–7，避免 logical/physical 编号混淆。

## 5. Fresh 精确命令（批准后才执行）

```bash
test "$(git -C "$WT" rev-parse HEAD)" = 4b609178d10d2534f3f972435ad972e4e015c392
test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$ROOT"
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test -d "$ROBOTWIN"
! pgrep -u "$(id -u)" -x raylet
! pgrep -u "$(id -u)" -x gcs_server
! nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'

FRESH_LOG=$ROOT/fresh
FRESH_EXP=dsrl-fresh-step1
mkdir -p "$FRESH_LOG"
FRESH_ARGS=(
  "${COMMON_ARGS[@]}"
  "runner.logger.log_path=$FRESH_LOG"
  "runner.logger.experiment_name=$FRESH_EXP"
  'runner.max_steps=1'
  'runner.resume_dir=null'
)

"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${FRESH_ARGS[@]}" --cfg job --resolve > "$FRESH_LOG/resolved.yaml"
sha256sum "$FRESH_LOG/resolved.yaml" > "$FRESH_LOG/resolved.yaml.sha256"
printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${FRESH_ARGS[@]}" > "$FRESH_LOG/command.txt"
printf '\n' >> "$FRESH_LOG/command.txt"

nohup setsid bash -c '
  log=$1; shift
  date --iso-8601=seconds > "$log/started_at.txt"
  "$@" > "$log/driver.log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$log/exit_code.txt"
  date --iso-8601=seconds > "$log/finished_at.txt"
  exit "$rc"
' _ "$FRESH_LOG" "$VENV/bin/python" \
  "$WT/examples/embodiment/train_embodied_agent.py" "${FRESH_ARGS[@]}" \
  < /dev/null > "$FRESH_LOG/wrapper.log" 2>&1 &
FRESH_PID=$!
printf '%s\n' "$FRESH_PID" > "$FRESH_LOG/driver.pid"
```

资源 observer 与 driver 同时启动，10 秒采样一次；精确脚本如下，不采集其他用户命令行或环境：

```bash
cat > "$FRESH_LOG/resource_observer.sh" <<'OBSERVER'
#!/usr/bin/env bash
set -u
pid=$1
out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,cgroup_memory_current_bytes,cgroup_oom,cgroup_oom_kill,mem_psi_some_avg10,mem_psi_full_avg10,gpu6_used_mib,gpu6_util_pct,gpu7_used_mib,gpu7_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  ts=$(date --iso-8601=seconds)
  mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$pid/cgroup" 2>/dev/null)
  cg=/sys/fs/cgroup$rel
  current=$(cat "$cg/memory.current" 2>/dev/null || printf '')
  oom=$(awk '$1=="oom" {print $2}' "$cg/memory.events" 2>/dev/null || printf '')
  oom_kill=$(awk '$1=="oom_kill" {print $2}' "$cg/memory.events" 2>/dev/null || printf '')
  psi_some=$(awk '$1=="some" {sub("avg10=", "", $2); print $2}' /proc/pressure/memory)
  psi_full=$(awk '$1=="full" {sub("avg10=", "", $2); print $2}' /proc/pressure/memory)
  mapfile -t gpu < <(nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,1,%s,%s,%s,%s,%s,%s' "$ts" "$mem" "$current" "$oom" "$oom_kill" "$psi_some" "$psi_full" >> "$out"
  for row in "${gpu[@]}"; do printf ',%s' "$row" >> "$out"; done
  printf '\n' >> "$out"
  sleep 10
done
printf '%s,0,%s,,,,,,,,' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
printf '\n' >> "$out"
OBSERVER
chmod 700 "$FRESH_LOG/resource_observer.sh"
nohup setsid bash "$FRESH_LOG/resource_observer.sh" "$FRESH_PID" "$FRESH_LOG/resource.csv" \
  > "$FRESH_LOG/resource_observer.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$FRESH_LOG/resource_observer.pid"
```

## 6. Fresh→resume 唯一中间验收

fresh driver 必须自然 `exit 0`，GPU/Ray 释放后才启动新进程。最小验收：

```bash
CKPT1=$ROOT/fresh/dsrl-fresh-step1/checkpoints/global_step_1
test -d "$CKPT1/actor"
for rank in 0 1; do
  test -f "$CKPT1/actor/sac_components/dsrl_trainer_state_rank_${rank}.pt"
  test -f "$CKPT1/actor/sac_components/target_model/checkpoint_rank_${rank}.pt"
  test -f "$CKPT1/actor/sac_components/replay_buffer/rank_${rank}/dsrl_transition_replay.pt"
done
test -d "$CKPT1/actor/sac_components/alpha"
```

同时从日志/TensorBoard确认：`global_new_transitions>=4`、`global_resident_transitions>=4`、`planned_optimizer_updates=20×global_new_transitions>0`；loss/Q/alpha/grad finite。sidecar 的两个 rank 必须均为 world size 2、phase 1、相同且正的 `update_step`。不做旧包那种整棵 32 GiB checkpoint SHA 扫描。

## 7. Fresh-process resume 精确命令（DCP1通过后才执行）

```bash
! pgrep -u "$(id -u)" -x raylet
! pgrep -u "$(id -u)" -x gcs_server
! nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'

RESUME_LOG=$ROOT/resume
RESUME_EXP=dsrl-resume-step2
mkdir -p "$RESUME_LOG"
RESUME_ARGS=(
  "${COMMON_ARGS[@]}"
  "runner.logger.log_path=$RESUME_LOG"
  "runner.logger.experiment_name=$RESUME_EXP"
  'runner.max_steps=2'
  "runner.resume_dir=$CKPT1"
)

"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${RESUME_ARGS[@]}" --cfg job --resolve > "$RESUME_LOG/resolved.yaml"
sha256sum "$RESUME_LOG/resolved.yaml" > "$RESUME_LOG/resolved.yaml.sha256"
printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${RESUME_ARGS[@]}" > "$RESUME_LOG/command.txt"
printf '\n' >> "$RESUME_LOG/command.txt"

nohup setsid bash -c '
  log=$1; shift
  date --iso-8601=seconds > "$log/started_at.txt"
  "$@" > "$log/driver.log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$log/exit_code.txt"
  date --iso-8601=seconds > "$log/finished_at.txt"
  exit "$rc"
' _ "$RESUME_LOG" "$VENV/bin/python" \
  "$WT/examples/embodiment/train_embodied_agent.py" "${RESUME_ARGS[@]}" \
  < /dev/null > "$RESUME_LOG/wrapper.log" 2>&1 &
RESUME_PID=$!
printf '%s\n' "$RESUME_PID" > "$RESUME_LOG/driver.pid"
```

resume 将上面的 observer 原样写入 `$RESUME_LOG/resource_observer.sh`，以 `$RESUME_PID` 启动并写 `$RESUME_LOG/resource.csv`。由于 runner 从路径后缀读到 `global_step_1`、而 `max_steps=2`，只会执行第二个 outer step并保存：

```text
/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/dsrl-resume-step2/checkpoints/global_step_2
```

## 8. 预计资源、时间与监控

旧 AutoDL 完全同预算（2×A800、4 env、H50/N20、GB256/MB64、warm4、UTD20）实测：fresh `40 transitions→800 updates`，resume `37→740`；每阶段约 12–13 分钟（含冷启动/eval/checkpoint），训练常态约 28–35 GiB/卡，smoke 峰 34.6 GiB/卡，formal DCP 峰 41.5 GiB/卡；anon 约 41–66 GiB，额外 raw RAM 多为 checkpoint file cache。

本 packet 关闭 eval，H100 资源首验预期：

- fresh/resume 各约 `10–15 min`，冷 checkpoint load/save 波动可更长；
- 常态约 `30–36 GiB/卡`，DCP 时约 `40–45 GiB/卡`；保守预算 `<50 GiB/卡`；
- 主机 active/anon 约 `40–80 GiB`，file cache 可能额外增加，但现场约 `1.9 TiB available`；
- 每个 DCP 约 `32–35 GiB`。

关键训练指标：

- `train/sac/global_new_transitions`
- `train/sac/global_resident_transitions`
- `train/sac/planned_optimizer_updates`
- `train/sac/critic_loss`, `train/sac/actor_loss`, `train/sac/alpha_loss`, `train/sac/alpha`
- `train/critic/grad_norm`, `train/actor/grad_norm`, `train/alpha/grad_norm`, `train/actor/entropy`
- `train/actor/q_value_0..9`, `train/actor/q_pi`
- `env/success_once`, `env/return`
- `time/generate_rollouts`, `time/actor/run_training`, `time/sync_weights`, `time/step`

## 9. 停止条件与通过条件

只处理本次 smoke 自己的 driver/Ray；不碰其他用户进程。

立即停并保留现场：CUDA OOM、Ray worker/driver crash、NCCL/仿真 fatal、NaN/Inf；GPU6/7 出现非本任务 compute app；主机出现 OOM/kill 或持续明显 PSI/full pressure；fresh 没跨到 phase1或 `planned != 20×new`；strict resume 报 sidecar/world-size/phase/shadow/target/replay mismatch；worker 已就绪后 20 分钟没有 macro/update 进度。

通过最低条件：

1. fresh 完成 Gaussian collect → compact ring → learned actor/Q/temperature update → target EMA/sync → DCP1；
2. 新进程严格加载 DCP1，phase 和 `update_step` 不回退、replay resident 不清零；
3. resume 再完成一次 collect/update并自然生成完整 DCP2；
4. 两阶段指标 finite、退出码 0，最后 GPU6/7 与 owned Ray 释放。
