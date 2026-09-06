# Sidney pi0.5 官方推理实施账本

## 授权

2026-09-02，用户明确授权停止当前 pi0.5 GRPO，并使用其释放的 GPU 做 Sidney pi0.5 官方推理；
同时授权停止当前 Fast-WAM256，在原 GPU6/7 做 pi0-style offload 资源 smoke。

## 计划锁定

- 官方 LeRobot 路径先于 RLinf 转换；
- `adjust_bottle` 与 `move_stapler_pad` 各 5 episodes、B=1；
- 不改 H50/M10、absolute qpos、processor/norm、camera contract；
- 环境、checkpoint、输出与 cache 均放在 chenyiteng 的独立路径；
- 不触碰其他用户进程或 shared Ray。

## 操作记录

### 21:42 CST：版本、网络与资源只读探针

- GPU4/5 各为 `0 MiB`，确认可用于本任务；GPU6/7 已由另一条已授权 Fast-WAM smoke 占用，本线不触碰。
- `/data` 剩余约 `1.6 TiB`，`/home` 剩余约 `1.4 TiB`。
- Hugging Face 直连 TLS reset；服务器本机 `https://hf-mirror.com` 返回 200。因此只使用服务器镜像，不经过本地 PC 代理。
- LeRobot tag 固定：
  - `v0.6.0 = 30da8e687a6dfc617fcd94afc367ac7071c376ce`；
  - `v0.4.3 = 0b067df57d21d3a02d6c511f1609172fa39ac29b`。
- Sidney checkpoint 固定为 `e49e2ab6c11f07511573b67261bd129e88d0a416`。

### 21:55 CST：上游 Python 合同冲突与窄路线

- LeRobot `v0.6.0` 的 RoboTwin 文档明确要求 Python3.10，但同 tag 的 `pyproject.toml` 要求 Python>=3.12；在真实 Python3.10 下，`processor/pipeline.py` 的 PEP695 泛型类语法产生 `SyntaxError`，因此不能强行安装。
- `v0.4.3` 原生支持 Python>=3.10，已有 PI05 policy 和 pre/postprocessor，但尚无 RoboTwin env wrapper。
- 选择的窄路线：在隔离 Python3.10 venv 中使用官方 `v0.4.3` PI05/processor；仅从固定 `v0.6.0` 回移官方 RoboTwin env 入口。必须先通过 checkpoint config、strict state load、pre/postprocessor、H50/M10、mean/std、absolute qpos 合同；不通过即停止，不叠加兼容层。

### 22:00 CST：模型与隔离环境开始落盘

- 完整 snapshot 在后台从服务器 `hf-mirror.com` 下载至：
  `/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab`。
- 下载使用固定 revision，而不是浮动 `main`；约五分钟时已到约 `1.1 GiB / 9.35 GiB`，八个小 metadata/processor 文件已完整落地。
- 隔离源码与环境目标：
  - source：`/data/chenyiteng/projects/lerobot-sidney/lerobot-0b067df57d21`；
  - venv：`/home/chenyiteng/venvs/lerobot-v043-sidney-py310`；
  - 不修改既有 Fast-WAM、RLinf 或 RoboTwin venv。

### 23:00--24:02 CST：锁定 official v0.6 主路径并完成首个有效回合

- LeRobot source 最终锁定 official `v0.6.0@30da8e687a6dfc617fcd94afc367ac7071c376ce`；
  checkpoint 固定为 `e49e2ab6c11f07511573b67261bd129e88d0a416`，完整 snapshot 9/9 files，
  `model.safetensors=9,354,050,752 bytes`。
- 因同一 tag 的 RoboTwin 文档要求 Python 3.10，而 source 使用少量 Python 3.12 typing 语法，隔离
  worktree 只做 13 files、`+29/-17` 的 typing/metadata 兼容；没有改 PI05 数学、processor 顺序、
  normalization、动作或环境语义。完整 `compileall src/lerobot` 与 import 通过。
- gated `google/paligemma-3b-pt-224` tokenizer 不能匿名下载；使用服务器已有的官方 OpenPI tokenizer
  资产构造本地 cache，并对两个实际任务 prompt + normalized state 比较 LeRobot 与 OpenPI token ids/
  masks，逐元素一致后才继续。
- checkpoint strict load 为 `813/813` keys，missing/unexpected 均为 0；resolved policy 为
  `H50/M10/BF16/absolute 14D/three cameras`。
- `adjust_bottle seed1000` 完成官方 B=1 推理，在 112 env steps 成功。随后同一个 5-episode CLI 在
  下一次 reset 遇到 `seed1001 UnStableError`；重新从 `seed1002` 启动也在策略执行前 reset 为
  `UnStableError`。两者是 RoboTwin 对象初态不稳定，不计作有效策略回合。

### 2026-09-03 09:48--10:18 CST：稳定种子探针与 adjust_bottle 5 个有效回合闭合

- 中断恢复后先只读确认：没有残留 Sidney eval/reset 进程，GPU4/5 均为 0 MiB；没有重复旧 run。
- 使用 exact `RoboTwinEnv(...).reset(seed=seed)` 做 reset-only 探针，不加载模型：
  - adjust stable：`1003,1004,1008,1009`；
  - adjust unstable：`1005,1006,1007`；此前 `1001,1002` 也为 unstable；
  - move_stapler_pad stable：`1000,1001,1002,1003,1004`。
- 四个新 adjust seed 均使用官方 `lerobot-eval`、每 seed 独立 `n_episodes=1/B=1`，分到 GPU4/5；
  三相机 rename、checkpoint processor/norm、absolute 14D、H50/M10 均不变：

| seed | outcome | env steps | eval wall | rc |
| --- | --- | ---: | ---: | ---: |
| 1000 | success | 112 | 约 12m51s | 0（随后 seed1001 reset 不稳定令多回合 wrapper rc1） |
| 1003 | success | 124 | 775.57s | 0 |
| 1004 | success | 121 | 763.94s | 0 |
| 1008 | success | 117 | 168.40s | 0 |
| 1009 | success | 116 | 166.27s | 0 |

- 结论：`adjust_bottle` 的 5 个有效回合为 `5/5`；这只是接通 oracle，不替代模型卡的 100 回合结果。
  前两个新 seed 包含 official `compile_mode=max-autotune` 的首次编译开销；后两个复用同 GPU cache，
  所以 wall 大幅下降，不能把该差异解释成策略快慢。

### 2026-09-03 10:18--11:00 CST：move_stapler_pad 官方推理闭合

- stable seeds `1000--1004`，仍每 seed 独立 B=1 official CLI；GPU4 跑 `1000,1002`，
  GPU5 跑 `1001,1003`，确认 GPU3 无其他用户进程后用 GPU3 跑 `1004`，仅缩短墙钟，不改变单回合协议。
- 10:23 CST 首批 seed1000/1001 分别推进到 env step151/101；GPU4/5 各约15.6 GiB、100% util，
  cache 命中、日志增长、无 OIDN/renderer/fatal。
- 每个 seed 的精确 official CLI 形状如下；仅 `CUDA_VISIBLE_DEVICES`、`--seed` 和 run-scoped
  `--output_dir` 随表中映射变化：

```bash
/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/lerobot-eval \
  --policy.path=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab \
  --env.type=robotwin --env.task=move_stapler_pad \
  --eval.batch_size=1 --eval.n_episodes=1 --seed=SEED \
  --rename_map='{"observation.images.head_camera":"observation.images.cam_high","observation.images.left_camera":"observation.images.cam_left_wrist","observation.images.right_camera":"observation.images.cam_right_wrist"}' \
  --output_dir=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/move_stapler_pad-official-valid-seedSEED-20260903/eval
```

| seed | GPU | outcome | env steps | eval wall | wrapper result |
| --- | ---: | --- | ---: | ---: | --- |
| 1000 | 4 | failure | 1200 | 786.38s | rc=0 |
| 1001 | 5 | failure | 1200 | 853.43s | rc=0 |
| 1002 | 4 | success | 1135 | 824.18s | official final metrics complete；外层顺序 wrapper 在其退出后被关闭，未再重复启动 seed1004 |
| 1003 | 5 | failure | 1200 | 739.95s | rc=0 |
| 1004 | 3 | success | 489 | 992.86s | rc=0 |

- `seed1004` 使用新的 GPU3 TorchInductor cache，`eval wall` 包含约 10 分钟的 official
  `max-autotune` 首次编译，不能与 cache-hit seed 的策略执行速度直接比较。
- `seed1002` 的 Python 进程已输出完整 `pc_success=100%`、`successes=[True]` 和视频路径；为了避免
  GPU3 已运行的 seed1004 被 GPU4 原顺序 wrapper 重复执行，只暂停并在 child 退出后关闭该 owned
  外层 wrapper。其 run 中保留 `manual_close_note.txt`，没有伪造未捕获的 wrapper exit code。
- 结论：`move_stapler_pad` 的 5 个有效回合为 `2/5`。加上前述 `adjust_bottle=5/5`，两任务
  official-native 小样本 oracle 已成立；这与模型卡的 100 回合统计不是同一统计精度。
- 11:00 CST 最终只读确认：GPU3/4/5 均为 0 MiB、0% util，没有残留 owned Sidney eval；
  全程未出现非 `UnStableError` 的运行错误，也未触碰 shared Ray 或其他用户任务。

使用的本地可审计入口：

- reset-only：`local_scripts/remote_commands/sz_sidney_adjust_stable_seed_probe_20260903.sh`、
  `local_scripts/remote_commands/sz_sidney_move_stable_seed_probe_20260903.sh`；
- adjust official：`sz_sidney_adjust_official_valid_gpu4_20260903.sh`、
  `sz_sidney_adjust_official_valid_gpu5_20260903.sh`；
- move official：`sz_sidney_move_official_valid_gpu4_20260903.sh`、
  `sz_sidney_move_official_valid_gpu5_20260903.sh`、
  `sz_sidney_move_official_valid_gpu3_seed1004_20260903.sh`。

### 2026-09-03 11:08 CST：源码与轻量产物只读快照

- LeRobot compat worktree exact HEAD为official
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`，detached `v0.6.0-dirty`；
- 仅13个未暂存Python3.10 typing/metadata兼容文件，diff stat=`+29/-17`，无staged/untracked；
  remote只有official `https://github.com/huggingface/lerobot.git`，本轮没有commit或push；
- Sidney run根共12个顶层run、69 files、约5.02 MiB；其中非视频轻量证据约1.89 MiB，10个视频约
  3.13 MiB。没有把模型权重或环境cache复制进本地文档树。

## 本轮停止边界

按用户要求，本轮在 official-native 推理闭合后停止：没有实现、转换或 smoke current RLinf adapter。
下一步先讨论权重/processor 转换、动作与 observation parity 以及 current Flow-SDE 接入边界。
