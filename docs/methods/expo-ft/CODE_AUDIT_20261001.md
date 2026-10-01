# EXPO-FT 官方源码与运行资产审计 · 2026-10-01

本轮仅研究和轻量源码审计；未操作服务器、安装项目环境、下载模型/数据、运行训练或项目测试。

**结论：算法和 DROID+π0.5 在线训练闭环已公开，能够作为移植依据；公开材料尚不足以一键复现论文所有真实机器人实验。** 不是只有 inference demo；也不是现成的 RLinf×RoboTwin 后端。当前同一仓库含 EXPO-FT 和 Real-Time EXPO-FT，须按原方法选 `EXPOLearner`、`scripts/pick/`、OpenPI `expo_ft` 分支，不能把 `dynamic_pick`/RTC 参数倒灌成原 EXPO-FT。

## 1. 来源固定

官方[项目官网](https://pd-perry.github.io/expo-ft/)的 Code 指向 [`pd-perry/expo-ft`](https://github.com/pd-perry/expo-ft)。审计快照：

| 项目 | 分支 | 固定 commit | commit 时间 |
|---|---|---|---|
| EXPO-FT 主仓库 | main | `023cf9cfcb09dab962b6e806fea47d2954b2b9bb` | 2026-09-30 10:04:07 −07:00 |
| 作者 OpenPI fork，原 EXPO-FT | expo_ft | `46407a41183b037313a383ff679683f2773b766d` | 2026-05-28 17:17:36 −07:00 |
| 作者 DROID fork，README 原 EXPO-FT 所选默认分支 | main | `04b7551ff8a4901b9914cc4dbca4d066771ae9a7` | 2026-09-10 14:12:59 −07:00 |

源码保存在 `local_scripts/expo_ft_20261001/official-repo`、`official-openpi-expo-ft`、`official-droid`。三份均 `git status --short` 空；不将嵌套 Git checkout 纳入我们的发布。

主仓库许可证 [MIT](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/LICENSE)；OpenPI fork 为 Apache-2.0。所取 DROID fork 根目录未见 LICENSE，不能将所有第三方依赖/模型/数据统称 MIT。当前[GitHub Releases](https://github.com/pd-perry/expo-ft/releases)页面无 release 包；因此 pin commit 比依赖 release tag 可操作。

## 2. 已公开哪些东西

| 内容 | 实际源码/配置证据 | 判断 |
|---|---|---|
| 原 EXPO-FT learner | [`expo_ft/agents/alg/expo_ft.py`](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L175) | critic、edit actor、base actor 更新、目标网络和恢复保存齐全 |
| 多候选采样与 Q 选优 | [L521–599](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L521) | π0.5 采 N 个 chunk，加 edit 候选，目标 Q 打分 argmax |
| edit actor SAC 损失 | [L694–730](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L694) | 修正归一化 action chunk，温度/熵项和 Q 梯度 |
| base VLA 在线训练 | [L734–755](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L734) | flow-matching BC 更新，target actor 参数 Polyak |
| online rollout→replay→update | [`train_pi_robo.py` L378–425](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/train_pi_robo.py#L378) | 收执行 action、reward/mask/done/HIL；标成功 episode；按触发模式更新 |
| 成功 episode base batch | [`batch_processor.py` L81–85、129–146](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/data/batch_processor.py#L81) | 当前默认 `actor_success_only=True`；成功 BC 含成功 demos/online |
| 人类介入 | [`client/run_client.py` L84–90、184–190](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/client/run_client.py#L84) | Spacemouse override，真实执行动作返回 learner；假设 7D |
| 同步/异步训练和 eval | 根 `train_pi_robo.py`、`train_pi_robo_async.py`、`eval_droid_policy.py` | 不是只开推理；async sampler/updater 分 GPU |
| 数据收集→LeRobot→norm→SFT→RL→eval | `scripts/pick/` 8 个脚本 | 原 EXPO-FT 可用的官方流程骨架 |
| RTC-SFT / Real-Time learner | `train_offline_rtc.py`、`agents/alg/realtime_expo_ft.py`、`scripts/dynamic_pick/` | 另一个方法的公开实现，原版迁移不必默认引入 |
| 任务代码 | `configs/task/{pick,light2,dynamic_pick}.py` 与 DROID pick/light 环境 | 部分真实任务示例，未覆盖论文全部八个任务的逐任务配置/检测器 |

### 代码事实，避免与论文口径混淆

- 原 EXPO-FT 示例 `N=8`、`n_edit_samples=8`、edit scale 0.2；Q ensemble 10、每次子采样 2；[`expo_ft_pi_config.py`](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/configs/model/expo_ft_pi_config.py#L9)。当前 `sample_actions` 的 edit 候选依附前 `n_edit_samples` 个 base；默认 N=M=8 保持一致，不应未经确认随意让 M>N。
- 原 `pick.py` 示例另开 `edit_action_xyzg=True`，在7D Cartesian动作中把rotation坐标3/4/5的edit置零，仅改xyz与gripper；actor仍输出完整chunk维，之后应用mask。见[任务配置](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/configs/task/pick.py#L33)、[mask实现](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L468)。RoboTwin双臂14D关节动作不能复用这些Cartesian下标。
- π0.5 padded action_dim=32、实际 DROID output action_dim=7、**模型 H=16**：[作者 OpenPI 配置 L875–904](https://github.com/pd-perry/openpi/blob/46407a41183b037313a383ff679683f2773b766d/src/openpi/training/config.py#L875)。**执行 C 默认 8**：[`train_pi_robo.py` L66](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/train_pi_robo.py#L66)。H 与 C 不同。
- 当前默认 B=64、UTD=20；每一次 `agent.update` 内有 20 个 critic minibatch 更新，然后 1 次 base 更新和 1 次 edit/temperature 更新：[L888–938](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L888)。这不等于每个环境步训练 20 次。
- 当前 `scripts/pick/run_server.sh` 指定每 episode **3 次 update call**；training loop 要求至少完成 10 episode 后才更新：[触发 L391–416](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/train_pi_robo.py#L391)。论文的训练调度与本次 main 示例应分别标明。
- bootstrap 用 `γ**C`，且 next action 也做候选筛选：[critic L775–804](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L775)。不能只在 rollout 加 best-of-N 而保持原 DSRL/RLT backup 不变后称完整 EXPO-FT。
- 示例训练 base 为 task-specific LoRA；独立 critic encoder 默认可训练。当前配置 `freeze_pi05_encoder=True`、`freeze_critic_encoder=False`，**前者不能单独证明图像编码器参数冻结**：它在 [`pi05.py` L618–631](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/vla/pi05.py#L618)控制多候选共享编码采样；配置构建没有据此增加 image 参数 freeze filter。base 更新仍按 TrainConfig.trainable_filter（[L171–176](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/vla/pi05.py#L171)）；所选作者 OpenPI [get_freeze_filter L88–117](https://github.com/pd-perry/openpi/blob/46407a41183b037313a383ff679683f2773b766d/src/openpi/models/pi0_config.py#L88)只过滤 llm 非 LoRA 参数，未匹配 img/SigLIP。静态预期 image 参数仍在 trainable set；本轮未跑梯度/权重变化来实证。论文冻结 encoder 的协议需与发布源码差异区分，移植时明确 parameter groups，不照旗标名称推断。

## 3. 缺什么：完整性的具体边界

1. **论文任务资产包未定位**：当前主仓库 `.gitignore` 排除了 data、assets、checkpoints；主树没有这些资产，也没有明确下载论文 task-specific SFT/最终 RL checkpoint、demo、norm 的脚本或仓库链接。基础 π0.5 指向 OpenPI 的公共 `gs://openpi-assets/checkpoints/pi05_base/params`；这不是 EXPO-FT 的任务训练 checkpoint。不能把 `expo_ft/droid_pick_cube_10` 这个本地转换用 dataset id 认成已公开 HF 数据集。
2. **逐任务复现配方有限**：pick/light2/dynamic_pick 三个任务 config、pick 和 light 两个 DROID 环境；未找到论文全部八任务对应 configs、检测器、随机初态与最终 checkpoints。一套方法源码完整，逐任务实验资产未齐。
3. **真实硬件是前置条件**：DROID/Polymetis 机械臂与 NUC、robot IP/类型/序列号、相机/校准、Spacemouse；ZED 使用还需系统 SDK。不是 GPU 服务器单独启动就能得到 rollout。[DROID 当前 fork 参数](https://github.com/pd-perry/droid/blob/04b7551ff8a4901b9914cc4dbca4d066771ae9a7/droid/misc/parameters.py#L4)为待填空值。
4. **没有现成 RoboTwin/仿真适配**：`client/envs` 仅 droid_env、工具和录制器；任务 `env_type=droid`；obs 是两外相机字段复用同一 side 图+一 wrist 图，Cartesian state 和 7D velocity action。MuJoCo/dm-control 出现在 DROID 的依赖中，不证明 EXPO-FT 已提供 MuJoCo/RoboTwin 训练环境。通用 client 结构可以承接新环境，但需实现。
5. **没有完整 π0/FastWAM 示例**：运行入口直接 `build_pi05`，公开唯一 VLA wrapper 为 `pi05.py`。OpenPI fork 里的 upstream π0/ALOHA/LIBERO configs 不等于 EXPO-FT 已接好相应 learner/环境。
6. **源码审计不等于运行验收**：本次没执行 `uv sync`，更没做真实机器人训练；不能将静态闭环存在描述为当前 pin 已实机验证。本机不安装大依赖，后续授权部署才在独立服务器目录验收。
7. **参数冻结待对齐**：上述当前代码的 `freeze_pi05_encoder` 是采样优化开关，静态未落实论文所称图像 encoder 参数冻结。核心算法已公开，不代表发布代码与论文每个训练细节完全一致。

## 4. 原 EXPO-FT 官方运行路径

以下是 Linux 官方 DROID 场景流程，经 pin 后可执行；实际运行前须配置硬件及对齐 data/assets/checkpoint。这里未执行。选 `pick`，不混 `dynamic_pick`：

```bash
git clone https://github.com/pd-perry/expo-ft.git
cd expo-ft
git checkout 023cf9cfcb09dab962b6e806fea47d2954b2b9bb
git clone -b expo_ft https://github.com/pd-perry/openpi.git expo_ft/agents/vla/openpi
git -C expo_ft/agents/vla/openpi checkout 46407a41183b037313a383ff679683f2773b766d
git clone https://github.com/pd-perry/droid.git client/droid
git -C client/droid checkout 04b7551ff8a4901b9914cc4dbca4d066771ae9a7
uv sync
cd client
uv sync
cd ..
# 使用 ZED 相机时，先按官方 SDK 要求安装，再执行：
bash client/install_pyzed.sh
```

NUC 上需独立启动 DROID server：`python scripts/server/run_server.py`。客户端所在机器先配好 cameras/robot task config，然后以下按顺序：

```bash
# robot/client
bash scripts/pick/collect_data.sh
# learner/GPU；不同机器需先将采集 data 同步到 learner
bash scripts/pick/convert_data.sh
bash scripts/pick/calculate_norm.sh
bash scripts/pick/finetune_droid.sh
```

脚本例子采 15 demonstrations、转换成功目录最多 10 个；SFT 4001 steps、save 2000。RL 脚本默认取 **2000/params**，不是自动最新最终 SFT；新实验需显式审 checkpoint/norm 一致。

```bash
# robot/client（独立终端）；端口与 learner 同为 8102
SERVER_HOST=<learner-host> bash scripts/pick/run_policy.sh
# learner/GPU（独立终端）；选一个训练入口
bash scripts/pick/run_server.sh
# 或 async，官方要求至少 2 GPU，不能与同步同时启动同一个任务
bash scripts/pick/run_server_async.sh
# 训练后，同 client/DROID 服务支持 eval
bash scripts/pick/eval_policy.sh
```

官方示例指定可见 GPU `0,1,2,3`，async `0,1,2`；它们是脚本的资源例子，不是部署到我们共享机器的授权。没有 RoboTwin 的即用官方启动命令；移植后才可列我们的完整 launch/control 差异。

参数需要统一：`dataset_path`、offline demo 数、SFT `weight_loader_path`、OpenPI config、assets dir/id、task config、N/M/edit scale/C、client port、checkpoint_dir/run_name。仓库 learner 用 JAX/Flax（锁 JAX 0.5.3、Flax 0.10.2），现有 RLinf PyTorch DSRL/RLT 不能无改动 import 这套 learner。

## 5. 当前公开问题：迁移前必审

[Issue #1](https://github.com/pd-perry/expo-ft/issues/1)（2026-08-03，页面仍 Open，未见作者回复）指出 n-step replay 的 `valids` 会使较早遇到 terminal 的 chunk loss 为 0，虽然 chunk return 含成功奖励。

当前 pin **并非完全没有应对代码**：[`replay_buffer.py` L515–531](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/data/replay_buffer.py#L515)写有 full executed chunk/backfill tail 理由，并提供 `valids_keep_terminal_windows` 分支；[`td_config.py` L14](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/configs/model/td_config.py#L14)默认 False。因此不应简单宣布“已确认官方 bug”或“已修”。RoboTwin 移植需独立明确：早终止 chunk 实际动作数、补齐动作表示、reward 累计、γ^实际执行步数/固定C、terminated/truncated、bootstrap 与 loss-valid mask 各自语义。针对这条边界写小型逻辑检查最有价值；不能直接复制旧 replay mask。

## 6. 对我们的实施含义

建议用官方 pin 作为算法参考，在现有 RLinf×RoboTwin×π0.5/DSRL 的 actor–critic、回放、恢复框架中实施 **原 EXPO-FT**；无需先重建 DROID 两环境，也无需默认 RTC/30Hz/prefix conditioning。

实现仍需完整组合：base N 候选；action-space residual actor+entropy；原+edit 联合 Q argmax；相同 candidate policy 的 TD backup；成功 episode BC 更新 base；target networks；chunk replay/终止语义；demo 和介入信息。本轮只有研究，尚未实现或启动。若不加入 human-in-loop，应明示 RoboTwin 的 autonomous EXPO-FT 变体，并保持 demo 数和互动预算可比，不能称原论文真机协议完全复现。
