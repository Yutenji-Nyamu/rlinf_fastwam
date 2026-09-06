# Sidney pi0.5 到 current RLinf：实现账本

更新时间：2026-09-03 CST

## 授权与边界

- 用户已授权实现、推送、必要检查与 GPU smoke。
- 本实现批次完成离线严格转换、Sidney 专用 dataconfig、工程接通 YAML、focused test、B=1 环境闭环与一轮真实两卡 GRPO smoke。
- 不修改 worker、schema、Builder、GRPO advantage/loss 或 checkpoint manager。
- GPU smoke 使用物理 GPU4/5；未触碰 GPU0 或 GPU6/7 上的既有任务。

## source lock

- current SZ pi0.5 base：`codex/sz-pi05-robotwin-rl@256eeeb4459b4bd5db85bfc6a0eb315771e8c38c`，服务器 worktree clean，`personal` remote 同 HEAD。
- official LeRobot：`v0.6.0@30da8e687a6dfc617fcd94afc367ac7071c376ce`。
- Sidney checkpoint：`SidneyXie/pi05_robotwin@e49e2ab6c11f07511573b67261bd129e88d0a416`；原生 strict load 已验证 813/813、missing/unexpected 均为 0。

## 2026-09-03 初始只读核对

- base worktree：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl`；branch clean。
- current 已有 OpenPI-PyTorch 转换器、`pi05_aloha_robotwin` dataconfig 和 RoboTwin pi0.5 GRPO 配置。
- Sidney snapshot 九个文件齐全；`model.safetensors=9,354,050,752 bytes`，pre/postprocessor JSON 与两份 norm safetensors 均在原目录。
- 当前 GPU6/7 为既有 Fast-WAM formal；GPU0 为其他用户模型服务。本实现不触碰任何运行任务。

## 实施记录

### 独立 worktree

- 从精确 base `256eeeb4...` 创建服务器分支 `codex/sz-sidney-pi05-current-rlinf`。
- worktree：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf`。
- 创建后 HEAD 与 base 一致、tracked/untracked 均为空；未修改既有 pi0.5 worktree 或运行任务。

### 真实 key / processor 合同

- Sidney `model.safetensors` 共 813 keys，全部只比 current RLinf target 多一层 `model.`；去掉该层后 target 也是 813 keys，名称与 shape 全部相等。
- 因此没有复用旧 OpenPI 重排数学，只新增 opt-in importer：严格要求唯一 `model.` wrapper，保留 tensor 数值及原生 BF16/FP32 dtype。
- Sidney processor 明确是 `STATE/ACTION=MEAN_STD`、三相机、14D absolute qpos；输出 OpenPI norm asset 为 `physical-intelligence/robotwin/norm_stats.json`。

### 代码改动

- 新增 `lerobot_pi05_to_openpi_rlinf.py`，负责 source config 验证、813-key/shape target accounting、单层 prefix 去除、mean/std norm 导出和 manifest。
- 在统一 `convert.py` 注册独立 mode；既有 converter 路径不变。
- `LeRobotAlohaDataConfig` 新增可选 `use_quantile_norm` override；默认 `None` 保持所有既有配置行为不变。
- 注册 `pi05_sidney_robotwin`：H50、absolute14D、`adapt_to_pi=false`、`extra_delta_transform=false`、`use_quantile_norm=false`。
- 新增 move-stapler 工程配置：C50/M10、G8、两卡 `64 env × rollout4`、GB1024/update2；外层 horizon 与 train/eval `task_config.step_lim` 均显式为 400；current RL 的 `train_expert_only=true` 保持不变。
- 新增 3 个 focused tests：single-prefix、拒绝未包装 key、14D mean/std 导出。
- 新增单文件 parity 工具 `toolkits/lerobot/sidney_pi05_parity.py`：以一个固定 NPZ 为边界，分别在
  native LeRobot 与 current RLinf runtime 导出三相机/normalized state/token、H50x32 model action
  和 H50x14 absolute qpos，再离线比较；两边模型路径、Sidney revision 与 converted manifest digest
  一并写入报告，避免把同一 backend 自比较成假阳性。

### 问题与窄修

- 首版 YAML 试图继承另一份带 `hydra.searchpath` 的 primary config，Hydra 明确拒绝；改为与官方配置相同的 component defaults，compose 通过，没有新增旁路。
- 首次真实转换把 freshly initialized FP32 target 当成 checkpoint dtype oracle，发现 436 个 BF16/FP32差异后在写输出前停止。target 仅用于结构 oracle；窄修为严格检查 key/shape，同时保留并记录 Sidney 原生 mixed dtype。未 cast 权重。

### 验证结果

- `pytest tests/unit_tests/test_lerobot_pi05_importer.py`：3/3 passed。
- focused `ruff check`：passed；`git diff --check`：passed。
- Hydra compose：`H50/M10, env64×rollout4, G8, GB1024/update2, move_stapler_pad/400` 全部断言通过。
- dataconfig registry：H50/pi05/absolute/no-adapt/mean-std/asset-id 全部断言通过。
- 真实转换输出：`/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab`，约 8.8 GiB。
- manifest：source/target 813/813，missing=0、unexpected=0、shape mismatch=0。
- 转换后逐 tensor 对照：813/813 `torch.equal`；模型文件 `9,354,045,872 bytes`。norm 为 state14/action14 mean/std。
- parity CLI 的两套 venv `py_compile`、`--help`、固定输入生成和 dummy compare roundtrip 均通过；固定输入为
  三路 480x640 RGB、state14、prompt 与 `noise[1,50,32]`。真实双 backend GPU action parity 留给 Stage A，
  本实现批次没有擅自启动 GPU。

### Git 终态

- branch：`codex/sz-sidney-pi05-current-rlinf`。
- 主实现 commit：`a0ad61e4585862391fc5750824c735b75cd73b3a`。
- horizon 精确修复 commit：`14d8d85f1ad9c221d588e7f837250d3ae166a61a`；避免继承 adjust-bottle 的 `step_lim=200`。
- current control 语义叶子修复：`c5ce08bb55c772c625c391f634aa409b9e1693c7`；随后以
  `63c06cbfcc45fb15015ce5488a80235ef4d92c10` 恢复已经验证的两卡 `64x4/GB1024/update2` 执行预算。
- cross-runtime parity 入口：`7aad93c567ba31dec1effaa0cc77fddd312ccb9f`；随后按 LeRobot 官方 parity
  路径显式把 native policy 移到所选 device，终态 commit 为
  `2e401955d2fd1bcde3c5b5c9275e3e6fe4bf1f6f`。Stage P 首跑随后暴露直接调用
  `PI05Config.from_pretrained` 不会消费 checkpoint 顶层 `type`；复用已跑通 oracle 的
  `PreTrainedConfig` 工厂分派并断言结果为 `PI05Config`，最终 commit 为
  `1f4d35c1335d759b9a2e8264e40c136ca08f9ce8`。真实 native forward 又确认官方 env 会在 policy
  processor 前把 HWC uint8 变成 CHW float；parity helper 绕过该 env helper，故只在 native 三图补
  `HWC -> CHW -> float/255`，RLinf 的原始 HWC 环境合同不变。CPU probe 得到三路
  `[1,3,480,640] -> [1,3,224,224]`，终态 commit 为
  `92fea8f72271b1ff37e58f3433a6fb64abe316cc`。P-v6 进一步证明 native 与 current 的 resize
  runtime 不相同：LeRobot 用 Torch bilinear float resize，current OpenPI transform 用 JAX resize
  并在 uint8 路径 round；480x640 高频 synthetic 输入不适合作为模型核心 action parity。按官方
  LeRobot/OpenPI parity 口径，工具默认生成 224x224、跳过 resize，只验证 processor/token/权重/M10
  forward；仍可显式指定 480x640 作为独立 resize diagnostic。没有修改 production resize。
  P-core-v7 的 224x224 双 backend forward 均成功：三相机最大差 `1.19e-7`，mask/token 完全一致；
  H50x32 action 最大/平均差 `0.00940/0.00153`，最终 H50x14 qpos 为 `0.00590/0.00149`，均通过
  官方 action `rtol=1e-2, atol=5e-3`。跨 PyTorch/NumPy 与序列化 mean/std 的 state 最大差
  `4.29e-6`；它不是官方同 runtime unit-test 的 bitwise 场景，parity helper 仅将 state gate 设为
  `atol=1e-5`，没有改 production normalization 或 action 容差。最终 commit 为
  `639444db8ad7bc9c934e056ad8511139ed94eba9`。复用 P-core-v7 两边 artifact 的最终 compare
  `exit 0`；报告位于服务器
  `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-core224-m10-compare-existing-v8/report.json`。
- Stage A 首次进入 `only_eval` 的真实 `validate_cfg` 后暴露上游官方 π0.5 YAML 未覆盖的缺口：
  训练校验选 `actor.model`，但 only-eval 校验选此前只有 path/precision 的 `rollout.model`，因缺
  `model_type` 在模型和环境启动前失败。专用 YAML 不复制易漂移的字段，而改成
  `rollout.model: ${actor.model}`，让两条路径共享完整 Sidney H/C/M、OpenPI、动作与精度合同。
  compose 全节点相等断言以及 training/only-eval 两次 `validate_cfg` 均通过；采样和优化预算未变。
  修复 commit 为 `bab221afb8bedc32a8f01b171901a347f0258063`。
- A-v12 随后越过模型加载，但 packet 只设 `runner.only_eval=true`，仍保留训练的
  `task_type=embodied` 与 `actor, env, rollout` placement；因此选中了 `EmbodiedRunner`，它尝试执行
  actor→rollout 权重同步，而 only-eval rollout 没有 weight-sync receiver。正确的 current eval 路径是
  `runner.task_type=embodied_eval` → `EmbodiedEvalRunner`，placement 只含 env/rollout，并由 rollout
  直接加载 checkpoint。该问题只修 Stage A 命令：Hydra 先
  `~cluster.component_placement`，再
  `+cluster.component_placement={env:4,rollout:4}`；真实 compose/`validate_cfg` 已通过。没有修改
  production YAML、正式训练 B 或任何采样/优化参数。
- A-v13r3 以该 official `EmbodiedEvalRunner` 路径完整通过：请求 seed 1001、1002 均触发
  RoboTwin `UnStableError` 后，由 current adapter 自动重试到实际 seed 1003；该回合在 400 actions
  合同内成功，进程 `exit 0`、无 fatal。至此 B=1 环境闭环以及坏 seed 重试语义均得到真实验证。
- B-v12 两卡真实 GRPO smoke 完整通过：物理 GPU4/5，`64 env x rollout1=64 trajectories`、
  8 个 G8 group、最多 512 query records、GB512/MB32/update2、fixed8、Step1 local-shard；
  rollout、两次 optimizer update、评估和保存均完成，进程自然 `exit 0`。训练成功率为
  `10/64=15.625%`，fixed8 为 `0/8`；KL/clip/grad=`0.353/0.250/22.370`，均为有限值。
  checkpoint 含 `rank0.pt`、`rank1.pt` 和 `full_weights.pt` 三个完整文件，合计约 28.83 GB。
  GPU4/5 峰值 `58,040/58,332 MiB`，主机 available RAM 最低约 1.80 TiB；无 fatal/OOM/Ray actor death。
  A/B 产物分别位于服务器
  `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3`
  与
  `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12`。
- 已推送 `personal/codex/sz-sidney-pi05-current-rlinf`，远端 ref 与本地 HEAD 一致；worktree clean。
- 最终实现代码 commit 为 `bab221afb8bedc32a8f01b171901a347f0258063`。随后按用户要求把轻量 smoke
  证据直接归档到同一算法分支的 `evidence/smoke_20260903/`：37 files、79,576 bytes，含 source lock、
  resolved、command、metrics/TensorBoard、resource CSV、关键日志摘要和 checkpoint 文件清单；不含
  checkpoint payload、视频、模型、数据或 Ray 全量日志，凭据扫描无命中。两个普通 evidence commit 为
  `e8998bdc`、`f50e235c`；最终 branch/remote HEAD=`f50e235c5ab1f4390f0ba92bfb13390ed0a86810`，worktree clean。
  本批没有影响 GPU0/6/7 上的既有任务。
