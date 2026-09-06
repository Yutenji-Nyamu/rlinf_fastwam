# GPU3 π0 在线成功 BC 短测合同（待用户确认，未启动）

2026-09-05状态：这是**旧的、未执行的小闭环提案**。用户要求smoke保留正式并发、只缩串行；最新讨论见[预算复核§3–4](BC_IMPLEMENTATION_AND_BUDGET_REVIEW_20260905.md#3-正式参数与-smoke-建议还没有改成启动合同)。尚未替换本页YAML或启动脚本，不能据此启动32环境容量测试。

## 1. 固定配置与目标

目标只验证：真实采集 → 成功数据入累计池 → FM 梯度更新 → 权重同步 → 下一轮采集／固定评估 → 实际保存。不是学习效果对比或长程稳定性测试。

完整参数：[服务器 compose + validate_cfg 后的 resolved YAML](smoke-resolved-compose.yaml)。
可编辑配置：[开发配方](../../../worktrees/pi0-online-bc/examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml)。

| 项目 | 本次拟定值 |
|---|---|
| 源码 | 官方 `dc9b87cc49334c7516487ead68ebeb060fd7c090`＋本专题9文件改动，独立 `codex/sz-pi0-online-bc`；尚未commit/push |
| 运行树 | `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc` |
| 已有依赖 | `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin`；不安装或升级 |
| RoboTwin | `/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support`，`0008ae6800df9f75fc8de7098bacb01735fd8fd2`；不引入Fast补丁，不改OIDN |
| 起点 | `/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`；不是旧GRPO checkpoint |
| 任务和输入 | adjust_bottle；三相机；无域随机化；C50、14维指令、10次flow去噪；沿用官方映射/归一化 |
| 可训练范围 | action expert＋动作/状态/时间投影；冻结PaliGemma；bf16，单rank FSDP no_shard |
| 采集 | 2并行环境×2串行轮＝4条尝试/轮；共2轮＝8条尝试；每条上限200个指令action槽，不是200次物理插值步 |
| 数据 | 累计成功episode；query均匀有放回采样；默认不混D0，`demo_weight=0` |
| 更新 | 每轮2次optimizer step，global batch4、micro1；有成功数据时共4次更新、16次replay query抽样；不等于遍历全池4遍 |
| 优化器 | LR2.5e-5 constant，warmup0；其余见完整YAML；这是开发预算，不继承旧GRPO正式超参 |
| 评估与保存 | 每轮2条fixed eval，共4条；每轮保存，共2个checkpoint目录；不据此判断成功率提升 |
| GPU | 物理GPU3，actor/env/rollout共置；不用CUDA_VISIBLE_DEVICES重编号，不抢4–7 |

当前RoboTwin对整个waypoint chunk做TOPP/重采样，成功可能提前停止物理执行。训练标签是实际提交的macro-command，排除成功后的后续query，不虚构原C50精确物理执行prefix。

示范参数化：`algorithm.online_bc.demo_weight=w` 和 `algorithm.online_bc.demo_data_path=/已有本地LeRobot数据集`；loss为 `(online_FM+w*demo_FM)/(1+w)`。`w`是loss权重，不是采集比例；真实D0加载未验收，本短测仍用0。

## 2. 精确启动命令与输出

完整环境变量、固定路径、禁止覆盖检查和后台启动命令均在[未执行的启动脚本](../../../local_scripts/remote_commands/sz_online_bc_smoke_launch_20260904.sh)。确认后先刷新GPU3及当前树，再由固定host-key Paramiko在普通账号下执行此脚本；不是重启Ray。

脚本设置 `PYTHONPATH=<BC树>:<RoboTwin树>`、`REPO_PATH`、`EMBODIED_PATH`、`ASSETS_PATH`、`PI0_MODEL_PATH`、`ONLINE_BC_RUN_DIR`、`RAY_ADDRESS=172.17.0.1:6389`、`RLINF_CODE_WORKING_DIR=<BC树>`、`TORCHINDUCTOR_COMPILE_THREADS=1`。

精确训练入口：

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc/examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_online_bc_openpi
```

输出根：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke-v1`。目录已存在则拒绝启动，不覆盖旧数据。

- driver日志/PID：根目录下 `driver.log`、`driver.pid`。
- 成功数据：`success_data/rank_0/batch_*.pt`。
- 模型/优化器及回放：`pi0-adjust-bottle-online-bc-smoke/checkpoints/global_step_{1,2}/actor/`；回放在其 `online_bc/rank_0/`。
- RoboTwin输出：根目录下 `robotwin_data`，关闭视频；不写共享相对`./data`。

## 3. 资源、监测和停止

历史预留估算为1张H100 80GB、显存40–60GiB、RAM128GiB以内、输出磁盘80GiB以内；**09-05复核撤回这些数字作为容量保证，BC至今没有GPU/RAM实测**。尤其不能外推至32并行、micro32/global1024或100轮。无新增模型下载；actor/rollout采用已有offload，首次加载/编译可能主导时间。

23:36 CST只读现场：GPU3为4MiB、0%、无compute进程，GPU1/2同样空闲；RAM available751GiB，swap6GiB已满。GPU4/5、6/7与shared Ray原PID均在。不推断它们的当前训练step、fatal或checkpoint。

监测：实际训练参数数目/冻结范围、成功episode和query入池数、有限FM loss/grad norm、optimizer update数、同步后下一轮、fixed eval、真实checkpoint文件、GPU显存/主机内存、fatal/OOM。

停止条件：2轮完成；或本任务fatal/OOM/NaN；或非加载编译阶段连续10分钟无进展；或总墙钟30分钟达到。后两项由本次执行者跟踪，不冒充配置已实现的自动超时。只识别并清理本次driver/job/worker，不停止shared Ray或其他任务。

若8次尝试没有成功，允许空池一致跳过更新，记录“采集已验、更新未验”；不自动扩大采集预算、不混入失败或老师动作。若需要依赖变更、下载、改变渲染/任务语义，停止并另行讨论。

## 4. 启动前已完成／仍未完成

已完成7项基础测试、Ruff、diff检查、AST、BC worker import、Hydra和真实validate_cfg；物理GPU映射确认。示范loss混合为替身模型测试，非真实数据验收。

未完成真实GPU模型梯度、环境闭环、权重同步效果、模型/优化器checkpoint保存与恢复；未启动短测、未启动正式长训。本合同需用户确认后执行。
