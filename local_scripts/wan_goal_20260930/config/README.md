# SZ3 Wan Goal OFT smoke 配置

2026-09-30。本目录仅准备本地配置／启动模板，未远程启动。固定上游 `RLinf/RLinf@d34d4c320d08cb982de034aa9a011f08dc0fa217`。

## 直接使用

将本目录两个OFT YAML复制到独立checkout `/data/chenyiteng/projects/wan-goal-sz3/RLinf/examples/embodiment/config/`。它们是以固定版本官方 `wan_libero_goal_grpo_openvlaoft.yaml`为底稿的独立primary配置，保留官方env/model/fsdp等defaults组；不再继承整个官方顶层YAML。原因是官方顶层定义了 `hydra.searchpath`，Hydra不允许该字段来自非primary配置。原上游YAML不修改。

以下模板由主实施器在借卡成功、私有Ray就绪后调用；GPU映射和Ray端口／namespace由主实施唯一控制。配置显式设置 `actor,env,rollout: "4-7"`，对应物理4–7上的4个actor、4个env和4个rollout rank，各卡colocated。不要把此模板发给已有共享Ray执行。

**这里必须写物理 `4-7`，不能写 `all` 或重新编号的 `0-3`。** 当前RLinf的 `NvidiaGPUManager.get_num_devices()`经Ray底层NVML计数枚举物理卡；NodeInfo可记录8卡，独立于Ray启动时的 `--num-gpus=4`。`FlexiblePlacementStrategy`直接把所选本机hardware rank变成worker的CUDA可见设备编号，并设置 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`。因此原生placement是本次实际约束；单设父进程 `CUDA_VISIBLE_DEVICES=4,5,6,7`并不能替代它。不需要额外node_group或覆盖NodeInfo为4卡，否则 `4-7`反而越界。

```bash
P=/data/chenyiteng/projects/wan-goal-sz3
export EMBODIED_PATH="$P/RLinf/examples/embodiment"
export PYTHONPATH="$P/RLinf:${PYTHONPATH:-}"
export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# 填主实施器已经下载完成的实际路径；路径可以不同于下面示例。
export WAN_GOAL_WM_PATH="$P/models/wan-libero-goal"
export WAN_GOAL_OFT_PATH="$P/models/oft-libero-goal-sft"
export WAN_GOAL_RUN_DIR="$P/runs/oft-smoke-<unique-run-id>"

source "$P/envs/oft-wan/bin/activate"
cd "$P/RLinf"
python "$EMBODIED_PATH/train_embodied_agent.py" \
  --config-path "$EMBODIED_PATH/config" \
  --config-name wan_goal_oft_smoke_sz3
```

上面直接调用官方Python入口，使YAML的独立输出路径生效。若用官方shell入口 `bash examples/embodiment/run_embodiment.sh wan_goal_oft_smoke_sz3 LIBERO`，shell会把 `runner.logger.log_path`覆盖为checkout内 `logs/<timestamp>-<config>`，应按该实际位置采集日志；它不会透传第三个以后的Hydra参数。

借卡前在服务器做一次官方Hydra解析即可确认配置组合和路径（不启动训练）：

```bash
python "$EMBODIED_PATH/train_embodied_agent.py" \
  --config-path "$EMBODIED_PATH/config" \
  --config-name wan_goal_oft_smoke_sz3 --cfg job --resolve
```

此解析不替代运行器的GPU资源与批量校验。主实施器保留resolved YAML及实际命令、stdout、退出码，不需要另写一套镜像配置测试。

私有Ray上的轻量placement核查可在主实施driver的同一address/namespace绑定内完成。Hydra compose并调用官方 `validate_cfg`后，在模型worker创建前取以下结果；它只启动scheduler管理／探针actor并读取硬件信息，不加载VLA或在0–3分配GPU张量：

```python
from dataclasses import asdict
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

# cfg来自本配置的官方Hydra解析，并已经validate_cfg(cfg)。
cluster = Cluster(cluster_cfg=cfg.cluster,
                  distributed_log_dir=cfg.runner.per_worker_log_path)
placement = HybridComponentPlacement(cfg, cluster)
for name in ("actor", "env", "rollout"):
    rows = placement.get_strategy(name).get_placement(cluster, True)
    assert placement.get_world_size(name) == 4
    assert [r.visible_accelerators for r in rows] == [["4"], ["5"], ["6"], ["7"]]
    assert [r.cluster_node_rank for r in rows] == [0, 0, 0, 0]
    print(name, [asdict(r) for r in rows], flush=True)
```

如单独运行该核查，结束仍由外层owner清理本次私有Ray；不要 `ray stop`。真实启动还应核worker进程的 `CUDA_VISIBLE_DEVICES`为对应4、5、6、7及owner token/namespace。driver启动后动态设置 `os.environ["PYTHONPATH"]`只影响子进程，当前进程导入源码还需启动前PYTHONPATH已包含checkout，或 `sys.path.insert(0, str(repo))`。

## 为什么选32环境

保留G8，用4张卡时每rank最少8环境才容纳一个完整GRPO组。`N=32`给每rank一个完整组，同时保留每条轨迹256动作；它验证完整rollout、同组优势、FSDP更新和第二轮权重同步。`N=64`给每rank两个组，更多组可能降低恰好全部被奖励过滤的机会，但采样工作量翻倍；先按32完成所要求的短验证。

| 口径 | 本smoke | 官方Goal预算参照 |
|---|---:|---:|
| GPU／actor rank | 4 | 本参照也放4 |
| GRPO G | 8 | 8 |
| total_num_envs | 32 | 64 |
| 每rank环境数／完整组 | 8／1 | 16／2 |
| rollout_epoch | 1 | 16 |
| 每轨迹动作数／chunk C | 256／8 | 256／8 |
| 每runner epoch全局动作数 | 8,192 | 262,144 |
| 每runner epoch全局chunk样本 | 1,024 | 32,768 |
| actor global／micro batch | 1,024／32 | 8,192／32 |
| 每rank每optimizer step累积微批 | 8 | 64 |
| 每runner epoch optimizer.step次数 | 1 | 4 |
| runner epochs | 2 | 1,000 |
| save_interval | 1 | 5 |
| 总optimizer.step调用数，update_epoch默认1 | 2 | 4,000 |

这里训练batch的单位是**chunk样本**，不是单动作。`32×1×(256÷8)=1024`；每rank收256样本，`1024÷4=256`；`256÷32=8`微批。官方预算参照保留上游资源与方法，不授权启动额外的OFT长训；用户正式目标仍是π0.5。

所有关键方法项按固定官方配置保留，未改变：C8、7D、G8、256×256、condition5／num_frames13、denoise5、KIR、Goal `TaskEmbedResnetRewModel`、relative reward、coef5、奖励过滤0.5–4.5、温度1.6、GRPO clip与lr、action-level reward／token-level logprob、offload与FSDP。这里是**task embedding奖励模型**，不是新引入的reward ensemble。

## 实际更新验收

保持官方 `filter_rewards=true`。固定源码的过滤通过loss mask完成，不物理删除样本，因此不会破坏上述整除。但若组被全过滤或组内优势为零，两次 `optimizer.step`调用本身不能证明有效RL学习。

验收读取真实运行输出：完成两个runner epoch；`train/actor/grad_norm`等训练指标有限，两轮均有非零有效梯度、有效mask与非零优势；checkpoint `global_step_1`与`global_step_2`均完成落盘；相同参数张量的对比存在非零变化，并结合GRPO损失／有效mask解释，不能只凭AdamW weight decay或模型文件时间认定有效策略更新。若没有有效组，保留失败口径交主实施处理，不关闭过滤来制造成功。

checkpoint原生路径为 `$WAN_GOAL_RUN_DIR/wan_goal_oft_smoke_sz3/checkpoints/global_step_{1,2}/actor/`。模型推理视频在继承的 `video/train`目录。`val_check_interval=-1`沿用官方，结束时也不会自动执行496环境真实LIBERO评测；真实环境评测需单列记录，不能把WM奖励等同于成功率。

只读验收脚本为上一级 `verify_smoke.py`，在训练进程退出并释放GPU后使用对应venv的Python：

```bash
python /data/chenyiteng/projects/wan-goal-sz3/scripts/verify_smoke.py \
  --run-dir "$WAN_GOAL_RUN_DIR" --experiment wan_goal_oft_smoke_sz3 \
  --output /data/chenyiteng/<本次control目录>/smoke-verification.json
```

成功exit0，否则exit2；只有新receipt会被写入，训练日志／checkpoint不修改，既有receipt不覆写。默认核TensorBoard step0/1（对应CP1/2）的有效梯度／loss／mask／adv，再检查 `actor/dcp_checkpoint/.metadata`引用的全部分片范围，并用CPU `torch.load(weights_only=True,mmap=True)`对 `actor/model_state_dict/full_weights.pt`作同参数样本比较。默认最多16个浮点tensor各8192元素，找不到样本差异返回未确认，不能反推全部权重未变。`wm-exit.json`存在时一并核正常退出。输出会说明这是结构完整性与样本变化证据，没有做整包checksum、optimizer恢复或LIBERO成功率测定。π0.5 smoke若沿用同一日志／checkpoint格式及两轮保存，可复用这个脚本；它还核额外SFT／entropy目标未启用，避免把其他目标的梯度归为GRPO。

当resolved配置为 `actor.model.openpi.train_expert_only=true`，比较范围限于固定d34源码实际保留训练的 `action_in_proj/action_out_proj/time_mlp_*`及 `llm.layers.*`的expert-1（norm、MLP、q/k/v/o投影、final norm）；前者优先。冻结的img、embedder与expert-0不占16个tensor名额。state_dict本身不保留`requires_grad`，因此这项范围来自已审计源码与resolved配置；若名字不匹配直接报未确认，不猜测参数是否冻结。此调整仅完成本地源码编辑，Hydra组合与真实checkpoint验收由服务器主实施执行。

源码依据：

- [官方Goal配置](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/wan_libero_goal_grpo_openvlaoft.yaml)。
- [环境组／步数整除校验](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/config.py#L1280)。
- [物理加速卡计数](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/scheduler/hardware/accelerators/nvidia_gpu.py#L347)、[原生range语法](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/scheduler/placement/placement.py#L287)、[placement生成可见设备](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/scheduler/placement/flexible.py#L207)、[worker最终写入设备环境](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/scheduler/worker/worker_group.py#L244)。
- [actor累积微批计算](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L91)。
- [过滤只改mask](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L251)。
- [flatten chunk样本、按global batch切分和optimizer.step](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L602)。
- [runner每epoch一步、checkpoint路径](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/runners/embodied_runner.py#L658)；[val_check_interval=-1不做最终eval](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/utils/runner_utils.py#L31)。
