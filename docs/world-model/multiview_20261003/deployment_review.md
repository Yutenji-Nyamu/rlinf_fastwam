# 10月3日授权改动：部署审查

本条只读审查源码和既有证据，不连接服务器、不修改训练源码、不启动训练。用户授权范围为 micro128→64、global2048保持、环境卸载等待、简要资源监控、save10及每10轮独立评估，未来评估C8的单/双相机两种输入。其余方法、采集量、预算保持。

## 最小文件清单

| 文件 | 变化 |
|---|---|
| `rlinf/runners/embodied_runner.py` | `rollout_handle.wait()`后增加`env_handle.wait()`，先完成视频flush/环境卸载再进入actor训练 |
| `rlinf/workers/env/env_worker.py` | env卸载前后资源记录 |
| `rlinf/workers/actor/embodied_fsdp_actor_worker.py` | actor加载前后及训练后资源记录，开始前重置统计峰值 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | rollout加载前后、卸载前后资源记录 |
| `rlinf/utils/resource_telemetry.py` | 新增可选记录器，无CUDA初始化或同步，诊断异常不改变训练 |
| `examples/embodiment/config/wan_goal_pi05_headonly_formal_sz3.yaml` | 仅micro128→64、save40→10；global2048及`val_check_interval=-1`保持 |
| 新执行包的`private_ray_driver.py` | 启用worker资源目录及外部监控启动接线 |
| 新执行包的`resource_monitor.py` | 每10秒GPU/每进程显存/RSS、每60秒PSS；随本run退出终止 |
| 独立评估协议文件 | 每10轮已完成CP入队；原生LIBERO、C8、head-only与head+wrist两路；复用现有外置入口 |

原r6 checkout和旧driver保持。新副本需要延续两份已经跑通的π0.5适配源码；不能从干净上游只打本次五文件patch就称继承了原实配。

## 精确基线

固定上游：`d34d4c320d08cb982de034aa9a011f08dc0fa217`。

`local_scripts/wan_goal_20261003/prepare_phase_barrier.py`对四份既有源码逐SHA校验，一次替换、语法检查后才允许全部写入，且禁止`--apply`到原名`RLinf-pi05`。本次五文件清单及改前/改后SHA在既有`phase-barrier-review/patch-manifest.json`。

已核本地文件与r6 manifest的SHA完全一致：

- `local_scripts/wan_goal_20260930/private_ray_driver.py`：`6b36a2b6d0e87c4f481a19530758a4716c07fb7239f10f3dc8d03d4eaab18242`。
- `local_scripts/wan_goal_20260930/wm_sequence.py`：`eb7db6e2eeeabc89ce5c607f78689087a1fea9a7bc0fa3b3c1b7211af60638a4`。
- `local_scripts/wan_goal_20260930/pi05/wan_goal_pi05_headonly_formal_sz3.yaml`：`7e650f37e95f72e0aed67cbd2f86c0ed345e3cc7b34f50d40964a0158725fbee`。

## 监控最小可靠接线

真实r6链路是`wm_sequence.py → run_stage(spec) → private_ray_driver.py → train_embodied_agent.py`。内层`stage_run`为`.../pi05-formal`，其下有`launch.json`、`managed-identities.json`及`wm-exit.json`，外部监控的`--run`必须指向这一层。

现有`resource_telemetry.py`默认关闭，唯一启用条件是worker可见`WAN_GOAL_RESOURCE_DIR`。真实driver已经包装`ray.init`，在`scoped_init()`中的`runtime_env.env_vars`注入owner、namespace、address。最小改法是在**新driver副本**中：

```python
# 在ray start之前，由该driver的log目录确定，不接受任意外部写路径。
resource_dir = log / "resources"
os.environ["WAN_GOAL_RESOURCE_DIR"] = str(resource_dir)

# 在现有scoped_init里，紧接既有env_vars.update之后。
runtime["env_vars"]["WAN_GOAL_RESOURCE_DIR"] = str(resource_dir)
```

固定d34d的scheduler明确保留job runtime环境继承；worker自己的env overlay没有这个键。这样无需改scheduler，且同时覆盖Ray启动前环境和job级显式传递。部署后CPU可用假的`orig_init`执行提取的`scoped_init`验证传递，真正worker日志需下次GPU运行再确认。

`resource_monitor.py`已有Linux只读采样实测，但文件部署不等于持续监控运行。启动时要在新run的`launch.json`存在后派生它，`--output`必须是该run直属的新`.jsonl`，建议`resource-usage.jsonl`；同run的`managed-identities.json`提供PID/UID/boot/start核验。复用WM owner token使它归本run管理，终态由`wm-exit.json`或精确owner清理结束，不新增独立长期调度器。

边界日志中actor/rollout的peak每轮重置；env当前只在卸载前后采样，不逐轮重置峰值，因此env的peak字段是累计峰值，不能误标为该轮独立峰值。外部10秒采样主要看驻留/趋势，瞬时actor峰值主要看torch边界统计。

## 评估频率

保留内置`val_check_interval=-1`，因为现有内置配置是500并发且复用训练输入接口；直接改10会引入用户没有要求的新评估执行路径和显存压力。

新协议应写：每完成10轮并验证CP完整后，排独立原生LIBERO评估；C8单相机、C8双相机分别记录；每模型每种输入500唯一初态、20环境×25批，任务/种子/上限320保持，资源不足就排队。原始SFT先在同路径建立对应基线。历史双相机/C5结果保留为既有结果，不混入C8曲线。本轮仅落协议及复用入口定位，不启动新评估系统。

既有可用入口位于服务器`evaluations/wm-official-20261002-1738-dc66e5a3/source/`。旧owner固化了原始/CP40/CP80和旧run身份，不能直接重放；应复用其中driver、worker、Mesa渲染和结果去重部分，给新run、CP与相机参数建立新owner。历史单相机/C8的原始0/500尚未解释，不能把该旧入口当作已通过基线。

## 部署验收边界

1. 对照旧r6精确SHA，保留原dirty和两个π0.5适配。
2. 在隔离副本应用五文件patch，检查最终配置仅两项指定数值变更。
3. 复用已有服务器CPU四项屏障验证；新增driver只检查环境传递/监控命令，不额外启动GPU。
4. 监控`--once`只读采样可证明采样器能执行，不能证明训练阶段曲线已经存在。
5. 云端只推审过源码、配置、说明和轻量证据；新GPU诊断及正式恢复仍未发生，发布文档必须如实注明。
