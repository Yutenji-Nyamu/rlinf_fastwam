# RoboTwin control trace：双仓调用、录像语义与产物

最后更新：2026-08-22  
范围：解释Idea2训练/推理中细粒度录像的实现；不改变训练算法或环境动作。

## 1. 一句话结论

control trace不是另起一套RoboTwin evaluator，也不是RLinf的六帧`RecordVideo`。它是在RLinf实际调用的
`RoboTwin_RLinf`环境控制循环中，读取**已经执行完当前physics/control step的head画面**，旁路写出MP4和
逐帧索引。动作chunk、路径压缩、TOPP、physics step数量和success判断仍走原调用链。

## 2. 两份源码怎样共同运行

```text
formal launcher
  PYTHONPATH =
    /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
    + /root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf
        |
        v
RLinf RoboTwinEnv
  from robotwin.envs.vector_env import VectorEnv
        |
        v
RoboTwin VectorEnv / Base_Task
  gen_sparse_reward_data(C50)
    -> compress_path(left/right)
    -> TOPP(left/right)
    -> repeated scene.step()
    -> selected frame -> control trace
```

- `PYTHONPATH`决定`robotwin.*`实际来自哪份环境源码；
- YAML中的`assets_path`经RLinf写入`ASSETS_PATH`，决定场景/模型/纹理资源根；
- `task_config.control_trace`决定是否开启旁路记录。

本次formal manifest记录两份版本：RLinf `3061872e30cf...`；RoboTwin/wamppo
`43696bbab85f...`。所以算法代码、环境代码和产物都有版本来源。

## 3. 精确改动放在哪里

### RLinf侧

- `rlinf/envs/robotwin/robotwin_env.py`：导入RoboTwin `VectorEnv`；开关开启时把RLinf worker identity写入
  control-trace配置。RLinf侧不负责逐帧编码。

### RoboTwin侧

- `envs/control_trace.py`：新增选择器、progress采样器、ffmpeg编码、`frames.csv`和`metadata.json`；
- `robotwin/envs/vector_env.py`：仅在开关开启时注入`env_slot`和该slot内episode序号；
- `envs/_base_task.py`：在现有`scene.step()`和render之后读取所选head frame；记录success、query边界和
  TOPP progress；episode结束时关闭编码器。

关键边界是：recorder不会再次调用policy、planner、`compress_path()`、TOPP、`scene.step()`或
`_update_render()`。因此它记录的是原控制过程，不是为录像重跑一条轨迹。

## 4. 当前formal怎样开启、保存多少

当前配置：

```yaml
control_trace:
  enabled: true
  output_dir: ${runner.logger.log_path}/control_trace
  worker_indices: [0]
  env_slots: [0]
  max_episodes_per_slot: 1
  camera: head_camera
  fps: 10
  crf: 32
  output_width: 160
  output_height: 120
  max_frames_per_episode: 200
  max_frames_per_query: 50
```

这表示整次100-step训练只抽样`worker 0 / env slot 0`的一条episode，而不是给所有rollout录像。现有formal
样本为111帧、约40 KiB；即使按200帧上限估算，当前单样本配置的存储量也可以忽略。若以后扩大录像数量，
主要开销是相机读取/编码和文件数量，而不是DVAC张量。

RLinf会在不同runner step重建环境。`.claims/`中的独占小标记让“一条episode”成为整次run的总预算，
不会每次重建后重新录一条。

## 5. 每个产物是什么

```text
control_trace/<task>/worker_000/env_slot_000/
  recording_0000_episode_0000_reset_<id>/
    head_camera.mp4
    frames.csv
    metadata.json
```

- `head_camera.mp4`：10 FPS、160×120、H.264/CRF32的低码率分析录像；
- `frames.csv`：每一编码帧对应的reset、query、action-slot、control/physics index、左右臂progress、
  `h_lo/h_hi/h_fraction`、success与capture reason；
- `metadata.json`：任务、相机、帧数、尺寸、编码配置、finish reason和encoder error。

progress采样器通常为C50的不同近似`h`区间各取第一帧，并为query end/success保留一帧；整episode最多200帧。
它因此比RLinf query-boundary视频细很多，但没有按250 Hz把每个physics step全部存下。

## 6. `h`对齐是什么精度

RoboTwin会先分别压缩左右臂路径，再分别TOPP重参数化；控制循环里两臂的实际控制点数可能不同。recorder用：

```text
progress = min(left_progress, right_progress)
h_float  = progress * (chunk_len - 1)
```

再保存`h_lo/h_hi/fraction`。它表示“双臂共同推进到chunk的哪个近似位置”，足以把moving、接近、接触和
首次success画面与query内进度对齐；它不是压缩前第`h`个模型waypoint的精确血缘。若以后确实需要精确映射，
要旁路保存路径压缩/TOPP的原始对应关系，而不能事后仅凭视频恢复。

## 7. 为什么不直接复用官方direct evaluator录像

官方direct evaluator会对policy返回的action逐个调用`take_action()`并录像。当前RLinf集成则把整个C50一次
交给`gen_sparse_reward_data()`，先对整条双臂路径压缩和TOPP，再进入底层控制循环。如果为了套用direct
recorder把C50拆成50次高层调用，规划和真实轨迹都会变化。

因此当前实现复用了RoboTwin已有相机与ffmpeg思路，但把取帧点放到RLinf实际使用的整chunk控制循环内。这是
“尽量少改原执行语义”下更合适的位置。

## 8. 对分析能提供什么、不能提供什么

能提供：

- 某个query内机器人何时接近、接触、抓取，以及首次success在哪一帧；
- frame与`query_idx/action_slot/control_idx/近似h`的对应；
- 将同query的DVAC `V(q,h)`曲线叠到更细的执行画面上。

不能单独提供：

- 更多独立DVAC时间点：一个query仍只有一次模型去噪与一条`V(q,0:49)`；
- 训练全量episode的task phase：当前只抽样一条；
- 精确model waypoint到physics frame的一一映射。

## 9. 本地源码入口

- [formal launcher](../../tmp/idea2_dvac_r_only_downweight_run_formal.sh)
- [formal YAML](../../tmp/idea2_residual_downweight_impl_source/rlinf/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_100step_formal.yaml)
- [RLinf RoboTwinEnv](../../tmp/idea2_train_impl_source/rlinf/rlinf/envs/robotwin/robotwin_env.py)
- [RoboTwin VectorEnv](../../tmp/idea2_train_impl_source/robotwin/robotwin/envs/vector_env.py)
- [RoboTwin Base_Task hooks](../../tmp/idea2_train_impl_source/robotwin/envs/_base_task.py)
- [control_trace.py](../../tmp/idea2_train_impl_source/robotwin/envs/control_trace.py)

