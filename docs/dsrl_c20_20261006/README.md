# 深圳1机 DSRL 两组 C20 从零重训

本轮授权：用户要求两组 DSRL 每次执行动作数统一改为20，重新训练。只处理 GPU6 Clean、GPU7 ＋U；GPU4 BC8/U5、GPU5 RLT 保持原进程。

## 配置与输出

- 继承两组 C10 正式运行的实际 `runtime/resolved.yaml`，源码仍为 `8bcd99a6df38a1700a988d8982bde9368bf18025`，没有修改算法源码。
- 两组仅改 `actor.model.num_action_chunks`、`actor.model.openpi.action_chunk`：10→20；＋U 同步模型和算法两处 `dsrl_u_spec/spec.chunk_length`：10→20。其余差异仅新输出目录和实验名。
- 原 Sidney π0.5 SFT、`resume_dir=null`、空回放，重新第0轮开始；正式仍200轮，GB=MB=256、UTD20、warmup500、ODE10/5、H50、潜变量32、4环境×200物理动作、每13轮评估/65轮保存均保持。
- 沿用当前 RoboTwin 环境；本轮没有另外修改任务指令或其他对照条件。
- 结果根 `/data/chenyiteng/results/rlinf-dsrl-pi05-u-20261006/`：`clean-c20-formal-200-mb256-v2`、`u-c20-formal-200-mb256-v2`。旧 C10 两组结果、日志和断点原位保留。
- 新控制目录 `/data/chenyiteng/deployment-20261006/dsrl-u-c20-lease-v2`；复用旧控制的已验证 GPU6/7 计算及 EGL 作用域、同一份尚未启动的 RLT 回接计划。

## 执行和停止条件

先精确停止旧 CPU lease owner（其 SIGTERM 处理只退出自身并保留训练），验收 `owner-stopped.json`；再对已核实 UID/PID/start 的两个旧 driver 发 SIGTERM，由原 finally 清理各自 Ray namespace。必须取得两卡计算/图形上下文、namespace、已记录进程均释放的回执，才启新 owner 和正式请求。共享 Ray 不重启。

启动复用命令：`python -u -B <repo>/tools/dsrl_u_20261006/ops/lease_owner.py --control <new-control> owner`；待 DEV_HOLD 后，分别 `lease_owner.py --control <new-control> submit --request <role>-c20-formal-200-mb256-v2-request.json`。

源码/配置 pin 不符、进程身份变化、4/5卡保护对象变化、释放不完整时停止切换，不扩大清理；正式结束或异常，沿用原 owner 的精确清理与旧 RLT 回接流程。新旧 owner 不同时持有6/7卡锁。

## 已验证

- `p119-c20-prepared`：逐叶差异限定在上述字段和输出；745项请求 source/config pins 核实。
- `p120-c20-cpu-check`：服务器 CPU 通过真实投影器及 replay 的 C20 合约检查，空缓冲→40条、256抽样、U形状 `[256,20]`、有限均值1权重、折扣 `.999**20`；拒绝 U spec 与执行长度不一致。此为 fixture 检查，实际采集验收在启动后记录。

## 正式启动验收

23:20完成旧C10释放；23:21:27/32新Clean/U正式启动。唯一owner `3811880/start418679574`；Clean6 driver `3812248/start418680648`、U7 driver `3812455/start418681164`。旧owner和driver均已退出，新owner独占原6/7锁。

23:25两组首轮均完成，各40条回放，轮耗时87.2/87.6秒；均为warmup，无SAC更新，未继承旧回放。首轮采集成功均0/4，不用于判断方法效果。实际计算/渲染分别在6/7，EnvGroup `3812860` / `3813699` 均在各自卡显示C+G；0–3无进程上下文。4/5原driver与worker身份保持。当前 `/data` 余405GiB，`/home` 余240GiB，日志无Traceback/OOM/ActorDiedError/NCCL error。

证据：`p121-c20-stop-old`、`p122-c20-launch`、`p125-c20-first-round`；服务器回执为新控制目录下的`c10-released.json`、`c20-formal-started.json`。本次只要求启动后正常采集；正式仍需采满500条回放才开始更新，预计约第13轮。

23:28二轮验收：两组回放均40→80，第二轮68.45/71.30秒；无SAC更新符合warmup。owner心跳4.65秒，4/5保护身份保持，无跨卡或日志致命错误；`c20-acceptance.json`已生成，原始证据`p127-c20-acceptance`。
