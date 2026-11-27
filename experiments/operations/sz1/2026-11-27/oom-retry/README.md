# 在共享 GPU 上恢复显存不足的训练

4、7 卡运行 FastWAM，5、6 卡运行 Dojo。GPU 显存不足导致任务退出时，先从原任务最近完整检查点恢复。七个正式 RLT 候补也沿用这套判断。普通启动保留原队列规则。

| 这次失败留下的证据 | 同一检查点如何处理 |
|---|---|
| 明确的 CUDA、Vulkan 或 GPU 渲染显存错误，同一张卡还有足以影响本次分配的其他用户占用 | 每次重新判断。证据持续时继续重试，间隔 30 秒、60 秒，随后每次 120 秒。|
| 有显著同卡占用但进程已消失，或双卡中只有一张有外部占用且故障卡未定位 | 额外尝试至多两次。|
| 已确认 GPU 显存不足，尚无争用证据 | 额外尝试一次，然后按原顺序交班。|
| 普通 RPC 断开、CPU 内存不足、单独的驱动 Xid | 沿用各自原有的故障处理。|

判据综合本次分配大小、报错中的空闲显存和其他 PID 的占用。1 GiB 是本机用于忽略零星图形上下文的判断下限。PyTorch 消息里的逻辑 GPU 0 由进程或作业信息映射至物理卡。已经写出新完整检查点后，重新计算连续失败次数。

双卡任务重新入场只等待与该任务匹配的旧 GPU 进程，清理预算 90 秒。查驱动限时 8 秒。清理超时留下占用进程并走原队列。其他用户可以留在卡上。发生 OOM 时才查询争用情况，正常训练期间没有新增 GPU 轮询。

`oom_classify_1127.py` 识别失败，`oom_contention.py` 决定是否重试及退避时间。`pair_gate.py` 负责双卡队列，`queue_admission.py` 核对旧进程，`patch_dojo_admission.py` 与 `fastwam_pair.patch` 对接两路启动器。`rlt_formal_oom_1127.py` 和 `patch_formal_owners.py` 接入单卡 RLT。无保存兜底沿用既有循环。

`install_contention.py` 是现有队列的在线升级入口，依赖原调度模块、实际启动器和上一版部署目录。`QUEUE_ROOT`、`FASTWAM_OWNER_ROOT`、`QUEUE_OPS`、`ACTIVE_RUN_INDEX` 分别指定这些现场路径。CPU 测试中将原队列的 `ops` 放入 `PYTHONPATH`，用 `OOM_RETRY_OWNER_SOURCES` 指向 RT2 与 Dojo RLT owner。现场回读见 [deployment.json](deployment.json)。

