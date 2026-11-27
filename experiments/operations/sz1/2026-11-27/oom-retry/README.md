# 显存不足后优先恢复原训练

深圳 1 机的 4–7 卡由两个双卡 GRPO 和后续七个正式 clean RLT 共用。正式任务因 GPU 显存不足退出时，队列保留原卡位，从最近完整检查点重新启动同一任务一次。重试结束后按既有顺序交班。RLT 首次保存前从原始训练配置启动。四项无保存兜底继续由原队列循环管理，训练与评估各用 24 个环境分两批执行。

- 双卡任务保留模型、优化器、采样设置和原有卡号。FastWAM 核对两个 DCP 分片，Dojo 核对两份本地分片及完成标记。
- 识别 CUDA、Vulkan 和渲染器明确报告的显存不足。CPU 观察进程读取任务退出记录和训练日志。队列等原任务释放后交接。人工停止、单独的通信失败和 GPU 驱动失去响应走各自原有处理。
- RT2 单卡 RLT 的仿真通信重试和显存重试共用一个额外启动名额。正式 Dojo RLT 重开本任务自己的仿真并从正式训练接回。
- 生产训练持续运行。验证使用 CPU 临时目录、模拟子进程、既有故障日志和真实恢复点。部署快照见 [deployment.json](deployment.json)。

## 代码入口

双卡接管用 `pair_gate.py` 和 `queue_admission.py`。FastWAM 原启动器的改动列于 `fastwam_pair.patch`。单卡 RLT 通过 `patch_formal_owners.py` 接入 `rlt_formal_oom_1127.py`。单次部署入口分别为 `arm_live_pairs.py` 和 `install_rlt.py`。它们复用现有队列的 `common`、计划文件及训练环境。

归档版用 `QUEUE_DEPLOYMENT`、`QUEUE_ROOT`、`FASTWAM_OWNER_ROOT`、`QUEUE_OPS`、`ACTIVE_RUN_INDEX` 指定现场目录。应用时保留改动前后的启动器用于回读。运行当前队列版本的 CPU 测试时，将原队列的 `ops` 加入 `PYTHONPATH`，用 `OOM_RETRY_OWNER_SOURCES` 指向对应版本的原始 RT2 和 Dojo RLT 启动器，再运行 `python -B -m unittest discover -p 'test_*.py'`。
