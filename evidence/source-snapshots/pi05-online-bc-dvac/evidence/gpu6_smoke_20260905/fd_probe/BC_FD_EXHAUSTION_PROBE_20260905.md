# 32＋32场景文件句柄耗尽：隔离现场证据

2026-09-05 12:52:57—13:00:54，GPU6独立进程579136，使用BC v5完整resolved、相同RoboTwin/三相机/OIDN；0策略查询、0动作step、0优化器更新。脚本`local_scripts/bc_env_fd_probe_20260905.py`及远程命令`local_scripts/remote_commands/sz_bc_env_fd_probe_20260905.sh`，终端原始输出本轮工具记录保留。没有改共享Ray、其他任务或生产源码。

| 阶段 | 文件句柄 | soft / hard |
|---|---:|---|
| 初始 | 39 | 1024 / 1048576 |
| 训练向量环境构造 | 152 | 不变 |
| 32训练场景reset并取三路RGB成功 | 632 | 不变 |
| 评估向量环境构造 | 632 | 不变 |
| 另外32评估场景reset/相机渲染 | 读取fd目录本身失败：Errno24 | 不变 |

关键终端原文：

```text
{"stage":"eval_reset_render_failed","open_fds":"errno=24: Too many open files","nofile":[1024,1048576],"error":"vk::Device::createFenceUnique: ErrorOutOfHostMemory"}
{"stage":"result","open_fds":"errno=24: Too many open files","nofile":[1024,1048576],"passed":false}
OSError: [Errno 24] Too many open files: '.../env-fd-probe-20260905/result.json'
```

该probe是**成功复现资源不足、程序退出1**，不是测试exit0。预设仅在getSemaphoreFdKHR错误时尝试提高上限，因此这次更早的createFenceUnique失败没有执行提高；finally连写result.json也无法打开文件。不存在成功写出的服务器result.json，不伪造下载结果。进程已结束，系统释放其资源；没有完整生产smoke或提高后32＋32的通过证据。

v5在相同首次评估阶段报getSemaphoreFdKHR；probe在邻近Vulkan fence调用报OutOfHostMemory，同时OS明确EMFILE，说明不能仅按Vulkan错误名字认定整机RAM不足。它确认本配置的1024 FD容量问题，但不证明所有同型Vulkan故障唯一同根因。

实际小修：BC显式`env.min_open_files:4096`，EnvWorker.init_worker内只抬本进程soft（保留hard及已有更高/无限soft），超过hard提前报错。共享Ray仍1024，不抬系统全局上限，不降train/eval32并发，不变更画面、种子、采样或学习预算。单元测试不实际改本机limits，而是mock验证opt-in/保留高值/非法hard边界；修后直接完整v6两轮验收，不另加独立采集试验。
