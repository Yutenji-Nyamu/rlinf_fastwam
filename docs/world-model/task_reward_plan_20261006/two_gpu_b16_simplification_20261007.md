# WMRL 外层精简与恢复 · 2026-10-07

本轮授权：充分精简不服务于训练的外层代码，保持训练行为，恢复深圳3 GPU4/5正式训练。当前入口为 `lift-two-gpu-b16-lean-20261007-v1`；实时启动证据见 [轻量状态](two_gpu_b16_lean_light.json)。

## 上次为何停训

13:00前已完成512条采样，累计384个WM小批次；B16最大已分配显存45.03GiB、预留52.92GiB，采样卡总占用约54.3GiB，未发现OOM。之后策略进入反向计算，13:06前后外层 `register_actors → Ray Dashboard HTTP读取` 超时；异常直接进入外层finally，终止了WM和训练。没有完整策略更新记录或新检查点。13:07原owner启动RLT4/5候补，13:28二者仍在运行。

这是管理查询被错误地作为训练故障处理。详见 [一小时四次观察记录](two_gpu_b16_monitor_20261007.json)。

## 实际删除与保留

| 项目 | 处理与原因 |
|---|---|
| 每10秒Ray全局actor列表查询 | 删除；正式运行及WM清理均不依赖Dashboard |
| 重复进程/显存扫描、双份nvidia-smi查询 | 合并为可失败的每分钟轻量观察；WM自身的阶段显存/RSS记录保持 |
| driver重复整套配置/历史源码校验 | 删除；准备时比对原配置，owner入口核一次实际源码 |
| formal结束清理一次、finally再清理一次 | 合并为finally唯一清理 |
| 结束后对将退出WM重复offload/health | 删除；训练阶段的真实WM卸载屏障保持在原框架内 |
| 外层等待RLT首轮并继续轮询Ray | 删除；恢复派发有回执，首轮状态由RLT自身日志证明，不宣称已验收 |
| 旧owner拼接生成器、两份过时审计脚本、4份本地重复源码 | 删除；原实验及发布版本保留历史证据 |
| pidfd、UID/PID/start、唯一worker token | 保留；精确停止本任务，防止PID复用或误伤其他实验 |
| 图形/计算绑卡、原生评估、阶段资源记录 | 保留；GPU4策略、GPU5 WM，0–3及6/7不新增本任务上下文 |
| RLT借卡、完整断点、部分借卡恢复、归还锁 | 复用原实现；这些直接承担资源交接，只在切换时运行 |

主owner从709行减为473行，不再由旧wrapper/模板拼装。后台仅一个owner、一个WM服务、一个训练driver；没有重试守护、定时重启或新增长期监控。资源统计超时只写 `RESOURCE_OBSERVATION_SKIPPED`。观测到明确越卡、真实训练/WM进程退出、用户终止或既有60天硬上限才中断训练。

清理通过唯一owner token发现Ray worker和本任务子进程，并核UID/PID/start后用pidfd发信号；无需全局Ray状态接口。清理未完成或GPU4/5仍占用时，不派发RLT。

## 参数与执行

- `lift_pot`；原始Sidney π0.5从0至200轮。上次没有完整检查点，不恢复半轮梯度。
- GPU4策略B64、actor micro16/global2048/U2；GPU5单份OpenDW真B16。
- N64/R8/G8，512轨迹/轮；C32、384动作；每10轮保存及原生32回合评估。
- 训练配置除输出目录/组名外与上次逐项一致；WM服务参数完全一致。不追加GPU smoke，不改RLinf运行源码、奖励、精度或算法。
- 准备时RLT完整断点4卡CP300、5卡CP325；停止时沿原流程冻结最新可用点。6/7 Norm保持。

控制目录：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-two-gpu-b16-lean-20261007-v1`。

输出目录：同父目录 `runs/lift-two-gpu-b16-lean-20261007-v1`；namespace `opendw_lift_b16_lean_1007`。

命令：`<rlinf-venv>/bin/python -u -B <控制>/code/owner.py --plan <控制>/prepared/owner-plan.json owner`。CPU加载时RLT继续，WM准备好后精确借4/5，完成/失败后归还。唯一回执为 `launch.json`；不要重放已启动入口。

## 验证与边界

6项服务器CPU回归通过，耗时2.17秒：Dashboard不调用、显存统计超时不打断训练、真实WM退出及训练非零退出不隐藏、观察到越卡仍拒绝、清理只停止带本任务token的进程且保留旁路进程、PID复用不发信号。配置和服务参数比对通过，原GPU smoke结论不重复测试。

这些证明了本次管理代码变更；正式有效梯度、学习收益和长期稳定性以新运行日志为准。
