> 2026-10-17更新：采集已从SZ1等待队列迁到SZ2物理4/5，使用sz2-v2事务；见[migration-sz2](migration-sz2/README.md)。以下为原始部署历史。

# U与完整执行视频补采 · 2026-10-17

已实现并排队，真实录像验收尚待前序推理结束。优先级固定：深圳1物理6/7当前强弱/预算推理 → 本轮U与完整视频 → 原逐卡RLT。4/5 Clean训练原样保持。

## 已确认的问题和复用

外部交接`E:/0school/研二上/cvpr/uaca/draw/10_U_VIDEO_RECOVERY_HANDOFF_20261017.md`确认旧DVCA头图来自FastWAM逐动作录像；近期π0.5采集仅有chunk前后帧，两者执行接口不同。此次从已验收的π0.5 DV50/U链及其精确100份实配继续，不移植FastWAM控制器，不把旧边界视频当作逐动作录像。

- 原50任务×32回合＝1600回合，100批；每卡B16、H50、主M10/旁M5、14真实动作维；任务步数、原始模型、任务种子和noise seed继承10/3原配置，不筛成功种子。
- 记录每个动作位置的U、两条完整求解终点、主链、完整初始噪声、实际动作和输入观察。
- 主相机按仿真时间25Hz连续采样，另存初始和query末帧，包括成功终止画面；覆盖整段执行，不声称每个250Hz物理步都有视频帧。每个物理控制步另有JSONL。
- 保留原动作、压缩路径及精确保留索引、TOPP/下采样后控制目标和夹爪路径。视频帧标query/slot/实际seed/物理tick/仿真时间/左右臂目标索引；模型h是带误差的路径投影，不能把它当精确TOPP逆映射。
- 原生控制器只在本进程插入两处观察回调，不改共享RoboTwin文件、不增加物理步、不更改动作与TOPP。

## 检查与运行边界

服务器CPU测试已通过：用安装现场的原生控制器，对照完整控制指令、步数、渲染更新次数、结果和NumPy RNG；合成录像完整解码，帧0/10/17对应控制步0/10/17；路径映射检查通过。没有占用GPU做额外测试。

两卡各自首个完成批会检查主动作和Torch RNG一致性，再做全部MP4完整解码、帧数/覆盖/控制步校验及U重算。两卡验收均通过才展开剩余任务。原生任务失败记失败后继续，不重试；录像/U校验或清场失败停止本路。保留600秒无query进展和3600秒单批截止。

这次不增加常驻监控；两个有限lane当前CPU等待原推理释放。唯一owner已重启读取新优先级，所有RLT fallback路径及4/5配置保持。

## 路径

- 源码分支：`codex/pi05-u-full-video-20261017`，实现`e105b2fd883659225d30232980bd1300843c0325`。
- 源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-u-full-video-20261017/tools/pi05_u_video/`。
- 结果：`/data/chenyiteng/results/pi05-u-video/20261017-v1`。
- 事务：`/data/chenyiteng/deployment-20261017/pi05-u-video-v1`；`enqueued.json`已生成，禁止重复交接。
- 后续先读两个lane状态、`acceptance-g6/7.json`和结果`summary.json`；有验收之前不能声称真实连续视频已通过。

绘图取每个episode的MP4、frames JSON、control JSONL、每query U NPZ和controller NPZ；同时保留task/实际seed/query，避免拿旧图混配新信号。未向其他聊天发送消息。
