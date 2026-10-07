# 两卡WMRL B16正式重启 · 2026-10-07

## 当前状态

2026-10-07T12:02:06.840619+08:00：正式入口存活，CPU加载WM；尚未借卡。owner PID3111039/start731673231；driver 尚未启动。真实完成WM batch数0；这不是完整RL轮数。

用户明确只改WM batch32→16、跳过smoke，直接正式。没有部署VAE拆批草稿，没有修改模型精度、动作/奖励接口或算法。

| 项目 | 本次配置 |
|---|---|
| 任务／起点 | lift_pot；原始Sidney π0.5，RL0开始，无optimizer resume |
| GPU4 | 单份策略推理B64；actor micro16 |
| GPU5 | 单份OpenDW＋小RM，WM真B16；N64拆成四批 |
| 采样 | N64/R8/G8，512轨迹/轮，G已包含在N中 |
| 训练 | global2048，U2，200轮 |
| 时间／评估 | C32，384动作上限；每10轮保存和原生32回合评估 |
| 故障行为 | 清理本任务后归还RLT4/5；不自动重复启动WM |

CPU配置比对通过：训练配置除输出位置/运行名外逐项一致；服务启动参数只从B32改B16。未新增GPU smoke。准备时RLT恢复点为GPU4 CP300、GPU5 CP325，停止后还会按原流程确认最新完整点。6/7 Norm实验保持。

## 唯一启动和归还入口

控制：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-two-gpu-b16-20261007-v1`；输出：同父目录`runs/lift-two-gpu-b16-20261007-v1`。

后台命令：`<rlinf-venv>/bin/python -u -B <控制>/code/owner.py --plan <控制>/prepared/owner-plan.json owner`。准备、启动只各执行一次；不要重放旧from0-v2、旧B32入口或RLT旧归还。新namespace为`opendw_lift_b16_200_1007`，RLT候补cycle在`<控制>/cycles/cycle`。

阶段为CPU加载→精确借4/5→正式训练→完成/失败清理→RLT续训。CPU加载期间RLT继续；GPU6/7、其他用户和共享Ray不动。关键日志为`owner-console.log`、`formal/driver.log`、`services/wm5/records/service-events.jsonl`；轻量状态见[two_gpu_b16_light.json](two_gpu_b16_light.json)。

## 外层脚本整理

- 有历史复杂度：多次启动修补留下包装层、旧smoke断言和重复校验；它们不是多份模型或多个并行WM守护进程。11:56现场仅见RLT4/5和另一实验的Norm owner，无存活旧WM owner。
- 本次将原fresh wrapper的配置检查和绑卡注入合并到原owner文件，后台仍只有一个owner；不再加新的adoption/retry包装层。
- 删除未使用的旧smoke配置检查和对历史smoke回执的运行依赖；短启动器不重复完整导入校验。顶层源码冻结清单71→32，仅保留当前实际依赖；RLT各自的断点及历史归还证据仍由原生命周期模块保护。
- 保留进程PID/start/namespace、计算/图形卡位、WM卸载等待、阶段资源监控、故障清理和RLT自动归还。这些直接关系共享服务器资源与恢复，不能随意删除。
- 清理、资源采样、进程登记和生命周期加载函数与已运行版本AST一致；本次仅搬入既有scope注入并改B16验证。旧实验日志/检查点不删除，历史脚本作为证据保留，不启动。

## 昨晚失败定位

物理GPU5 OpenDW进程1190340，在VAE single_decode→F.pad第56批失败；之前55批B32完成。需要5.58GiB、剩2.05GiB，WM进程占76.45GiB。首末已完成批的存活张量同为23.207GiB，预留74.734→75.795GiB；显存余量过小明确，碎片是否为直接触发因素尚未独立验证。新B16正式运行仍需后续观察，不能先称长跑稳定。旧from0-v2无完整RL更新、无新CP，四卡历史CP10保留。
