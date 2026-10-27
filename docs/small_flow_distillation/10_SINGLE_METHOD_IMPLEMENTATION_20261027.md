# RLinf单任务π0.5：五方法复用与短测

2026-10-27，用户授权：先完成无卡实现和检查，再暂时暂停1机6/7的多任务FQL，测试五种方法及适合的训练/评估并行；全部完成后恢复原FQL，最后接逐卡RLT。1机4/5现有直接蒸馏继续。本轮不启动单任务长训。

## 原始对照与模型接口

adjust_bottle原生老师N5/C50：111/128=86.72%；原始N1/C50：98/128=76.56%。各约20.5分钟，32环境×4批；配对128个官方held-out种子。差10.16个百分点，没有触发此前“差距<10点才追加C10”的条件。

保留该RLinf固定版本的`pi05_aloha_robotwin`工厂、原norm、delta关节/绝对夹爪变换、三相机、H50/C50和200动作上限。它与Sidney多任务的绝对动作接口不同，不能只换权重文件。当前单任务模型仍是完整π0.5，动作专家大小不会因任务少而变小。

## 复用与有意改变

|部分|依据与处理|
|---|---|
|模型、动作、观测|复用刚完成128回合原始评估的RLinf工厂/实配；所有训练动作均经过该模型的原生input_transform|
|直接端点|复用已跑通端点MSE；允许原生N5老师，学生N1，H50×真实14维；2.5e-5最高学习率、原AdamW/裁剪/衰减|
|CP|复用已验收FM适配，80时间格、EMA组合一致性加专家FM；两项仍按原实现H50×32；2.5e-5与EMA方案不另改|
|DMD new|复用新版本，学生1e-5、辅助2e-4、辅助5次/学生1次；学生计14维，辅助仍拟合32维|
|MFD|复用已验收FM积分实现，步长0.02、批量积分、辅助速度回归；其余同DMD new|
|FQL|复用已验收固定老师+双Q/目标Q。N5端点约束；Q宏动作由C10参数化为C50，gamma=.99、alpha=10、Q学习率3e-4、512×4与目标EMA=.005保持|
|数据|四种非Q方法取已有adjust_bottle 50条Clean示范、4738窗口；沿用冻结VLM，只训动作部分。FQL另采原生C50转移，不能把Sidney的归一化动作作为这里的Q输入|

方法出处和数学适配边界沿用[09](09_METHOD_COORDINATE_AUDIT_20261027.md)、[五方法依据](03_METHOD_ADAPTERS_AND_PARAMETERS.md)、[MFD实施](../pi05_distillation/66_MFD_IMPLEMENTATION_20261026.md)及[新DMD实测](../pi05_distillation/59_DMD_NEW_SMOKE_AND_FQL_HANDOFF_20261026.md)。没有加入新损失或新的机器人控制技巧。

## 短测、并行与停止条件

- 先在服务器CPU运行协议/梯度/转移边界测试。原多任务仍运行。
- 训练从已测并行起步：端点、DMD、MFD、FQL每卡192；CP每卡64（旧128曾OOM）。4步容量测量；有余量只加试一个有依据的更大档，比较窗口吞吐和峰值显存。
- 选定并行后，每方法12步保存→完整续训到14→原生N1导出；有限损失/梯度、冻结老师、单次动作专家调用都检查。CP仍导出EMA。
- 评估容量原定比较每卡32环境×2批与64环境×1批；64环境在reset渲染取图失败后，采用已完成的32环境一批，并取消无必要的第二批。有效32回合（27成功、5失败）共102条C50转移，仅供FQL功能短测；种子与全部128个held-out种子隔离。五种导出策略分别用双卡×32环境检查真实执行。
- 8线程解码、预取1批沿用多任务实测；不录视频、U、逐动作控制信号。
- 部署使用相关算法、训练、评估及模型子目录的完整源码集合，不能只覆盖本轮diff而漏掉新依赖；GPU交接前先实际运行两个trainer的无卡`--help`入口。
- 本次容量探测不写大权重；每方法仅保留验证后的完整CP14及导出，CP12在CP14落盘后淘汰。旧FQL的完整恢复点与已完成评估不改。
- MFD每卡192在第3步prefix分配时OOM：已分配38.11 GiB、保留未用36.76 GiB。仅对MFD重试`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，保持batch与算法；依据[PyTorch显存分配器说明](https://docs.pytorch.org/docs/2.9/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)。结果记在11，不将重试配置默认应用到原多任务FQL。
- 非有限数、非OOM错误或续训不一致停止本次suite并恢复旧FQL；只按精确PID/UID/start或唯一token清理自己的进程。

## 执行入口与产物

源码：`tools/pi05_distillation/single_task_prepare.py`、`single_task_suite.py`、`resume_existing_fql.py`。现有trainer默认仍是N10/C10；仅显式配置新的N5/C50路径。

1机部署`/data/chenyiteng/deployment-20261027/single-methods-sz1-g67`，产物`/data/chenyiteng/results/pi05-distill/single-methods-sz1-g67-20261027`。

入口：`python -u -B <source>/tools/pi05_distillation/single_task_suite.py <deployment>`。内部仍调用原双进程`torch.distributed.run`及既有resident评估器；计算卡与渲染卡都限定物理6/7。

新队列登记候补owner后才停止旧FQL，接续轮询0.25秒；结束先启动原FQL恢复进程、确认其登记，再退出临时owner。记录进程交接间隔，模型加载时间另计。原FQL恢复使用原源码、原配置、原完整状态和原评估进度，而不是重新采集或从零训练。

运行结果与吞吐另记[11](11_SINGLE_METHOD_SMOKE_RESULTS_20261027.md)。
