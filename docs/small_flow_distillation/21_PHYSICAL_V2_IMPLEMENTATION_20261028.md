# CP v2 / DMD v2 / MFD v2：独立版本实施

2026-10-28。用户授权实现三个独立版本、使用靠后空卡做smoke、推送，完成后停止。实现自`884845328`起，分支`codex/pi05-physical-v2-20261028`。正式部署目录与当前4–7训练完全分开。

## 本次变化

|版本|内部方法|学生目标|辅助/其他目标|
|---|---|---|---|
|`cp_v2`|CP|一致性只比H50×14|专家FM只比H50×14；EMA仍是学生平滑副本|
|`dmd_v2`|OneDP-S的现有DMD new|维持H50×14指导|辅助epsilon回归由32改14|
|`mfd_v2`|现有MFD|维持H50×14平均速度指导|辅助velocity回归由32改14|

只评分真实坐标，不裁H50时间长度。内部仍生成32维，老师积分和MFD两条积分仍保留32维；不逐步清零、不替换填充轨迹。不新增dropout，本轮三个版本只处理评分范围。

## 精确代码范围

- `rlinf/algorithms/pi05_distillation/variants.py`：三个独立名称解析成原方法，要求`loss_coordinate_scope=physical`和已验证14维接口。历史名字继续走旧评分方式；新名字遇到冲突维数时在训练前报错。
- `losses.py`：`action_coordinates`在求平均前选真实维；CP一致性/FM和DMD辅助增加可选参数，缺省保留旧目标。CP原距离形式保持，delta按H50×14计算，约0.01429。
- `mfd.py`：辅助真正裁loss，而非仅分组日志。短测计时模式额外记录两条积分的真实/填充位置、速度最大值及填充起终点RMS；复用已有前向，不增加网络调用。
- `tools/pi05_distillation/train.py`：传递真实评分范围，resolved注明学生/辅助范围；v2另记裁剪比例。保存完整配置/模型/优化器/各rank RNG；旧恢复校验保持严格。
- `physical_v2_smoke.py`：复用原`run_queue.train/accept_smoke`，三方法依次12→14，验证原生N1导出后退出。没有正式训练或RLT入口。
- `test_physical_v2.py`：物理坐标归一化、CP两项目标、EMA间接梯度、两辅助隔离、积分诊断不改计算，以及版本/续训边界。

导出仍记录原方法名（CP/OneDP/MFD），另加`variant`标明v2。未来接入仿真评估时，CP v2的模型配置仍须`distill_target_time=true`，不能仅按新字符串`cp_v2`与`cp`相等判断；本轮已用正确模型结构检查权重导出和原生N1推理，未新增正式评估队列。

Direct和FQL已经使用真实14维，本轮未修改它们。未修改共享OpenPI/Gemma环境、其他部署源码、正在运行的训练配置或RLT登记。

## 参数与来源

使用原生RLinf `adjust_bottle`单任务π0.5，复用已跑通的归一化/动作接口、4738个完整窗口和固定数据；不换成Sidney多任务老师。代码本身支持既有多任务数据路径，本轮实际GPU短测使用最近验收的单任务配置。

|项|CP v2|DMD v2|MFD v2|
|---|---|---|---|
|每卡microbatch / 全局batch|80 / 160|192 / 384|192 / 384|
|学生学习率|峰值2.5e-5、旧预热/余弦|固定1e-5|固定1e-5|
|辅助学习率|EMA，无辅助优化器|2e-4|2e-4|
|辅助:学生更新|EMA按旧计划|5:1|5:1|
|H / 推理检查C|50 / 50|50 / 50|50 / 50|
|短测|12步→完整恢复至14步|同左|同左|

batch与优化器来自相应已验收实配。配置中保留原100预算轮以保持学习率计划；实际入口有`--stop-step 12/14`，不会跑满该预算。供数继续8线程、预取1批。CP 80、DMD/MFD 192已有容量依据，损失裁维不改变主要激活规模，不重复做从小batch起的容量爬坡。

DMD v2首试在第3步构造视觉前缀时OOM：已用38.11 GiB、缓存未用36.77 GiB，申请5.67 GiB失败。只在独立DMD v2配置启用现有启动器支持的`cuda_allocator_conf=expandable_segments:True`重测，batch、数学目标和学习率保持。该选项针对显存分配碎片，依据[PyTorch官方说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)。旧日志和失败配置单独留档；非自动缩batch。

## 本轮资源与命令

1机2/3卡经计算/图形进程检查为空；4–7有任务，未占用。2/3机本地SSH问候前断开，经1机TCP超时，未进入认证，内部状态未知。新增大产物放Home：

- 部署：`/home/chenyiteng/deployment-20261028/physical-v2-sz1-g23`
- 独立源码：部署下`source/`
- 输出：`/home/chenyiteng/results/pi05-distill/physical-v2-sz1-g23-20261028`
- 每方法仅保留最近1个本次smoke完整点，最终14步含优化器/RNG；不引用或清理其他实验输出。

调用：`python tools/pi05_distillation/physical_v2_smoke.py <deployment>`；末尾可列方法名只补测失败项，合并既有验收回执。重测前保留失败输出和日志，入口不覆盖已存在的段日志。内部沿用既有双rank启动，`CUDA_VISIBLE_DEVICES=DISTILL_GPU_PAIR=2,3`；本轮是离线训练与数组推理，不创建模拟器或渲染器。每段启动前再次确认选定卡无人占用，异常只记录本方法失败并尝试后续空闲段，禁止改卡、缩batch或重启其他任务。

## 验证状态

服务器CPU检查40项通过，Ruff格式/检查与训练CLI通过；同时覆盖旧损失默认、旧MFD以及旧完整恢复边界。GPU验收结果见[22](22_PHYSICAL_V2_SMOKE_RESULTS_20261028.md)，未完成前不把“已实现”写成“短测通过”。

设计依据：[19源码审计](19_UNIFIED_PHYSICAL_LOSS_CODE_AUDIT_20261028.md)、[20计划](20_UNIFIED_PHYSICAL_LOSS_PLAN_20261028.md)、[11既有单任务短测](11_SINGLE_METHOD_SMOKE_RESULTS_20261027.md)。14维只是统一目标口径，smoke不证明成功率提高。
