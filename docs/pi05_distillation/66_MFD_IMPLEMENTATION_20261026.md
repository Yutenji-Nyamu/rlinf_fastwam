# MFD：π0.5实现、依据与短验收

2026-10-26。继承DMD new `0fd7dcf11`，独立`method=mfd`；本轮仅短训练、完整续训和一步导出，不启动正式训练。方法从[64号规划](64_MFD_REFINED_PLAN_20261026.md)落实。

## 改动切口

- `rlinf/algorithms/pi05_distillation/mfd.py`：时间采样、两路批量Euler积分、学生指导、辅助速度回归。
- `tools/pi05_distillation/train.py`：接入MFD；复用已有模型、DDP、供数、5:1、优化器、保存恢复与导出。CP/OneDP/直接蒸馏原损失不变。
- `prepare_mfd.py`：从已验收DMD new实配生成独立配置，核对关键参数；不覆盖已有配置。
- `mfd_smoke.py`：复用已有pair launcher与验收器，仅12→14；脚本没有正式训练或RLT启动分支。
- `test_mfd.py`：独立解析例子验证方向、批量积分、短区间、梯度归属、物理维掩码、采样和配置继承。

## 一次更新

1. 按任务均衡抽384个观察窗口，两rank各192；8线程解码、预取1批。同一批的冻结视觉语言前缀只计算一次。
2. 学生参数暂时不变，辅助连续更新5次。每次学生用新噪声生成动作，另采噪声和均匀时间，辅助预测`noise - generated.detach()`。损失拟合完整32维，日志分14真实维和18填充维。
3. 学生重新生成动作并加噪。老师与辅助从同一个含噪起点出发，各沿自己的速度场积分到结束时间。二者共享观察前缀，各自状态独立。
4. 学生损失为`mean(stopgrad(u_teacher-u_aux) * x_t)`，只计H50×14。梯度只沿`x_t→学生动作→学生参数`，不经过两条积分，也不更新老师或辅助。
5. 更新学生一次。记录指导RMS、积分步数、两优化器梯度范数和裁剪；短计时模式再记录输出层的相对参数改变量。这里是输出层指标，不能称全网络更新范数。

## 每项依据

|设置|采用值与依据|
|---|---|
|老师/学生/辅助|既有Sidney π0.5动作专家，冻结VLM及老师；复用DMD new的三个模型和前缀缓存|
|动作|内部H50×32，学生只计14维；评估执行C10，原生N1输出H50×14|
|batch/供数|每卡192、全局384；8线程、预取1批，来自已验收DMD new容量与计时|
|学生/辅助优化器|继承DMD new：AdamW/Adam，学习率1e-5/2e-4，beta=(0,.999)，clip=1|
|更新比例|辅助5次、学生1次；用户已接受的DMD new对照设置，不称作者原版比例|
|学生目标|官方MFD平均速度差与含噪动作内积；不保留OneDP额外`t(1-t)`权重|
|辅助目标|官方直接速度MSE；不保留OneDP epsilon目标的额外时间加权|
|时间方向|π0.5与SANA实现均为1噪声→0数据；用实现的速度符号，未直接照搬论文相反坐标|
|时间采样|两个独立连续Uniform时间排序的单样本分布；采用下面的合批适配|
|积分|Euler，最大步长0.02，来自官方默认；无梯度，平均速度/时间/位置FP32，模型沿用BF16计算|
|短验收|12步保存→同配置续到14→加载实际导出、原生N1；沿用已成功DMD验收链|
|配置预算|保留20预算轮用于学习率/预算上下文；本轮launcher明确在14步停止，不运行完整预算|

## 批量时间与数值适配

官方脚本有`.item()`和batch=1断言，直接搬到192会失败。先按`P(K=k)=(1-(k-1)h)^2-(1-min(kh,1))^2`抽全批K，两rank广播同步；每个样本再在该长度档内独立采起止时间。每个样本仍保持原连续均匀排序分布，样本之间因共享K而相关。这样整批合算，无需循环192个样本或补齐到50次。

每条路径使用等距K步。因此`mean(各步velocity)`就是Euler端点位移除以区间长；实现直接累计前者，避免短区间的两个大端点相减、除以很小时间。零区间自然退化为当前位置速度。两路位置均保持FP32；不积累各步反向图。

SANA的时间乘1000再取整、文本CFG=4.5、图像输出和LoRA适配器不搬入π0.5。π0.5已有连续时间和原生条件动作接口。没有新增JVP、双时间网络、判别器、专家动作FM或Q。

## 资源、命令与停止边界

独立部署目录和独立输出目录；`prepared.json`明确授权的两个物理GPU，沿用既有launcher同时检查占用。纯离线训练和原生数组推理不启动模拟器或渲染器。运行前复核计算与图形进程，完整保存至少预留35GiB。

```bash
python tools/pi05_distillation/prepare_mfd.py --base-config <accepted-dmd-new.json> --output <deployment>/mfd.json
python tools/pi05_distillation/mfd_smoke.py <deployment>
# 内部：torchrun --standalone --nproc_per_node=2 train.py ...
# --stop-step 12；然后 --resume checkpoint-000012.pt --stop-step 14
# 两段均 --decode-workers 8 --prefetch-batches 1 --profile-stages
```

OOM、非有限损失/梯度、配置或恢复不匹配均失败退出，不自动缩batch、改学习率或换到其他卡。最终须有两rank的一步导出回执、14次学生/70次辅助更新、有限指标与正确物理卡绑定；完成后进程退出。性能以实际计时为准，不用34/2推算整轮放大倍数。

阶段计时中`mfd_guidance`包含于`student_update`，二者不能重复相加。稳态每步应相加`data_wait + observation_prefix + auxiliary_updates + student_update`。启动和保存另计。

## 来源

- [MFD正式论文](https://proceedings.mlr.press/v306/zhao26bx.html)。
- [官方固定源码](https://github.com/happyw1nd/MFD/blob/bd882924fe24ce59fac0e225f7c82db10db29cb8/t2i_SANA/scripts/train_sana_mfd.py)：辅助速度MSE、平均速度积分、学生内积指导、步长与时间调用。
- [既有DMD new验收](59_DMD_NEW_SMOKE_AND_FQL_HANDOFF_20261026.md)：模型、batch、供数、5:1与恢复链依据。

本轮结果另记于轻量验收回执；短smoke不提供成功率提升结论。
