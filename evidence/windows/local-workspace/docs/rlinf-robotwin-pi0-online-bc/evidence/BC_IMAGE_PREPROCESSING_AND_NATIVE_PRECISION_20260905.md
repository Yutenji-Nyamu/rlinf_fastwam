# BC 图像、增强与精度：来源、更正和当前选择

## 1. 什么是我们的方法必需的

成功BC只要求：当前策略收集 → 完整成功episode入池 → 决策前观察和实际提交动作chunk → 原生FM更新。图像增强不是方法定义的一部分。

| 阶段 | 实际行为 | 当前处理 |
|---|---|---|
| 保存 | 成功池保存原始三路RGB uint8 240×320、状态、文本、提交命令 | 不保存随机增强后的图片，不覆盖原图 |
| 模型输入 | π0原生尺寸调整、像素范围转换、状态/动作归一化和padding | 保留；这是模型输入合同，不能等同于随意增强 |
| 训练图像增强 | 官方OpenPI SFT内部train=True时随机裁剪、轻微旋转、颜色变化 | 用户09-05明确要求关闭；仅本BC配置关闭 |
| FM噪声/时间 | 对动作目标采样噪声与时间，学习流匹配向量场 | 保留；属于π0 FM训练目标，不是图像增强 |

环境domain randomization与SFT图像增强是两个开关。关闭增强不改变成功BC定义，但可能改变拟合和泛化，不能保证“效果完全一样”；该选择明确记录，后续BC+DVAC应沿用同一输入合同。

## 2. 错误是谁引入的

这是本次集成配置的问题，不是原始图像存错或成功BC必然带来的问题。我们显式设置model/FSDP precision=BF16；固定官方π0 DAgger实际是null，保留OpenPI原生BF16主干及FP32投影。FSDP被强制BF16后，原生训练路径的FP32增强网格、状态/动作投影输入与计算参数类型失配。

更正前面的口头表述：`precision_processor`只搬设备，并没有把图像转BF16；实际转型在FSDP入口。BF16/FP32是浮点数精度，不是图片文件格式。也不能从这个配置字段或通用warning推断“所有Adam状态实测均BF16”。

v2先在grid_sample失败；v3曾只恢复图像FP32，随后state_proj仍失败。这个图像cast补丁不完整，现已撤掉；当前恢复官方null精度，不逐个投影补cast，也不修改共享OpenPI依赖。

来源：[原生增强调用代码](SFT_DTYPE_BOUNDARY_20260905.txt:136)、[官方DAgger精度与优化器](PRECISION_PROVENANCE_20260905.txt:279)、[v3投影失败及真实model factory](FULL_SFT_BOUNDARY_20260905.txt:98)。

## 3. 干净关闭的方法

本分支`OpenPi0Config.image_augmentation`默认true，保持其他配置原行为；`_preprocess_observation`只门控原生预处理的train标志，不调用model.eval、不关闭梯度、不改变FM目标。BC YAML显式：

```yaml
actor:
  model:
    precision: null
    openpi:
      image_augmentation: false
```

FSDP三个dtype继续引用model.precision，故也为null；不是把全部权重强行变FP32。关闭增强时原生resize、mask、状态和文本处理仍在，推理本来就是train=False。

对应本地编辑副本：[模型开关](../../../../worktrees/pi0-online-bc/rlinf/models/embodiment/openpi/openpi_action_model.py:378)、[BC配置](../../../../worktrees/pi0-online-bc/examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml:119)。

## 4. 验证边界

回归覆盖：默认兼容、BC显式off、关增强不改变已达目标尺寸的像素/不消耗增强随机数、eval始终不增强、开启仍有增强、关闭时仍resize。9/9测试及配置/导入通过。真实模型/FSDP单卡诊断也已通过：MB32/GB1024，32次微批完成1次optimizer，FM均值0.04079、梯度0.5401，动作输出投影确有变化，冻结VLM无梯度；[原始结果](NATIVE_SFT_GPU_PROBE_20260905.json)。该诊断0环境、不保存权重，不能替代完整32并行容量smoke。

已推源码`9876c28de25bba429d10e2c8b4c85fb19b24c387`，远端核对一致；v4完整两轮smoke正在接续，实际结果以[本轮账本](GPU6_SMOKE_LEDGER_20260905.md)为准。
