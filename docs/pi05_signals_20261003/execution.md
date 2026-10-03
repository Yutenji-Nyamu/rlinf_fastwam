# 七信号＋DV：实现与深圳1执行记录

本页维护实际执行证据；方法定义以[12 v0.4](signal-plan.md)为准。更新日期2026-10-03，北京时间。

## 固定范围

- 七个新信号加DV；不运行训练，不改动作权重。
- 原DV50模型与源码基线 `d74e7edc93d346caa9758dadd4537197eeaf0081`，模型 `sidney-pi05-robotwin-e49e2ab`。
- 50任务×32条=1600条目标轨迹，100批；每批16并行环境，H50、主M10，U-GROW额外M5。沿原生任务步长上限与种子规则。
- DV/Fresco末5轮；SHIFT全层首末；Norm末5轮×深3层；SR末轮5动作邻域；GeoAAC前缀增长，首位置缺测。
- 物理4/5/6/7为授权范围；原监督链负责信号后恢复Dojo，再恢复原RLT。未验收前不把四卡写成已运行。

## 已有证据

服务器16项信号数学检查通过；合成数据验收器正常通过，故意改坏DV后能拒绝，恢复后再次通过。

首轮真实B16模型检查通过：记录开关前后，环境动作、x/z/t、完整32维去噪链完全相同；Python/NumPy/Torch CPU/CUDA随机状态完全相同。18层×10轮hook各一次。旁路时间从1完整走到0，5次求值。

第一次环境初始化失败：从旧深圳3环境快照沿用了不存在的 `/usr/share/vulkan/icd.d/nvidia_icd.json`。已确认深圳1实际文件为 `/etc/vulkan/icd.d/nvidia_icd.json` 并修正本进程环境。失败批、日志、清场凭据保留在attempts，不算有效轨迹。23:18左右开始第二次smoke。

第二次环境已正常初始化，随后记录器把原始相机分辨率误写死为224×224。已改为保存原生图像尺寸，模型自身的图像预处理不变。第二次失败也保留在attempts。

第三次smoke两个任务全部通过：turn_switch 16条、8次批量决策、7成功，用时327.36秒；adjust_bottle 16条、8次批量决策、11成功，用时305.28秒。32个视频可解码，八条分数逐项可复算，验收warnings=0。GeoAAC首位置NaN/无效是定义上的缺测，不算记录异常。C+G进程均在物理7卡。真实smoke已结束，不再追加重复试跑。

23:46正式队列已放行。6/7卡运行；4/5共用原owner，等待当前最后N36扩容档结束后自动接管。正式计划100批，已通过的两个smoke批直接计入，不重复采样。GPU6已完成N4/N9的扩容结果被保留，N16/25/36按冻结计划排在信号之后、原Dojo之前。随后仍由原owner恢复原RLT与next-six门槛。

当前服务器检查共34项：信号数值16、队列5、接管/逐卡恢复13；另有合成数据正常/篡改拒绝/恢复验收与真实模型/环境检查。尚未做新信号训练，不把记录成功解释为训练有效。

此前SZ1 RLT next-six已发布：[`3f35e89`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz1-pi05-rlt-next-six-20261002)。此前Dojo补齐46份代码、630份日志/JSON凭据、7篇文档；8份大日志只提供索引与hash，未声称上传原件：[`0834ff3`](https://github.com/Yutenji-Nyamu/robodojo_openwam/tree/codex/sz1-dojo-scope-scaling-20261003/docs/experiments/20261003_sz1_scope_scaling)。均已核对远端SHA。

## 路径与入口

实现：本地源码目录（原本地研究资料）。远端分支 `codex/sz1-pi05-signals-20261003`，独立worktree位于 `rlinf-shenzhen/worktrees/pi05-signals-sz1-20261003`。

结果根：`/srv/research/results/pi05-signals/20261003-v1`。八条分数、有效掩码、原始x/v/z/t、完整初噪声、hidden、norm网格、同噪声M5动作、观测与视频均按batch/query保存。

`done.json`只表示采集结束；`validation.json.passed=true`才表示数据验收通过。终止成功chunk内的实际物理动作前缀仍不可由旧TOPP接口精确反推，只保存可证实的提交掩码，不伪称全部H动作执行。

历史计时与原队列：18审计（原本地研究资料）。上次深圳3两卡1408条有效轨迹约9小时5分钟；本次四卡初估5–8小时，需用真实smoke速度校正。
