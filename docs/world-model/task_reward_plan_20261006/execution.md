# 单任务奖励模型执行 · 2026-10-06

**14:15实查：v2已于14:10完成，error/recovery_error均为null，GPU4已归还且原RLT新driver身份匹配存活。128回合56成功/72失败；采集82.36分钟，训练阶段26.51秒（训练器内部5.34秒）。新RM测试TP8/FN3/FP1/TN196；OpenDW验证待做，不代表新任务WMRL已启动。** 详细结果、提速候选和下一步见[结果记录](results_20261006.md)，后文保留执行来历。

## 当前路由

用户授权在深圳3实现、实验、训练新任务RM；可停旧WM，RLT低优先候补。旧 `click-bell-v2` 已在完整CP50（两份shard与full_weights ZIP核验）后精确SIGTERM；原finally清理完成、归还RLT4–7，`recovery_error=null`。旧owner记录signal15为本次主动停止，不是新的训练故障。

当前新任务目录 `/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/task-reward-v2`；入口 `code/rm_owner.py`、计划 `prepared/plan.json`、输出 `run/`。只借GPU4；5–7卡既有RLT身份保持。每一阶段超时/失败或全部结束时，按原完整checkpoint和精确进程回执归还GPU4。不得重放旧WM启动或另建RLT协调器。

v1已完成摆瓶子pilot，native启动时发现继承的 `RLINF_CODE_WORKING_DIR` 指向无capture hook的旧bell仓库；12:38主动停止，12:39:03归还GPU4完成，`recovery_error=null`，5–7原driver保持。没有计入有效抬锅采集。v2只修代码同步目录与日志验证路径，继承相同采样/模型/预算，不重复pilot；v1完整结果保留在 `task-reward-v1/pilot`。

当前顺序：已完成摆瓶子32回合训练pilot → 原SFT抬锅128回合采集 → 采集核验与数据转换 → 抬锅RM训练/留出评估 → GPU4归还RLT。实际启动与完成以同目录轻量结果的时间和回执为准。

## 模型与输入输出

- 架构：ImageNet ResNet18，512→256→1分类头、ReLU/Dropout0.1；约11M参数，远小于8B视频模型。
- 输入：主相机RGB，uint8或[0,1]float批图，NCHW/NHWC；统一224×224与ImageNet归一化；`compute_reward`接受图像或RLinf的`main_images`字典。
- 输出：`compute_reward(images, instructions)`返回[B]成功概率，仍可按固定阈值变0/1。单任务权重忽略文字，保留调用参数；不能直接加载旧ResNet＋T5权重。
- 训练：AdamW，学习率1e-4、weight decay1e-5、BCE、global batch64；FP32。显存缓存预处理图，实测micro32/64后选更快者；推理测128/256。microbatch改变BatchNorm统计，明确记实际配置。
- pilot最多20epochs/patience8；新任务最多100epochs/patience15，以验证损失选择权重。验证集独立选阈值，优先限制整回合提前误报，再看召回；测试只在选择冻结后运行。

代码沿WorldArena/RLinf的纯ResNet结构和默认训练配方做轻量单卡实现，不称完整复现发布的文字RM。旧运行仓库及旧WM源码不变。

## 新任务和数据

首任务 `lift_pot`，继承原生Clean/三图π0.5/H50执行C32/384动作；原SFT权重，不使用RLT或旧按铃WM的策略。N16×8=128回合。使用既有训练seedbank，排除原有20个eval种子；不重跑专家筛种子，不为了凑50/50增加采集。既有seedbank的专家筛选来源保留在元数据，它不表示当前策略必成功。

原生成功条件：锅高度>0.82m、左右末端分别距把手<0.03m、锅朝向指标>0.8。主图能观察抬高和姿态，但视觉模型是否精确判断把手距离仍靠留出数据检查。

每C32保存原生主图、success、动作时刻、seed与首次成功位置。未知reset标签排除；只保留首次成功及之前，不用锁存后的全部画面当正例。失败回合必须执行完整384动作。按task+seed成组划分，近似60/20/20；训练帧负正约2:1，验证/测试保留全部可用观察。

采集验证检查128条完整记录、计划seed覆盖、重复UID、初始图重复、图像标签哈希和重试日志。recorder记录的是请求seed；旧仿真reset有不稳定时自动换seed逻辑，故只在没有重复初图/重试证据时继续，并不宣称单独记录了实际seed。

## 已完成验证与结果边界

- 服务器CPU的5个针对性测试通过：reset/锁存标签、K8索引去重、按seed隔离、提前误报阈值、checkpoint严格加载和批量输出。
- 摆瓶子32回合转换成功：训练16回合92帧（8正）、验证8回合50帧（3正）、测试8回合48帧（4正）。小样本只用于工程验证。
- 摆瓶子pilot20epoch/20update：纯训练5.18秒，含加载/调度整阶段41.10秒；best epoch18。micro64每64张前反向12.65ms，micro32为15.20ms，约缩短17%；选择micro64和推理batch256。profile Torch分配峰值约1.56GiB，小模型单卡足够，无需第二卡。
- pilot选出的阈值>1，召回为0，表示在此小验证集的假成功限制下没有找到可用成功阈值。即使test AUC约0.994，也不能上线。这里验证了工程链，不用排序AUC代替成败判定。
- 新任务采集、GPU profile及训练结果由 `light_results.json` 更新；尚未有结果的阶段不得说成通过。
- 留出原生图上的准确度不能直接代替OpenDW生成图准确度；本次RM完成也不等于新任务WMRL已启动。

## 输出与发布

每阶段：`run/<phase>/command.log`、`command-result.json`；GPU资源：`run/resources.jsonl`与心跳。模型目录保存 `best.pt/config.json/profile.json/history.csv/report.json` 和预测明细；原生捕获留服务器。Git仅推审过的源码、配置说明、轻量指标及回执，不推权重、NPZ、完整日志、环境文件或凭据。
