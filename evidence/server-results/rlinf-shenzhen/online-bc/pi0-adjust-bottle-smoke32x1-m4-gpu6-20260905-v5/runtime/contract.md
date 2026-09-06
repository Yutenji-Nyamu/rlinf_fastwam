# 已授权GPU6容量smoke：π0在线成功BC

10:48补充：v1在init暴露资产根误设为`RoboTwin/assets`，未采样。已按源码及旧π0合同改为RoboTwin仓库根；v2只改资产根和新输出路径，所有模型/方法/并发/batch/评估预算不变。v1原始产物保留。

历史v3：曾仅恢复SFT图像FP32，随后原生state_proj仍遇Float/BFloat16错误。该不完整转换补丁现已撤掉。

当前v4：恢复官方DAgger的`precision:null`及对应FSDP null，保留OpenPI原生BF16主干/FP32投影，不再强制全模型入口BF16。用户最新明确要求关闭BC图像增强，故`openpi.image_augmentation:false`；仍保留resize/像素与动作归一化。采样32×1、M4、MB32/GB1024/U2及固定评估不变。先用v3已有成功数据独立验证一次真实FSDP更新，不采集、不保存或复用诊断权重，再从原SFT启动v4容量smoke。

v5接续：v4于11:48失败在训练后权重同步，未通过完整smoke。隔离真实原生SFT+Adam验证已通过连续2次MB32/GB1024更新、778键导出、offload/onload、local_shard保存及权重/Adam恢复。v5仅恢复use_orig_params=False，并通过既有wrap_policy匹配联合SFT实际调用的MLP/norm/qkv/o及投影；checkpoint_format显式local_shard、save/load对称透传。不改数据/模型/方法预算，不升级依赖。

v5输出为`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v5`，入口命令同下；最新完整resolved仍见同名文件，旧v4服务器runtime保留。GPU6仍32×1、micro32/global1024/U2、两轮、fixed32每轮、每轮保存、90分钟上限。v4采样峰值60.13GiB只覆盖其已运行阶段，未验证eval/save峰值；目前主机资源仍需启动前只读刷新。用户本轮已明确授权“放正式训练”，故v5完整通过且资源允许后，另从原SFT/空池新起100轮；不得复用smoke或诊断权重。

## 1. 本次执行合同

用户已授权smoke、GPU6共用（liwenbo的评估在其他卡主计算，GPU6仅轻量上下文），GPU7预留。Fast保持停止，不迁移/终止他人有效评估或Sidney，不重启shared Ray、不升级依赖。

完整[resolved配置](GPU6_SMOKE_RESOLVED_20260905.yaml)已在服务器Hydra与validate_cfg校验，物理placement为[[6]]；[精确环境与命令](../../../local_scripts/bc_gpu6_smoke_wrapper_20260905.sh)。下列预算已向用户展示；本轮新增正式授权，以v5完整通过为前提另从原SFT启动100轮。

| 项目 | 值 | 依据 |
|---|---|---|
| 源码 | 官方dc9b87c＋独立BC改动 | 不继承Fast/Sidney算法或渲染补丁 |
| 模型 | 已有adjust_bottle π0 SFT @92684e50 | 用户明确选择；模型config为gemma_2b＋gemma_300m、H50、内部D32、bf16 |
| 模型推理 | M4、C50、环境D14、三相机、原norm/映射 | M4继承官方model/pi0.yaml；官方π0 DAgger与旧同模型GRPO相同。M不是checkpoint不可变结构参数；无“4步蒸馏模型”的证据 |
| 更新范围 | action expert＋相关投影；VLM冻结 | 用户要求；native train_expert_only，启动核查实际参数 |
| 数据/方法 | 完整成功episode累计回放，query均匀有放回；w=0 | 成功过滤BC；DAgger只借监督更新，不取teacher标签；保留D0参数但本次不加载 |
| train并发与串行 | 单卡32环境×1批；2轮 | 用户要求单卡并发32、正式也每轮32条；smoke只缩总轮数 |
| rollout预算 | 64训练尝试，最多256新query、12,800指令action槽 | 每episode最多4次C50查询；不是物理插值步数 |
| FM更新 | micro32/global1024、U2；最多4optimizer更新、4096次query抽样 | MB/GB同官方π0 DAgger与旧GRPO；U2是本次明确预算，官方DAgger默认U1；无成功时跳过，不算更新通过 |
| optimizer | LR2.5e-5，Adam(.9,.95)、eps1e-8、wd1e-10、clip1 | 官方RoboTwin **π0** DAgger监督FM配置 |
| scheduler | constant、warmup0、total4 | 本次短预算；不复制官方1000warmup/30000total |
| 环境/行为 | adjust_bottle、horizon200、G1、无DR；原生随机初始噪声ODE | 任务同旧π0；无GRPO组内优势或中间SDE噪声；不改OIDN |
| eval/save | 每轮fixed32、共64评估；每轮保存、共2代 | 测试实际训练/评估并发与第二轮reset/sync/save，非成功率统计实验 |
| 精度/增强 | 官方precision:null，原生混合精度；BC图像增强off | 修正全局BF16类型冲突；增强off为用户最新选择，并非方法必需项 |
| 资源策略 | no_shard/use_orig_params:false；SFT叶模块wrap；local_shard；actor/rollout offload；train/eval env不offload | 单卡冻结模型适配、复用已有offload；32+32场景可能共存，需要本次实测 |
| 资源/时长 | 一张H100 80GB；GPU6峰值/RAM未知；最多90分钟＝1.5 GPUh | 采集、FM、eval、save各阶段实际测量；不假定40–60GiB显存保证 |

正式仅改100轮、eval5/save10、optimizer total200及run名/路径；也是32×1＝每轮32条，共3,200尝试、最多200次更新、640评估和10代保存。正式从原SFT开始，不承接smoke已训练权重；须先验收v5，再展示正式resolved/命令/资源后按本轮授权启动。

精度依据更正：OpenPI factory保留原生BF16主干和FP32投影，`actor.model.precision`并非将所有权重直接转换；此前显式BF16主要经FSDP param_dtype改变计算/输入精度。不能把该字段或框架通用warning直接当作“所有Adam状态均BF16”的实测。v4遵循锁定官方null配方；不另加AMP（官方配置也明确不应同时开启FSDP混合精度与AMP）。

## 2. 精确入口和产物

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc/examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_online_bc_openpi
```

输出：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v4`。存在则拒绝启动。wrapper独立PGID，连接既有Ray 6389，RLinf独立namespace，显式代码工作树和绝对输出路径；ASSETS_PATH指向RoboTwin仓库根，不是assets子目录；不设全局CUDA_VISIBLE_DEVICES重编号，不加载Fast原生shim。

`driver.log`、`resource.csv`、`runtime/resolved.yaml`、`runtime/wrapper.sh`、`success_data/rank_0`、`exit_code.txt`及实验名下的checkpoint留服务器。无模型下载。轻量配置、源码、测试、指标与边界日志可进Git；不提交数据池、checkpoint、视频、完整Ray日志或他人进程信息。

## 3. 验收与停止

先行精度诊断：`local_scripts/bc_native_sft_gpu_probe_20260905.py`经`sz_bc_native_sft_probe_stdin_20260905.sh`在GPU6运行：同模型/FSDP配置、MB32/GB1024，只有1个optimizer更新，0环境/0策略采集，不保存模型；输出`.../online-bc/native-sft-probe-20260905/result.json`，最多20分钟，任何非有限值/OOM/异常即退出。GPU无场景负载，显存测量不能冒充完整smoke容量；它验证完整真实FM前后向，不是继续逐层补cast。

观察：真实训练参数范围、32环境实际启动、成功池episode/query计数、有限FM loss/grad norm、4次optimizer更新、权重同步后的第2轮、fixed评估与checkpoint实体、GPU6峰值/available RAM/PSI和其他任务状态。资源每5秒采样，不能冒称连续精确峰值。

正常两轮完成即停止；fatal/OOM/NaN不调小并发掩盖失败，不自动循环重试；总墙钟90分钟由timeout约束。若非加载/编译阶段15分钟无进展，执行者定位后只处理本任务。host内存压力或其他任务明显受影响时先停止本任务并报告。checkpoint恢复未实际做就不声称通过；无成功池则更新未验收。RoboTwin原生长期稳定性不由两轮smoke保证。
