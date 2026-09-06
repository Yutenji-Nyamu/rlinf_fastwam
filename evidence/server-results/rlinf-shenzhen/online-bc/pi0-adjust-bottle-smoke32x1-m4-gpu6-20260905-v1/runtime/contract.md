# 已授权GPU6容量smoke：π0在线成功BC

## 1. 本次执行合同

用户已授权smoke、GPU6共用（liwenbo的评估在其他卡主计算，GPU6仅轻量上下文），GPU7预留。Fast保持停止，不迁移/终止他人有效评估或Sidney，不重启shared Ray、不升级依赖。

完整[resolved配置](GPU6_SMOKE_RESOLVED_20260905.yaml)已在服务器Hydra与validate_cfg校验，物理placement为[[6]]；[精确环境与命令](../../../local_scripts/bc_gpu6_smoke_wrapper_20260905.sh)。下列预算已向用户展示后按其本轮授权执行，不自动启动100轮长训。

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
| 资源策略 | bf16，no_shard/use_orig_params，actor/rollout offload；train/eval env不offload | 单卡冻结模型适配、复用已有offload；32+32场景可能共存，需要本次实测 |
| 资源/时长 | 一张H100 80GB；GPU6峰值/RAM未知；最多90分钟＝1.5 GPUh | 采集、FM、eval、save各阶段实际测量；不假定40–60GiB显存保证 |

正式候选仅改100轮、eval5/save10、optimizer total200及run名/路径；也是32×1＝每轮32条，共3,200尝试、最多200次更新、640评估和10代保存。正式从原SFT开始，不承接smoke已训练权重；本次不自动启动。

## 2. 精确入口和产物

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc/examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_online_bc_openpi
```

输出：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v1`。存在则拒绝启动。wrapper独立PGID，连接既有Ray 6389，RLinf独立namespace，显式代码工作树和绝对输出路径；不设全局CUDA_VISIBLE_DEVICES重编号，不加载Fast原生shim。

`driver.log`、`resource.csv`、`runtime/resolved.yaml`、`runtime/wrapper.sh`、`success_data/rank_0`、`exit_code.txt`及实验名下的checkpoint留服务器。无模型下载。轻量配置、源码、测试、指标与边界日志可进Git；不提交数据池、checkpoint、视频、完整Ray日志或他人进程信息。

## 3. 验收与停止

观察：真实训练参数范围、32环境实际启动、成功池episode/query计数、有限FM loss/grad norm、4次optimizer更新、权重同步后的第2轮、fixed评估与checkpoint实体、GPU6峰值/available RAM/PSI和其他任务状态。资源每5秒采样，不能冒称连续精确峰值。

正常两轮完成即停止；fatal/OOM/NaN不调小并发掩盖失败，不自动循环重试；总墙钟90分钟由timeout约束。若非加载/编译阶段15分钟无进展，执行者定位后只处理本任务。host内存压力或其他任务明显受影响时先停止本任务并报告。checkpoint恢复未实际做就不声称通过；无成功池则更新未验收。RoboTwin原生长期稳定性不由两轮smoke保证。
