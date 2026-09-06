# H100 80GB 与 A800 80GB：官方规格口径和可复现 benchmark 计划

> 日期：2026-08-22。本文只做规格审计和实验设计，不代表已经运行 benchmark。深圳现场数据由本轮
> 只读硬件检查提供；AutoDL 型号来自既有现场账本。正式比较前仍需在两台机器各自重新锁定动态状态。

## 1. 先给结论

1. 本项目应比较的实际形态是 **深圳 8×H100 80GB HBM3、完整 NVSwitch/NVLink fabric** 与
   **AutoDL 2×A800-SXM4-80GB**，不是 H100 PCIe 对 A800 PCIe。
2. 以两张实际卡型的**稠密峰值**比较，H100 SXM 相对 A800 SXM 的官方理论上限约为：FP32
   `3.44×`、TF32/BF16/FP16 Tensor `3.17×`、HBM 带宽 `1.64×`、每 GPU NVLink 总带宽
   `2.25×`；显存容量同为 80 GB。
3. 这些比值只描述不同资源的峰值包络。π0/Fast-WAM 的大矩阵更可能接近计算收益，FSDP collective
   更受 NVLink/NVSwitch 影响；RoboTwin reset/render/primitive step 主要是 simulator/CPU/同步路径，
   不能用 `3.17×` Tensor 峰值外推端到端加速。
4. H100 的 FP8 不是当前 BF16 π0/Fast-WAM 自动获得的收益。只有显式启用 Transformer Engine/FP8、
   固定 recipe 并验证精度后，才可把 FP8 作为 H100-only 特性测试；A800 无原生 FP8，不制造
   “A800 FP8”对照。

## 2. 实际硬件锁定与规格口径

### 2.1 两台机器实际是什么

| 机器 | 已有现场事实 | 本文采用的规格形态 | 边界 |
|---|---|---|---|
| 深圳 | 8 卡均报告 `NVIDIA H100 80GB HBM3`、81559 MiB、device ID `0x2330:0x10DE`；任意 GPU 对在 `nvidia-smi topo -m` 均为 `NV18`，每卡 Link0–17 均 active、速率 26.562 GB/s；GPU0–3 属 NUMA0，GPU4–7 属 NUMA1 | H100 SXM / HGX NVSwitch 形态 | `nvidia-smi` product name 本身没有写 `SXM`；这是由 device ID、18 links 和全互联拓扑共同支持的形态判断，不把服务器厂商/具体 HGX 型号一并猜出 |
| AutoDL | 历史账本精确报告 `NVIDIA A800-SXM4-80GB` | A800 80GB SXM4 | 只锁定了卡型；正式通信对比前仍需现场重取 topology、link 状态、NUMA 和 power limit |

[NVIDIA supported-chips 表](https://download.nvidia.com/solaris/570.133.07/README/supportedchips.html)
将 `0x2330` 列为 `NVIDIA H100 80GB HBM3`，而 H100 PCIe 是 `0x2331`；
[NVIDIA `nvidia-smi` 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html) 中 `NV#` 表示由该数量
NVLink 组成的 bonded path。深圳的 `NV18` 与 18 条 active link 因而支持 SXM/NVSwitch 形态判断，
但这里不把 `26.562×18` 当作实测应用带宽；带宽必须由 NCCL/nvbandwidth 实测。

### 2.2 实际形态的同口径稠密峰值

| 指标（每 GPU） | 深圳 H100 SXM | AutoDL A800 SXM4 | H100/A800 | 口径 |
|---|---:|---:|---:|---|
| FP32（非 Tensor） | 67 TFLOPS | 19.5 TFLOPS | 3.44× | 均为官方未标 sparsity 的峰值 |
| TF32 Tensor | 494.5 TFLOPS | 156 TFLOPS | 3.17× | 稠密；H100 由官方 `989*` 除 2，A800 取表中第一项 |
| BF16 Tensor | 989.5 TFLOPS | 312 TFLOPS | 3.17× | 稠密 |
| FP16 Tensor | 989.5 TFLOPS | 312 TFLOPS | 3.17× | 稠密 |
| FP8 Tensor | 1,979 TFLOPS | 不支持原生 FP8 | 不作比值 | H100 稠密；只有显式 FP8 路径才有意义 |
| HBM 容量 | 80 GB HBM3 | 80 GB HBM2e | 1.00× | 容量相同，不产生单卡“能装更大模型”的优势 |
| HBM 峰值带宽 | 3.35 TB/s | 2.039 TB/s | 1.64× | 规格峰值，不等于 kernel 实测 |
| 每 GPU NVLink 总带宽 | 900 GB/s | 400 GB/s | 2.25× | 规格总带宽；实际 collective 受拓扑、消息大小和 NCCL 影响 |
| Host 接口 | PCIe Gen5，128 GB/s aggregate | PCIe Gen4，64 GB/s aggregate | 2.00× | 双向 aggregate；不能和单向实测速率混写 |
| 功耗规格 | up to 700 W、可配置 | OEM SXM 形态依系统而异 | 不作比值 | A800 来源中的 500 W 是 Lenovo 水冷 4-GPU board，不自动等于 AutoDL power limit；benchmark 使用各机现场 stock default |

H100 行来自 [NVIDIA 当前 H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)；该页 Tensor
数值带 `* With sparsity`，所以表中只对带星号项除以 2，FP32 67 不除。A800 行来自 Lenovo 的
[A800 官方 OEM 产品指南](https://lenovopress.lenovo.com/lp1813-thinksystem-nvidia-a800-pcie-gpu)：其表把
稠密/结构化稀疏写为第一项/第二项，并给出 SXM 的 2,039 GB/s、400 GB/s NVLink。Lenovo 同页正文明确
为 HBM2e，规格表中的 `HBM2` 简写不据此改成另一种实物。

### 2.3 为什么不能把不同 H100 表格拼在一起

NVIDIA 当前网页的第二列是 **H100 NVL 94GB**，不是 H100 PCIe 80GB，不能拿它补 PCIe 数据。若只为
理解 form factor 差异，可使用 NVIDIA 的 [2022 H100 SXM/PCIe 数据表](https://dam-cdn.nvd.orangelogic.com/AssetLink/mfj81tsm68n0ne632upmuvirso3ta3g3.pdf)：

| 2022 官方数据表快照（每 GPU，稠密） | H100 SXM 80GB | H100 PCIe 80GB |
|---|---:|---:|
| FP32 | 60 TFLOPS | 48 TFLOPS |
| TF32 Tensor | 500 TFLOPS | 400 TFLOPS |
| BF16/FP16 Tensor | 1,000 TFLOPS | 800 TFLOPS |
| FP8 Tensor | 2,000 TFLOPS | 1,600 TFLOPS |
| HBM | 80 GB / 3 TB/s | 80 GB / 2 TB/s |
| NVLink | 900 GB/s | 600 GB/s（仅相邻卡 bridge） |
| PCIe | Gen5 / 128 GB/s aggregate | Gen5 / 128 GB/s aggregate |
| TDP | 700 W | 350 W |

这张表只作为**同一发布日期内的 SXM/PCIe 结构对照**；深圳实际 H100 的主表采用当前产品页 shipping
数值，不把旧的 60/500/1000 与新的 67/494.5/989.5 混成一个 SKU。NVIDIA 的
[H100 PCIe product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf)
另确认 PCIe 卡是 device ID `0x2331`、80 GB HBM2e、2,000 GB/s、350 W，且最多只 bridge 相邻两卡。

A800 也要区分 form factor：同一 Lenovo 表中 PCIe/SXM 的稠密 compute 相同，但 HBM 带宽是
1,935/2,039 GB/s，PCIe 卡 300 W、其特定水冷 SXM board 500 W；两者都列 400 GB/s NVLink，但
PCIe 只通过双卡 bridge，SXM 才能部署在 HGX/NVSwitch board。**卡名不能替代系统 topology 实测。**

## 3. 对当前工作负载意味着什么

### 3.1 π0 / Fast-WAM 模型计算

- BF16/FP16 大 GEMM 的理论上限约 3.17×，HBM-bound kernel 的上限更接近 1.64×；真实模型会落在两者
  之间或更低。batch 1、小矩阵、三路图像预处理、VAE、逐步 denoising、Python/launch/synchronization
  都会压低 Tensor Core 利用率。
- 两卡都为 80 GB，所以 H100 的主要收益是吞吐/延迟和带宽，不是容量。π0 `B=1` 延迟与大 batch
  throughput 必须分别测；不能用大 GEMM 结果替代 policy query。
- Fast-WAM/π0 当前共同主比较固定 BF16。H100-only FP8 是第二张特性表，必须记录 Transformer Engine
  recipe、实际选中 kernel、量化/反量化开销和精度；不改变当前 baseline dtype 来追求规格数字。

### 3.2 FSDP 与多卡通信

[PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) 在 forward/backward 周围涉及 parameter
all-gather 与 gradient reduce-scatter。H100 的 900 GB/s NVLink 和完整 8-GPU NVSwitch fabric 对这类
collective 的强扩展更有利，A800 每 GPU 400 GB/s 是更低的通信上限。

但这不等于 FSDP 固定快 2.25×：小消息看 latency，较大消息才接近 bandwidth；NCCL algorithm、rank
数量、bucket size、计算/通信 overlap 也会改变结果。RLinf 若有 CPU offload、CPU-staged weight sync
或 host copy，其瓶颈还经过 PCIe 和 NUMA，不能只看 NVLink。深圳 GPU4–7 同属 NUMA1，四卡测试应把
CPU affinity 固定在 NUMA1；跨 NUMA 测试另列，不和同 NUMA 主结果混合。

### 3.3 RoboTwin simulator 与端到端 PPO

现有深圳 formal 的同配置现场证据已显示 `generate_rollouts≈1506–1517s`，actor training 约
`23s`；参见 [`11_FORMAL_CONFIG_AND_MEMORY_ANALYSIS.md`](11_FORMAL_CONFIG_AND_MEMORY_ANALYSIS.md)。
主要时间在 RoboTwin reset/render/primitive interaction 和同步等待，而不是 optimizer。于是：

- 更快 H100 可以缩短 policy query/训练段，却不会按 Tensor 峰值同比缩短 SAPIEN/CuRobo/CPU/renderer；
- 只有把 model-only、collective、simulator-only、episode total 分层计时，才能解释端到端差异；
- AutoDL 历史 step 与深圳 step 的 env 数、rollout wave、CPU、renderer、软件版本不同，不能拿旧日志直接
  算硬件比。端到端跨机器结果是**整机系统 benchmark**，不是 GPU-only benchmark。

## 4. 可复现 benchmark 设计（本轮不执行）

### 4.1 共同控制与一次性硬件锁定

两机使用相同 source/checkpoint/assets、相同 CUDA/PyTorch/NCCL/Transformer Engine 版本和相同 benchmark
commit。每个 run 前保存：

- `nvidia-smi -L`、`nvidia-smi -q -x`、`nvidia-smi topo -m`、`nvidia-smi nvlink -s`；
- UUID、PCI/device ID、MIG/ECC、driver、GPU/SM/memory clocks、P-state、temperature、当前/默认 power
  limit、active throttling reason；同时保存 `lscpu`/`numactl -H`；
- GPU 无其他 compute process、MIG 关闭；CPU thread 数和 affinity 固定。深圳主四卡固定 GPU4–7 +
  NUMA1；AutoDL 先根据 live topology 选择对应的同 NUMA pair。

主结果让每张卡维持自己的 **stock default power limit**，在全部重复中不变，并报告实测 median/p95
power 与 clocks；不把 H100/A800 强制成同瓦数。若以后需要能效对照，再另做等功耗副实验。所有输入使用
固定 seed 的非零随机值；GPU kernel 用 CUDA events 并在每次 sample 前后 synchronize，端到端用
monotonic wall clock。除下文特殊说明外，每 case 为 20 次 warmup、100 次 measured、5 个 outer
repeat，汇总全部 measured samples 的 median/p95；吞吐和延迟同时报告。

### 4.2 分层矩阵

| 层 | 固定 case | 工具/同步 | 报告 |
|---|---|---|---|
| GEMM | dense `m,n,k`=`8192,8192,8192`、`2048,4096,4096`、`128,4096,4096`；FP32 CUDA-core、TF32/BF16/FP16 Tensor | [NVIDIA CUTLASS profiler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/profiler.md)，固定 op class/layout；30 warmup + 100 measured × 5 | latency median/p95、TFLOP/s、相对**对应稠密规格**的效率 |
| Transformer | `B=8,S=512,H=2048,heads=16,FFN=8192`；BF16，forward 与 forward+backward 分开 | 同一 PyTorch/[Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/getting_started/index.html)，记录实际 attention backend | latency median/p95、tokens/s、peak VRAM；H100-only FP8 另表 |
| HBM | local SM read/write，固定 512 MiB（再补 2 GiB 大 buffer），20 internal samples × 5 | [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth/blob/main/README.md)，先 `-l` 锁定当前版本 test name | read/write GB/s median/p95；不把 read+write“有效字节”混成 HBM 理论值 |
| NCCL all-reduce | 公平主比较 `N=2`，消息 `1 MiB / 64 MiB / 1 GiB`，BF16；H100 再做 N=4/8 形态表 | [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)，默认 NCCL 自动拓扑算法，20 warmup + 100 iter × 10 | latency median/p95、`algbw`、`busbw`；all-reduce `busbw=algbw×2(N-1)/N` |
| π0 model-only | 同一 checkpoint/norm/preprocessed GPU-resident observation；BF16；`B=1` 与 `B=32`；固定 H/C 和 resolved denoise steps | `inference_mode`，排除模型 load 与 CPU preprocess；10 warmup + 50 queries × 5 | query latency median/p95、queries/s、peak VRAM；load/preprocess 另计 |
| Fast-WAM model-only | official checkpoint，BF16，`B=1,H=32`、10 denoise steps、replan 24；同一 GPU-resident 三相机输入 | 同一 source/依赖和 inference entry；10 warmup + 50 queries × 5 | query latency median/p95、queries/s、peak VRAM |
| RoboTwin end-to-end | 同一 RoboTwin commit/assets、`adjust_bottle`、固定 8 seeds、同 camera/resolution；1 env 串行主结果，可补固定 8-env throughput | sensor rendering 保留；video encode/file write 关闭；固定 CPU threads/affinity，分别打点 reset/render/sim step/policy/wait | primitive sim steps/s、policy queries/s、query latency median/p95、episode wall、各阶段占比；success 仅作功能 sanity |

GEMM 的 FP8 只在 H100 上用明确的 Transformer Engine/CUTLASS FP8 recipe 做 feature characterization；
A800 记 `N/A`。NCCL 的 N=8 H100 与 N=2 A800 不是纯硬件公平比，主表只比较共同的 N=2，扩展规模单独
展示。HBM 与 P2P 也分表：local HBM 读/写回答显存子系统，all-pairs P2P/NCCL 才回答 fabric。

### 4.3 统一输出与判读规则

每台机器输出同一目录结构：`hardware.json`、`software.json`、`cases.csv`、`telemetry.csv`、
`summary.json` 和原始工具日志。`cases.csv` 至少包含 machine/GPU UUID、case、shape、dtype、batch、
warmup、iteration、latency、throughput；telemetry 记录 timestamp、power、clocks、temperature、memory。

最后只给出以下几类比值，不做一个“总加速比”：

1. GEMM/Transformer 与 model-only 的 compute 比；
2. local HBM 的 bandwidth 比；
3. 同 rank 数、同 topology 类别的 NCCL 比；
4. simulator-only 与完整 episode 的系统比；
5. 各层 H100/A800 median 比值，p95 单列，不用平均值掩盖长尾。

如果 model-only 接近 3×而 episode total 接近 1×，结论应是 simulator-bound，而不是“更快 GPU 无效”；
如果 NCCL 与规格差距大，先看 topology/NUMA/消息大小/软件版本，再讨论 FSDP。任何结论都引用 exact
case，而不把 sparse 峰值、FP8 特性、不同 power cap 或不同 GPU 数混进同一个比值。

## 5. 官方一手入口

- [NVIDIA H100 当前产品规格](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H100 SXM/PCIe 数据表（2022 快照）](https://dam-cdn.nvd.orangelogic.com/AssetLink/mfj81tsm68n0ne632upmuvirso3ta3g3.pdf)
- [NVIDIA H100 PCIe product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf)
- [NVIDIA Hopper architecture in depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA supported GPU device IDs](https://download.nvidia.com/solaris/570.133.07/README/supportedchips.html)
- [Lenovo official OEM A800 product guide](https://lenovopress.lenovo.com/lp1813-thinksystem-nvidia-a800-pcie-gpu)
- [NVIDIA `nvidia-smi` documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [NVIDIA CUTLASS profiler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/profiler.md)
- [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth)
- [NVIDIA nccl-tests performance semantics](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [NVIDIA Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)
- [PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html)
