# StarVLA / RoboDojo GPU占用只读核对（2026-09-05）

用户自述管理员并确认对方同意查看；本轮普通账号读取进程/GPU，sudo只读限定进程设备环境与两个启动入口。没有停止、改配置或迁移任何liwenbo进程，没有读数据集、权重内容或凭据；不把这些进程信息提交到BC公开源码分支。

## 现场用途

10:36–10:37，PID201345是RoboDojo/Isaac Sim的`build_tower`、arx_x5、单环境、三相机、seed0、headless评估，策略StarVLA OFT。命令显式`--device cuda:3 --device_id 3`，禁用renderer multiGpu；GPU3约8224MiB且SM24–26%，GPU0还有约2685MiB图形分配；CPU约217%，三路ffmpeg正在编码。因此不是一个只占256MiB、无事可做的空进程。

同轮StarVLA模型服务PID200301在GPU1约9880MiB，模型是RoboDojo OFT step100000。GPU0另有独立LIBERO step50000服务PID916861，约9786MiB，已运行两天多；它不是本次build_tower仿真。该旧服务明确CUDA_VISIBLE_DEVICES=0、idle_timeout=-1，空闲驻留是其启动行为，不能仅凭0%称挂死。

评估PID201345当时在GPU1/2/4/5/6/7各有256MiB CUDA上下文，GPU6/7三次pmon采样均0%计算。10:41管理员限定探针发现PID201345已退出；后续出现同类新PID218322。10:44:57常规检查时两者及对应短期模型服务都不在，GPU1/2/3/6/7已释放，仅旧GPU0服务和Sidney仍在。**本轮没有杀它们**，不猜其自然结束原因或成功率。

原始非管理员快照：[进程与pmon](../rlinf-robotwin-pi0-online-bc/evidence/LIWENBO_GPU_PROCESS_20260905.txt)。

## 是否配置错，能否集中单卡

可以说设备隔离仍有改进空间，不能说“八张卡各256MiB就是八卡训练或必然配错”。已有`device`、`device_id`、renderer multiGpu=false，并非完全没配置；还需把策略模型、物理仿真、渲染器和进程可见设备一起对齐。官方Isaac Sim分别提供active_gpu、physics_gpu、multi_gpu，因此仅限制PyTorch CUDA编号不保证Vulkan/渲染器同卡。[官方SimulationApp接口](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html)

从该次模型约9.65GiB＋主要仿真约8.03GiB＋额外图形分配看，**合并到一张80GB卡很有希望**；这是容量推断，不是实测并发峰值或吞吐保证。独立旧LIBERO服务是否也要并入应单独决定，不能把所有进程都归作同一次评估。

干净做法是在下一次启动合同中统一模型/物理/渲染的目标卡，保持服务端口与客户端一致，然后检查各GPU的实际分配。进程已建立的CUDA对象不能靠事后修改环境变量迁移；通常需要结束本次评估后重新启动，不能无损“搬现有PID”。当前6/7已释放，没有为了回收这点上下文中断有效评估的必要。
