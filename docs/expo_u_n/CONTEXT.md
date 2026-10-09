# EXPO U / Norm 实施合同

2026-10-09，用户授权：深圳3机物理6/7，先暂停原候补逐卡RLT；实现、针对性测试、最短smoke、推送，再恢复原候补；不启动正式训练。

## 已确认方法

- 基线3685f54df：SZ2成功Clean的缓存/渲染修正版；π0.5、H50/C10、ODE10、8原+8编辑、B64、Q20→FM1→editor1→alpha1、N1采集。2卡复制模型并行；动作专家FP32主参数/BF16计算，视觉语言冻结。
- 信号沿用采集时的父base proposal：U同cast初噪声ODE10/5（仅被选父候选补算ODE5）；Norm原ODE10最后5步×3层post-residual norm。信号不参与选动作。edited j归父base j。
- 每个已执行物理动作仅保存对应位置的标量信号和来源。任意起点H50/C10窗口按同一物理索引取信号，允许跨采集query，不广播未执行尾部。demo无原trace，权重1；在线有信号窗口参与比较域。
- FM-U/FM-Norm：BC-DVCA逐动作内层×H50窗口外层，权重[B,50]广播到原native D32误差，再原式mean；不引入BC的D14 mask或成功chunk数量上限。
- Editor-U/Editor-Norm：C10窗口外层权重[B]乘完整αlogπ−Q；Q、温度α、回放概率、候选排序不变。粒度参考RLT-Q，目标参考DSRL。
- 两种目标可独立开关；FM τ_local=τ_chunk=2.5，Editor仅τ_chunk=2.5。exp/mean映射复用BC函数。
- dropout p=.2，抽中整个监督窗口权重恢复1，不删除、不重归一；私有seed42和已保存update计数。
- 退火按**已完成采集回合**：R1系数1，R200及以后0；FM内外层分别退火，Editor外层退火。预热10回合也计入R。独立开关。

## 验收与运行

- CPU：真实索引对齐、demo中性、全batch/微批梯度、整SAC项重权且alpha不变、R200/dropout恢复与合同兼容。服务器隐藏GPU执行。
- GPU：一次短测进程覆盖U和Norm的真实采集信号与FM/editor更新；复用同一模型加载，保持B64/Q20及2卡；成功即止，不为调度终点重复smoke。若采集没有成功H50，明确FM在线覆盖未验收，不能伪造成功。
- 输出/操作回执：/data/chenyiteng/deployment-20261009/expo-u-n-smoke-v1；源码独立目录；现场配置、命令及停止条件在EXECUTION.md登记后启动。
- 原RLT精确driver/namespace/CP complete/hash保留；CPU owner暂时只保留4/5槽。smoke释放核C/G后，6/7恢复原Clean/Combo各最新完整CP，唯一新namespace，验证owner和进程身份。
- 64GiB byte-LRU复用；不新增逐轨迹反复校验或缓存限制。熵尺度、短H50补齐和正式实验预算均未改动。

源码复用：BC τ映射/双trick、U及Norm producer取自`bc_signal_sweep_20261008/publication`；Norm额外隔离DataParallel replica的hook字典，避免线程共享hook状态。EXPO保留原主采样器，仅增加可选ODE5与Norm观测入口。
