# BC U / Norm τ扫描 · 2026-10-08

本轮授权：收尾SZ1 BC-U、RLT-U-BC、DSRL Clean/U，SZ2 RLT Q-U/Q-Norm，SZ3 BC-Norm、DSRL-Norm共8组，发布轻量记录；8卡fresh启动BC U/Norm各τ1/2/3/4；所有指定卡逐卡RLT候补；清理SZ1本人可再生smoke权重/下载缓存及无恢复引用的非终态大文件。EXPO、WM、共享Ray、其他用户和0–3卡保持。

当前BC基线：Sidney π0.5、move_pillbottle_pad、H50/M10/D14、N8/U5、global1024/micro32、LR2.5e-5、200动作、成功≤3chunk入池、修复指令环境、每5轮固定32条真实评估、300轮。U生产器为ODE10/5，Norm为ODE末5步×深3层残差hidden范数；保持生产器与主动作/RNG。

待执行卡位：SZ1 4=Uτ1、5=Normτ1、6=Uτ2、7=Normτ2；SZ2 6=Uτ3、7=Normτ3；SZ3 6=Uτ4、7=Normτ4。同τ两信号同机，输出/namespace独立，从原SFT与空池开始。

旧U/Norm使用历史5轮校准+bounded_linear[0,5]，没有DVCA两层τ及双trick。本次复用2026-09-21已跑通的DVCA两层log/minmax/exp-mean映射，将raw signal替换为U或Norm。local/chunk温度同为指定τ；τ越小权重差异越大。两trick沿旧BC定义：20% sampled chunk恢复等权；local/chunk alpha在第1→200轮从1→0，第200轮后等权。它是新映射实验，不能称只调旧U/Norm的一个参数。

先核checkpoint/目标身份和归还控制，再终止；本次切换期间RLT候补不得启动。新任务命令/环境/模型/种子/卡位及图形scope写入各卡manifest。小范围CPU合同检查通过后直接正式，验收首轮采集、一次真实更新和权重差异；不追加长smoke。训练非有限/绑定越界则精确停止该卡任务并记录；释放核验后同卡RLT才接替。

RLT候补只使用原卡原任务已保存断点、原预算与原配置。单卡优先任务退出并清理其namespace/进程后才能接替；未知占用时等待，不结束别人进程。EXPO与WM本身是两卡任务，它们所占卡须等该任务退出释放，但不阻塞同机其他卡。

清理保留最新完整可恢复checkpoint、评估需要的checkpoint、全部日志/配置/指标/轻量证据；先写精确路径+大小+引用核对清单再删。不按模糊通配删除，不删除在跑任务输入/模型/数据。

状态：源码与配置准备中；00:37固定host-key/chenyiteng身份及8组现场已刷新。动态操作结果另记EXECUTION.md。
