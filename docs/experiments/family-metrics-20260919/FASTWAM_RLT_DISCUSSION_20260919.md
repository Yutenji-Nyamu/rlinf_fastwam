# FastWAM RLT：已有实现与本次候选

09-19已读取SZ1实际源码：fastwam-rlt-20260913 HEAD 4a42c30de9de8e0c19e0156a40b834a0ad21e8ab。用户修正：现役GPU4 τ2.5成功倍率2先继续，讨论决定后才撤，FastWAM本轮不启动。

已有实现与09-13短验收：取FastWAM首帧video expert最终hidden（120×3072），原两层RLT encoder与causal AR decoder；z=2048。Stage2冻结FastWAM和Stage1 encoder，训练小student/Q，保留原BC/Q/回放机制。真实在线smoke3轮通过，但完整Stage1及正式RLT从未验收；不能借FastWAM BC结果代替RLT效果。

|项目|旧FastWAM RLT正式候选|这次建议|
|---|---|---|
|任务|adjust_bottle|保持，与π0 RLT同任务|
|轮数/采样|600/每轮8|800/每轮4，继承当前Clean4|
|Replay/初始池/初始化更新|80k/20k/30k|80k/10k/15k|
|B/micro/UTD/每轮更新上限|512/256/5/1600|512/256/5/800|
|BC/Q课程|旧20k+50k|当前10k+25k；BC7→2.5，Q0.05→0.45；critic:actor=2|
|student/Q优化|RLT设置|LR1e-4 constant，clip10，gamma.99，target tau.005|
|模型原生接口|H32/C24/D14/M10，shift5，bf16|保持；teacher前向micro1|
|动作坐标|FastWAM z-score，student identity|保持；不能直接套π0 tanh或归一化|
|执行上限|192动作（8×24）|保持模型整除适配|
|Stage1|clean50，2000更新，B32/micro1，LR2.5e-5|必须新训FastWAM专属Stage1；不能复用π0权重或3步smoke|
|DVAC|Clean无DVAC|建议先无DVAC；先验证模型迁移，再比较τ2.5|

方法部分参考当前π0 Clean4，模型接口参考已跑通FastWAM，用户的理解正确。注意C10→24减少每条轨迹query次数，所以相同4轨迹不等于相同回放新增量/teacher调用量；到10k初始池所需轮次也可能更晚。不可直接套π0的800轮耗时预估。

Stage1训练的是压缩器，不微调FastWAM大模型。Stage2也只训小student/Q；这一点与之前FastWAM BC训练action expert不同。真正要补的是完整Stage1、当前Clean4配置迁移和训练入口；已有模型/环境/适配实现可复用。

读取证据：E同任务experiment-refresh-20260918-1420/sz1/fastwam-live-0919.json。历史验证详见FASTWAM_IMPLEMENTATION_20260913.md。以上为候选讨论，不代表用户已确定800轮、4采样或授权启动。
