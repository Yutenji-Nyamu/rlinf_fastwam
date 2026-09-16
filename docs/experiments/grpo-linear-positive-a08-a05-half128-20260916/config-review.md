# GRPO128 linear positive alpha0.8/0.5 配置核对

2026-09-16，继承已跑通的positive τ1源码和Clean128实际预算。

最终有效值为`mapping=linear_centered`、`scope=positive`、`alpha_local=0.8`、`alpha_chunk=0.5`。线性分支忽略继承的temperature_local/chunk=1；旧strength/warmup/window字段同样不参与两层路径。

新recipe为examples/embodiment/config/dvac_grpo/adv_linear_positive_a08_a05.yaml。与现役positive τ1在身份路径归一后，只有mapping和两α共3叶差异；scope原本已positive。与Clean128全部差异均为方法或身份路径，unexpected_nonmethod_diff为空。训练/评估seed文件SHA与Clean相同。

不变项：原Sidney π0.5模型、move_pillbottle_pad；fresh200，无resume；64环境×2=128轨迹/轮、G8；最多200动作、H50、14维、M10、noise.5、flow_sde；B512/micro32、U2、LR5e-6、grad clip1、PPO clip.2；rollout42/43、actor1234、env train/eval0；fixed32每5轮、保存每10轮；物理GPU4/5。只对A>0额外加权，失败保持原GRPO。

服务器97项已有CPU检查通过。首次v1误用了不被实现接受的枚举`linear`；这些既有测试没有从新增recipe解析该值，因此首启在actor构造时被生产契约拒绝，exit255，未完成首轮。失败run、日志和初次切换回执保留。

修正只将recipe值改为`linear_centered`，不改算法源码。新增的执行前检查从实际recipe读取mapping/α/scope，调用生产dvac_mapping_contract和compute_dvac_two_level_weights，验证正侧非均权、失败侧逐项1及全部有限。随后准备独立v2 run/namespace，保留预算并重新启动；不重放旧实验停止操作。

当前正式run以retry-v2的specs、contract、STARTUP_VERIFIED及发布回执为准。v1清卡到发起进程0.185秒仅描述第一次进程切换，不能当作最终正常运行的全部空闲时间；v1发起到v2发起相隔约6分33秒，需在最终交付中说明配置错误带来的重启延迟。
