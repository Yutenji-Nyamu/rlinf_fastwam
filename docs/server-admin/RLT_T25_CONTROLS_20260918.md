# SZ1 RLT τ2.5 controls / 2026-09-18

用户授权：基于τ2.5，GPU6只加dropout0.2，GPU7只加退火至R500为0；简要测试、推送、跳过GPU smoke直接fresh正式训练。

13:10:54/57完成逐卡切换；释放设备至提交新启动分别0.140/0.136秒（模型加载另需时间）。0–5卡原GPU进程均保持存活，共享Ray未重启。
两组均完成前2轮teacher采集，各12 actors存活、实际配置diff为空、无fatal。尚在teacher阶段，方法更新尚未开始；方法行为通过CPU worker/梯度测试验证。

|GPU|配置|轮数|
|---|---|---|
|6|τ2.5；成功倍率1；仅成功chunk dropout0.2|fresh800|
|7|τ2.5；成功倍率1；内外α R1=1线性→R500=0；无dropout|fresh800|

保持Clean4的4env/轮、B512/micro256/U5、critic_actor_ratio2、学习率1e-4、80k池、10k预热池/15k初始化更新、Stage1 CP2000、原train/eval种子、每25轮20条固定评估及保存。原Clean4为600轮，累计已续到800；本次沿用τ2.5的fresh800预算。
退火按总runner轮次，包含teacher收集；温度τ不退火。Dropout在整批replay权重生成后、切microbatch前，让被选成功query的10个位置统一回到1；不删除BC梯度、不缩放1/(1-p)、不重新归一化；Q/失败BC不变。
随机种子使用独立生成器结合持久化update_step，避免扰动数据或模型RNG；resume已有方法配置契约拒绝控制不一致。

实现提交：c42cbd500d945593f707350f6cdbc4a4600d5d5c。94 CPU tests passed。复用GRPO控制语义，适配RLT B×H replay域；旧AST测试夹具同步引入新函数。
源码工作树：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-t25-controls-half4-800-20260918`。
维护入口：`/data/chenyiteng/results/server-maintenance-20260918/rlt-t25-controls/ops.py`。
配置、命令、测试日志和首轮回执见同目录evidence/rlt-t25-controls-20260918。运行至800或用户明确停止/不可恢复错误，不设性能自动中止。

旧τ2.5/τ3分别结束于R637/R634，均保留CP625；收尾与全族8实验ZIP推送到原τ2.5/τ3分支，服务器权重/经验池未删除。
